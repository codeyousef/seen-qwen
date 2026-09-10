#include "seen_qwen_cuda.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#define CHECK_STATUS(expr) do { SeenCudaStatus s_ = (expr); if (s_.code != SEEN_CUDA_OK) { \
    std::fprintf(stderr, "FAIL:%d: %s code=%d native=%d op=%s message=%s\n", \
        __LINE__, #expr, s_.code, s_.native_code, s_.operation, s_.message); return 1; } } while (0)
#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "FAIL:%d: %s\n", __LINE__, #expr); return 1; } } while (0)

namespace {
constexpr uint64_t kKvHeads = 4;
constexpr uint64_t kHeadDim = 256;
constexpr uint64_t kCapacity = 8;
constexpr uint64_t kTokenWidth = kKvHeads * kHeadDim;
constexpr uint64_t kCacheElements = kCapacity * kTokenWidth;
constexpr uint64_t kCacheBytes = kCacheElements * sizeof(float);

SeenQwenCudaBufferView view(void *address, uint64_t bytes, int32_t device = 0) {
    return SeenQwenCudaBufferView{SEEN_QWEN_CUDA_REFERENCE_ABI_VERSION,
        device, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address)), bytes};
}

SeenCudaStreamLaunchToken borrow(SeenCudaHandle stream) {
    SeenCudaStreamLaunchToken token{};
    const SeenCudaStatus result = seen_cuda_stream_borrow_launch_token(stream, 0, &token);
    if (result.code != SEEN_CUDA_OK) std::abort();
    return token;
}

struct CacheOwner {
    SeenCudaHandle keys = 0;
    SeenCudaHandle values = 0;
    void *key_data = nullptr;
    void *value_data = nullptr;
    uint64_t used = 0;

    int open() {
        SeenCudaStatus status = seen_cuda_malloc(0, kCacheBytes, &keys);
        if (status.code != SEEN_CUDA_OK) return status.code;
        status = seen_cuda_malloc(0, kCacheBytes, &values);
        if (status.code != SEEN_CUDA_OK) {
            (void)seen_cuda_free(&keys);
            return status.code;
        }
        uint64_t bytes = 0; int32_t device = -1;
        status = seen_cuda_allocation_address(keys, &key_data, &bytes, &device);
        if (status.code != SEEN_CUDA_OK || bytes != kCacheBytes || device != 0) return 1;
        status = seen_cuda_allocation_address(values, &value_data, &bytes, &device);
        return status.code == SEEN_CUDA_OK && bytes == kCacheBytes && device == 0 ? 0 : 1;
    }

    SeenCudaStatus append(const SeenCudaStreamLaunchToken *token,
                          SeenQwenCudaBufferView key,
                          SeenQwenCudaBufferView value, uint64_t token_count,
                          uint64_t start_position) {
        if (start_position != used || token_count == 0 ||
            used > kCapacity || token_count > kCapacity - used) {
            SeenCudaStatus rejected{};
            rejected.code = SEEN_CUDA_INVALID_ARGUMENT;
            return rejected;
        }
        const SeenCudaStatus status = seen_qwen_kv_append_f32(token, key, value,
            view(key_data, kCacheBytes), view(value_data, kCacheBytes),
            token_count, kKvHeads, kHeadDim, start_position, kCapacity);
        if (status.code == SEEN_CUDA_OK) used += token_count;
        return status;
    }

    void reset() { used = 0; }

    int close() {
        SeenCudaStatus value_status = seen_cuda_free(&values);
        SeenCudaStatus key_status = seen_cuda_free(&keys);
        key_data = nullptr; value_data = nullptr; used = 0;
        return value_status.code == SEEN_CUDA_OK && key_status.code == SEEN_CUDA_OK ? 0 : 1;
    }
};
}  // namespace

int main() {
    SeenCudaHandle stream = 0, event = 0, input_keys = 0, input_values = 0;
    CHECK_STATUS(seen_cuda_stream_create(0, &stream));
    CHECK_STATUS(seen_cuda_event_create(0, &event));
    CHECK_STATUS(seen_cuda_malloc(0, 3 * kTokenWidth * sizeof(float), &input_keys));
    CHECK_STATUS(seen_cuda_malloc(0, 3 * kTokenWidth * sizeof(float), &input_values));
    void *key_input = nullptr, *value_input = nullptr;
    uint64_t bytes = 0; int32_t device = -1;
    CHECK_STATUS(seen_cuda_allocation_address(input_keys, &key_input, &bytes, &device));
    CHECK_STATUS(seen_cuda_allocation_address(input_values, &value_input, &bytes, &device));
    CacheOwner cache; CHECK(cache.open() == 0);
    void *original_keys = cache.key_data, *original_values = cache.value_data;

    std::vector<float> keys(3 * kTokenWidth), values(3 * kTokenWidth);
    for (uint64_t index = 0; index < keys.size(); ++index) {
        keys[index] = static_cast<float>(static_cast<int64_t>(index % 257) - 128) / 19.0f;
        values[index] = -keys[index] * 0.5f;
    }
    CHECK_STATUS(seen_cuda_memcpy_async(key_input, keys.data(), keys.size() * sizeof(float),
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(value_input, values.data(), values.size() * sizeof(float),
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));

    auto token = borrow(stream);
    CHECK_STATUS(cache.append(&token, view(key_input, keys.size() * sizeof(float)),
        view(value_input, values.size() * sizeof(float)), 3, 0));
    CHECK(cache.used == 3);

    // A rejected non-sequential or overflowing update cannot advance ownership state.
    token = borrow(stream);
    CHECK(cache.append(&token, view(key_input, keys.size() * sizeof(float)),
        view(value_input, values.size() * sizeof(float)), 1, 2).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(cache.used == 3);
    CHECK(cache.append(&token, view(key_input, keys.size() * sizeof(float)),
        view(value_input, values.size() * sizeof(float)), 6, 3).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(cache.used == 3);
    SeenCudaStreamLaunchToken incompatible = token; incompatible.generation = 0;
    CHECK(cache.append(&incompatible, view(key_input, keys.size() * sizeof(float)),
        view(value_input, values.size() * sizeof(float)), 1, 3).code == SEEN_CUDA_INCOMPATIBLE);
    CHECK(cache.used == 3);

    // Append two more official-geometry positions on the same Seen-owned stream.
    token = borrow(stream);
    CHECK_STATUS(cache.append(&token,
        view(static_cast<float *>(key_input), 2 * kTokenWidth * sizeof(float)),
        view(static_cast<float *>(value_input), 2 * kTokenWidth * sizeof(float)), 2, 3));
    CHECK(cache.used == 5);
    std::vector<float> actual(kCacheElements, 777.0f);
    CHECK_STATUS(seen_cuda_memcpy_async(actual.data(), cache.key_data,
        5 * kTokenWidth * sizeof(float),
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (uint64_t index = 0; index < 3 * kTokenWidth; ++index)
        CHECK(actual[index] == keys[index]);
    for (uint64_t index = 0; index < 2 * kTokenWidth; ++index)
        CHECK(actual[3 * kTokenWidth + index] == keys[index]);

    // Reset preserves stable allocations; graph capture/replay rewrites position zero.
    cache.reset();
    CHECK(cache.used == 0 && cache.key_data == original_keys &&
        cache.value_data == original_values);
    CHECK_STATUS(seen_cuda_graph_begin_capture(stream)); token = borrow(stream);
    CHECK_STATUS(cache.append(&token, view(key_input, kTokenWidth * sizeof(float)),
        view(value_input, kTokenWidth * sizeof(float)), 1, 0));
    SeenCudaHandle graph = 0, graph_exec = 0;
    CHECK_STATUS(seen_cuda_graph_end_capture(stream, &graph));
    CHECK_STATUS(seen_cuda_graph_instantiate(graph, &graph_exec));
    CHECK_STATUS(seen_cuda_graph_launch(graph_exec, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    CHECK(cache.used == 1);
    CHECK_STATUS(seen_cuda_graph_exec_destroy(&graph_exec));
    CHECK_STATUS(seen_cuda_graph_destroy(&graph));

    CHECK(cache.close() == 0);
    CHECK(cache.close() == 0);
    CHECK_STATUS(seen_cuda_free(&input_values));
    CHECK_STATUS(seen_cuda_free(&input_keys));
    CHECK_STATUS(seen_cuda_event_destroy(&event));
    CHECK_STATUS(seen_cuda_stream_destroy(&stream));
    std::printf("PASS: FEL-1437 Seen-owned deterministic KV cache update reset capture cleanup\n");
    return 0;
}
