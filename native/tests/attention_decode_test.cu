#include "seen_qwen_cuda.h"

#include <cmath>
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
constexpr uint64_t kQueryHeads = 24;
constexpr uint64_t kKvHeads = 4;
constexpr uint64_t kHeadDim = 256;
constexpr uint64_t kCacheLength = 5;
constexpr uint64_t kCacheCapacity = 7;
constexpr uint64_t kQueryCount = kQueryHeads * kHeadDim;
constexpr uint64_t kCacheCount = kCacheCapacity * kKvHeads * kHeadDim;
constexpr uint64_t kLiveCount = kCacheLength * kKvHeads * kHeadDim;

SeenQwenCudaBufferView view(void *address, uint64_t bytes, int32_t device = 0) {
    return SeenQwenCudaBufferView{SEEN_QWEN_CUDA_REFERENCE_ABI_VERSION,
        device, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address)), bytes};
}

SeenCudaStreamLaunchToken borrow(SeenCudaHandle stream) {
    SeenCudaStreamLaunchToken token{};
    const SeenCudaStatus status = seen_cuda_stream_borrow_launch_token(stream, 0, &token);
    if (status.code != SEEN_CUDA_OK) std::abort();
    return token;
}

void reference(const std::vector<float> &query, const std::vector<float> &keys,
               const std::vector<float> &values, std::vector<float> *output) {
    const uint64_t repeat = kQueryHeads / kKvHeads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    for (uint64_t head = 0; head < kQueryHeads; ++head) {
        const uint64_t kv_head = head / repeat;
        std::vector<float> scores(kCacheLength);
        float maximum = 0.0f;
        for (uint64_t position = 0; position < kCacheLength; ++position) {
            float dot = 0.0f;
            const uint64_t cache_base = (position * kKvHeads + kv_head) * kHeadDim;
            for (uint64_t dimension = 0; dimension < kHeadDim; ++dimension)
                dot += query[head * kHeadDim + dimension] *
                    keys[cache_base + dimension];
            scores[position] = dot * scale;
            if (position == 0 || scores[position] > maximum)
                maximum = scores[position];
        }
        float denominator = 0.0f;
        for (float &score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        for (uint64_t dimension = 0; dimension < kHeadDim; ++dimension) {
            float sum = 0.0f;
            for (uint64_t position = 0; position < kCacheLength; ++position) {
                const uint64_t index =
                    (position * kKvHeads + kv_head) * kHeadDim + dimension;
                sum += (scores[position] / denominator) * values[index];
            }
            (*output)[head * kHeadDim + dimension] = sum;
        }
    }
}

bool near(float actual, float expected) {
    return std::fabs(actual - expected) <= 3.0e-4f +
        3.0e-4f * std::fabs(expected);
}
}  // namespace

int main() {
    const uint64_t query_bytes = kQueryCount * sizeof(float);
    const uint64_t cache_bytes = kCacheCount * sizeof(float);
    SeenCudaHandle stream = 0, event = 0, allocations[4]{};
    CHECK_STATUS(seen_cuda_stream_create(0, &stream));
    CHECK_STATUS(seen_cuda_event_create(0, &event));
    const uint64_t sizes[4] = {query_bytes, cache_bytes, cache_bytes, query_bytes};
    for (int index = 0; index < 4; ++index)
        CHECK_STATUS(seen_cuda_malloc(0, sizes[index], &allocations[index]));
    void *device[4]{};
    for (int index = 0; index < 4; ++index) {
        uint64_t bytes = 0; int32_t ordinal = -1;
        CHECK_STATUS(seen_cuda_allocation_address(
            allocations[index], &device[index], &bytes, &ordinal));
        CHECK(bytes == sizes[index] && ordinal == 0);
    }

    std::vector<float> query(kQueryCount), keys(kCacheCount, 91.0f);
    std::vector<float> values(kCacheCount, -73.0f), expected(kQueryCount);
    std::vector<float> actual(kQueryCount);
    for (uint64_t index = 0; index < query.size(); ++index)
        query[index] = static_cast<float>(static_cast<int64_t>((index * 13) % 67) - 33) / 31.0f;
    for (uint64_t index = 0; index < kLiveCount; ++index) {
        keys[index] = static_cast<float>(static_cast<int64_t>((index * 17) % 89) - 44) / 29.0f;
        values[index] = static_cast<float>(static_cast<int64_t>((index * 23) % 97) - 48) / 37.0f;
    }
    reference(query, keys, values, &expected);
    CHECK_STATUS(seen_cuda_memcpy_async(device[0], query.data(), query_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[1], keys.data(), cache_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[2], values.data(), cache_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    auto token = borrow(stream);
    CHECK_STATUS(seen_qwen_attention_decode_f32(&token,
        view(device[0], query_bytes), view(device[1], cache_bytes),
        view(device[2], cache_bytes), view(device[3], query_bytes),
        kQueryHeads, kKvHeads, kHeadDim, kCacheLength, kCacheCapacity));
    CHECK_STATUS(seen_cuda_memcpy_async(actual.data(), device[3], query_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (uint64_t index = 0; index < kQueryCount; ++index) {
        CHECK(std::isfinite(actual[index]));
        CHECK(near(actual[index], expected[index]));
    }

    // Geometry, extent, alias, device, and token failures reject before launch.
    token = borrow(stream);
    auto call = [&](uint64_t query_heads, uint64_t kv_heads,
                    uint64_t head_dim, uint64_t cache_length,
                    uint64_t capacity, SeenQwenCudaBufferView output) {
        return seen_qwen_attention_decode_f32(&token,
            view(device[0], query_bytes), view(device[1], cache_bytes),
            view(device[2], cache_bytes), output, query_heads, kv_heads,
            head_dim, cache_length, capacity);
    };
    CHECK(call(23, kKvHeads, kHeadDim, kCacheLength, kCacheCapacity,
        view(device[3], query_bytes)).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(call(kQueryHeads, kKvHeads, kHeadDim, 0, kCacheCapacity,
        view(device[3], query_bytes)).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(call(kQueryHeads, kKvHeads, kHeadDim, kCacheCapacity + 1,
        kCacheCapacity, view(device[3], query_bytes)).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(call(std::numeric_limits<uint64_t>::max(), 1, kHeadDim, 1, 1,
        view(device[3], query_bytes)).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(call(kQueryHeads, kKvHeads, kHeadDim, kCacheLength, kCacheCapacity,
        view(device[0], query_bytes)).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(call(kQueryHeads, kKvHeads, kHeadDim, kCacheLength, kCacheCapacity,
        view(device[3], query_bytes - sizeof(float))).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_attention_decode_f32(&token,
        view(device[0], query_bytes, 1), view(device[1], cache_bytes),
        view(device[2], cache_bytes), view(device[3], query_bytes),
        kQueryHeads, kKvHeads, kHeadDim, kCacheLength,
        kCacheCapacity).code == SEEN_CUDA_INVALID_ARGUMENT);
    SeenCudaStreamLaunchToken incompatible = token;
    incompatible.generation = 0;
    CHECK(seen_qwen_attention_decode_f32(&incompatible,
        view(device[0], query_bytes), view(device[1], cache_bytes),
        view(device[2], cache_bytes), view(device[3], query_bytes),
        kQueryHeads, kKvHeads, kHeadDim, kCacheLength,
        kCacheCapacity).code == SEEN_CUDA_INCOMPATIBLE);

    // Capture and replay the exact same borrowed-stream decode launch.
    CHECK_STATUS(seen_cuda_graph_begin_capture(stream));
    token = borrow(stream);
    CHECK_STATUS(seen_qwen_attention_decode_f32(&token,
        view(device[0], query_bytes), view(device[1], cache_bytes),
        view(device[2], cache_bytes), view(device[3], query_bytes),
        kQueryHeads, kKvHeads, kHeadDim, kCacheLength, kCacheCapacity));
    SeenCudaHandle graph = 0, graph_exec = 0;
    CHECK_STATUS(seen_cuda_graph_end_capture(stream, &graph));
    CHECK_STATUS(seen_cuda_graph_instantiate(graph, &graph_exec));
    CHECK_STATUS(seen_cuda_graph_launch(graph_exec, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(actual.data(), device[3], query_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (uint64_t index = 0; index < kQueryCount; ++index)
        CHECK(near(actual[index], expected[index]));
    CHECK_STATUS(seen_cuda_graph_exec_destroy(&graph_exec));
    CHECK_STATUS(seen_cuda_graph_destroy(&graph));

    for (int index = 3; index >= 0; --index)
        CHECK_STATUS(seen_cuda_free(&allocations[index]));
    CHECK_STATUS(seen_cuda_event_destroy(&event));
    CHECK_STATUS(seen_cuda_stream_destroy(&stream));
    std::printf("PASS: FEL-1440 official grouped-query stable attention decode cleanup\n");
    return 0;
}
