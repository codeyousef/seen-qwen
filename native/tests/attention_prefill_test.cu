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
constexpr uint64_t kTokens = 3;
constexpr uint64_t kQueryHeads = 24;
constexpr uint64_t kKvHeads = 4;
constexpr uint64_t kHeadDim = 256;
constexpr uint64_t kCapacity = 7;
constexpr uint64_t kQueryWidth = kQueryHeads * kHeadDim;
constexpr uint64_t kKvWidth = kKvHeads * kHeadDim;
constexpr uint64_t kQueryCount = kTokens * kQueryWidth;
constexpr uint64_t kInputCount = kTokens * kKvWidth;
constexpr uint64_t kCacheCount = kCapacity * kKvWidth;

SeenQwenCudaBufferView view(void *address, uint64_t bytes, int32_t device = 0) {
    return SeenQwenCudaBufferView{SEEN_QWEN_CUDA_REFERENCE_ABI_VERSION,
        device, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address)), bytes};
}

SeenQwenCudaBufferView offset_view(void *address, uint64_t element_offset,
                                   uint64_t element_count) {
    auto *base = static_cast<float *>(address);
    return view(base + element_offset, element_count * sizeof(float));
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
    for (uint64_t token = 0; token < kTokens; ++token) {
        for (uint64_t head = 0; head < kQueryHeads; ++head) {
            const uint64_t kv_head = head / repeat;
            std::vector<float> scores(token + 1);
            float maximum = 0.0f;
            for (uint64_t position = 0; position <= token; ++position) {
                float dot = 0.0f;
                const uint64_t cache_base =
                    (position * kKvHeads + kv_head) * kHeadDim;
                const uint64_t query_base =
                    (token * kQueryHeads + head) * kHeadDim;
                for (uint64_t dimension = 0; dimension < kHeadDim; ++dimension)
                    dot += query[query_base + dimension] *
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
                for (uint64_t position = 0; position <= token; ++position) {
                    const uint64_t index =
                        (position * kKvHeads + kv_head) * kHeadDim + dimension;
                    sum += scores[position] / denominator * values[index];
                }
                (*output)[(token * kQueryHeads + head) * kHeadDim + dimension] = sum;
            }
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
    const uint64_t input_bytes = kInputCount * sizeof(float);
    const uint64_t cache_bytes = kCacheCount * sizeof(float);
    SeenCudaHandle stream = 0, event = 0, allocations[9]{};
    CHECK_STATUS(seen_cuda_stream_create(0, &stream));
    CHECK_STATUS(seen_cuda_event_create(0, &event));
    const uint64_t sizes[9] = {query_bytes, input_bytes, input_bytes,
        cache_bytes, cache_bytes, query_bytes, cache_bytes, cache_bytes,
        query_bytes};
    void *device[9]{};
    for (int index = 0; index < 9; ++index) {
        CHECK_STATUS(seen_cuda_malloc(0, sizes[index], &allocations[index]));
        uint64_t bytes = 0; int32_t ordinal = -1;
        CHECK_STATUS(seen_cuda_allocation_address(
            allocations[index], &device[index], &bytes, &ordinal));
        CHECK(bytes == sizes[index] && ordinal == 0);
    }

    std::vector<float> query(kQueryCount), keys(kInputCount), values(kInputCount);
    std::vector<float> expected(kQueryCount), actual(kQueryCount), chunked(kQueryCount);
    std::vector<float> sentinel(kCacheCount, 91.0f), key_cache(kCacheCount);
    std::vector<float> value_cache(kCacheCount), chunk_keys(kCacheCount);
    std::vector<float> chunk_values(kCacheCount);
    for (uint64_t index = 0; index < query.size(); ++index)
        query[index] = static_cast<float>(static_cast<int64_t>((index * 13) % 67) - 33) / 31.0f;
    for (uint64_t index = 0; index < keys.size(); ++index) {
        keys[index] = static_cast<float>(static_cast<int64_t>((index * 17) % 89) - 44) / 29.0f;
        values[index] = static_cast<float>(static_cast<int64_t>((index * 23) % 97) - 48) / 37.0f;
    }
    reference(query, keys, values, &expected);
    CHECK_STATUS(seen_cuda_memcpy_async(device[0], query.data(), query_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[1], keys.data(), input_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[2], values.data(), input_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    for (int index : {3, 4, 6, 7})
        CHECK_STATUS(seen_cuda_memcpy_async(device[index], sentinel.data(), cache_bytes,
            SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));

    auto token = borrow(stream);
    CHECK_STATUS(seen_qwen_attention_prefill_f32(&token,
        view(device[0], query_bytes), view(device[1], input_bytes),
        view(device[2], input_bytes), view(device[3], cache_bytes),
        view(device[4], cache_bytes), view(device[5], query_bytes), kTokens,
        kQueryHeads, kKvHeads, kHeadDim, 0, kCapacity));

    // Arbitrary chunking preserves causal outputs and the exact cache state.
    token = borrow(stream);
    CHECK_STATUS(seen_qwen_attention_prefill_f32(&token,
        offset_view(device[0], 0, kQueryWidth),
        offset_view(device[1], 0, kKvWidth),
        offset_view(device[2], 0, kKvWidth), view(device[6], cache_bytes),
        view(device[7], cache_bytes), offset_view(device[8], 0, kQueryWidth),
        1, kQueryHeads, kKvHeads, kHeadDim, 0, kCapacity));
    token = borrow(stream);
    CHECK_STATUS(seen_qwen_attention_prefill_f32(&token,
        offset_view(device[0], kQueryWidth, 2 * kQueryWidth),
        offset_view(device[1], kKvWidth, 2 * kKvWidth),
        offset_view(device[2], kKvWidth, 2 * kKvWidth),
        view(device[6], cache_bytes), view(device[7], cache_bytes),
        offset_view(device[8], kQueryWidth, 2 * kQueryWidth), 2,
        kQueryHeads, kKvHeads, kHeadDim, 1, kCapacity));
    CHECK_STATUS(seen_cuda_memcpy_async(actual.data(), device[5], query_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(chunked.data(), device[8], query_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(key_cache.data(), device[3], cache_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(value_cache.data(), device[4], cache_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(chunk_keys.data(), device[6], cache_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(chunk_values.data(), device[7], cache_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (uint64_t index = 0; index < kQueryCount; ++index) {
        CHECK(std::isfinite(actual[index]));
        CHECK(near(actual[index], expected[index]));
        CHECK(near(chunked[index], actual[index]));
    }
    for (uint64_t index = 0; index < kInputCount; ++index) {
        CHECK(key_cache[index] == keys[index]);
        CHECK(value_cache[index] == values[index]);
        CHECK(chunk_keys[index] == key_cache[index]);
        CHECK(chunk_values[index] == value_cache[index]);
    }
    for (uint64_t index = kInputCount; index < kCacheCount; ++index) {
        CHECK(key_cache[index] == 91.0f);
        CHECK(value_cache[index] == 91.0f);
        CHECK(chunk_keys[index] == 91.0f);
        CHECK(chunk_values[index] == 91.0f);
    }

    // Geometry, extent, alias, device, and token failures reject before launch.
    token = borrow(stream);
    auto call = [&](uint64_t count, uint64_t query_heads, uint64_t kv_heads,
                    uint64_t start, uint64_t capacity,
                    SeenQwenCudaBufferView output) {
        return seen_qwen_attention_prefill_f32(&token,
            view(device[0], query_bytes), view(device[1], input_bytes),
            view(device[2], input_bytes), view(device[3], cache_bytes),
            view(device[4], cache_bytes), output, count, query_heads, kv_heads,
            kHeadDim, start, capacity);
    };
    CHECK(call(0, kQueryHeads, kKvHeads, 0, kCapacity,
        view(device[5], query_bytes)).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(call(kTokens, 23, kKvHeads, 0, kCapacity,
        view(device[5], query_bytes)).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(call(kTokens, kQueryHeads, kKvHeads, kCapacity, kCapacity,
        view(device[5], query_bytes)).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(call(std::numeric_limits<uint64_t>::max(), kQueryHeads, kKvHeads,
        0, kCapacity, view(device[5], query_bytes)).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(call(kTokens, kQueryHeads, kKvHeads, 0, kCapacity,
        view(device[0], query_bytes)).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(call(kTokens, kQueryHeads, kKvHeads, 0, kCapacity,
        view(device[5], query_bytes - sizeof(float))).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_attention_prefill_f32(&token,
        view(device[0], query_bytes, 1), view(device[1], input_bytes),
        view(device[2], input_bytes), view(device[3], cache_bytes),
        view(device[4], cache_bytes), view(device[5], query_bytes), kTokens,
        kQueryHeads, kKvHeads, kHeadDim, 0,
        kCapacity).code == SEEN_CUDA_INVALID_ARGUMENT);
    SeenCudaStreamLaunchToken incompatible = token;
    incompatible.generation = 0;
    CHECK(seen_qwen_attention_prefill_f32(&incompatible,
        view(device[0], query_bytes), view(device[1], input_bytes),
        view(device[2], input_bytes), view(device[3], cache_bytes),
        view(device[4], cache_bytes), view(device[5], query_bytes), kTokens,
        kQueryHeads, kKvHeads, kHeadDim, 0,
        kCapacity).code == SEEN_CUDA_INCOMPATIBLE);

    CHECK_STATUS(seen_cuda_graph_begin_capture(stream));
    token = borrow(stream);
    CHECK_STATUS(seen_qwen_attention_prefill_f32(&token,
        view(device[0], query_bytes), view(device[1], input_bytes),
        view(device[2], input_bytes), view(device[3], cache_bytes),
        view(device[4], cache_bytes), view(device[5], query_bytes), kTokens,
        kQueryHeads, kKvHeads, kHeadDim, 0, kCapacity));
    SeenCudaHandle graph = 0, graph_exec = 0;
    CHECK_STATUS(seen_cuda_graph_end_capture(stream, &graph));
    CHECK_STATUS(seen_cuda_graph_instantiate(graph, &graph_exec));
    CHECK_STATUS(seen_cuda_graph_launch(graph_exec, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(actual.data(), device[5], query_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (uint64_t index = 0; index < kQueryCount; ++index)
        CHECK(near(actual[index], expected[index]));
    CHECK_STATUS(seen_cuda_graph_exec_destroy(&graph_exec));
    CHECK_STATUS(seen_cuda_graph_destroy(&graph));

    for (int index = 8; index >= 0; --index)
        CHECK_STATUS(seen_cuda_free(&allocations[index]));
    CHECK_STATUS(seen_cuda_event_destroy(&event));
    CHECK_STATUS(seen_cuda_stream_destroy(&stream));
    std::printf("PASS: FEL-1443 causal attention prefill chunk equivalence cleanup\n");
    return 0;
}
