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
constexpr uint64_t kTokens = 2;
constexpr uint64_t kQueryHeads = 24;
constexpr uint64_t kKvHeads = 4;
constexpr uint64_t kHeadDim = 256;
constexpr uint64_t kRotaryDim = 64;
constexpr uint64_t kPositionOffset = 262142;
constexpr uint64_t kMaxPosition = 262144;
constexpr float kTheta = 10000000.0f;
constexpr float kEpsilon = 1.0e-6f;

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

void reference_rows(const float *input, const float *weight, float *output,
                    uint64_t tokens, uint64_t heads, uint64_t head_dim,
                    uint64_t rotary_dim, uint64_t position_offset,
                    float theta, float epsilon, bool packed_query,
                    float *gate) {
    for (uint64_t token = 0; token < tokens; ++token) {
        for (uint64_t head = 0; head < heads; ++head) {
            const uint64_t row_index = token * heads + head;
            const float *row = input + row_index * head_dim * (packed_query ? 2 : 1);
            float sum = 0.0f;
            for (uint64_t dimension = 0; dimension < head_dim; ++dimension)
                sum += row[dimension] * row[dimension];
            const float inverse = 1.0f / std::sqrt(sum / static_cast<float>(head_dim) + epsilon);
            for (uint64_t dimension = 0; dimension < head_dim; ++dimension) {
                const uint64_t index = row_index * head_dim + dimension;
                const float direct = row[dimension] * inverse * (1.0f + weight[dimension]);
                if (dimension >= rotary_dim) {
                    output[index] = direct;
                } else {
                    const uint64_t half = rotary_dim / 2;
                    const uint64_t paired = dimension < half
                        ? dimension + half : dimension - half;
                    float rotated = row[paired] * inverse * (1.0f + weight[paired]);
                    if (dimension < half) rotated = -rotated;
                    const float exponent = static_cast<float>((dimension % half) * 2) /
                        static_cast<float>(rotary_dim);
                    const float angle = static_cast<float>(position_offset + token) /
                        std::pow(theta, exponent);
                    output[index] = direct * std::cos(angle) + rotated * std::sin(angle);
                }
                if (packed_query) gate[index] = row[head_dim + dimension];
            }
        }
    }
}

bool near(float actual, float expected) {
    return std::fabs(actual - expected) <= 2.0e-4f;
}
}  // namespace

int main() {
    constexpr uint64_t query_count = kTokens * kQueryHeads * kHeadDim;
    constexpr uint64_t key_count = kTokens * kKvHeads * kHeadDim;
    constexpr uint64_t query_projection_count = query_count * 2;
    constexpr uint64_t query_bytes = query_count * sizeof(float);
    constexpr uint64_t key_bytes = key_count * sizeof(float);
    constexpr uint64_t query_projection_bytes = query_projection_count * sizeof(float);
    constexpr uint64_t weight_bytes = kHeadDim * sizeof(float);
    const uint64_t allocation_bytes[7] = {
        query_projection_bytes, key_bytes, weight_bytes, weight_bytes,
        query_bytes, key_bytes, query_bytes};

    SeenCudaHandle stream = 0, event = 0, allocations[7]{};
    CHECK_STATUS(seen_cuda_stream_create(0, &stream));
    CHECK_STATUS(seen_cuda_event_create(0, &event));
    for (int index = 0; index < 7; ++index)
        CHECK_STATUS(seen_cuda_malloc(0, allocation_bytes[index], &allocations[index]));
    void *device[7]{};
    for (int index = 0; index < 7; ++index) {
        uint64_t actual_bytes = 0; int32_t ordinal = -1;
        CHECK_STATUS(seen_cuda_allocation_address(
            allocations[index], &device[index], &actual_bytes, &ordinal));
        CHECK(actual_bytes >= allocation_bytes[index] && ordinal == 0);
    }

    std::vector<float> query_projection(query_projection_count), key_projection(key_count);
    std::vector<float> query_weight(kHeadDim), key_weight(kHeadDim);
    std::vector<float> expected_query(query_count), expected_key(key_count), expected_gate(query_count);
    std::vector<float> actual_query(query_count), actual_key(key_count), actual_gate(query_count);
    for (uint64_t index = 0; index < query_projection_count; ++index)
        query_projection[index] = static_cast<float>(static_cast<int64_t>((index * 29) % 101) - 50) / 37.0f;
    for (uint64_t index = 0; index < key_count; ++index)
        key_projection[index] = static_cast<float>(static_cast<int64_t>((index * 17) % 79) - 39) / 23.0f;
    for (uint64_t index = 0; index < kHeadDim; ++index) {
        query_weight[index] = static_cast<float>(static_cast<int64_t>(index % 19) - 9) / 64.0f;
        key_weight[index] = static_cast<float>(static_cast<int64_t>(index % 23) - 11) / 80.0f;
    }
    reference_rows(query_projection.data(), query_weight.data(), expected_query.data(),
        kTokens, kQueryHeads, kHeadDim, kRotaryDim, kPositionOffset,
        kTheta, kEpsilon, true, expected_gate.data());
    reference_rows(key_projection.data(), key_weight.data(), expected_key.data(),
        kTokens, kKvHeads, kHeadDim, kRotaryDim, kPositionOffset,
        kTheta, kEpsilon, false, nullptr);

    CHECK_STATUS(seen_cuda_memcpy_async(device[0], query_projection.data(),
        query_projection_bytes, SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[1], key_projection.data(),
        key_bytes, SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[2], query_weight.data(),
        weight_bytes, SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[3], key_weight.data(),
        weight_bytes, SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    auto token = borrow(stream);
    CHECK_STATUS(seen_qwen_attention_qk_rope_f32(&token,
        view(device[0], query_projection_bytes), view(device[1], key_bytes),
        view(device[2], weight_bytes), view(device[3], weight_bytes),
        view(device[4], query_bytes), view(device[5], key_bytes),
        view(device[6], query_bytes), kTokens, kQueryHeads, kKvHeads,
        kHeadDim, kRotaryDim, kPositionOffset, kMaxPosition, kTheta, kEpsilon));
    CHECK_STATUS(seen_cuda_memcpy_async(actual_query.data(), device[4], query_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(actual_key.data(), device[5], key_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(actual_gate.data(), device[6], query_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (uint64_t index = 0; index < query_count; ++index) {
        CHECK(near(actual_query[index], expected_query[index]));
        CHECK(actual_gate[index] == expected_gate[index]);
    }
    for (uint64_t index = 0; index < key_count; ++index)
        CHECK(near(actual_key[index], expected_key[index]));

    // Partial RoPE leaves dimensions 64..255 at their normalized learned scale.
    const uint64_t pass_index = (kQueryHeads + 3) * kHeadDim + 255;
    CHECK(near(actual_query[pass_index], expected_query[pass_index]));
    CHECK(actual_query[pass_index] != query_projection[(kQueryHeads + 3) * kHeadDim * 2 + 255]);

    // Invalid bounds, geometry, overlap, device identity, and tokens fail before enqueue.
    token = borrow(stream);
    auto call = [&](uint64_t tokens, uint64_t query_heads, uint64_t kv_heads,
                    uint64_t head_dim, uint64_t rotary_dim,
                    uint64_t position_offset, uint64_t max_position,
                    SeenQwenCudaBufferView query_output) {
        return seen_qwen_attention_qk_rope_f32(&token,
            view(device[0], query_projection_bytes), view(device[1], key_bytes),
            view(device[2], weight_bytes), view(device[3], weight_bytes),
            query_output, view(device[5], key_bytes), view(device[6], query_bytes),
            tokens, query_heads, kv_heads, head_dim, rotary_dim,
            position_offset, max_position, kTheta, kEpsilon);
    };
    CHECK(call(0, kQueryHeads, kKvHeads, kHeadDim, kRotaryDim,
        kPositionOffset, kMaxPosition, view(device[4], query_bytes)).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(call(kTokens, 23, kKvHeads, kHeadDim, kRotaryDim,
        kPositionOffset, kMaxPosition, view(device[4], query_bytes)).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(call(kTokens, kQueryHeads, kKvHeads, kHeadDim, 65,
        kPositionOffset, kMaxPosition, view(device[4], query_bytes)).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(call(kTokens, kQueryHeads, kKvHeads, kHeadDim, kRotaryDim,
        kPositionOffset + 1, kMaxPosition, view(device[4], query_bytes)).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(call(std::numeric_limits<uint64_t>::max(), kQueryHeads, kKvHeads,
        kHeadDim, kRotaryDim, 0, kMaxPosition, view(device[4], query_bytes)).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(call(kTokens, kQueryHeads, kKvHeads, kHeadDim, kRotaryDim,
        kPositionOffset, kMaxPosition,
        view(static_cast<float *>(device[0]) + 1, query_bytes)).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_attention_qk_rope_f32(&token,
        view(device[0], query_projection_bytes, 1), view(device[1], key_bytes),
        view(device[2], weight_bytes), view(device[3], weight_bytes),
        view(device[4], query_bytes), view(device[5], key_bytes), view(device[6], query_bytes),
        kTokens, kQueryHeads, kKvHeads, kHeadDim, kRotaryDim,
        kPositionOffset, kMaxPosition, kTheta, kEpsilon).code == SEEN_CUDA_INVALID_ARGUMENT);
    SeenCudaStreamLaunchToken incompatible = token; incompatible.generation = 0;
    CHECK(seen_qwen_attention_qk_rope_f32(&incompatible,
        view(device[0], query_projection_bytes), view(device[1], key_bytes),
        view(device[2], weight_bytes), view(device[3], weight_bytes),
        view(device[4], query_bytes), view(device[5], key_bytes), view(device[6], query_bytes),
        kTokens, kQueryHeads, kKvHeads, kHeadDim, kRotaryDim,
        kPositionOffset, kMaxPosition, kTheta, kEpsilon).code == SEEN_CUDA_INCOMPATIBLE);

    // Capture and replay both launches on the same Seen-owned stream.
    CHECK_STATUS(seen_cuda_graph_begin_capture(stream)); token = borrow(stream);
    CHECK_STATUS(seen_qwen_attention_qk_rope_f32(&token,
        view(device[0], query_projection_bytes), view(device[1], key_bytes),
        view(device[2], weight_bytes), view(device[3], weight_bytes),
        view(device[4], query_bytes), view(device[5], key_bytes), view(device[6], query_bytes),
        kTokens, kQueryHeads, kKvHeads, kHeadDim, kRotaryDim,
        kPositionOffset, kMaxPosition, kTheta, kEpsilon));
    SeenCudaHandle graph = 0, graph_exec = 0;
    CHECK_STATUS(seen_cuda_graph_end_capture(stream, &graph));
    CHECK_STATUS(seen_cuda_graph_instantiate(graph, &graph_exec));
    CHECK_STATUS(seen_cuda_graph_launch(graph_exec, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(actual_query.data(), device[4], query_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    CHECK_STATUS(seen_cuda_graph_exec_destroy(&graph_exec));
    CHECK_STATUS(seen_cuda_graph_destroy(&graph));
    for (uint64_t index = 0; index < query_count; ++index)
        CHECK(near(actual_query[index], expected_query[index]));

    for (int index = 6; index >= 0; --index)
        CHECK_STATUS(seen_cuda_free(&allocations[index]));
    CHECK_STATUS(seen_cuda_event_destroy(&event));
    CHECK_STATUS(seen_cuda_stream_destroy(&stream));
    std::printf("PASS: FEL-1439 official Q/K projection split RMSNorm partial RoPE geometry cleanup\n");
    return 0;
}
