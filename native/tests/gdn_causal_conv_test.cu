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

constexpr uint64_t kKernel = 4;
constexpr uint64_t kMaximumChannels = 10240;
constexpr uint64_t kMaximumTokens = 7;

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

void reference(const float *input, const float *weights, float *history,
               float *output, uint64_t tokens, uint64_t channels) {
    for (uint64_t token = 0; token < tokens; ++token) {
        for (uint64_t channel = 0; channel < channels; ++channel) {
            float sum = 0.0f;
            for (uint64_t tap = 0; tap + 1 < kKernel; ++tap)
                sum += history[tap * channels + channel] *
                    weights[channel * kKernel + tap];
            const float current = input[token * channels + channel];
            sum += current * weights[channel * kKernel + kKernel - 1];
            output[token * channels + channel] = sum / (1.0f + std::exp(-sum));
            for (uint64_t tap = 0; tap + 2 < kKernel; ++tap)
                history[tap * channels + channel] =
                    history[(tap + 1) * channels + channel];
            history[(kKernel - 2) * channels + channel] = current;
        }
    }
}

bool near(float actual, float expected, float tolerance = 4.0e-6f) {
    return std::fabs(actual - expected) <= tolerance;
}

}  // namespace

int main() {
    const uint64_t input_capacity = kMaximumTokens * kMaximumChannels;
    const uint64_t weight_capacity = kKernel * kMaximumChannels;
    const uint64_t history_capacity = (kKernel - 1) * kMaximumChannels;
    SeenCudaHandle stream = 0, event = 0;
    SeenCudaHandle allocations[4]{};
    CHECK_STATUS(seen_cuda_stream_create(0, &stream));
    CHECK_STATUS(seen_cuda_event_create(0, &event));
    CHECK_STATUS(seen_cuda_malloc(0, input_capacity * sizeof(float), &allocations[0]));
    CHECK_STATUS(seen_cuda_malloc(0, weight_capacity * sizeof(float), &allocations[1]));
    CHECK_STATUS(seen_cuda_malloc(0, history_capacity * sizeof(float), &allocations[2]));
    CHECK_STATUS(seen_cuda_malloc(0, input_capacity * sizeof(float), &allocations[3]));
    void *device[4]{}; uint64_t actual = 0; int32_t ordinal = -1;
    for (int i = 0; i < 4; ++i)
        CHECK_STATUS(seen_cuda_allocation_address(
            allocations[i], &device[i], &actual, &ordinal));

    constexpr uint64_t channels = 5;
    constexpr uint64_t tokens = 7;
    const uint64_t input_bytes = tokens * channels * sizeof(float);
    const uint64_t weight_bytes = kKernel * channels * sizeof(float);
    const uint64_t history_bytes = (kKernel - 1) * channels * sizeof(float);
    std::vector<float> input(tokens * channels), weights(kKernel * channels);
    std::vector<float> expected(tokens * channels), expected_history((kKernel - 1) * channels, 0.0f);
    std::vector<float> actual_values(tokens * channels), full_gpu(tokens * channels);
    std::vector<float> actual_history((kKernel - 1) * channels);
    for (uint64_t i = 0; i < input.size(); ++i)
        input[i] = static_cast<float>(static_cast<int64_t>((i * 17) % 29) - 14) / 9.0f;
    for (uint64_t i = 0; i < weights.size(); ++i)
        weights[i] = static_cast<float>(static_cast<int64_t>((i * 7) % 19) - 9) / 13.0f;
    reference(input.data(), weights.data(), expected_history.data(),
              expected.data(), tokens, channels);
    CHECK_STATUS(seen_cuda_memcpy_async(device[0], input.data(), input_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[1], weights.data(), weight_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memset_async(device[2], 0, history_bytes, stream));
    auto token = borrow(stream);
    CHECK_STATUS(seen_qwen_causal_conv_silu_f32(&token,
        view(device[0], input_bytes), view(device[1], weight_bytes),
        view(device[2], history_bytes), view(device[3], input_bytes),
        tokens, channels, kKernel, 0, 0));
    CHECK_STATUS(seen_cuda_memcpy_async(actual_values.data(), device[3], input_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(actual_history.data(), device[2], history_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (uint64_t i = 0; i < expected.size(); ++i)
        CHECK(near(actual_values[i], expected[i]));
    full_gpu = actual_values;
    for (uint64_t i = 0; i < expected_history.size(); ++i)
        CHECK(actual_history[i] == expected_history[i]);

    // Arbitrary chunk boundaries preserve output and state exactly.
    CHECK_STATUS(seen_cuda_memset_async(device[2], 0, history_bytes, stream));
    CHECK_STATUS(seen_cuda_memset_async(device[3], 0, input_bytes, stream));
    uint64_t processed = 0;
    for (const uint64_t chunk : {2u, 1u, 4u}) {
        auto *chunk_input = static_cast<float *>(device[0]) + processed * channels;
        auto *chunk_output = static_cast<float *>(device[3]) + processed * channels;
        token = borrow(stream);
        CHECK_STATUS(seen_qwen_causal_conv_silu_f32(&token,
            view(chunk_input, input_bytes - processed * channels * sizeof(float)),
            view(device[1], weight_bytes), view(device[2], history_bytes),
            view(chunk_output, input_bytes - processed * channels * sizeof(float)),
            chunk, channels, kKernel, processed, processed));
        processed += chunk;
    }
    CHECK_STATUS(seen_cuda_memcpy_async(actual_values.data(), device[3], input_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(actual_history.data(), device[2], history_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (uint64_t i = 0; i < expected.size(); ++i)
        CHECK(actual_values[i] == full_gpu[i]);
    for (uint64_t i = 0; i < expected_history.size(); ++i)
        CHECK(actual_history[i] == expected_history[i]);

    // Exact input/output aliasing is supported and still preserves raw history.
    CHECK_STATUS(seen_cuda_memcpy_async(device[3], input.data(), input_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memset_async(device[2], 0, history_bytes, stream));
    token = borrow(stream);
    CHECK_STATUS(seen_qwen_causal_conv_silu_f32(&token,
        view(device[3], input_bytes), view(device[1], weight_bytes),
        view(device[2], history_bytes), view(device[3], input_bytes),
        tokens, channels, kKernel, 0, 0));
    CHECK_STATUS(seen_cuda_memcpy_async(actual_values.data(), device[3], input_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (uint64_t i = 0; i < expected.size(); ++i)
        CHECK(near(actual_values[i], expected[i]));

    // The exact official 10,240-channel convolution geometry is admitted.
    constexpr uint64_t official_tokens = 2;
    const uint64_t official_input_bytes = official_tokens * kMaximumChannels * sizeof(float);
    const uint64_t official_weight_bytes = kKernel * kMaximumChannels * sizeof(float);
    const uint64_t official_history_bytes = (kKernel - 1) * kMaximumChannels * sizeof(float);
    std::vector<float> official_input(official_tokens * kMaximumChannels, 0.25f);
    std::vector<float> official_weights(kKernel * kMaximumChannels, 0.125f);
    CHECK_STATUS(seen_cuda_memcpy_async(device[0], official_input.data(),
        official_input_bytes, SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[1], official_weights.data(),
        official_weight_bytes, SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memset_async(device[2], 0, official_history_bytes, stream));
    token = borrow(stream);
    CHECK_STATUS(seen_qwen_causal_conv_silu_f32(&token,
        view(device[0], official_input_bytes), view(device[1], official_weight_bytes),
        view(device[2], official_history_bytes), view(device[3], official_input_bytes),
        official_tokens, kMaximumChannels, kKernel, 0, 0));
    CHECK_STATUS(seen_cuda_memcpy_async(actual_values.data(), device[3], sizeof(float),
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    const float first_sum = 0.25f * 0.125f;
    CHECK(near(actual_values[0], first_sum / (1.0f + std::exp(-first_sum))));

    // Host-side rejection occurs before state mutation.
    CHECK_STATUS(seen_cuda_memset_async(device[2], 0x3f, history_bytes, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    token = borrow(stream);
    CHECK(seen_qwen_causal_conv_silu_f32(&token,
        view(device[0], input_bytes), view(device[1], weight_bytes),
        view(device[2], history_bytes), view(device[3], input_bytes),
        tokens, channels, kKernel, 1, 0).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_causal_conv_silu_f32(&token,
        view(device[0], input_bytes), view(device[1], weight_bytes),
        view(device[2], history_bytes), view(device[3], input_bytes),
        tokens, channels, 3, 0, 0).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_causal_conv_silu_f32(&token,
        view(device[0], input_bytes), view(device[1], weight_bytes),
        view(device[2], history_bytes - sizeof(float)), view(device[3], input_bytes),
        tokens, channels, kKernel, 0, 0).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_causal_conv_silu_f32(&token,
        view(device[0], input_bytes), view(device[1], weight_bytes),
        view(device[2], history_bytes), view(static_cast<float *>(device[0]) + 1,
            input_bytes - sizeof(float)), tokens, channels, kKernel, 0, 0).code ==
        SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_causal_conv_silu_f32(&token,
        view(device[0], input_bytes), view(device[1], weight_bytes),
        view(device[2], history_bytes), view(device[3], input_bytes),
        2, channels, kKernel, std::numeric_limits<uint64_t>::max() - 1,
        std::numeric_limits<uint64_t>::max() - 1).code == SEEN_CUDA_INVALID_ARGUMENT);
    SeenCudaStreamLaunchToken incompatible = token; incompatible.reserved = 1;
    CHECK(seen_qwen_causal_conv_silu_f32(&incompatible,
        view(device[0], input_bytes), view(device[1], weight_bytes),
        view(device[2], history_bytes), view(device[3], input_bytes),
        tokens, channels, kKernel, 0, 0).code == SEEN_CUDA_INCOMPATIBLE);
    CHECK(seen_qwen_causal_conv_silu_f32(nullptr,
        view(device[0], input_bytes), view(device[1], weight_bytes),
        view(device[2], history_bytes), view(device[3], input_bytes),
        tokens, channels, kKernel, 0, 0).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK_STATUS(seen_cuda_memcpy_async(actual_history.data(), device[2], history_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (float value : actual_history) CHECK(value != 0.0f);

    // Capture and replay use the same Seen-owned stream.
    CHECK_STATUS(seen_cuda_memset_async(device[2], 0, history_bytes, stream));
    CHECK_STATUS(seen_cuda_graph_begin_capture(stream));
    token = borrow(stream);
    CHECK_STATUS(seen_qwen_causal_conv_silu_f32(&token,
        view(device[0], input_bytes), view(device[1], weight_bytes),
        view(device[2], history_bytes), view(device[3], input_bytes),
        tokens, channels, kKernel, 0, 0));
    SeenCudaHandle graph = 0, graph_exec = 0;
    CHECK_STATUS(seen_cuda_graph_end_capture(stream, &graph));
    CHECK_STATUS(seen_cuda_graph_instantiate(graph, &graph_exec));
    CHECK_STATUS(seen_cuda_graph_launch(graph_exec, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    CHECK_STATUS(seen_cuda_graph_exec_destroy(&graph_exec));
    CHECK_STATUS(seen_cuda_graph_destroy(&graph));

    for (int i = 3; i >= 0; --i) CHECK_STATUS(seen_cuda_free(&allocations[i]));
    CHECK_STATUS(seen_cuda_event_destroy(&event));
    CHECK_STATUS(seen_cuda_stream_destroy(&stream));
    std::printf("PASS: FEL-1433 GDN state causal convolution differential chunks bounds graph cleanup\n");
    return 0;
}
