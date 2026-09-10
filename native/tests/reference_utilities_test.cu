#include "seen_qwen_cuda.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

#define CHECK_STATUS(expr) do { SeenCudaStatus s_ = (expr); if (s_.code != SEEN_CUDA_OK) { \
    std::fprintf(stderr, "FAIL:%d: %s code=%d native=%d op=%s message=%s\n", \
        __LINE__, #expr, s_.code, s_.native_code, s_.operation, s_.message); return 1; } } while (0)
#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "FAIL:%d: %s\n", __LINE__, #expr); return 1; } } while (0)

namespace {

constexpr uint64_t kFloats = 96;
constexpr uint64_t kBytes = kFloats * sizeof(float);

SeenQwenCudaBufferView view(void *address, uint64_t bytes, int32_t device = 0) {
    return SeenQwenCudaBufferView{SEEN_QWEN_CUDA_REFERENCE_ABI_VERSION, device,
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address)), bytes};
}

SeenCudaStreamLaunchToken borrow(SeenCudaHandle stream) {
    SeenCudaStreamLaunchToken token{};
    const SeenCudaStatus result = seen_cuda_stream_borrow_launch_token(stream, 0, &token);
    if (result.code != SEEN_CUDA_OK) std::abort();
    return token;
}

bool near(float actual, float expected, float tolerance = 3.0e-6f) {
    return std::fabs(actual - expected) <= tolerance;
}

}  // namespace

int main() {
    SeenCudaHandle stream = 0, event = 0, host_handle = 0;
    SeenCudaHandle handles[7]{};
    CHECK_STATUS(seen_cuda_stream_create(0, &stream));
    CHECK_STATUS(seen_cuda_event_create(0, &event));
    CHECK_STATUS(seen_cuda_host_alloc(kBytes, &host_handle));
    for (auto &handle : handles) CHECK_STATUS(seen_cuda_malloc(0, kBytes, &handle));

    void *host_raw = nullptr; uint64_t actual = 0; int32_t device = -1;
    CHECK_STATUS(seen_cuda_host_allocation_address(host_handle, &host_raw, &actual));
    auto *host = static_cast<float *>(host_raw);
    void *buffers[7]{};
    for (int i = 0; i < 7; ++i)
        CHECK_STATUS(seen_cuda_allocation_address(handles[i], &buffers[i], &actual, &device));

    // Offset-weight RMSNorm matches the existing FP32 CPU contract.
    const float rms_input[8] = {1.0f, 2.0f, -3.0f, 4.0f, -2.0f, 0.5f, 1.5f, 3.0f};
    const float rms_weight[4] = {0.1f, -0.2f, 0.25f, -0.5f};
    CHECK_STATUS(seen_cuda_memcpy_async(buffers[0], rms_input, sizeof(rms_input), SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(buffers[1], rms_weight, sizeof(rms_weight), SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    auto token = borrow(stream);
    CHECK_STATUS(seen_qwen_rms_norm_f32(&token, view(buffers[0], sizeof(rms_input)),
        view(buffers[1], sizeof(rms_weight)), view(buffers[2], sizeof(rms_input)), 2, 4, 1.0e-6f));
    CHECK_STATUS(seen_cuda_memcpy_async(host, buffers[2], sizeof(rms_input), SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream)); CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (int row = 0; row < 2; ++row) {
        float sum = 0.0f; for (int col = 0; col < 4; ++col) sum += rms_input[row * 4 + col] * rms_input[row * 4 + col];
        const float inverse = 1.0f / std::sqrt(sum / 4.0f + 1.0e-6f);
        for (int col = 0; col < 4; ++col)
            CHECK(near(host[row * 4 + col], rms_input[row * 4 + col] * inverse * (1.0f + rms_weight[col])));
    }

    token = borrow(stream);
    CHECK_STATUS(seen_qwen_l2_norm_f32(&token, view(buffers[0], sizeof(rms_input)),
        view(buffers[2], sizeof(rms_input)), 2, 4, 1.0e-12f));
    CHECK_STATUS(seen_cuda_memcpy_async(host, buffers[2], sizeof(rms_input), SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream)); CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (int row = 0; row < 2; ++row) {
        float norm = 0.0f; for (int col = 0; col < 4; ++col) norm += host[row * 4 + col] * host[row * 4 + col];
        CHECK(near(norm, 1.0f, 5.0e-6f));
    }

    const float activation[7] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 4.0f, -4.0f};
    const float up[7] = {0.5f, 1.5f, -3.0f, 2.0f, -0.25f, 0.75f, 3.0f};
    CHECK_STATUS(seen_cuda_memcpy_async(buffers[0], activation, sizeof(activation), SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(buffers[1], up, sizeof(up), SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    token = borrow(stream); CHECK_STATUS(seen_qwen_silu_f32(&token,
        view(buffers[0], sizeof(activation)), view(buffers[2], sizeof(activation)), 7));
    CHECK_STATUS(seen_cuda_memcpy_async(host, buffers[2], sizeof(activation), SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream)); CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (int i = 0; i < 7; ++i) CHECK(near(host[i], activation[i] / (1.0f + std::exp(-activation[i]))));
    token = borrow(stream); CHECK_STATUS(seen_qwen_swiglu_f32(&token,
        view(buffers[0], sizeof(activation)), view(buffers[1], sizeof(up)), view(buffers[2], sizeof(activation)), 7));
    CHECK_STATUS(seen_cuda_memcpy_async(host, buffers[2], sizeof(activation), SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream)); CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (int i = 0; i < 7; ++i) CHECK(near(host[i], activation[i] / (1.0f + std::exp(-activation[i])) * up[i]));
    token = borrow(stream); CHECK_STATUS(seen_qwen_sigmoid_gate_f32(&token,
        view(buffers[1], sizeof(up)), view(buffers[0], sizeof(activation)), view(buffers[2], sizeof(up)), 7));
    CHECK_STATUS(seen_cuda_memcpy_async(host, buffers[2], sizeof(up), SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream)); CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (int i = 0; i < 7; ++i) CHECK(near(host[i], up[i] / (1.0f + std::exp(-activation[i]))));

    const float rope_input[8] = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.5f, 7.0f, 8.0f};
    CHECK_STATUS(seen_cuda_memcpy_async(buffers[0], rope_input, sizeof(rope_input), SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    token = borrow(stream); CHECK_STATUS(seen_qwen_partial_rope_f32(&token,
        view(buffers[0], sizeof(rope_input)), view(buffers[2], sizeof(rope_input)),
        1, 2, 4, 2, 1, 128, 10000000.0f));
    CHECK_STATUS(seen_cuda_memcpy_async(host, buffers[2], sizeof(rope_input), SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream)); CHECK_STATUS(seen_cuda_event_synchronize(event));
    CHECK(near(host[0], -1.14263964f) && near(host[1], 1.92207563f));
    CHECK(host[2] == 3.0f && host[3] == 4.0f && near(host[4], -0.96103781f));

    const float keys[8] = {1,2,3,4,5,6,7,8};
    const float values[8] = {-1,-2,-3,-4,-5,-6,-7,-8};
    CHECK_STATUS(seen_cuda_memcpy_async(buffers[0], keys, sizeof(keys), SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(buffers[1], values, sizeof(values), SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    token = borrow(stream); CHECK_STATUS(seen_qwen_fill_f32(&token, view(buffers[2], 12 * sizeof(float)), 12, 99.0f));
    token = borrow(stream); CHECK_STATUS(seen_qwen_fill_f32(&token, view(buffers[3], 12 * sizeof(float)), 12, 99.0f));
    token = borrow(stream); CHECK_STATUS(seen_qwen_kv_append_f32(&token,
        view(buffers[0], sizeof(keys)), view(buffers[1], sizeof(values)),
        view(buffers[2], 12 * sizeof(float)), view(buffers[3], 12 * sizeof(float)),
        2, 1, 4, 1, 3));
    CHECK_STATUS(seen_cuda_memcpy_async(host, buffers[2], 12 * sizeof(float), SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream)); CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (int i = 0; i < 4; ++i) CHECK(host[i] == 99.0f);
    for (int i = 0; i < 8; ++i) CHECK(host[4 + i] == keys[i]);

    const float logits[12] = {1,5,5,2,-1,3, -4,-2,-3,-2,-5,-6};
    CHECK_STATUS(seen_cuda_memcpy_async(buffers[0], logits, sizeof(logits), SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    token = borrow(stream); CHECK_STATUS(seen_qwen_greedy_argmax_f32(&token,
        view(buffers[0], sizeof(logits)), view(buffers[4], 2 * sizeof(int32_t)), 2, 6, 6));
    token = borrow(stream); CHECK_STATUS(seen_qwen_top_k_f32(&token,
        view(buffers[0], sizeof(logits)), view(buffers[5], 6 * sizeof(int32_t)),
        view(buffers[6], 6 * sizeof(float)), 2, 6, 6, 3));
    CHECK_STATUS(seen_cuda_memcpy_async(host, buffers[4], 2 * sizeof(int32_t), SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream)); CHECK_STATUS(seen_cuda_event_synchronize(event));
    auto *ids = reinterpret_cast<int32_t *>(host); CHECK(ids[0] == 1 && ids[1] == 1);
    CHECK_STATUS(seen_cuda_memcpy_async(host, buffers[5], 6 * sizeof(int32_t), SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream)); CHECK_STATUS(seen_cuda_event_synchronize(event));
    CHECK(ids[0] == 1 && ids[1] == 2 && ids[2] == 5);
    CHECK(ids[3] == 1 && ids[4] == 3 && ids[5] == 2);

    token = borrow(stream);
    CHECK(seen_qwen_rms_norm_f32(&token, view(buffers[0], 32), view(buffers[1], 16), view(buffers[2], 32), 2, 4, 0.0f).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_partial_rope_f32(&token, view(buffers[0], 32), view(buffers[0], 32), 1, 2, 4, 2, 1, 128, 10000000.0f).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_partial_rope_f32(&token, view(buffers[0], 32), view(buffers[2], 32), 1, 2, 4, 3, 1, 128, 10000000.0f).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_kv_append_f32(&token, view(buffers[0], 32), view(buffers[1], 32), view(buffers[2], 48), view(buffers[3], 48), 2, 1, 4, 2, 3).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_greedy_argmax_f32(&token, view(buffers[0], 48), view(buffers[4], 8), 2, 6, 7).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_top_k_f32(&token, view(buffers[0], 48), view(buffers[5], 24), view(buffers[6], 24), 2, 6, 6, 7).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_silu_f32(&token, view(buffers[0], 4), view(buffers[2], 4), 0).code == SEEN_CUDA_INVALID_ARGUMENT);

    CHECK_STATUS(seen_cuda_graph_begin_capture(stream));
    token = borrow(stream); CHECK_STATUS(seen_qwen_silu_f32(&token,
        view(buffers[0], sizeof(logits)), view(buffers[2], sizeof(logits)), 12));
    SeenCudaHandle graph = 0, graph_exec = 0;
    CHECK_STATUS(seen_cuda_graph_end_capture(stream, &graph));
    CHECK_STATUS(seen_cuda_graph_instantiate(graph, &graph_exec));
    CHECK_STATUS(seen_cuda_graph_launch(graph_exec, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream)); CHECK_STATUS(seen_cuda_event_synchronize(event));
    CHECK_STATUS(seen_cuda_graph_exec_destroy(&graph_exec)); CHECK_STATUS(seen_cuda_graph_destroy(&graph));

    for (int i = 6; i >= 0; --i) CHECK_STATUS(seen_cuda_free(&handles[i]));
    CHECK_STATUS(seen_cuda_host_free(&host_handle)); CHECK_STATUS(seen_cuda_event_destroy(&event));
    CHECK_STATUS(seen_cuda_stream_destroy(&stream));
    std::printf("PASS: FEL-1430 normalization activation RoPE KV sampling ordering capture negatives cleanup\n");
    return 0;
}
