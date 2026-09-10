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
constexpr uint64_t kOfficialHeads = 48;
constexpr uint64_t kOfficialValueDim = 128;
constexpr uint64_t kOfficialTokens = 2;
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

void reference(const float *core, const float *gate, const float *weight,
               float *output, uint64_t rows, uint64_t width, float epsilon) {
    for (uint64_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (uint64_t column = 0; column < width; ++column) {
            const float value = core[row * width + column];
            sum += value * value;
        }
        const float inverse = 1.0f / std::sqrt(sum / static_cast<float>(width) + epsilon);
        for (uint64_t column = 0; column < width; ++column) {
            const uint64_t index = row * width + column;
            const float silu = gate[index] / (1.0f + std::exp(-gate[index]));
            output[index] = weight[column] * core[index] * inverse * silu;
        }
    }
}

bool near(float actual, float expected, float tolerance = 4.0e-6f) {
    return std::fabs(actual - expected) <= tolerance;
}
}  // namespace

int main() {
    constexpr uint64_t rows = kOfficialTokens * kOfficialHeads;
    constexpr uint64_t width = kOfficialValueDim;
    constexpr uint64_t count = rows * width;
    constexpr uint64_t bytes = count * sizeof(float);
    constexpr uint64_t weight_bytes = width * sizeof(float);
    SeenCudaHandle stream = 0, event = 0, allocations[4]{};
    CHECK_STATUS(seen_cuda_stream_create(0, &stream));
    CHECK_STATUS(seen_cuda_event_create(0, &event));
    CHECK_STATUS(seen_cuda_malloc(0, bytes, &allocations[0]));
    CHECK_STATUS(seen_cuda_malloc(0, bytes, &allocations[1]));
    CHECK_STATUS(seen_cuda_malloc(0, weight_bytes, &allocations[2]));
    CHECK_STATUS(seen_cuda_malloc(0, bytes, &allocations[3]));
    void *device[4]{}; uint64_t actual_bytes = 0; int32_t ordinal = -1;
    for (int index = 0; index < 4; ++index)
        CHECK_STATUS(seen_cuda_allocation_address(
            allocations[index], &device[index], &actual_bytes, &ordinal));

    std::vector<float> core(count), gate(count), weight(width), expected(count), actual(count);
    for (uint64_t index = 0; index < count; ++index) {
        core[index] = static_cast<float>(static_cast<int64_t>((index * 17) % 67) - 33) / 31.0f;
        gate[index] = static_cast<float>(static_cast<int64_t>((index * 11) % 43) - 21) / 9.0f;
    }
    for (uint64_t index = 0; index < width; ++index)
        weight[index] = 0.25f + static_cast<float>(index % 13) / 17.0f;
    reference(core.data(), gate.data(), weight.data(), expected.data(), rows, width, kEpsilon);
    CHECK_STATUS(seen_cuda_memcpy_async(device[0], core.data(), bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[1], gate.data(), bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[2], weight.data(), weight_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    auto token = borrow(stream);
    CHECK_STATUS(seen_qwen_gdn_gated_rms_norm_f32(&token,
        view(device[0], bytes), view(device[1], bytes), view(device[2], weight_bytes),
        view(device[3], bytes), rows, width, kEpsilon));
    CHECK_STATUS(seen_cuda_memcpy_async(actual.data(), device[3], bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (uint64_t index = 0; index < count; ++index) CHECK(near(actual[index], expected[index]));

    // Exact core/output alias preserves ownership and matches the separate-output result.
    CHECK_STATUS(seen_cuda_memcpy_async(device[0], core.data(), bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    token = borrow(stream);
    CHECK_STATUS(seen_qwen_gdn_gated_rms_norm_f32(&token,
        view(device[0], bytes), view(device[1], bytes), view(device[2], weight_bytes),
        view(device[0], bytes), rows, width, kEpsilon));
    CHECK_STATUS(seen_cuda_memcpy_async(actual.data(), device[0], bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (uint64_t index = 0; index < count; ++index) CHECK(near(actual[index], expected[index]));

    // Invalid geometry, bounds, device identity, overlap, and launch tokens fail closed.
    token = borrow(stream);
    CHECK(seen_qwen_gdn_gated_rms_norm_f32(&token,
        view(device[0], bytes), view(device[1], bytes), view(device[2], weight_bytes),
        view(device[3], bytes), 0, width, kEpsilon).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_gdn_gated_rms_norm_f32(&token,
        view(device[0], bytes), view(device[1], bytes), view(device[2], weight_bytes),
        view(device[3], bytes), rows, width, 0.0f).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_gdn_gated_rms_norm_f32(&token,
        view(device[0], bytes), view(device[1], bytes), view(device[2], weight_bytes - 1),
        view(device[3], bytes), rows, width, kEpsilon).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_gdn_gated_rms_norm_f32(&token,
        view(device[0], bytes), view(device[1], bytes), view(device[2], weight_bytes),
        view(static_cast<float *>(device[0]) + 1, bytes - sizeof(float)),
        rows, width, kEpsilon).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_gdn_gated_rms_norm_f32(&token,
        view(device[0], bytes, 1), view(device[1], bytes), view(device[2], weight_bytes),
        view(device[3], bytes), rows, width, kEpsilon).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_gdn_gated_rms_norm_f32(&token,
        view(device[0], bytes), view(device[1], bytes), view(device[2], weight_bytes),
        view(device[3], bytes), std::numeric_limits<uint64_t>::max(), width,
        kEpsilon).code == SEEN_CUDA_INVALID_ARGUMENT);
    SeenCudaStreamLaunchToken incompatible = token; incompatible.generation = 0;
    CHECK(seen_qwen_gdn_gated_rms_norm_f32(&incompatible,
        view(device[0], bytes), view(device[1], bytes), view(device[2], weight_bytes),
        view(device[3], bytes), rows, width, kEpsilon).code == SEEN_CUDA_INCOMPATIBLE);
    CHECK(seen_qwen_gdn_gated_rms_norm_f32(nullptr,
        view(device[0], bytes), view(device[1], bytes), view(device[2], weight_bytes),
        view(device[3], bytes), rows, width, kEpsilon).code == SEEN_CUDA_INVALID_ARGUMENT);

    // Capture, replay, and a following copy stay ordered on the same Seen-owned stream.
    CHECK_STATUS(seen_cuda_graph_begin_capture(stream)); token = borrow(stream);
    CHECK_STATUS(seen_qwen_gdn_gated_rms_norm_f32(&token,
        view(device[0], bytes), view(device[1], bytes), view(device[2], weight_bytes),
        view(device[3], bytes), rows, width, kEpsilon));
    SeenCudaHandle graph = 0, graph_exec = 0;
    CHECK_STATUS(seen_cuda_graph_end_capture(stream, &graph));
    CHECK_STATUS(seen_cuda_graph_instantiate(graph, &graph_exec));
    CHECK_STATUS(seen_cuda_graph_launch(graph_exec, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(actual.data(), device[3], bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    CHECK_STATUS(seen_cuda_graph_exec_destroy(&graph_exec));
    CHECK_STATUS(seen_cuda_graph_destroy(&graph));

    for (int index = 3; index >= 0; --index) CHECK_STATUS(seen_cuda_free(&allocations[index]));
    CHECK_STATUS(seen_cuda_event_destroy(&event));
    CHECK_STATUS(seen_cuda_stream_destroy(&stream));
    std::printf("PASS: FEL-1438 gated GDN RMSNorm differential bounds graph ordering cleanup\n");
    return 0;
}
