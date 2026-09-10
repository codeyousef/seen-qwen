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
constexpr uint64_t kHeadDim = 256;
constexpr uint64_t kCount = kTokens * kQueryHeads * kHeadDim;
constexpr uint64_t kBytes = kCount * sizeof(float);

SeenQwenCudaBufferView view(void *address, uint64_t bytes, int32_t device = 0) {
    return SeenQwenCudaBufferView{SEEN_QWEN_CUDA_REFERENCE_ABI_VERSION,
        device, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address)), bytes};
}

SeenCudaStreamLaunchToken borrow(SeenCudaHandle stream) {
    SeenCudaStreamLaunchToken token{};
    const SeenCudaStatus status =
        seen_cuda_stream_borrow_launch_token(stream, 0, &token);
    if (status.code != SEEN_CUDA_OK) std::abort();
    return token;
}

float cpu_reference(float attended, float gate) {
    return attended / (1.0f + std::exp(-gate));
}

bool near(float actual, float expected) {
    return std::fabs(actual - expected) <= 4.0e-6f +
        4.0e-6f * std::fabs(expected);
}
}  // namespace

int main() {
    SeenCudaHandle stream = 0, event = 0, allocations[5]{};
    CHECK_STATUS(seen_cuda_stream_create(0, &stream));
    CHECK_STATUS(seen_cuda_event_create(0, &event));
    void *device[5]{};
    for (int index = 0; index < 5; ++index) {
        CHECK_STATUS(seen_cuda_malloc(0, kBytes, &allocations[index]));
        uint64_t bytes = 0; int32_t ordinal = -1;
        CHECK_STATUS(seen_cuda_allocation_address(
            allocations[index], &device[index], &bytes, &ordinal));
        CHECK(bytes == kBytes && ordinal == 0);
    }

    std::vector<float> attended(kCount), gate(kCount), expected(kCount);
    std::vector<float> actual(kCount), input_in_place(kCount), gate_in_place(kCount);
    for (uint64_t index = 0; index < kCount; ++index) {
        attended[index] = static_cast<float>(
            static_cast<int64_t>((index * 17) % 103) - 51) / 19.0f;
        gate[index] = static_cast<float>(
            static_cast<int64_t>((index * 29) % 89) - 44) / 7.0f;
        expected[index] = cpu_reference(attended[index], gate[index]);
    }
    gate[0] = -100.0f; gate[1] = 100.0f;
    expected[0] = cpu_reference(attended[0], gate[0]);
    expected[1] = cpu_reference(attended[1], gate[1]);

    CHECK_STATUS(seen_cuda_memcpy_async(device[0], attended.data(), kBytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[1], gate.data(), kBytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[3], attended.data(), kBytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[4], gate.data(), kBytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));

    auto token = borrow(stream);
    CHECK_STATUS(seen_qwen_sigmoid_gate_f32(&token, view(device[0], kBytes),
        view(device[1], kBytes), view(device[2], kBytes), kCount));
    // Exact input in-place and exact gate in-place preserve the same flattened
    // projection input because each element is read before its output write.
    token = borrow(stream);
    CHECK_STATUS(seen_qwen_sigmoid_gate_f32(&token, view(device[3], kBytes),
        view(device[1], kBytes), view(device[3], kBytes), kCount));
    token = borrow(stream);
    CHECK_STATUS(seen_qwen_sigmoid_gate_f32(&token, view(device[0], kBytes),
        view(device[4], kBytes), view(device[4], kBytes), kCount));
    CHECK_STATUS(seen_cuda_memcpy_async(actual.data(), device[2], kBytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(input_in_place.data(), device[3], kBytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(gate_in_place.data(), device[4], kBytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (uint64_t index = 0; index < kCount; ++index) {
        CHECK(std::isfinite(actual[index]));
        CHECK(near(actual[index], expected[index]));
        CHECK(near(input_in_place[index], actual[index]));
        CHECK(near(gate_in_place[index], actual[index]));
    }

    // Empty/overflowing counts, extents, partial aliases, devices, and stale
    // launch-token identities reject before enqueue.
    token = borrow(stream);
    CHECK(seen_qwen_sigmoid_gate_f32(&token, view(device[0], kBytes),
        view(device[1], kBytes), view(device[2], kBytes), 0).code ==
        SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_sigmoid_gate_f32(&token, view(device[0], kBytes),
        view(device[1], kBytes), view(device[2], kBytes),
        std::numeric_limits<uint64_t>::max()).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_sigmoid_gate_f32(&token, view(device[0], kBytes),
        view(device[1], kBytes), view(device[2], kBytes - sizeof(float)),
        kCount).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_sigmoid_gate_f32(&token,
        view(device[0], kBytes - sizeof(float)),
        view(device[1], kBytes - sizeof(float)),
        view(static_cast<float *>(device[0]) + 1, kBytes - sizeof(float)),
        kCount - 1).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_sigmoid_gate_f32(&token, view(device[0], kBytes, 1),
        view(device[1], kBytes), view(device[2], kBytes), kCount).code ==
        SEEN_CUDA_INVALID_ARGUMENT);
    SeenCudaStreamLaunchToken incompatible = token;
    incompatible.generation = 0;
    CHECK(seen_qwen_sigmoid_gate_f32(&incompatible, view(device[0], kBytes),
        view(device[1], kBytes), view(device[2], kBytes), kCount).code ==
        SEEN_CUDA_INCOMPATIBLE);

    CHECK_STATUS(seen_cuda_graph_begin_capture(stream));
    token = borrow(stream);
    CHECK_STATUS(seen_qwen_sigmoid_gate_f32(&token, view(device[0], kBytes),
        view(device[1], kBytes), view(device[2], kBytes), kCount));
    SeenCudaHandle graph = 0, graph_exec = 0;
    CHECK_STATUS(seen_cuda_graph_end_capture(stream, &graph));
    CHECK_STATUS(seen_cuda_graph_instantiate(graph, &graph_exec));
    CHECK_STATUS(seen_cuda_graph_launch(graph_exec, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(actual.data(), device[2], kBytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (uint64_t index = 0; index < kCount; ++index)
        CHECK(near(actual[index], expected[index]));
    CHECK_STATUS(seen_cuda_graph_exec_destroy(&graph_exec));
    CHECK_STATUS(seen_cuda_graph_destroy(&graph));

    for (int index = 4; index >= 0; --index)
        CHECK_STATUS(seen_cuda_free(&allocations[index]));
    CHECK_STATUS(seen_cuda_event_destroy(&event));
    CHECK_STATUS(seen_cuda_stream_destroy(&stream));
    std::printf("PASS: FEL-1441 official gated attention projection differential cleanup\n");
    return 0;
}
