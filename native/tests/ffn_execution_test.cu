#include "seen_qwen_cuda.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#define CHECK_STATUS(expr) do { SeenCudaStatus s_ = (expr); if (s_.code != SEEN_CUDA_OK) { \
    std::fprintf(stderr, "FAIL:%d: %s code=%d native=%d op=%s message=%s\n", \
        __LINE__, #expr, s_.code, s_.native_code, s_.operation, s_.message); return false; } } while (0)
#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "FAIL:%d: %s\n", __LINE__, #expr); return false; } } while (0)

namespace {

constexpr uint64_t kTokens = 3;
constexpr uint64_t kHidden = 8;
constexpr uint64_t kIntermediate = 12;
constexpr uint64_t kWorkspaceBytes = 8ull << 20;

SeenCudaMatmulDesc descriptor(uint64_t input, uint64_t output, int32_t type) {
    return SeenCudaMatmulDesc{SEEN_CUDA_ABI_VERSION, type, 1, 0,
        output, kTokens, input, static_cast<int64_t>(input),
        static_cast<int64_t>(input), static_cast<int64_t>(output),
        kWorkspaceBytes};
}

SeenQwenCudaBufferView view(void *address, uint64_t bytes,
                            int32_t device = 0) {
    return SeenQwenCudaBufferView{SEEN_QWEN_CUDA_REFERENCE_ABI_VERSION,
        device, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address)),
        bytes};
}

SeenCudaStreamLaunchToken borrow(SeenCudaHandle stream) {
    SeenCudaStreamLaunchToken token{};
    const SeenCudaStatus status =
        seen_cuda_stream_borrow_launch_token(stream, 0, &token);
    if (status.code != SEEN_CUDA_OK) std::abort();
    return token;
}

template <typename T> T encode(float value);
template <> __half encode(float value) { return __float2half_rn(value); }
template <> __nv_bfloat16 encode(float value) { return __float2bfloat16(value); }

template <typename T> float decode(T value);
template <> float decode(__half value) { return __half2float(value); }
template <> float decode(__nv_bfloat16 value) { return __bfloat162float(value); }

template <typename T>
std::vector<T> encoded(uint64_t count, uint64_t multiplier, int64_t offset,
                       float divisor) {
    std::vector<T> result(count);
    for (uint64_t index = 0; index < count; ++index) {
        const int64_t integer = static_cast<int64_t>((index * multiplier) % 29) +
            offset;
        result[index] = encode<T>(static_cast<float>(integer) / divisor);
    }
    return result;
}

template <typename T>
std::vector<T> projection(const std::vector<T> &input,
                          const std::vector<T> &weight, uint64_t rows,
                          uint64_t input_width, uint64_t output_width) {
    std::vector<T> result(rows * output_width);
    for (uint64_t row = 0; row < rows; ++row) {
        for (uint64_t output = 0; output < output_width; ++output) {
            float sum = 0.0f;
            for (uint64_t input_column = 0; input_column < input_width;
                 ++input_column) {
                sum += decode(input[row * input_width + input_column]) *
                    decode(weight[output * input_width + input_column]);
            }
            result[row * output_width + output] = encode<T>(sum);
        }
    }
    return result;
}

template <typename T>
std::vector<T> reference(const std::vector<T> &input,
                         const std::vector<T> &gate_weight,
                         const std::vector<T> &up_weight,
                         const std::vector<T> &down_weight) {
    auto gate = projection(input, gate_weight, kTokens, kHidden, kIntermediate);
    auto up = projection(input, up_weight, kTokens, kHidden, kIntermediate);
    for (uint64_t index = 0; index < gate.size(); ++index) {
        const float value = decode(gate[index]);
        gate[index] = encode<T>(value / (1.0f + std::exp(-value)) *
            decode(up[index]));
    }
    return projection(gate, down_weight, kTokens, kIntermediate, kHidden);
}

template <typename T>
bool run_type(int32_t data_type, float tolerance) {
    const uint64_t input_bytes = kTokens * kHidden * sizeof(T);
    const uint64_t intermediate_bytes = kTokens * kIntermediate * sizeof(T);
    const uint64_t output_bytes = kTokens * kHidden * sizeof(T);
    const uint64_t gate_weight_bytes = kIntermediate * kHidden * sizeof(T);
    const uint64_t down_weight_bytes = kHidden * kIntermediate * sizeof(T);
    const uint64_t sizes[8] = {input_bytes, gate_weight_bytes,
        gate_weight_bytes, down_weight_bytes, intermediate_bytes,
        intermediate_bytes, output_bytes, kWorkspaceBytes};
    SeenCudaHandle stream = 0, event = 0, cublas = 0, allocations[8]{};
    CHECK_STATUS(seen_cuda_stream_create(0, &stream));
    CHECK_STATUS(seen_cuda_event_create(0, &event));
    CHECK_STATUS(seen_cublaslt_create(0, &cublas));
    void *device[8]{};
    for (int index = 0; index < 8; ++index) {
        CHECK_STATUS(seen_cuda_malloc(0, sizes[index], &allocations[index]));
        uint64_t bytes = 0; int32_t ordinal = -1;
        CHECK_STATUS(seen_cuda_allocation_address(
            allocations[index], &device[index], &bytes, &ordinal));
        CHECK(bytes == sizes[index] && ordinal == 0);
    }

    const auto input = encoded<T>(kTokens * kHidden, 7, -13, 19.0f);
    const auto gate_weight = encoded<T>(kIntermediate * kHidden, 11, -14, 31.0f);
    const auto up_weight = encoded<T>(kIntermediate * kHidden, 13, -12, 29.0f);
    const auto down_weight = encoded<T>(kHidden * kIntermediate, 17, -15, 37.0f);
    const auto expected = reference(input, gate_weight, up_weight, down_weight);
    CHECK_STATUS(seen_cuda_memcpy_async(device[0], input.data(), input_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[1], gate_weight.data(),
        gate_weight_bytes, SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[2], up_weight.data(),
        gate_weight_bytes, SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[3], down_weight.data(),
        down_weight_bytes, SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));

    auto gate_desc = descriptor(kHidden, kIntermediate, data_type);
    auto up_desc = descriptor(kHidden, kIntermediate, data_type);
    auto down_desc = descriptor(kIntermediate, kHidden, data_type);
    SeenCudaAlgorithm gate_algorithm{}, up_algorithm{}, down_algorithm{};
    CHECK_STATUS(seen_cublaslt_select_algorithm(cublas, &gate_desc,
        &gate_algorithm));
    CHECK_STATUS(seen_cublaslt_select_algorithm(cublas, &up_desc,
        &up_algorithm));
    CHECK_STATUS(seen_cublaslt_select_algorithm(cublas, &down_desc,
        &down_algorithm));
    CHECK(gate_algorithm.workspace_bytes <= kWorkspaceBytes);
    CHECK(up_algorithm.workspace_bytes <= kWorkspaceBytes);
    CHECK(down_algorithm.workspace_bytes <= kWorkspaceBytes);

    auto enqueue = [&]() -> bool {
        CHECK_STATUS(seen_cublaslt_matmul(cublas, &gate_desc, &gate_algorithm,
            device[1], device[0], device[4], device[7], stream));
        CHECK_STATUS(seen_cublaslt_matmul(cublas, &up_desc, &up_algorithm,
            device[2], device[0], device[5], device[7], stream));
        auto token = borrow(stream);
        // Reuse the gate projection as the activation buffer. The adapter
        // reads each gate element before replacing that same element.
        CHECK_STATUS(seen_qwen_swiglu_low_precision(&token,
            view(device[4], intermediate_bytes),
            view(device[5], intermediate_bytes),
            view(device[4], intermediate_bytes),
            kTokens * kIntermediate, data_type));
        CHECK_STATUS(seen_cublaslt_matmul(cublas, &down_desc, &down_algorithm,
            device[3], device[4], device[6], device[7], stream));
        return true;
    };

    CHECK(enqueue());
    std::vector<T> actual(kTokens * kHidden);
    CHECK_STATUS(seen_cuda_memcpy_async(actual.data(), device[6], output_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (uint64_t index = 0; index < actual.size(); ++index) {
        CHECK(std::isfinite(decode(actual[index])));
        CHECK(std::fabs(decode(actual[index]) - decode(expected[index])) <=
            tolerance + tolerance * std::fabs(decode(expected[index])));
    }

    // All four operations must remain ordered and capture-compatible on the
    // exact same Seen-owned stream.
    CHECK_STATUS(seen_cuda_graph_begin_capture(stream));
    CHECK(enqueue());
    SeenCudaHandle graph = 0, graph_exec = 0;
    CHECK_STATUS(seen_cuda_graph_end_capture(stream, &graph));
    CHECK_STATUS(seen_cuda_graph_instantiate(graph, &graph_exec));
    CHECK_STATUS(seen_cuda_graph_launch(graph_exec, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(actual.data(), device[6], output_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (uint64_t index = 0; index < actual.size(); ++index)
        CHECK(std::fabs(decode(actual[index]) - decode(expected[index])) <=
            tolerance + tolerance * std::fabs(decode(expected[index])));
    CHECK_STATUS(seen_cuda_graph_exec_destroy(&graph_exec));
    CHECK_STATUS(seen_cuda_graph_destroy(&graph));

    auto token = borrow(stream);
    CHECK(seen_qwen_swiglu_low_precision(nullptr,
        view(device[4], intermediate_bytes), view(device[5], intermediate_bytes),
        view(device[4], intermediate_bytes), kTokens * kIntermediate,
        data_type).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_swiglu_low_precision(&token,
        view(device[4], intermediate_bytes), view(device[5], intermediate_bytes),
        view(device[4], intermediate_bytes), 0, data_type).code ==
        SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_swiglu_low_precision(&token,
        view(device[4], intermediate_bytes), view(device[5], intermediate_bytes),
        view(device[4], intermediate_bytes), kTokens * kIntermediate,
        SEEN_CUDA_F32).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_swiglu_low_precision(&token,
        view(device[4], intermediate_bytes),
        view(device[5], intermediate_bytes - sizeof(T)),
        view(device[4], intermediate_bytes), kTokens * kIntermediate,
        data_type).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_swiglu_low_precision(&token,
        view(device[4], intermediate_bytes - sizeof(T)),
        view(device[5], intermediate_bytes - sizeof(T)),
        view(static_cast<T *>(device[4]) + 1, intermediate_bytes - sizeof(T)),
        kTokens * kIntermediate - 1, data_type).code ==
        SEEN_CUDA_INVALID_ARGUMENT);
    SeenCudaStreamLaunchToken incompatible = token;
    incompatible.generation = 0;
    CHECK(seen_qwen_swiglu_low_precision(&incompatible,
        view(device[4], intermediate_bytes), view(device[5], intermediate_bytes),
        view(device[4], intermediate_bytes), kTokens * kIntermediate,
        data_type).code == SEEN_CUDA_INCOMPATIBLE);

    // The required short-session soak reuses fixed allocations and performs
    // no host or device allocation in the FFN path.
    for (int iteration = 0; iteration < 1000; ++iteration) CHECK(enqueue());
    CHECK_STATUS(seen_cuda_stream_synchronize(stream));

    for (int index = 7; index >= 0; --index)
        CHECK_STATUS(seen_cuda_free(&allocations[index]));
    CHECK_STATUS(seen_cublaslt_destroy(&cublas));
    CHECK_STATUS(seen_cublaslt_destroy(&cublas));
    CHECK_STATUS(seen_cuda_event_destroy(&event));
    CHECK_STATUS(seen_cuda_event_destroy(&event));
    CHECK_STATUS(seen_cuda_stream_destroy(&stream));
    CHECK_STATUS(seen_cuda_stream_destroy(&stream));
    return true;
}

}  // namespace

int main() {
    if (!run_type<__half>(SEEN_CUDA_F16, 2.5e-2f)) return 1;
    if (!run_type<__nv_bfloat16>(SEEN_CUDA_BF16, 4.5e-2f)) return 1;
    std::printf("PASS: FEL-1444 exact F16/BF16 Qwen FFN ordering capture negatives 1000-session cleanup\n");
    return 0;
}
