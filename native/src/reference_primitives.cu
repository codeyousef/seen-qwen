#include "seen_qwen_cuda.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace {

constexpr uint32_t kThreads = 256;

SeenCudaStatus status(int32_t code, int32_t native_code, int32_t device,
                      const char *operation, const char *message) {
    SeenCudaStatus result{};
    result.code = code;
    result.native_code = native_code;
    result.device_ordinal = device;
    result.maturity = SEEN_CUDA_MATURITY_EXPERIMENTAL_HARDWARE;
    result.operation = operation;
    result.message = message;
    return result;
}

SeenCudaStatus invalid(int32_t device, const char *operation,
                       const char *message) {
    return status(SEEN_CUDA_INVALID_ARGUMENT, 0, device, operation, message);
}

bool checked_multiply(uint64_t left, uint64_t right, uint64_t *result) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left)
        return false;
    *result = left * right;
    return true;
}

SeenCudaStatus validate_token(const SeenCudaStreamLaunchToken *token,
                              const char *operation, cudaStream_t *stream) {
    if (!token)
        return invalid(-1, operation, "missing borrowed stream launch token");
    if (token->abi_version != SEEN_CUDA_STREAM_LAUNCH_TOKEN_ABI_VERSION ||
        token->reserved != 0 || token->native_stream == 0 ||
        token->generation == 0 || token->device_ordinal < 0 ||
        (token->flags & ~(SEEN_CUDA_STREAM_LAUNCH_CAPTURE_COMPATIBLE |
                          SEEN_CUDA_STREAM_LAUNCH_CAPTURE_ACTIVE)) != 0 ||
        !(token->flags & SEEN_CUDA_STREAM_LAUNCH_CAPTURE_COMPATIBLE))
        return status(SEEN_CUDA_INCOMPATIBLE, 0, token->device_ordinal,
                      operation, "incompatible borrowed stream launch token");
    int current_device = -1;
    const cudaError_t queried = cudaGetDevice(&current_device);
    if (queried != cudaSuccess)
        return status(SEEN_CUDA_RUNTIME_ERROR, static_cast<int32_t>(queried),
                      token->device_ordinal, operation,
                      cudaGetErrorString(queried));
    if (current_device != token->device_ordinal)
        return status(SEEN_CUDA_INCOMPATIBLE, 0, current_device, operation,
                      "borrowed stream token device is not current");
    *stream = reinterpret_cast<cudaStream_t>(
        static_cast<uintptr_t>(token->native_stream));
    return status(SEEN_CUDA_OK, 0, token->device_ordinal, operation, "ok");
}

SeenCudaStatus validate_view(const SeenQwenCudaBufferView &view,
                             uint64_t required_bytes, int32_t device,
                             const char *operation) {
    if (view.abi_version != SEEN_QWEN_CUDA_REFERENCE_ABI_VERSION ||
        view.device_ordinal != device || view.address == 0 ||
        required_bytes == 0 || view.byte_length < required_bytes)
        return invalid(device, operation, "invalid bounded device buffer view");
    cudaPointerAttributes attributes{};
    const cudaError_t queried = cudaPointerGetAttributes(
        &attributes, reinterpret_cast<const void *>(
                         static_cast<uintptr_t>(view.address)));
    if (queried != cudaSuccess)
        return status(SEEN_CUDA_INVALID_ARGUMENT,
                      static_cast<int32_t>(queried), device, operation,
                      "buffer address is not a CUDA allocation");
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != device)
        return status(SEEN_CUDA_INCOMPATIBLE, 0, device, operation,
                      "buffer allocation belongs to another memory domain");
    CUdeviceptr allocation_base = 0;
    size_t allocation_bytes = 0;
    const CUresult ranged = cuMemGetAddressRange(
        &allocation_base, &allocation_bytes,
        static_cast<CUdeviceptr>(view.address));
    if (ranged != CUDA_SUCCESS)
        return status(SEEN_CUDA_INVALID_ARGUMENT,
                      static_cast<int32_t>(ranged), device, operation,
                      "buffer allocation extent is unavailable");
    const uintptr_t address = static_cast<uintptr_t>(view.address);
    const uintptr_t base = static_cast<uintptr_t>(allocation_base);
    if (address < base || address - base > allocation_bytes ||
        view.byte_length > allocation_bytes - (address - base) ||
        required_bytes > allocation_bytes - (address - base))
        return invalid(device, operation,
                       "buffer view exceeds its CUDA allocation");
    return status(SEEN_CUDA_OK, 0, device, operation, "ok");
}

SeenCudaStatus launch_status(cudaError_t error, int32_t device,
                             const char *operation) {
    return error == cudaSuccess
        ? status(SEEN_CUDA_OK, 0, device, operation, "ok")
        : status(SEEN_CUDA_RUNTIME_ERROR, static_cast<int32_t>(error), device,
                 operation, cudaGetErrorString(error));
}

bool launch_shape(uint64_t count, uint32_t *blocks) {
    if (count == 0) return false;
    const uint64_t value = (count + kThreads - 1) / kThreads;
    if (value == 0 || value > std::numeric_limits<uint32_t>::max()) return false;
    *blocks = static_cast<uint32_t>(value);
    return true;
}

template <typename T>
T *pointer(const SeenQwenCudaBufferView &view) {
    return reinterpret_cast<T *>(static_cast<uintptr_t>(view.address));
}

bool overlaps(const SeenQwenCudaBufferView &left, uint64_t left_bytes,
              const SeenQwenCudaBufferView &right, uint64_t right_bytes) {
    return left.address < right.address + right_bytes &&
           right.address < left.address + left_bytes;
}

__global__ void fill_f32(float *output, uint64_t count, float value) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < count) output[index] = value;
}

__global__ void add_f32(const float *left, const float *right, float *output,
                        uint64_t count) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < count) output[index] = left[index] + right[index];
}

__global__ void copy_f32(const float *input, float *output, uint64_t count) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < count) output[index] = input[index];
}

__global__ void transpose_f32(const float *input, float *output,
                              uint64_t rows, uint64_t columns,
                              uint64_t count) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < count) {
        const uint64_t row = index / columns;
        const uint64_t column = index - row * columns;
        output[column * rows + row] = input[index];
    }
}

__global__ void row_sum_f32(const float *input, float *output,
                            uint64_t rows, uint64_t columns) {
    const uint64_t row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) return;
    float sum = 0.0f;
    for (uint64_t column = 0; column < columns; ++column)
        sum += input[row * columns + column];
    output[row] = sum;
}

__global__ void embedding_gather_f32(const float *table,
                                     const int32_t *token_ids, float *output,
                                     uint64_t token_count,
                                     uint64_t vocabulary_size,
                                     uint64_t width, uint64_t count) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= count) return;
    const uint64_t token = index / width;
    const uint64_t column = index - token * width;
    const int32_t id = token_ids[token];
    output[index] = id >= 0 && static_cast<uint64_t>(id) < vocabulary_size
        ? table[static_cast<uint64_t>(id) * width + column]
        : __int_as_float(0x7fffffff);
}

}  // namespace

extern "C" SeenCudaStatus seen_qwen_fill_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView output,
    uint64_t count, float value) {
    constexpr const char *op = "seen_qwen_fill_f32";
    cudaStream_t stream{};
    SeenCudaStatus checked = validate_token(token, op, &stream);
    if (checked.code != SEEN_CUDA_OK) return checked;
    uint64_t bytes = 0; uint32_t blocks = 0;
    if (!checked_multiply(count, sizeof(float), &bytes) ||
        !launch_shape(count, &blocks)) return invalid(token->device_ordinal, op, "invalid element count");
    checked = validate_view(output, bytes, token->device_ordinal, op);
    if (checked.code != SEEN_CUDA_OK) return checked;
    fill_f32<<<blocks, kThreads, 0, stream>>>(pointer<float>(output), count, value);
    return launch_status(cudaPeekAtLastError(), token->device_ordinal, op);
}

extern "C" SeenCudaStatus seen_qwen_add_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView left,
    SeenQwenCudaBufferView right, SeenQwenCudaBufferView output,
    uint64_t count) {
    constexpr const char *op = "seen_qwen_add_f32";
    cudaStream_t stream{}; SeenCudaStatus checked = validate_token(token, op, &stream);
    if (checked.code != SEEN_CUDA_OK) return checked;
    uint64_t bytes = 0; uint32_t blocks = 0;
    if (!checked_multiply(count, sizeof(float), &bytes) || !launch_shape(count, &blocks))
        return invalid(token->device_ordinal, op, "invalid element count");
    for (const auto &view : {left, right, output}) {
        checked = validate_view(view, bytes, token->device_ordinal, op);
        if (checked.code != SEEN_CUDA_OK) return checked;
    }
    if ((overlaps(left, bytes, output, bytes) && left.address != output.address) ||
        (overlaps(right, bytes, output, bytes) && right.address != output.address))
        return invalid(token->device_ordinal, op,
                       "partially overlapping add buffers are unsupported");
    add_f32<<<blocks, kThreads, 0, stream>>>(pointer<const float>(left), pointer<const float>(right), pointer<float>(output), count);
    return launch_status(cudaPeekAtLastError(), token->device_ordinal, op);
}

extern "C" SeenCudaStatus seen_qwen_row_sum_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView input,
    SeenQwenCudaBufferView output, uint64_t rows, uint64_t columns) {
    constexpr const char *op = "seen_qwen_row_sum_f32";
    cudaStream_t stream{}; SeenCudaStatus checked = validate_token(token, op, &stream);
    if (checked.code != SEEN_CUDA_OK) return checked;
    uint64_t count = 0, input_bytes = 0, output_bytes = 0;
    if (!checked_multiply(rows, columns, &count) || !checked_multiply(count, sizeof(float), &input_bytes) ||
        !checked_multiply(rows, sizeof(float), &output_bytes) || rows == 0 || columns == 0 || rows > UINT32_MAX)
        return invalid(token->device_ordinal, op, "invalid matrix geometry");
    checked = validate_view(input, input_bytes, token->device_ordinal, op); if (checked.code != SEEN_CUDA_OK) return checked;
    checked = validate_view(output, output_bytes, token->device_ordinal, op); if (checked.code != SEEN_CUDA_OK) return checked;
    if (overlaps(input, input_bytes, output, output_bytes))
        return invalid(token->device_ordinal, op,
                       "overlapping reduction buffers are unsupported");
    row_sum_f32<<<static_cast<uint32_t>(rows), 1, 0, stream>>>(pointer<const float>(input), pointer<float>(output), rows, columns);
    return launch_status(cudaPeekAtLastError(), token->device_ordinal, op);
}

extern "C" SeenCudaStatus seen_qwen_transpose_2d_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView input,
    SeenQwenCudaBufferView output, uint64_t rows, uint64_t columns) {
    constexpr const char *op = "seen_qwen_transpose_2d_f32";
    cudaStream_t stream{}; SeenCudaStatus checked = validate_token(token, op, &stream);
    if (checked.code != SEEN_CUDA_OK) return checked;
    uint64_t count = 0, bytes = 0; uint32_t blocks = 0;
    if (!checked_multiply(rows, columns, &count) || !checked_multiply(count, sizeof(float), &bytes) || !launch_shape(count, &blocks))
        return invalid(token->device_ordinal, op, "invalid matrix geometry");
    checked = validate_view(input, bytes, token->device_ordinal, op); if (checked.code != SEEN_CUDA_OK) return checked;
    checked = validate_view(output, bytes, token->device_ordinal, op); if (checked.code != SEEN_CUDA_OK) return checked;
    if (overlaps(input, bytes, output, bytes))
        return invalid(token->device_ordinal, op,
                       "overlapping transpose buffers are unsupported");
    transpose_f32<<<blocks, kThreads, 0, stream>>>(pointer<const float>(input), pointer<float>(output), rows, columns, count);
    return launch_status(cudaPeekAtLastError(), token->device_ordinal, op);
}

extern "C" SeenCudaStatus seen_qwen_copy_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView input,
    SeenQwenCudaBufferView output, uint64_t count) {
    constexpr const char *op = "seen_qwen_copy_f32";
    cudaStream_t stream{}; SeenCudaStatus checked = validate_token(token, op, &stream);
    if (checked.code != SEEN_CUDA_OK) return checked;
    uint64_t bytes = 0; uint32_t blocks = 0;
    if (!checked_multiply(count, sizeof(float), &bytes) || !launch_shape(count, &blocks))
        return invalid(token->device_ordinal, op, "invalid element count");
    checked = validate_view(input, bytes, token->device_ordinal, op); if (checked.code != SEEN_CUDA_OK) return checked;
    checked = validate_view(output, bytes, token->device_ordinal, op); if (checked.code != SEEN_CUDA_OK) return checked;
    if (overlaps(input, bytes, output, bytes) && input.address != output.address)
        return invalid(token->device_ordinal, op,
                       "partially overlapping copy buffers are unsupported");
    copy_f32<<<blocks, kThreads, 0, stream>>>(pointer<const float>(input), pointer<float>(output), count);
    return launch_status(cudaPeekAtLastError(), token->device_ordinal, op);
}

extern "C" SeenCudaStatus seen_qwen_embedding_gather_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView table,
    SeenQwenCudaBufferView token_ids, SeenQwenCudaBufferView output,
    uint64_t token_count, uint64_t vocabulary_size, uint64_t width) {
    constexpr const char *op = "seen_qwen_embedding_gather_f32";
    cudaStream_t stream{}; SeenCudaStatus checked = validate_token(token, op, &stream);
    if (checked.code != SEEN_CUDA_OK) return checked;
    uint64_t table_count = 0, output_count = 0, table_bytes = 0, ids_bytes = 0, output_bytes = 0; uint32_t blocks = 0;
    if (!checked_multiply(vocabulary_size, width, &table_count) || !checked_multiply(token_count, width, &output_count) ||
        !checked_multiply(table_count, sizeof(float), &table_bytes) || !checked_multiply(token_count, sizeof(int32_t), &ids_bytes) ||
        !checked_multiply(output_count, sizeof(float), &output_bytes) || !launch_shape(output_count, &blocks))
        return invalid(token->device_ordinal, op, "invalid embedding geometry");
    for (const auto &pair : {std::pair<SeenQwenCudaBufferView, uint64_t>{table, table_bytes}, {token_ids, ids_bytes}, {output, output_bytes}}) {
        checked = validate_view(pair.first, pair.second, token->device_ordinal, op);
        if (checked.code != SEEN_CUDA_OK) return checked;
    }
    if (overlaps(table, table_bytes, output, output_bytes) ||
        overlaps(token_ids, ids_bytes, output, output_bytes))
        return invalid(token->device_ordinal, op,
                       "overlapping embedding buffers are unsupported");
    embedding_gather_f32<<<blocks, kThreads, 0, stream>>>(pointer<const float>(table), pointer<const int32_t>(token_ids), pointer<float>(output), token_count, vocabulary_size, width, output_count);
    return launch_status(cudaPeekAtLastError(), token->device_ordinal, op);
}
