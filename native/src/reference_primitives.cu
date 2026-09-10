#include "seen_qwen_cuda.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cmath>
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

__global__ void rms_norm_f32(const float *input, const float *weight,
                             float *output, uint64_t rows, uint64_t width,
                             float epsilon) {
    const uint64_t row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) return;
    float sum = 0.0f;
    for (uint64_t column = 0; column < width; ++column) {
        const float value = input[row * width + column];
        sum = __fadd_rn(sum, __fmul_rn(value, value));
    }
    const float inverse = __fdividef(1.0f, sqrtf(__fadd_rn(
        __fdividef(sum, static_cast<float>(width)), epsilon)));
    for (uint64_t column = 0; column < width; ++column) {
        const uint64_t index = row * width + column;
        output[index] = __fmul_rn(__fmul_rn(input[index], inverse),
                                  __fadd_rn(1.0f, weight[column]));
    }
}

__global__ void gdn_gated_rms_norm_f32(const float *core, const float *gate,
                                       const float *weight, float *output,
                                       uint64_t rows, uint64_t width,
                                       float epsilon) {
    const uint64_t row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) return;
    float sum = 0.0f;
    for (uint64_t column = 0; column < width; ++column) {
        const float value = core[row * width + column];
        sum = __fadd_rn(sum, __fmul_rn(value, value));
    }
    const float inverse = __fdividef(1.0f, sqrtf(__fadd_rn(
        __fdividef(sum, static_cast<float>(width)), epsilon)));
    for (uint64_t column = 0; column < width; ++column) {
        const uint64_t index = row * width + column;
        const float gate_value = gate[index];
        const float sigmoid = __fdividef(1.0f, __fadd_rn(1.0f, expf(-gate_value)));
        const float silu = __fmul_rn(gate_value, sigmoid);
        output[index] = __fmul_rn(weight[column],
            __fmul_rn(__fmul_rn(core[index], inverse), silu));
    }
}

__global__ void l2_norm_f32(const float *input, float *output,
                            uint64_t rows, uint64_t width, float epsilon) {
    const uint64_t row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) return;
    float sum = 0.0f;
    for (uint64_t column = 0; column < width; ++column) {
        const float value = input[row * width + column];
        sum = __fadd_rn(sum, __fmul_rn(value, value));
    }
    const float inverse = __fdividef(1.0f, sqrtf(__fadd_rn(sum, epsilon)));
    for (uint64_t column = 0; column < width; ++column) {
        const uint64_t index = row * width + column;
        output[index] = __fmul_rn(input[index], inverse);
    }
}

__global__ void silu_f32(const float *input, float *output, uint64_t count) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < count)
        output[index] = __fdividef(input[index], __fadd_rn(1.0f, expf(-input[index])));
}

__global__ void swiglu_f32(const float *gate, const float *up, float *output,
                           uint64_t count) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < count) {
        const float activated = __fdividef(gate[index],
            __fadd_rn(1.0f, expf(-gate[index])));
        output[index] = __fmul_rn(activated, up[index]);
    }
}

__global__ void sigmoid_gate_f32(const float *input, const float *gate,
                                 float *output, uint64_t count) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < count) {
        const float sigmoid = __fdividef(1.0f,
            __fadd_rn(1.0f, expf(-gate[index])));
        output[index] = __fmul_rn(input[index], sigmoid);
    }
}

__global__ void partial_rope_f32(const float *input, float *output,
                                 uint64_t count, uint64_t heads,
                                 uint64_t head_dim, uint64_t rotary_dim,
                                 uint64_t position_offset, float theta) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= count) return;
    const uint64_t dimension = index % head_dim;
    if (dimension >= rotary_dim) { output[index] = input[index]; return; }
    const uint64_t half = rotary_dim / 2;
    const uint64_t token = index / (heads * head_dim);
    const uint64_t row_base = index - dimension;
    const uint64_t frequency_index = dimension % half;
    const float exponent = static_cast<float>(frequency_index * 2) /
        static_cast<float>(rotary_dim);
    const float angle = static_cast<float>(position_offset + token) /
        powf(theta, exponent);
    const uint64_t rotated_dimension = dimension < half
        ? dimension + half : dimension - half;
    const float rotated = dimension < half
        ? -input[row_base + rotated_dimension]
        : input[row_base + rotated_dimension];
    output[index] = __fadd_rn(__fmul_rn(input[index], cosf(angle)),
                              __fmul_rn(rotated, sinf(angle)));
}

__device__ float attention_rope_value(const float *row, const float *weight,
                                      uint64_t dimension,
                                      uint64_t rotary_dim, uint64_t position,
                                      float theta, float inverse) {
    const float direct = __fmul_rn(__fmul_rn(row[dimension], inverse),
                                   __fadd_rn(1.0f, weight[dimension]));
    if (dimension >= rotary_dim) return direct;
    const uint64_t half = rotary_dim / 2;
    const uint64_t frequency_index = dimension % half;
    const float exponent = static_cast<float>(frequency_index * 2) /
        static_cast<float>(rotary_dim);
    const float angle = static_cast<float>(position) / powf(theta, exponent);
    const uint64_t paired = dimension < half
        ? dimension + half : dimension - half;
    const float normalized_pair = __fmul_rn(
        __fmul_rn(row[paired], inverse),
        __fadd_rn(1.0f, weight[paired]));
    const float rotated = dimension < half
        ? -normalized_pair : normalized_pair;
    return __fadd_rn(__fmul_rn(direct, cosf(angle)),
                     __fmul_rn(rotated, sinf(angle)));
}

__global__ void attention_query_rope_f32(
    const float *query_gate_projection, const float *weight,
    float *query_output, float *gate_output, uint64_t rows,
    uint64_t query_heads, uint64_t head_dim, uint64_t rotary_dim,
    uint64_t position_offset, float theta, float epsilon) {
    const uint64_t row_index = blockIdx.x;
    if (row_index >= rows || threadIdx.x != 0) return;
    const float *row = query_gate_projection + row_index * head_dim * 2;
    float sum = 0.0f;
    for (uint64_t dimension = 0; dimension < head_dim; ++dimension)
        sum = __fadd_rn(sum, __fmul_rn(row[dimension], row[dimension]));
    const float inverse = __fdividef(1.0f, sqrtf(__fadd_rn(
        __fdividef(sum, static_cast<float>(head_dim)), epsilon)));
    const uint64_t position = position_offset + row_index / query_heads;
    for (uint64_t dimension = 0; dimension < head_dim; ++dimension) {
        const uint64_t output_index = row_index * head_dim + dimension;
        query_output[output_index] = attention_rope_value(
            row, weight, dimension, rotary_dim, position, theta, inverse);
        gate_output[output_index] = row[head_dim + dimension];
    }
}

__global__ void attention_key_rope_f32(
    const float *key_projection, const float *weight, float *key_output,
    uint64_t rows, uint64_t kv_heads, uint64_t head_dim,
    uint64_t rotary_dim, uint64_t position_offset, float theta,
    float epsilon) {
    const uint64_t row_index = blockIdx.x;
    if (row_index >= rows || threadIdx.x != 0) return;
    const float *row = key_projection + row_index * head_dim;
    float sum = 0.0f;
    for (uint64_t dimension = 0; dimension < head_dim; ++dimension)
        sum = __fadd_rn(sum, __fmul_rn(row[dimension], row[dimension]));
    const float inverse = __fdividef(1.0f, sqrtf(__fadd_rn(
        __fdividef(sum, static_cast<float>(head_dim)), epsilon)));
    const uint64_t position = position_offset + row_index / kv_heads;
    for (uint64_t dimension = 0; dimension < head_dim; ++dimension) {
        const uint64_t output_index = row_index * head_dim + dimension;
        key_output[output_index] = attention_rope_value(
            row, weight, dimension, rotary_dim, position, theta, inverse);
    }
}

__global__ void kv_append_f32(const float *keys, const float *values,
                              float *key_cache, float *value_cache,
                              uint64_t count, uint64_t destination_offset) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < count) {
        key_cache[destination_offset + index] = keys[index];
        value_cache[destination_offset + index] = values[index];
    }
}

__global__ void attention_decode_f32(
    const float *query, const float *key_cache, const float *value_cache,
    float *output, uint64_t query_heads, uint64_t kv_heads,
    uint64_t head_dim, uint64_t cache_length) {
    const uint64_t query_head = blockIdx.x;
    if (query_head >= query_heads) return;
    const uint64_t kv_head = query_head / (query_heads / kv_heads);
    const uint64_t query_base = query_head * head_dim;
    const uint64_t output_base = query_head * head_dim;
    __shared__ float reduction[kThreads];
    __shared__ float running_max;
    __shared__ float denominator;
    __shared__ float previous_scale;
    __shared__ float current_scale;

    for (uint64_t dimension = threadIdx.x; dimension < head_dim;
         dimension += blockDim.x)
        output[output_base + dimension] = 0.0f;
    if (threadIdx.x == 0) {
        running_max = -INFINITY;
        denominator = 0.0f;
    }
    __syncthreads();

    const float scale = rsqrtf(static_cast<float>(head_dim));
    for (uint64_t position = 0; position < cache_length; ++position) {
        const uint64_t cache_base =
            (position * kv_heads + kv_head) * head_dim;
        float partial = 0.0f;
        for (uint64_t dimension = threadIdx.x; dimension < head_dim;
             dimension += blockDim.x)
            partial = __fadd_rn(partial, __fmul_rn(
                query[query_base + dimension],
                key_cache[cache_base + dimension]));
        reduction[threadIdx.x] = partial;
        __syncthreads();
        for (uint32_t width = kThreads / 2; width != 0; width /= 2) {
            if (threadIdx.x < width)
                reduction[threadIdx.x] = __fadd_rn(
                    reduction[threadIdx.x], reduction[threadIdx.x + width]);
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float score = __fmul_rn(reduction[0], scale);
            const float next_max = fmaxf(running_max, score);
            previous_scale = position == 0 ? 0.0f
                : expf(running_max - next_max);
            current_scale = expf(score - next_max);
            denominator = __fadd_rn(__fmul_rn(
                denominator, previous_scale), current_scale);
            running_max = next_max;
        }
        __syncthreads();
        for (uint64_t dimension = threadIdx.x; dimension < head_dim;
             dimension += blockDim.x) {
            const uint64_t index = output_base + dimension;
            output[index] = __fadd_rn(__fmul_rn(output[index], previous_scale),
                __fmul_rn(value_cache[cache_base + dimension], current_scale));
        }
        __syncthreads();
    }
    for (uint64_t dimension = threadIdx.x; dimension < head_dim;
         dimension += blockDim.x)
        output[output_base + dimension] = __fdividef(
            output[output_base + dimension], denominator);
}

__global__ void causal_conv_silu_f32(const float *input, const float *weights,
                                     float *history, float *output,
                                     uint64_t token_count, uint64_t channels,
                                     uint64_t kernel_length) {
    const uint64_t channel = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (channel >= channels) return;
    const uint64_t history_slots = kernel_length - 1;
    for (uint64_t token = 0; token < token_count; ++token) {
        float sum = 0.0f;
        for (uint64_t tap = 0; tap < history_slots; ++tap)
            sum = __fadd_rn(sum, __fmul_rn(
                history[tap * channels + channel],
                weights[channel * kernel_length + tap]));
        const uint64_t input_index = token * channels + channel;
        const float current = input[input_index];
        sum = __fadd_rn(sum, __fmul_rn(current,
            weights[channel * kernel_length + history_slots]));
        output[input_index] = __fdividef(sum, __fadd_rn(1.0f, expf(-sum)));
        for (uint64_t tap = 0; tap + 1 < history_slots; ++tap)
            history[tap * channels + channel] =
                history[(tap + 1) * channels + channel];
        if (history_slots != 0)
            history[(history_slots - 1) * channels + channel] = current;
    }
}

__device__ void gdn_recurrent_step_f32(
    const float *query, const float *key, const float *value,
    const float *beta, const float *log_decay, float *state, float *output,
    uint64_t head, uint64_t key_base, uint64_t value_base,
    uint64_t scalar_index, uint64_t key_dim, uint64_t value_dim) {
    const uint64_t state_base = head * key_dim * value_dim;
    const float decay = expf(log_decay[scalar_index]);
    const float step = beta[scalar_index];
    for (uint64_t column = 0; column < value_dim; ++column) {
        float memory = 0.0f;
        for (uint64_t row = 0; row < key_dim; ++row) {
            const uint64_t index = state_base + row * value_dim + column;
            const float decayed = __fmul_rn(state[index], decay);
            state[index] = decayed;
            memory = __fadd_rn(memory,
                __fmul_rn(decayed, key[key_base + row]));
        }
        const float delta = __fmul_rn(
            __fsub_rn(value[value_base + column], memory), step);
        float result = 0.0f;
        for (uint64_t row = 0; row < key_dim; ++row) {
            const uint64_t index = state_base + row * value_dim + column;
            const float updated = __fadd_rn(state[index],
                __fmul_rn(key[key_base + row], delta));
            state[index] = updated;
            result = __fadd_rn(result,
                __fmul_rn(updated, query[key_base + row]));
        }
        output[value_base + column] = result;
    }
}

__global__ void gdn_recurrent_decode_f32(
    const float *query, const float *key, const float *value,
    const float *beta, const float *log_decay, float *state, float *output,
    uint64_t value_heads, uint64_t key_dim, uint64_t value_dim) {
    const uint64_t head = static_cast<uint64_t>(blockIdx.x);
    if (head >= value_heads || threadIdx.x != 0) return;
    gdn_recurrent_step_f32(query, key, value, beta, log_decay, state, output,
        head, head * key_dim, head * value_dim, head, key_dim, value_dim);
}

__global__ void gdn_recurrent_prefill_f32(
    const float *query, const float *key, const float *value,
    const float *beta, const float *log_decay, float *state, float *output,
    uint64_t token_count, uint64_t value_heads, uint64_t key_dim,
    uint64_t value_dim) {
    const uint64_t head = static_cast<uint64_t>(blockIdx.x);
    if (head >= value_heads || threadIdx.x != 0) return;
    for (uint64_t token = 0; token < token_count; ++token) {
        const uint64_t scalar_index = token * value_heads + head;
        gdn_recurrent_step_f32(query, key, value, beta, log_decay, state,
            output, head, scalar_index * key_dim,
            scalar_index * value_dim, scalar_index, key_dim, value_dim);
    }
}

__global__ void greedy_argmax_f32(const float *logits, int32_t *token_ids,
                                  uint64_t rows, uint64_t width,
                                  uint64_t vocabulary_size) {
    const uint64_t row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) return;
    const float *values = logits + row * width;
    uint64_t best = 0;
    for (uint64_t token = 1; token < vocabulary_size; ++token) {
        if ((isnan(values[best]) && !isnan(values[token])) ||
            values[token] > values[best]) best = token;
    }
    token_ids[row] = static_cast<int32_t>(best);
}

__global__ void top_k_f32(const float *logits, int32_t *token_ids,
                          float *top_values, uint64_t rows, uint64_t width,
                          uint64_t vocabulary_size, uint64_t top_k) {
    const uint64_t row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) return;
    const float *values = logits + row * width;
    for (uint64_t rank = 0; rank < top_k; ++rank) {
        uint64_t best = UINT64_MAX;
        for (uint64_t token = 0; token < vocabulary_size; ++token) {
            bool selected = false;
            for (uint64_t previous = 0; previous < rank; ++previous)
                if (token_ids[row * top_k + previous] == static_cast<int32_t>(token))
                    selected = true;
            if (selected) continue;
            if (best == UINT64_MAX ||
                (isnan(values[best]) && !isnan(values[token])) ||
                values[token] > values[best]) best = token;
        }
        token_ids[row * top_k + rank] = static_cast<int32_t>(best);
        top_values[row * top_k + rank] = values[best];
    }
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

extern "C" SeenCudaStatus seen_qwen_rms_norm_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView input,
    SeenQwenCudaBufferView weight, SeenQwenCudaBufferView output,
    uint64_t rows, uint64_t width, float epsilon) {
    constexpr const char *op = "seen_qwen_rms_norm_f32";
    cudaStream_t stream{}; SeenCudaStatus checked = validate_token(token, op, &stream);
    if (checked.code != SEEN_CUDA_OK) return checked;
    uint64_t count = 0, bytes = 0, weight_bytes = 0;
    if (!checked_multiply(rows, width, &count) ||
        !checked_multiply(count, sizeof(float), &bytes) ||
        !checked_multiply(width, sizeof(float), &weight_bytes) ||
        rows == 0 || width == 0 || rows > UINT32_MAX ||
        !(epsilon > 0.0f) || !std::isfinite(epsilon))
        return invalid(token->device_ordinal, op, "invalid RMSNorm geometry or epsilon");
    for (const auto &pair : {std::pair<SeenQwenCudaBufferView, uint64_t>{input, bytes},
             {weight, weight_bytes}, {output, bytes}}) {
        checked = validate_view(pair.first, pair.second, token->device_ordinal, op);
        if (checked.code != SEEN_CUDA_OK) return checked;
    }
    if ((overlaps(input, bytes, output, bytes) && input.address != output.address) ||
        overlaps(weight, weight_bytes, output, bytes))
        return invalid(token->device_ordinal, op, "unsupported RMSNorm buffer overlap");
    rms_norm_f32<<<static_cast<uint32_t>(rows), 1, 0, stream>>>(
        pointer<const float>(input), pointer<const float>(weight),
        pointer<float>(output), rows, width, epsilon);
    return launch_status(cudaPeekAtLastError(), token->device_ordinal, op);
}

extern "C" SeenCudaStatus seen_qwen_gdn_gated_rms_norm_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView core,
    SeenQwenCudaBufferView gate, SeenQwenCudaBufferView weight,
    SeenQwenCudaBufferView output, uint64_t rows, uint64_t width,
    float epsilon) {
    constexpr const char *op = "seen_qwen_gdn_gated_rms_norm_f32";
    cudaStream_t stream{};
    SeenCudaStatus checked = validate_token(token, op, &stream);
    if (checked.code != SEEN_CUDA_OK) return checked;
    uint64_t count = 0, bytes = 0, weight_bytes = 0;
    if (!checked_multiply(rows, width, &count) ||
        !checked_multiply(count, sizeof(float), &bytes) ||
        !checked_multiply(width, sizeof(float), &weight_bytes) ||
        rows == 0 || width == 0 || rows > UINT32_MAX ||
        !(epsilon > 0.0f) || !std::isfinite(epsilon))
        return invalid(token->device_ordinal, op,
                       "invalid gated GDN RMSNorm geometry or epsilon");
    for (const auto &pair : {
             std::pair<SeenQwenCudaBufferView, uint64_t>{core, bytes},
             {gate, bytes}, {weight, weight_bytes}, {output, bytes}}) {
        checked = validate_view(pair.first, pair.second,
                                token->device_ordinal, op);
        if (checked.code != SEEN_CUDA_OK) return checked;
    }
    const bool exact_core_alias = core.address == output.address;
    if ((overlaps(core, bytes, output, bytes) && !exact_core_alias) ||
        overlaps(gate, bytes, output, bytes) ||
        overlaps(weight, weight_bytes, output, bytes) ||
        overlaps(core, bytes, gate, bytes) ||
        overlaps(core, bytes, weight, weight_bytes) ||
        overlaps(gate, bytes, weight, weight_bytes))
        return invalid(token->device_ordinal, op,
                       "unsupported gated GDN RMSNorm buffer overlap");
    gdn_gated_rms_norm_f32<<<static_cast<uint32_t>(rows), 1, 0, stream>>>(
        pointer<const float>(core), pointer<const float>(gate),
        pointer<const float>(weight), pointer<float>(output),
        rows, width, epsilon);
    return launch_status(cudaPeekAtLastError(), token->device_ordinal, op);
}

extern "C" SeenCudaStatus seen_qwen_l2_norm_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView input,
    SeenQwenCudaBufferView output, uint64_t rows, uint64_t width,
    float epsilon) {
    constexpr const char *op = "seen_qwen_l2_norm_f32";
    cudaStream_t stream{}; SeenCudaStatus checked = validate_token(token, op, &stream);
    if (checked.code != SEEN_CUDA_OK) return checked;
    uint64_t count = 0, bytes = 0;
    if (!checked_multiply(rows, width, &count) ||
        !checked_multiply(count, sizeof(float), &bytes) || rows == 0 ||
        width == 0 || rows > UINT32_MAX || !(epsilon > 0.0f) ||
        !std::isfinite(epsilon))
        return invalid(token->device_ordinal, op, "invalid L2 normalization geometry or epsilon");
    checked = validate_view(input, bytes, token->device_ordinal, op);
    if (checked.code != SEEN_CUDA_OK) return checked;
    checked = validate_view(output, bytes, token->device_ordinal, op);
    if (checked.code != SEEN_CUDA_OK) return checked;
    if (overlaps(input, bytes, output, bytes) && input.address != output.address)
        return invalid(token->device_ordinal, op, "partially overlapping L2 buffers are unsupported");
    l2_norm_f32<<<static_cast<uint32_t>(rows), 1, 0, stream>>>(
        pointer<const float>(input), pointer<float>(output), rows, width, epsilon);
    return launch_status(cudaPeekAtLastError(), token->device_ordinal, op);
}

extern "C" SeenCudaStatus seen_qwen_silu_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView input,
    SeenQwenCudaBufferView output, uint64_t count) {
    constexpr const char *op = "seen_qwen_silu_f32";
    cudaStream_t stream{}; SeenCudaStatus checked = validate_token(token, op, &stream);
    if (checked.code != SEEN_CUDA_OK) return checked;
    uint64_t bytes = 0; uint32_t blocks = 0;
    if (!checked_multiply(count, sizeof(float), &bytes) || !launch_shape(count, &blocks))
        return invalid(token->device_ordinal, op, "invalid element count");
    checked = validate_view(input, bytes, token->device_ordinal, op);
    if (checked.code != SEEN_CUDA_OK) return checked;
    checked = validate_view(output, bytes, token->device_ordinal, op);
    if (checked.code != SEEN_CUDA_OK) return checked;
    if (overlaps(input, bytes, output, bytes) && input.address != output.address)
        return invalid(token->device_ordinal, op, "partially overlapping SiLU buffers are unsupported");
    silu_f32<<<blocks, kThreads, 0, stream>>>(pointer<const float>(input), pointer<float>(output), count);
    return launch_status(cudaPeekAtLastError(), token->device_ordinal, op);
}

extern "C" SeenCudaStatus seen_qwen_swiglu_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView gate,
    SeenQwenCudaBufferView up, SeenQwenCudaBufferView output, uint64_t count) {
    constexpr const char *op = "seen_qwen_swiglu_f32";
    cudaStream_t stream{}; SeenCudaStatus checked = validate_token(token, op, &stream);
    if (checked.code != SEEN_CUDA_OK) return checked;
    uint64_t bytes = 0; uint32_t blocks = 0;
    if (!checked_multiply(count, sizeof(float), &bytes) || !launch_shape(count, &blocks))
        return invalid(token->device_ordinal, op, "invalid element count");
    for (const auto &view : {gate, up, output}) {
        checked = validate_view(view, bytes, token->device_ordinal, op);
        if (checked.code != SEEN_CUDA_OK) return checked;
    }
    if ((overlaps(gate, bytes, output, bytes) && gate.address != output.address) ||
        (overlaps(up, bytes, output, bytes) && up.address != output.address))
        return invalid(token->device_ordinal, op, "partially overlapping SwiGLU buffers are unsupported");
    swiglu_f32<<<blocks, kThreads, 0, stream>>>(pointer<const float>(gate), pointer<const float>(up), pointer<float>(output), count);
    return launch_status(cudaPeekAtLastError(), token->device_ordinal, op);
}

extern "C" SeenCudaStatus seen_qwen_sigmoid_gate_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView input,
    SeenQwenCudaBufferView gate, SeenQwenCudaBufferView output, uint64_t count) {
    constexpr const char *op = "seen_qwen_sigmoid_gate_f32";
    cudaStream_t stream{}; SeenCudaStatus checked = validate_token(token, op, &stream);
    if (checked.code != SEEN_CUDA_OK) return checked;
    uint64_t bytes = 0; uint32_t blocks = 0;
    if (!checked_multiply(count, sizeof(float), &bytes) || !launch_shape(count, &blocks))
        return invalid(token->device_ordinal, op, "invalid element count");
    for (const auto &view : {input, gate, output}) {
        checked = validate_view(view, bytes, token->device_ordinal, op);
        if (checked.code != SEEN_CUDA_OK) return checked;
    }
    if ((overlaps(input, bytes, output, bytes) && input.address != output.address) ||
        (overlaps(gate, bytes, output, bytes) && gate.address != output.address))
        return invalid(token->device_ordinal, op, "partially overlapping gate buffers are unsupported");
    sigmoid_gate_f32<<<blocks, kThreads, 0, stream>>>(pointer<const float>(input), pointer<const float>(gate), pointer<float>(output), count);
    return launch_status(cudaPeekAtLastError(), token->device_ordinal, op);
}

extern "C" SeenCudaStatus seen_qwen_partial_rope_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView input,
    SeenQwenCudaBufferView output, uint64_t tokens, uint64_t heads,
    uint64_t head_dim, uint64_t rotary_dim, uint64_t position_offset,
    uint64_t max_position, float theta) {
    constexpr const char *op = "seen_qwen_partial_rope_f32";
    cudaStream_t stream{}; SeenCudaStatus checked = validate_token(token, op, &stream);
    if (checked.code != SEEN_CUDA_OK) return checked;
    uint64_t rows = 0, count = 0, bytes = 0; uint32_t blocks = 0;
    if (!checked_multiply(tokens, heads, &rows) ||
        !checked_multiply(rows, head_dim, &count) ||
        !checked_multiply(count, sizeof(float), &bytes) ||
        !launch_shape(count, &blocks) || rotary_dim == 0 ||
        rotary_dim > head_dim || rotary_dim % 2 != 0 ||
        !(theta > 0.0f) || !std::isfinite(theta) ||
        position_offset > max_position || tokens > max_position - position_offset)
        return invalid(token->device_ordinal, op, "invalid bounded RoPE geometry");
    checked = validate_view(input, bytes, token->device_ordinal, op);
    if (checked.code != SEEN_CUDA_OK) return checked;
    checked = validate_view(output, bytes, token->device_ordinal, op);
    if (checked.code != SEEN_CUDA_OK) return checked;
    if (overlaps(input, bytes, output, bytes))
        return invalid(token->device_ordinal, op, "overlapping RoPE buffers are unsupported");
    partial_rope_f32<<<blocks, kThreads, 0, stream>>>(pointer<const float>(input),
        pointer<float>(output), count, heads, head_dim, rotary_dim,
        position_offset, theta);
    return launch_status(cudaPeekAtLastError(), token->device_ordinal, op);
}

extern "C" SeenCudaStatus seen_qwen_attention_qk_rope_f32(
    const SeenCudaStreamLaunchToken *token,
    SeenQwenCudaBufferView query_gate_projection,
    SeenQwenCudaBufferView key_projection,
    SeenQwenCudaBufferView query_norm_weight,
    SeenQwenCudaBufferView key_norm_weight,
    SeenQwenCudaBufferView query_output,
    SeenQwenCudaBufferView key_output,
    SeenQwenCudaBufferView gate_output,
    uint64_t tokens, uint64_t query_heads, uint64_t kv_heads,
    uint64_t head_dim, uint64_t rotary_dim, uint64_t position_offset,
    uint64_t max_position, float theta, float epsilon) {
    constexpr const char *op = "seen_qwen_attention_qk_rope_f32";
    cudaStream_t stream{};
    SeenCudaStatus checked = validate_token(token, op, &stream);
    if (checked.code != SEEN_CUDA_OK) return checked;
    uint64_t query_rows = 0, key_rows = 0, query_count = 0, key_count = 0;
    uint64_t query_projection_count = 0, query_bytes = 0, key_bytes = 0;
    uint64_t query_projection_bytes = 0, weight_bytes = 0;
    if (tokens == 0 || query_heads == 0 || kv_heads == 0 ||
        query_heads % kv_heads != 0 || head_dim == 0 || rotary_dim == 0 ||
        rotary_dim > head_dim || rotary_dim % 2 != 0 ||
        position_offset >= max_position || tokens > max_position - position_offset ||
        !(theta > 0.0f) || !std::isfinite(theta) || !(epsilon > 0.0f) ||
        !std::isfinite(epsilon) ||
        !checked_multiply(tokens, query_heads, &query_rows) ||
        !checked_multiply(tokens, kv_heads, &key_rows) ||
        query_rows > UINT32_MAX || key_rows > UINT32_MAX ||
        !checked_multiply(query_rows, head_dim, &query_count) ||
        !checked_multiply(key_rows, head_dim, &key_count) ||
        !checked_multiply(query_count, 2, &query_projection_count) ||
        !checked_multiply(query_count, sizeof(float), &query_bytes) ||
        !checked_multiply(key_count, sizeof(float), &key_bytes) ||
        !checked_multiply(query_projection_count, sizeof(float),
                          &query_projection_bytes) ||
        !checked_multiply(head_dim, sizeof(float), &weight_bytes))
        return invalid(token->device_ordinal, op,
                       "invalid attention Q/K projection or RoPE geometry");
    const std::pair<SeenQwenCudaBufferView, uint64_t> views[] = {
        {query_gate_projection, query_projection_bytes},
        {key_projection, key_bytes},
        {query_norm_weight, weight_bytes}, {key_norm_weight, weight_bytes},
        {query_output, query_bytes}, {key_output, key_bytes},
        {gate_output, query_bytes},
    };
    for (const auto &pair : views) {
        checked = validate_view(pair.first, pair.second,
                                token->device_ordinal, op);
        if (checked.code != SEEN_CUDA_OK) return checked;
    }
    for (size_t left = 0; left < sizeof(views) / sizeof(views[0]); ++left)
        for (size_t right = left + 1;
             right < sizeof(views) / sizeof(views[0]); ++right)
            if (overlaps(views[left].first, views[left].second,
                         views[right].first, views[right].second))
                return invalid(token->device_ordinal, op,
                               "attention Q/K buffers must be disjoint");
    attention_query_rope_f32<<<static_cast<uint32_t>(query_rows), 1, 0, stream>>>(
                  pointer<const float>(query_gate_projection),
                  pointer<const float>(query_norm_weight),
                  pointer<float>(query_output), pointer<float>(gate_output),
                  query_rows, query_heads, head_dim, rotary_dim,
                  position_offset, theta, epsilon);
    cudaError_t launched = cudaPeekAtLastError();
    if (launched != cudaSuccess)
        return launch_status(launched, token->device_ordinal, op);
    attention_key_rope_f32<<<static_cast<uint32_t>(key_rows), 1, 0, stream>>>(
                  pointer<const float>(key_projection),
                  pointer<const float>(key_norm_weight),
                  pointer<float>(key_output), key_rows, kv_heads, head_dim,
                  rotary_dim, position_offset, theta, epsilon);
    return launch_status(cudaPeekAtLastError(), token->device_ordinal, op);
}

extern "C" SeenCudaStatus seen_qwen_kv_append_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView keys,
    SeenQwenCudaBufferView values, SeenQwenCudaBufferView key_cache,
    SeenQwenCudaBufferView value_cache, uint64_t token_count,
    uint64_t kv_heads, uint64_t head_dim, uint64_t start_position,
    uint64_t capacity) {
    constexpr const char *op = "seen_qwen_kv_append_f32";
    cudaStream_t stream{}; SeenCudaStatus checked = validate_token(token, op, &stream);
    if (checked.code != SEEN_CUDA_OK) return checked;
    uint64_t token_width = 0, count = 0, cache_count = 0, bytes = 0;
    uint64_t cache_bytes = 0, destination_offset = 0; uint32_t blocks = 0;
    if (!checked_multiply(kv_heads, head_dim, &token_width) ||
        !checked_multiply(token_count, token_width, &count) ||
        !checked_multiply(capacity, token_width, &cache_count) ||
        !checked_multiply(count, sizeof(float), &bytes) ||
        !checked_multiply(cache_count, sizeof(float), &cache_bytes) ||
        !checked_multiply(start_position, token_width, &destination_offset) ||
        !launch_shape(count, &blocks) || start_position > capacity ||
        token_count > capacity - start_position)
        return invalid(token->device_ordinal, op, "invalid bounded KV append geometry");
    for (const auto &pair : {std::pair<SeenQwenCudaBufferView, uint64_t>{keys, bytes},
             {values, bytes}, {key_cache, cache_bytes}, {value_cache, cache_bytes}}) {
        checked = validate_view(pair.first, pair.second, token->device_ordinal, op);
        if (checked.code != SEEN_CUDA_OK) return checked;
    }
    if (overlaps(keys, bytes, key_cache, cache_bytes) ||
        overlaps(keys, bytes, value_cache, cache_bytes) ||
        overlaps(values, bytes, key_cache, cache_bytes) ||
        overlaps(values, bytes, value_cache, cache_bytes) ||
        overlaps(key_cache, cache_bytes, value_cache, cache_bytes))
        return invalid(token->device_ordinal, op, "overlapping KV buffers are unsupported");
    kv_append_f32<<<blocks, kThreads, 0, stream>>>(pointer<const float>(keys),
        pointer<const float>(values), pointer<float>(key_cache),
        pointer<float>(value_cache), count, destination_offset);
    return launch_status(cudaPeekAtLastError(), token->device_ordinal, op);
}

extern "C" SeenCudaStatus seen_qwen_attention_decode_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView query,
    SeenQwenCudaBufferView key_cache, SeenQwenCudaBufferView value_cache,
    SeenQwenCudaBufferView output, uint64_t query_heads, uint64_t kv_heads,
    uint64_t head_dim, uint64_t cache_length, uint64_t cache_capacity) {
    constexpr const char *op = "seen_qwen_attention_decode_f32";
    cudaStream_t stream{};
    SeenCudaStatus checked = validate_token(token, op, &stream);
    if (checked.code != SEEN_CUDA_OK) return checked;
    uint64_t query_count = 0, cache_count = 0;
    uint64_t query_bytes = 0, cache_bytes = 0;
    if (query_heads == 0 || kv_heads == 0 ||
        query_heads % kv_heads != 0 || head_dim == 0 ||
        cache_length == 0 || cache_length > cache_capacity ||
        query_heads > INT32_MAX ||
        !checked_multiply(query_heads, head_dim, &query_count) ||
        !checked_multiply(cache_capacity, kv_heads, &cache_count) ||
        !checked_multiply(cache_count, head_dim, &cache_count) ||
        !checked_multiply(query_count, sizeof(float), &query_bytes) ||
        !checked_multiply(cache_count, sizeof(float), &cache_bytes))
        return invalid(token->device_ordinal, op,
                       "invalid bounded attention decode geometry");
    const std::pair<SeenQwenCudaBufferView, uint64_t> views[] = {
        {query, query_bytes}, {key_cache, cache_bytes},
        {value_cache, cache_bytes}, {output, query_bytes},
    };
    for (const auto &pair : views) {
        checked = validate_view(pair.first, pair.second,
                                token->device_ordinal, op);
        if (checked.code != SEEN_CUDA_OK) return checked;
    }
    for (size_t left = 0; left < sizeof(views) / sizeof(views[0]); ++left)
        for (size_t right = left + 1;
             right < sizeof(views) / sizeof(views[0]); ++right)
            if (overlaps(views[left].first, views[left].second,
                         views[right].first, views[right].second))
                return invalid(token->device_ordinal, op,
                               "attention decode buffers must be disjoint");
    attention_decode_f32<<<static_cast<uint32_t>(query_heads), kThreads, 0, stream>>>(
        pointer<const float>(query), pointer<const float>(key_cache),
        pointer<const float>(value_cache), pointer<float>(output),
        query_heads, kv_heads, head_dim, cache_length);
    return launch_status(cudaPeekAtLastError(), token->device_ordinal, op);
}

extern "C" SeenCudaStatus seen_qwen_greedy_argmax_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView logits,
    SeenQwenCudaBufferView token_id, uint64_t rows, uint64_t width,
    uint64_t vocabulary_size) {
    constexpr const char *op = "seen_qwen_greedy_argmax_f32";
    cudaStream_t stream{}; SeenCudaStatus checked = validate_token(token, op, &stream);
    if (checked.code != SEEN_CUDA_OK) return checked;
    uint64_t count = 0, bytes = 0, id_bytes = 0;
    if (!checked_multiply(rows, width, &count) ||
        !checked_multiply(count, sizeof(float), &bytes) ||
        !checked_multiply(rows, sizeof(int32_t), &id_bytes) || rows == 0 ||
        rows > UINT32_MAX || vocabulary_size == 0 ||
        vocabulary_size > width || vocabulary_size > INT32_MAX)
        return invalid(token->device_ordinal, op, "invalid greedy sampling geometry");
    checked = validate_view(logits, bytes, token->device_ordinal, op);
    if (checked.code != SEEN_CUDA_OK) return checked;
    checked = validate_view(token_id, id_bytes, token->device_ordinal, op);
    if (checked.code != SEEN_CUDA_OK) return checked;
    if (overlaps(logits, bytes, token_id, id_bytes))
        return invalid(token->device_ordinal, op, "overlapping greedy buffers are unsupported");
    greedy_argmax_f32<<<static_cast<uint32_t>(rows), 1, 0, stream>>>(
        pointer<const float>(logits), pointer<int32_t>(token_id), rows, width,
        vocabulary_size);
    return launch_status(cudaPeekAtLastError(), token->device_ordinal, op);
}

extern "C" SeenCudaStatus seen_qwen_top_k_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView logits,
    SeenQwenCudaBufferView token_ids, SeenQwenCudaBufferView values,
    uint64_t rows, uint64_t width, uint64_t vocabulary_size, uint64_t top_k) {
    constexpr const char *op = "seen_qwen_top_k_f32";
    cudaStream_t stream{}; SeenCudaStatus checked = validate_token(token, op, &stream);
    if (checked.code != SEEN_CUDA_OK) return checked;
    uint64_t count = 0, bytes = 0, output_count = 0, id_bytes = 0, value_bytes = 0;
    if (!checked_multiply(rows, width, &count) ||
        !checked_multiply(count, sizeof(float), &bytes) ||
        !checked_multiply(rows, top_k, &output_count) ||
        !checked_multiply(output_count, sizeof(int32_t), &id_bytes) ||
        !checked_multiply(output_count, sizeof(float), &value_bytes) ||
        rows == 0 || rows > UINT32_MAX || top_k == 0 ||
        top_k > vocabulary_size || vocabulary_size > width ||
        vocabulary_size > INT32_MAX)
        return invalid(token->device_ordinal, op, "invalid top-k sampling geometry");
    for (const auto &pair : {std::pair<SeenQwenCudaBufferView, uint64_t>{logits, bytes},
             {token_ids, id_bytes}, {values, value_bytes}}) {
        checked = validate_view(pair.first, pair.second, token->device_ordinal, op);
        if (checked.code != SEEN_CUDA_OK) return checked;
    }
    if (overlaps(logits, bytes, token_ids, id_bytes) ||
        overlaps(logits, bytes, values, value_bytes) ||
        overlaps(token_ids, id_bytes, values, value_bytes))
        return invalid(token->device_ordinal, op, "overlapping top-k buffers are unsupported");
    top_k_f32<<<static_cast<uint32_t>(rows), 1, 0, stream>>>(
        pointer<const float>(logits), pointer<int32_t>(token_ids),
        pointer<float>(values), rows, width, vocabulary_size, top_k);
    return launch_status(cudaPeekAtLastError(), token->device_ordinal, op);
}

extern "C" SeenCudaStatus seen_qwen_causal_conv_silu_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView input,
    SeenQwenCudaBufferView weights, SeenQwenCudaBufferView history,
    SeenQwenCudaBufferView output, uint64_t token_count, uint64_t channels,
    uint64_t kernel_length, uint64_t start_position,
    uint64_t processed_position) {
    constexpr const char *op = "seen_qwen_causal_conv_silu_f32";
    cudaStream_t stream{};
    SeenCudaStatus checked = validate_token(token, op, &stream);
    if (checked.code != SEEN_CUDA_OK) return checked;
    uint64_t element_count = 0, weight_count = 0, history_count = 0;
    uint64_t input_bytes = 0, weight_bytes = 0, history_bytes = 0;
    uint32_t blocks = 0;
    if (kernel_length != 4 || token_count == 0 || channels == 0 ||
        start_position != processed_position ||
        token_count > std::numeric_limits<uint64_t>::max() - processed_position ||
        !checked_multiply(token_count, channels, &element_count) ||
        !checked_multiply(channels, kernel_length, &weight_count) ||
        !checked_multiply(channels, kernel_length - 1, &history_count) ||
        !checked_multiply(element_count, sizeof(float), &input_bytes) ||
        !checked_multiply(weight_count, sizeof(float), &weight_bytes) ||
        !checked_multiply(history_count, sizeof(float), &history_bytes) ||
        !launch_shape(channels, &blocks))
        return invalid(token->device_ordinal, op,
                       "invalid causal-convolution state or geometry");
    for (const auto &pair : {
             std::pair<SeenQwenCudaBufferView, uint64_t>{input, input_bytes},
             {weights, weight_bytes}, {history, history_bytes},
             {output, input_bytes}}) {
        checked = validate_view(pair.first, pair.second,
                                token->device_ordinal, op);
        if (checked.code != SEEN_CUDA_OK) return checked;
    }
    const bool exact_in_place = input.address == output.address;
    if ((overlaps(input, input_bytes, output, input_bytes) && !exact_in_place) ||
        overlaps(input, input_bytes, weights, weight_bytes) ||
        overlaps(input, input_bytes, history, history_bytes) ||
        overlaps(output, input_bytes, weights, weight_bytes) ||
        overlaps(output, input_bytes, history, history_bytes) ||
        overlaps(weights, weight_bytes, history, history_bytes))
        return invalid(token->device_ordinal, op,
                       "unsupported causal-convolution buffer overlap");
    causal_conv_silu_f32<<<blocks, kThreads, 0, stream>>>(
        pointer<const float>(input), pointer<const float>(weights),
        pointer<float>(history), pointer<float>(output), token_count, channels,
        kernel_length);
    return launch_status(cudaPeekAtLastError(), token->device_ordinal, op);
}

extern "C" SeenCudaStatus seen_qwen_gdn_recurrent_decode_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView query,
    SeenQwenCudaBufferView key, SeenQwenCudaBufferView value,
    SeenQwenCudaBufferView beta, SeenQwenCudaBufferView log_decay,
    SeenQwenCudaBufferView state, SeenQwenCudaBufferView output,
    uint64_t value_heads, uint64_t key_dim, uint64_t value_dim,
    uint64_t start_position, uint64_t processed_position) {
    constexpr const char *op = "seen_qwen_gdn_recurrent_decode_f32";
    cudaStream_t stream{};
    SeenCudaStatus checked = validate_token(token, op, &stream);
    if (checked.code != SEEN_CUDA_OK) return checked;
    uint64_t key_count = 0, value_count = 0, state_count = 0;
    uint64_t key_bytes = 0, value_bytes = 0, state_bytes = 0;
    if (value_heads == 0 || value_heads > UINT32_MAX || key_dim == 0 ||
        value_dim == 0 || start_position != processed_position ||
        processed_position == std::numeric_limits<uint64_t>::max() ||
        !checked_multiply(value_heads, key_dim, &key_count) ||
        !checked_multiply(value_heads, value_dim, &value_count) ||
        !checked_multiply(key_count, value_dim, &state_count) ||
        !checked_multiply(key_count, sizeof(float), &key_bytes) ||
        !checked_multiply(value_count, sizeof(float), &value_bytes) ||
        !checked_multiply(state_count, sizeof(float), &state_bytes))
        return invalid(token->device_ordinal, op,
                       "invalid recurrent decode state or geometry");
    const uint64_t scalar_bytes = value_heads * sizeof(float);
    for (const auto &pair : {
             std::pair<SeenQwenCudaBufferView, uint64_t>{query, key_bytes},
             {key, key_bytes}, {value, value_bytes}, {beta, scalar_bytes},
             {log_decay, scalar_bytes}, {state, state_bytes},
             {output, value_bytes}}) {
        checked = validate_view(pair.first, pair.second,
                                token->device_ordinal, op);
        if (checked.code != SEEN_CUDA_OK) return checked;
    }
    const std::pair<SeenQwenCudaBufferView, uint64_t> views[] = {
        {query, key_bytes}, {key, key_bytes}, {value, value_bytes},
        {beta, scalar_bytes}, {log_decay, scalar_bytes},
        {state, state_bytes}, {output, value_bytes}};
    for (size_t left = 0; left < sizeof(views) / sizeof(views[0]); ++left)
        for (size_t right = left + 1;
             right < sizeof(views) / sizeof(views[0]); ++right)
            if (overlaps(views[left].first, views[left].second,
                         views[right].first, views[right].second))
                return invalid(token->device_ordinal, op,
                               "overlapping recurrent decode buffers are unsupported");
    gdn_recurrent_decode_f32<<<static_cast<uint32_t>(value_heads), 1, 0, stream>>>(
        pointer<const float>(query), pointer<const float>(key),
        pointer<const float>(value), pointer<const float>(beta),
        pointer<const float>(log_decay), pointer<float>(state),
        pointer<float>(output), value_heads, key_dim, value_dim);
    return launch_status(cudaPeekAtLastError(), token->device_ordinal, op);
}

extern "C" SeenCudaStatus seen_qwen_gdn_recurrent_prefill_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView query,
    SeenQwenCudaBufferView key, SeenQwenCudaBufferView value,
    SeenQwenCudaBufferView beta, SeenQwenCudaBufferView log_decay,
    SeenQwenCudaBufferView state, SeenQwenCudaBufferView output,
    uint64_t token_count, uint64_t value_heads, uint64_t key_dim,
    uint64_t value_dim, uint64_t start_position,
    uint64_t processed_position) {
    constexpr const char *op = "seen_qwen_gdn_recurrent_prefill_f32";
    cudaStream_t stream{};
    SeenCudaStatus checked = validate_token(token, op, &stream);
    if (checked.code != SEEN_CUDA_OK) return checked;
    uint64_t token_heads = 0, key_count = 0, value_count = 0;
    uint64_t state_count = 0, state_rows = 0;
    uint64_t key_bytes = 0, value_bytes = 0, scalar_bytes = 0, state_bytes = 0;
    if (token_count == 0 || value_heads == 0 || value_heads > UINT32_MAX ||
        key_dim == 0 || value_dim == 0 ||
        start_position != processed_position ||
        token_count > std::numeric_limits<uint64_t>::max() - processed_position ||
        !checked_multiply(token_count, value_heads, &token_heads) ||
        !checked_multiply(token_heads, key_dim, &key_count) ||
        !checked_multiply(token_heads, value_dim, &value_count) ||
        !checked_multiply(value_heads, key_dim, &state_rows) ||
        !checked_multiply(state_rows, value_dim, &state_count) ||
        !checked_multiply(key_count, sizeof(float), &key_bytes) ||
        !checked_multiply(value_count, sizeof(float), &value_bytes) ||
        !checked_multiply(token_heads, sizeof(float), &scalar_bytes) ||
        !checked_multiply(state_count, sizeof(float), &state_bytes))
        return invalid(token->device_ordinal, op,
                       "invalid recurrent prefill state or geometry");
    for (const auto &pair : {
             std::pair<SeenQwenCudaBufferView, uint64_t>{query, key_bytes},
             {key, key_bytes}, {value, value_bytes}, {beta, scalar_bytes},
             {log_decay, scalar_bytes}, {state, state_bytes},
             {output, value_bytes}}) {
        checked = validate_view(pair.first, pair.second,
                                token->device_ordinal, op);
        if (checked.code != SEEN_CUDA_OK) return checked;
    }
    const std::pair<SeenQwenCudaBufferView, uint64_t> views[] = {
        {query, key_bytes}, {key, key_bytes}, {value, value_bytes},
        {beta, scalar_bytes}, {log_decay, scalar_bytes},
        {state, state_bytes}, {output, value_bytes}};
    for (size_t left = 0; left < sizeof(views) / sizeof(views[0]); ++left)
        for (size_t right = left + 1;
             right < sizeof(views) / sizeof(views[0]); ++right)
            if (overlaps(views[left].first, views[left].second,
                         views[right].first, views[right].second))
                return invalid(token->device_ordinal, op,
                               "overlapping recurrent prefill buffers are unsupported");
    gdn_recurrent_prefill_f32<<<static_cast<uint32_t>(value_heads), 1, 0, stream>>>(
        pointer<const float>(query), pointer<const float>(key),
        pointer<const float>(value), pointer<const float>(beta),
        pointer<const float>(log_decay), pointer<float>(state),
        pointer<float>(output), token_count, value_heads, key_dim, value_dim);
    return launch_status(cudaPeekAtLastError(), token->device_ordinal, op);
}
