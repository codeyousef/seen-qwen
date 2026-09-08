#ifndef SEEN_QWEN_CUDA_H
#define SEEN_QWEN_CUDA_H

#include "seen_cuda.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SEEN_QWEN_CUDA_REFERENCE_ABI_VERSION 1u

typedef struct SeenQwenCudaBufferView {
    uint32_t abi_version;
    int32_t device_ordinal;
    uint64_t address;
    uint64_t byte_length;
} SeenQwenCudaBufferView;

SeenCudaStatus seen_qwen_fill_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView output,
    uint64_t count, float value);
SeenCudaStatus seen_qwen_add_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView left,
    SeenQwenCudaBufferView right, SeenQwenCudaBufferView output,
    uint64_t count);
SeenCudaStatus seen_qwen_row_sum_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView input,
    SeenQwenCudaBufferView output, uint64_t rows, uint64_t columns);
SeenCudaStatus seen_qwen_transpose_2d_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView input,
    SeenQwenCudaBufferView output, uint64_t rows, uint64_t columns);
SeenCudaStatus seen_qwen_copy_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView input,
    SeenQwenCudaBufferView output, uint64_t count);
SeenCudaStatus seen_qwen_embedding_gather_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView table,
    SeenQwenCudaBufferView token_ids, SeenQwenCudaBufferView output,
    uint64_t token_count, uint64_t vocabulary_size, uint64_t width);

#ifdef __cplusplus
}
#endif

#endif
