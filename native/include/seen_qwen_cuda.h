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
SeenCudaStatus seen_qwen_rms_norm_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView input,
    SeenQwenCudaBufferView weight, SeenQwenCudaBufferView output,
    uint64_t rows, uint64_t width, float epsilon);
SeenCudaStatus seen_qwen_l2_norm_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView input,
    SeenQwenCudaBufferView output, uint64_t rows, uint64_t width,
    float epsilon);
SeenCudaStatus seen_qwen_silu_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView input,
    SeenQwenCudaBufferView output, uint64_t count);
SeenCudaStatus seen_qwen_swiglu_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView gate,
    SeenQwenCudaBufferView up, SeenQwenCudaBufferView output, uint64_t count);
SeenCudaStatus seen_qwen_sigmoid_gate_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView input,
    SeenQwenCudaBufferView gate, SeenQwenCudaBufferView output, uint64_t count);
SeenCudaStatus seen_qwen_partial_rope_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView input,
    SeenQwenCudaBufferView output, uint64_t tokens, uint64_t heads,
    uint64_t head_dim, uint64_t rotary_dim, uint64_t position_offset,
    uint64_t max_position, float theta);
SeenCudaStatus seen_qwen_attention_qk_rope_f32(
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
    uint64_t max_position, float theta, float epsilon);
SeenCudaStatus seen_qwen_kv_append_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView keys,
    SeenQwenCudaBufferView values, SeenQwenCudaBufferView key_cache,
    SeenQwenCudaBufferView value_cache, uint64_t token_count,
    uint64_t kv_heads, uint64_t head_dim, uint64_t start_position,
    uint64_t capacity);
SeenCudaStatus seen_qwen_attention_decode_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView query,
    SeenQwenCudaBufferView key_cache, SeenQwenCudaBufferView value_cache,
    SeenQwenCudaBufferView output, uint64_t query_heads, uint64_t kv_heads,
    uint64_t head_dim, uint64_t cache_length, uint64_t cache_capacity);
SeenCudaStatus seen_qwen_attention_prefill_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView query,
    SeenQwenCudaBufferView keys, SeenQwenCudaBufferView values,
    SeenQwenCudaBufferView key_cache, SeenQwenCudaBufferView value_cache,
    SeenQwenCudaBufferView output, uint64_t token_count,
    uint64_t query_heads, uint64_t kv_heads, uint64_t head_dim,
    uint64_t start_position, uint64_t cache_capacity);
SeenCudaStatus seen_qwen_greedy_argmax_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView logits,
    SeenQwenCudaBufferView token_id, uint64_t rows, uint64_t width,
    uint64_t vocabulary_size);
SeenCudaStatus seen_qwen_top_k_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView logits,
    SeenQwenCudaBufferView token_ids, SeenQwenCudaBufferView values,
    uint64_t rows, uint64_t width, uint64_t vocabulary_size, uint64_t top_k);
SeenCudaStatus seen_qwen_causal_conv_silu_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView input,
    SeenQwenCudaBufferView weights, SeenQwenCudaBufferView history,
    SeenQwenCudaBufferView output, uint64_t token_count, uint64_t channels,
    uint64_t kernel_length, uint64_t start_position,
    uint64_t processed_position);
SeenCudaStatus seen_qwen_gdn_recurrent_decode_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView query,
    SeenQwenCudaBufferView key, SeenQwenCudaBufferView value,
    SeenQwenCudaBufferView beta, SeenQwenCudaBufferView log_decay,
    SeenQwenCudaBufferView state, SeenQwenCudaBufferView output,
    uint64_t value_heads, uint64_t key_dim, uint64_t value_dim,
    uint64_t start_position, uint64_t processed_position);
SeenCudaStatus seen_qwen_gdn_recurrent_prefill_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView query,
    SeenQwenCudaBufferView key, SeenQwenCudaBufferView value,
    SeenQwenCudaBufferView beta, SeenQwenCudaBufferView log_decay,
    SeenQwenCudaBufferView state, SeenQwenCudaBufferView output,
    uint64_t token_count, uint64_t value_heads, uint64_t key_dim,
    uint64_t value_dim, uint64_t start_position,
    uint64_t processed_position);
SeenCudaStatus seen_qwen_gdn_gated_rms_norm_f32(
    const SeenCudaStreamLaunchToken *token, SeenQwenCudaBufferView core,
    SeenQwenCudaBufferView gate, SeenQwenCudaBufferView weight,
    SeenQwenCudaBufferView output, uint64_t rows, uint64_t width,
    float epsilon);

#ifdef __cplusplus
}
#endif

#endif
