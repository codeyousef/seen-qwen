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
constexpr uint64_t kOfficialKeyDim = 128;
constexpr uint64_t kOfficialValueDim = 128;
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

void reference(const float *query, const float *key, const float *value,
               const float *beta, const float *log_decay, float *state,
               float *output, uint64_t tokens, uint64_t heads,
               uint64_t key_dim, uint64_t value_dim) {
    for (uint64_t token = 0; token < tokens; ++token) {
        for (uint64_t head = 0; head < heads; ++head) {
            const uint64_t scalar = token * heads + head;
            const uint64_t key_base = scalar * key_dim;
            const uint64_t value_base = scalar * value_dim;
            const uint64_t state_base = head * key_dim * value_dim;
            const float decay = std::exp(log_decay[scalar]);
            for (uint64_t column = 0; column < value_dim; ++column) {
                float memory = 0.0f;
                for (uint64_t row = 0; row < key_dim; ++row) {
                    const uint64_t index = state_base + row * value_dim + column;
                    state[index] *= decay;
                    memory += state[index] * key[key_base + row];
                }
                const float delta = (value[value_base + column] - memory) * beta[scalar];
                float result = 0.0f;
                for (uint64_t row = 0; row < key_dim; ++row) {
                    const uint64_t index = state_base + row * value_dim + column;
                    state[index] += key[key_base + row] * delta;
                    result += state[index] * query[key_base + row];
                }
                output[value_base + column] = result;
            }
        }
    }
}

bool near(float actual, float expected, float tolerance = 3.0e-5f) {
    return std::fabs(actual - expected) <= tolerance;
}
}  // namespace

int main() {
    constexpr uint64_t max_key_count = kMaximumTokens * kOfficialHeads * kOfficialKeyDim;
    constexpr uint64_t max_value_count = kMaximumTokens * kOfficialHeads * kOfficialValueDim;
    constexpr uint64_t max_scalar_count = kMaximumTokens * kOfficialHeads;
    constexpr uint64_t max_state_count = kOfficialHeads * kOfficialKeyDim * kOfficialValueDim;
    const uint64_t max_key_bytes = max_key_count * sizeof(float);
    const uint64_t max_value_bytes = max_value_count * sizeof(float);
    const uint64_t max_scalar_bytes = max_scalar_count * sizeof(float);
    const uint64_t max_state_bytes = max_state_count * sizeof(float);
    SeenCudaHandle stream = 0, event = 0, allocations[7]{};
    CHECK_STATUS(seen_cuda_stream_create(0, &stream));
    CHECK_STATUS(seen_cuda_event_create(0, &event));
    CHECK_STATUS(seen_cuda_malloc(0, max_key_bytes, &allocations[0]));
    CHECK_STATUS(seen_cuda_malloc(0, max_key_bytes, &allocations[1]));
    CHECK_STATUS(seen_cuda_malloc(0, max_value_bytes, &allocations[2]));
    CHECK_STATUS(seen_cuda_malloc(0, max_scalar_bytes, &allocations[3]));
    CHECK_STATUS(seen_cuda_malloc(0, max_scalar_bytes, &allocations[4]));
    CHECK_STATUS(seen_cuda_malloc(0, max_state_bytes, &allocations[5]));
    CHECK_STATUS(seen_cuda_malloc(0, max_value_bytes, &allocations[6]));
    void *device[7]{}; uint64_t actual_bytes = 0; int32_t ordinal = -1;
    for (int index = 0; index < 7; ++index)
        CHECK_STATUS(seen_cuda_allocation_address(
            allocations[index], &device[index], &actual_bytes, &ordinal));

    constexpr uint64_t tokens = 7, heads = 3, key_dim = 4, value_dim = 5;
    const uint64_t key_count = tokens * heads * key_dim;
    const uint64_t value_count = tokens * heads * value_dim;
    const uint64_t scalar_count = tokens * heads;
    const uint64_t state_count = heads * key_dim * value_dim;
    const uint64_t key_bytes = key_count * sizeof(float);
    const uint64_t value_bytes = value_count * sizeof(float);
    const uint64_t scalar_bytes = scalar_count * sizeof(float);
    const uint64_t state_bytes = state_count * sizeof(float);
    std::vector<float> query(key_count), key(key_count), value(value_count);
    std::vector<float> beta(scalar_count), log_decay(scalar_count);
    std::vector<float> expected(value_count), expected_state(state_count);
    std::vector<float> actual(value_count), full_gpu(value_count);
    std::vector<float> actual_state(state_count), full_gpu_state(state_count);
    for (uint64_t index = 0; index < key_count; ++index) {
        query[index] = static_cast<float>(static_cast<int64_t>((index * 7) % 23) - 11) / 29.0f;
        key[index] = static_cast<float>(static_cast<int64_t>((index * 5) % 19) - 9) / 31.0f;
    }
    for (uint64_t index = 0; index < value_count; ++index)
        value[index] = static_cast<float>(static_cast<int64_t>((index * 3) % 17) - 8) / 13.0f;
    for (uint64_t index = 0; index < scalar_count; ++index) {
        beta[index] = 0.1f + static_cast<float>(index % 4) * 0.2f;
        log_decay[index] = -0.025f - static_cast<float>(index % 5) * 0.05f;
    }
    reference(query.data(), key.data(), value.data(), beta.data(), log_decay.data(),
              expected_state.data(), expected.data(), tokens, heads, key_dim, value_dim);
    CHECK_STATUS(seen_cuda_memcpy_async(device[0], query.data(), key_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[1], key.data(), key_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[2], value.data(), value_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[3], beta.data(), scalar_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[4], log_decay.data(), scalar_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memset_async(device[5], 0, state_bytes, stream));
    auto token = borrow(stream);
    CHECK_STATUS(seen_qwen_gdn_recurrent_prefill_f32(&token,
        view(device[0], key_bytes), view(device[1], key_bytes),
        view(device[2], value_bytes), view(device[3], scalar_bytes),
        view(device[4], scalar_bytes), view(device[5], state_bytes),
        view(device[6], value_bytes), tokens, heads, key_dim, value_dim, 0, 0));
    CHECK_STATUS(seen_cuda_memcpy_async(actual.data(), device[6], value_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(actual_state.data(), device[5], state_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (uint64_t index = 0; index < value_count; ++index)
        CHECK(near(actual[index], expected[index]));
    for (uint64_t index = 0; index < state_count; ++index)
        CHECK(near(actual_state[index], expected_state[index]));
    full_gpu = actual; full_gpu_state = actual_state;

    // Sequential decode is bit-identical to one prefill launch.
    CHECK_STATUS(seen_cuda_memset_async(device[5], 0, state_bytes, stream));
    CHECK_STATUS(seen_cuda_memset_async(device[6], 0, value_bytes, stream));
    const uint64_t token_key_bytes = heads * key_dim * sizeof(float);
    const uint64_t token_value_bytes = heads * value_dim * sizeof(float);
    const uint64_t token_scalar_bytes = heads * sizeof(float);
    for (uint64_t position = 0; position < tokens; ++position) {
        token = borrow(stream);
        CHECK_STATUS(seen_qwen_gdn_recurrent_decode_f32(&token,
            view(static_cast<float *>(device[0]) + position * heads * key_dim, token_key_bytes),
            view(static_cast<float *>(device[1]) + position * heads * key_dim, token_key_bytes),
            view(static_cast<float *>(device[2]) + position * heads * value_dim, token_value_bytes),
            view(static_cast<float *>(device[3]) + position * heads, token_scalar_bytes),
            view(static_cast<float *>(device[4]) + position * heads, token_scalar_bytes),
            view(device[5], state_bytes),
            view(static_cast<float *>(device[6]) + position * heads * value_dim,
                 token_value_bytes), heads, key_dim, value_dim, position, position));
    }
    CHECK_STATUS(seen_cuda_memcpy_async(actual.data(), device[6], value_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(actual_state.data(), device[5], state_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (uint64_t index = 0; index < value_count; ++index) CHECK(actual[index] == full_gpu[index]);
    for (uint64_t index = 0; index < state_count; ++index)
        CHECK(actual_state[index] == full_gpu_state[index]);

    // Arbitrary chunk boundaries are also bit-identical.
    CHECK_STATUS(seen_cuda_memset_async(device[5], 0, state_bytes, stream));
    CHECK_STATUS(seen_cuda_memset_async(device[6], 0, value_bytes, stream));
    uint64_t processed = 0;
    for (const uint64_t chunk : {2u, 1u, 4u}) {
        token = borrow(stream);
        CHECK_STATUS(seen_qwen_gdn_recurrent_prefill_f32(&token,
            view(static_cast<float *>(device[0]) + processed * heads * key_dim,
                 key_bytes - processed * token_key_bytes),
            view(static_cast<float *>(device[1]) + processed * heads * key_dim,
                 key_bytes - processed * token_key_bytes),
            view(static_cast<float *>(device[2]) + processed * heads * value_dim,
                 value_bytes - processed * token_value_bytes),
            view(static_cast<float *>(device[3]) + processed * heads,
                 scalar_bytes - processed * token_scalar_bytes),
            view(static_cast<float *>(device[4]) + processed * heads,
                 scalar_bytes - processed * token_scalar_bytes),
            view(device[5], state_bytes),
            view(static_cast<float *>(device[6]) + processed * heads * value_dim,
                 value_bytes - processed * token_value_bytes),
            chunk, heads, key_dim, value_dim, processed, processed));
        processed += chunk;
    }
    CHECK_STATUS(seen_cuda_memcpy_async(actual.data(), device[6], value_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(actual_state.data(), device[5], state_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (uint64_t index = 0; index < value_count; ++index) CHECK(actual[index] == full_gpu[index]);
    for (uint64_t index = 0; index < state_count; ++index)
        CHECK(actual_state[index] == full_gpu_state[index]);

    // The exact official 48 x 128 x 128 geometry admits a bounded two-token prefill.
    constexpr uint64_t official_tokens = 2;
    const uint64_t official_key_bytes = official_tokens * kOfficialHeads * kOfficialKeyDim * sizeof(float);
    const uint64_t official_value_bytes = official_tokens * kOfficialHeads * kOfficialValueDim * sizeof(float);
    const uint64_t official_scalar_bytes = official_tokens * kOfficialHeads * sizeof(float);
    std::vector<float> official_key(official_key_bytes / sizeof(float), 0.03125f);
    std::vector<float> official_value(official_value_bytes / sizeof(float), 0.25f);
    std::vector<float> official_beta(official_scalar_bytes / sizeof(float), 0.5f);
    std::vector<float> official_decay(official_scalar_bytes / sizeof(float), -0.125f);
    CHECK_STATUS(seen_cuda_memcpy_async(device[0], official_key.data(), official_key_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[1], official_key.data(), official_key_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[2], official_value.data(), official_value_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[3], official_beta.data(), official_scalar_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[4], official_decay.data(), official_scalar_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memset_async(device[5], 0, max_state_bytes, stream));
    token = borrow(stream);
    CHECK_STATUS(seen_qwen_gdn_recurrent_prefill_f32(&token,
        view(device[0], official_key_bytes), view(device[1], official_key_bytes),
        view(device[2], official_value_bytes), view(device[3], official_scalar_bytes),
        view(device[4], official_scalar_bytes), view(device[5], max_state_bytes),
        view(device[6], official_value_bytes), official_tokens, kOfficialHeads,
        kOfficialKeyDim, kOfficialValueDim, 262142, 262142));

    // Host-side rejection occurs before prefill state mutation.
    std::vector<float> sentinel(state_count, 0.75f), unchanged(state_count);
    CHECK_STATUS(seen_cuda_memcpy_async(device[5], sentinel.data(), state_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    token = borrow(stream);
    CHECK(seen_qwen_gdn_recurrent_prefill_f32(&token,
        view(device[0], key_bytes), view(device[1], key_bytes), view(device[2], value_bytes),
        view(device[3], scalar_bytes), view(device[4], scalar_bytes), view(device[5], state_bytes),
        view(device[6], value_bytes), tokens, heads, key_dim, value_dim, 1, 0).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_gdn_recurrent_prefill_f32(&token,
        view(device[0], key_bytes), view(device[1], key_bytes), view(device[2], value_bytes),
        view(device[3], scalar_bytes), view(device[4], scalar_bytes), view(device[5], state_bytes),
        view(device[6], value_bytes), 0, heads, key_dim, value_dim, 0, 0).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_gdn_recurrent_prefill_f32(&token,
        view(device[0], key_bytes - 1), view(device[1], key_bytes), view(device[2], value_bytes),
        view(device[3], scalar_bytes), view(device[4], scalar_bytes), view(device[5], state_bytes),
        view(device[6], value_bytes), tokens, heads, key_dim, value_dim, 0, 0).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_gdn_recurrent_prefill_f32(&token,
        view(device[5], state_bytes), view(device[1], key_bytes), view(device[2], value_bytes),
        view(device[3], scalar_bytes), view(device[4], scalar_bytes), view(device[5], state_bytes),
        view(device[6], value_bytes), tokens, heads, key_dim, value_dim, 0, 0).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_gdn_recurrent_prefill_f32(&token,
        view(device[0], key_bytes), view(device[1], key_bytes), view(device[2], value_bytes),
        view(device[3], scalar_bytes), view(device[4], scalar_bytes), view(device[5], state_bytes),
        view(device[6], value_bytes), 2, heads, key_dim, value_dim,
        std::numeric_limits<uint64_t>::max() - 1,
        std::numeric_limits<uint64_t>::max() - 1).code == SEEN_CUDA_INVALID_ARGUMENT);
    SeenCudaStreamLaunchToken incompatible = token; incompatible.generation = 0;
    CHECK(seen_qwen_gdn_recurrent_prefill_f32(&incompatible,
        view(device[0], key_bytes), view(device[1], key_bytes), view(device[2], value_bytes),
        view(device[3], scalar_bytes), view(device[4], scalar_bytes), view(device[5], state_bytes),
        view(device[6], value_bytes), tokens, heads, key_dim, value_dim, 0, 0).code == SEEN_CUDA_INCOMPATIBLE);
    CHECK(seen_qwen_gdn_recurrent_prefill_f32(nullptr,
        view(device[0], key_bytes), view(device[1], key_bytes), view(device[2], value_bytes),
        view(device[3], scalar_bytes), view(device[4], scalar_bytes), view(device[5], state_bytes),
        view(device[6], value_bytes), tokens, heads, key_dim, value_dim, 0, 0).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK_STATUS(seen_cuda_memcpy_async(unchanged.data(), device[5], state_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (uint64_t index = 0; index < state_count; ++index) CHECK(unchanged[index] == sentinel[index]);

    // Capture and replay use only the same Seen-owned stream.
    CHECK_STATUS(seen_cuda_memset_async(device[5], 0, state_bytes, stream));
    CHECK_STATUS(seen_cuda_graph_begin_capture(stream)); token = borrow(stream);
    CHECK_STATUS(seen_qwen_gdn_recurrent_prefill_f32(&token,
        view(device[0], key_bytes), view(device[1], key_bytes), view(device[2], value_bytes),
        view(device[3], scalar_bytes), view(device[4], scalar_bytes), view(device[5], state_bytes),
        view(device[6], value_bytes), tokens, heads, key_dim, value_dim, 0, 0));
    SeenCudaHandle graph = 0, graph_exec = 0;
    CHECK_STATUS(seen_cuda_graph_end_capture(stream, &graph));
    CHECK_STATUS(seen_cuda_graph_instantiate(graph, &graph_exec));
    CHECK_STATUS(seen_cuda_graph_launch(graph_exec, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    CHECK_STATUS(seen_cuda_graph_exec_destroy(&graph_exec));
    CHECK_STATUS(seen_cuda_graph_destroy(&graph));

    for (int index = 6; index >= 0; --index) CHECK_STATUS(seen_cuda_free(&allocations[index]));
    CHECK_STATUS(seen_cuda_event_destroy(&event));
    CHECK_STATUS(seen_cuda_stream_destroy(&stream));
    std::printf("PASS: FEL-1435 chunked GDN prefill differential decode equivalence bounds graph cleanup\n");
    return 0;
}
