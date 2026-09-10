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
               float *output, uint64_t heads, uint64_t key_dim,
               uint64_t value_dim) {
    for (uint64_t head = 0; head < heads; ++head) {
        const uint64_t key_base = head * key_dim;
        const uint64_t value_base = head * value_dim;
        const uint64_t state_base = head * key_dim * value_dim;
        const float decay = std::exp(log_decay[head]);
        for (uint64_t column = 0; column < value_dim; ++column) {
            float memory = 0.0f;
            for (uint64_t row = 0; row < key_dim; ++row) {
                const uint64_t index = state_base + row * value_dim + column;
                state[index] *= decay;
                memory += state[index] * key[key_base + row];
            }
            const float delta = (value[value_base + column] - memory) * beta[head];
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

bool near(float actual, float expected, float tolerance = 2.0e-5f) {
    return std::fabs(actual - expected) <= tolerance;
}

}  // namespace

int main() {
    constexpr uint64_t max_key_count = kOfficialHeads * kOfficialKeyDim;
    constexpr uint64_t max_value_count = kOfficialHeads * kOfficialValueDim;
    constexpr uint64_t max_state_count = max_key_count * kOfficialValueDim;
    const uint64_t max_key_bytes = max_key_count * sizeof(float);
    const uint64_t max_value_bytes = max_value_count * sizeof(float);
    const uint64_t max_scalar_bytes = kOfficialHeads * sizeof(float);
    const uint64_t max_state_bytes = max_state_count * sizeof(float);

    SeenCudaHandle stream = 0, event = 0;
    SeenCudaHandle allocations[7]{};
    CHECK_STATUS(seen_cuda_stream_create(0, &stream));
    CHECK_STATUS(seen_cuda_event_create(0, &event));
    CHECK_STATUS(seen_cuda_malloc(0, max_key_bytes, &allocations[0]));
    CHECK_STATUS(seen_cuda_malloc(0, max_key_bytes, &allocations[1]));
    CHECK_STATUS(seen_cuda_malloc(0, max_value_bytes, &allocations[2]));
    CHECK_STATUS(seen_cuda_malloc(0, max_scalar_bytes, &allocations[3]));
    CHECK_STATUS(seen_cuda_malloc(0, max_scalar_bytes, &allocations[4]));
    CHECK_STATUS(seen_cuda_malloc(0, max_state_bytes, &allocations[5]));
    CHECK_STATUS(seen_cuda_malloc(0, max_value_bytes, &allocations[6]));
    void *device[7]{};
    uint64_t actual_bytes = 0;
    int32_t ordinal = -1;
    for (int index = 0; index < 7; ++index)
        CHECK_STATUS(seen_cuda_allocation_address(
            allocations[index], &device[index], &actual_bytes, &ordinal));

    constexpr uint64_t heads = 3, key_dim = 4, value_dim = 5;
    const uint64_t key_count = heads * key_dim;
    const uint64_t value_count = heads * value_dim;
    const uint64_t state_count = key_count * value_dim;
    const uint64_t key_bytes = key_count * sizeof(float);
    const uint64_t value_bytes = value_count * sizeof(float);
    const uint64_t scalar_bytes = heads * sizeof(float);
    const uint64_t state_bytes = state_count * sizeof(float);
    std::vector<float> query(key_count), key(key_count), value(value_count);
    std::vector<float> beta(heads), log_decay(heads), expected_state(state_count);
    std::vector<float> expected(value_count), actual_state(state_count), actual(value_count);
    for (uint64_t index = 0; index < key_count; ++index) {
        query[index] = static_cast<float>(static_cast<int64_t>((index * 7) % 13) - 6) / 17.0f;
        key[index] = static_cast<float>(static_cast<int64_t>((index * 5) % 11) - 5) / 19.0f;
    }
    for (uint64_t index = 0; index < value_count; ++index)
        value[index] = static_cast<float>(static_cast<int64_t>((index * 3) % 17) - 8) / 13.0f;
    for (uint64_t head = 0; head < heads; ++head) {
        beta[head] = 0.2f + static_cast<float>(head) * 0.25f;
        log_decay[head] = -0.05f - static_cast<float>(head) * 0.1f;
    }
    reference(query.data(), key.data(), value.data(), beta.data(), log_decay.data(),
              expected_state.data(), expected.data(), heads, key_dim, value_dim);
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
    CHECK_STATUS(seen_qwen_gdn_recurrent_decode_f32(&token,
        view(device[0], key_bytes), view(device[1], key_bytes),
        view(device[2], value_bytes), view(device[3], scalar_bytes),
        view(device[4], scalar_bytes), view(device[5], state_bytes),
        view(device[6], value_bytes), heads, key_dim, value_dim, 0, 0));
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

    // A second decode step consumes exactly the state produced by the first.
    reference(query.data(), key.data(), value.data(), beta.data(), log_decay.data(),
              expected_state.data(), expected.data(), heads, key_dim, value_dim);
    token = borrow(stream);
    CHECK_STATUS(seen_qwen_gdn_recurrent_decode_f32(&token,
        view(device[0], key_bytes), view(device[1], key_bytes),
        view(device[2], value_bytes), view(device[3], scalar_bytes),
        view(device[4], scalar_bytes), view(device[5], state_bytes),
        view(device[6], value_bytes), heads, key_dim, value_dim, 1, 1));
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

    // The exact official 48 x 128 x 128 decode-state geometry is admitted.
    std::vector<float> official_key(max_key_count, 0.03125f);
    std::vector<float> official_value(max_value_count, 0.25f);
    std::vector<float> official_beta(kOfficialHeads, 0.5f);
    std::vector<float> official_decay(kOfficialHeads, -0.125f);
    CHECK_STATUS(seen_cuda_memcpy_async(device[0], official_key.data(), max_key_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[1], official_key.data(), max_key_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[2], official_value.data(), max_value_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[3], official_beta.data(), max_scalar_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[4], official_decay.data(), max_scalar_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memset_async(device[5], 0, max_state_bytes, stream));
    token = borrow(stream);
    CHECK_STATUS(seen_qwen_gdn_recurrent_decode_f32(&token,
        view(device[0], max_key_bytes), view(device[1], max_key_bytes),
        view(device[2], max_value_bytes), view(device[3], max_scalar_bytes),
        view(device[4], max_scalar_bytes), view(device[5], max_state_bytes),
        view(device[6], max_value_bytes), kOfficialHeads, kOfficialKeyDim,
        kOfficialValueDim, 262143, 262143));

    // Host-side rejection occurs before recurrent state mutation.
    std::vector<float> sentinel(state_count, 0.75f), unchanged(state_count);
    CHECK_STATUS(seen_cuda_memcpy_async(device[5], sentinel.data(), state_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    token = borrow(stream);
    CHECK(seen_qwen_gdn_recurrent_decode_f32(&token,
        view(device[0], key_bytes), view(device[1], key_bytes),
        view(device[2], value_bytes), view(device[3], scalar_bytes),
        view(device[4], scalar_bytes), view(device[5], state_bytes),
        view(device[6], value_bytes), heads, key_dim, value_dim, 2, 1).code ==
        SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_gdn_recurrent_decode_f32(&token,
        view(device[0], key_bytes - 1), view(device[1], key_bytes),
        view(device[2], value_bytes), view(device[3], scalar_bytes),
        view(device[4], scalar_bytes), view(device[5], state_bytes),
        view(device[6], value_bytes), heads, key_dim, value_dim, 1, 1).code ==
        SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_gdn_recurrent_decode_f32(&token,
        view(device[5], state_bytes), view(device[1], key_bytes),
        view(device[2], value_bytes), view(device[3], scalar_bytes),
        view(device[4], scalar_bytes), view(device[5], state_bytes),
        view(device[6], value_bytes), heads, key_dim, value_dim, 1, 1).code ==
        SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_gdn_recurrent_decode_f32(&token,
        view(device[0], key_bytes), view(device[1], key_bytes),
        view(device[2], value_bytes), view(device[3], scalar_bytes),
        view(device[4], scalar_bytes), view(device[5], state_bytes),
        view(device[6], value_bytes), 0, key_dim, value_dim, 1, 1).code ==
        SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_gdn_recurrent_decode_f32(&token,
        view(device[0], key_bytes), view(device[1], key_bytes),
        view(device[2], value_bytes), view(device[3], scalar_bytes),
        view(device[4], scalar_bytes), view(device[5], state_bytes),
        view(device[6], value_bytes), heads, key_dim, value_dim,
        std::numeric_limits<uint64_t>::max(),
        std::numeric_limits<uint64_t>::max()).code == SEEN_CUDA_INVALID_ARGUMENT);
    SeenCudaStreamLaunchToken incompatible = token;
    incompatible.generation = 0;
    CHECK(seen_qwen_gdn_recurrent_decode_f32(&incompatible,
        view(device[0], key_bytes), view(device[1], key_bytes),
        view(device[2], value_bytes), view(device[3], scalar_bytes),
        view(device[4], scalar_bytes), view(device[5], state_bytes),
        view(device[6], value_bytes), heads, key_dim, value_dim, 1, 1).code ==
        SEEN_CUDA_INCOMPATIBLE);
    CHECK(seen_qwen_gdn_recurrent_decode_f32(nullptr,
        view(device[0], key_bytes), view(device[1], key_bytes),
        view(device[2], value_bytes), view(device[3], scalar_bytes),
        view(device[4], scalar_bytes), view(device[5], state_bytes),
        view(device[6], value_bytes), heads, key_dim, value_dim, 1, 1).code ==
        SEEN_CUDA_INVALID_ARGUMENT);
    CHECK_STATUS(seen_cuda_memcpy_async(unchanged.data(), device[5], state_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (uint64_t index = 0; index < state_count; ++index)
        CHECK(unchanged[index] == sentinel[index]);

    // Capture and replay enqueue only on the same Seen-owned stream.
    CHECK_STATUS(seen_cuda_memset_async(device[5], 0, state_bytes, stream));
    CHECK_STATUS(seen_cuda_graph_begin_capture(stream));
    token = borrow(stream);
    CHECK_STATUS(seen_qwen_gdn_recurrent_decode_f32(&token,
        view(device[0], key_bytes), view(device[1], key_bytes),
        view(device[2], value_bytes), view(device[3], scalar_bytes),
        view(device[4], scalar_bytes), view(device[5], state_bytes),
        view(device[6], value_bytes), heads, key_dim, value_dim, 0, 0));
    SeenCudaHandle graph = 0, graph_exec = 0;
    CHECK_STATUS(seen_cuda_graph_end_capture(stream, &graph));
    CHECK_STATUS(seen_cuda_graph_instantiate(graph, &graph_exec));
    CHECK_STATUS(seen_cuda_graph_launch(graph_exec, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    CHECK_STATUS(seen_cuda_graph_exec_destroy(&graph_exec));
    CHECK_STATUS(seen_cuda_graph_destroy(&graph));

    for (int index = 6; index >= 0; --index)
        CHECK_STATUS(seen_cuda_free(&allocations[index]));
    CHECK_STATUS(seen_cuda_event_destroy(&event));
    CHECK_STATUS(seen_cuda_stream_destroy(&stream));
    std::printf("PASS: FEL-1436 recurrent GDN decode differential state bounds graph cleanup\n");
    return 0;
}
