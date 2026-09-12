#include "seen_qwen_cuda.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_STATUS(expr) do { SeenCudaStatus s_ = (expr); if (s_.code != SEEN_CUDA_OK) { \
    std::fprintf(stderr, "FAIL:%d: %s code=%d native=%d op=%s message=%s\n", \
        __LINE__, #expr, s_.code, s_.native_code, s_.operation, s_.message); return false; } } while (0)
#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "FAIL:%d: %s\n", __LINE__, #expr); return false; } } while (0)

namespace {

constexpr uint64_t kTokens = 2;
constexpr uint64_t kHidden = 8;
constexpr uint64_t kVocabulary = 16;
constexpr uint64_t kOfficialVocabulary = 248320;
constexpr uint64_t kWorkspaceBytes = 8ull << 20;

SeenCudaMatmulDesc descriptor(uint64_t tokens, uint64_t hidden,
                              uint64_t vocabulary, int32_t type) {
    return SeenCudaMatmulDesc{SEEN_CUDA_ABI_VERSION, type, 1, 0,
        vocabulary, tokens, hidden, static_cast<int64_t>(hidden),
        static_cast<int64_t>(hidden), static_cast<int64_t>(vocabulary),
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

bool same_algorithm(const SeenCudaAlgorithm &left,
                    const SeenCudaAlgorithm &right) {
    return left.algorithm_id == right.algorithm_id &&
        left.tile_id == right.tile_id && left.split_k == right.split_k &&
        left.reduction_scheme == right.reduction_scheme &&
        left.workspace_bytes == right.workspace_bytes &&
        left.cache_identity == right.cache_identity;
}

template <typename T> T encode(float value);
template <> __half encode(float value) { return __float2half_rn(value); }
template <> __nv_bfloat16 encode(float value) { return __float2bfloat16(value); }

template <typename T> float decode(T value);
template <> float decode(__half value) { return __half2float(value); }
template <> float decode(__nv_bfloat16 value) { return __bfloat162float(value); }

template <typename T>
bool run_type(int32_t data_type, float tolerance) {
    const uint64_t hidden_bytes = kTokens * kHidden * sizeof(T);
    const uint64_t weight_bytes = kVocabulary * kHidden * sizeof(T);
    const uint64_t logits_bytes = kTokens * kVocabulary * sizeof(T);
    const uint64_t token_bytes = kTokens * sizeof(int32_t);
    const uint64_t sizes[5] = {hidden_bytes, weight_bytes, logits_bytes,
                               token_bytes, kWorkspaceBytes};
    SeenCudaHandle stream = 0, event = 0, cublas = 0, allocations[5]{};
    CHECK_STATUS(seen_cuda_stream_create(0, &stream));
    CHECK_STATUS(seen_cuda_event_create(0, &event));
    CHECK_STATUS(seen_cublaslt_create(0, &cublas));
    void *device[5]{};
    for (int index = 0; index < 5; ++index) {
        CHECK_STATUS(seen_cuda_malloc(0, sizes[index], &allocations[index]));
        uint64_t bytes = 0; int32_t ordinal = -1;
        CHECK_STATUS(seen_cuda_allocation_address(
            allocations[index], &device[index], &bytes, &ordinal));
        CHECK(bytes == sizes[index] && ordinal == 0);
    }

    std::vector<T> hidden(kTokens * kHidden, encode<T>(0.0f));
    std::vector<T> weight(kVocabulary * kHidden, encode<T>(0.0f));
    hidden[0] = encode<T>(1.0f);
    hidden[kHidden + 1] = encode<T>(1.0f);
    weight[3 * kHidden] = encode<T>(2.0f);
    weight[5 * kHidden] = encode<T>(2.0f);  // exact tie: lower ID wins
    weight[7 * kHidden + 1] = encode<T>(3.0f);
    const int32_t expected_tokens[2] = {3, 7};
    std::vector<T> expected(kTokens * kVocabulary, encode<T>(0.0f));
    for (uint64_t row = 0; row < kTokens; ++row)
        for (uint64_t output = 0; output < kVocabulary; ++output) {
            float sum = 0.0f;
            for (uint64_t column = 0; column < kHidden; ++column)
                sum += decode(hidden[row * kHidden + column]) *
                    decode(weight[output * kHidden + column]);
            expected[row * kVocabulary + output] = encode<T>(sum);
        }
    CHECK_STATUS(seen_cuda_memcpy_async(device[0], hidden.data(), hidden_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(device[1], weight.data(), weight_bytes,
        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));

    auto desc = descriptor(kTokens, kHidden, kVocabulary, data_type);
    SeenCudaAlgorithm algorithm{}, repeated{};
    CHECK_STATUS(seen_cublaslt_select_algorithm(cublas, &desc, &algorithm));
    CHECK_STATUS(seen_cublaslt_select_algorithm(cublas, &desc, &repeated));
    CHECK(same_algorithm(algorithm, repeated));
    CHECK(algorithm.workspace_bytes <= kWorkspaceBytes);
    auto official = descriptor(1, 5120, kOfficialVocabulary, data_type);
    SeenCudaAlgorithm official_algorithm{}, official_repeated{};
    CHECK_STATUS(seen_cublaslt_select_algorithm(cublas, &official,
        &official_algorithm));
    CHECK_STATUS(seen_cublaslt_select_algorithm(cublas, &official,
        &official_repeated));
    CHECK(same_algorithm(official_algorithm, official_repeated));
    CHECK(official_algorithm.workspace_bytes <= kWorkspaceBytes);

    auto enqueue = [&]() -> bool {
        CHECK_STATUS(seen_cublaslt_matmul(cublas, &desc, &algorithm,
            device[1], device[0], device[2], device[4], stream));
        auto token = borrow(stream);
        CHECK_STATUS(seen_qwen_greedy_argmax_low_precision(&token,
            view(device[2], logits_bytes), view(device[3], token_bytes),
            kTokens, kVocabulary, kVocabulary, data_type));
        return true;
    };

    CHECK(enqueue());
    std::vector<T> actual(kTokens * kVocabulary);
    int32_t actual_tokens[2] = {-1, -1};
    CHECK_STATUS(seen_cuda_memcpy_async(actual.data(), device[2], logits_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(actual_tokens, device[3], token_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (uint64_t index = 0; index < actual.size(); ++index)
        CHECK(std::fabs(decode(actual[index]) - decode(expected[index])) <=
            tolerance);
    CHECK(actual_tokens[0] == expected_tokens[0]);
    CHECK(actual_tokens[1] == expected_tokens[1]);

    // NaNs cannot displace a finite candidate; a wholly NaN row has the
    // stable token-zero result documented by the public contract.
    std::vector<T> nan_logits(kTokens * kVocabulary,
        encode<T>(std::numeric_limits<float>::quiet_NaN()));
    nan_logits[kVocabulary + 4] = encode<T>(1.0f);
    CHECK_STATUS(seen_cuda_memcpy_async(device[2], nan_logits.data(),
        logits_bytes, SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    auto nan_token = borrow(stream);
    CHECK_STATUS(seen_qwen_greedy_argmax_low_precision(&nan_token,
        view(device[2], logits_bytes), view(device[3], token_bytes),
        kTokens, kVocabulary, kVocabulary, data_type));
    CHECK_STATUS(seen_cuda_memcpy_async(actual_tokens, device[3], token_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    CHECK(actual_tokens[0] == 0);
    CHECK(actual_tokens[1] == 4);

    CHECK_STATUS(seen_cuda_graph_begin_capture(stream));
    CHECK(enqueue());
    SeenCudaHandle graph = 0, graph_exec = 0;
    CHECK_STATUS(seen_cuda_graph_end_capture(stream, &graph));
    CHECK_STATUS(seen_cuda_graph_instantiate(graph, &graph_exec));
    CHECK_STATUS(seen_cuda_graph_launch(graph_exec, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(actual_tokens, device[3], token_bytes,
        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    CHECK(actual_tokens[0] == expected_tokens[0]);
    CHECK(actual_tokens[1] == expected_tokens[1]);
    CHECK_STATUS(seen_cuda_graph_exec_destroy(&graph_exec));
    CHECK_STATUS(seen_cuda_graph_destroy(&graph));

    auto token = borrow(stream);
    CHECK(seen_qwen_greedy_argmax_low_precision(nullptr,
        view(device[2], logits_bytes), view(device[3], token_bytes),
        kTokens, kVocabulary, kVocabulary, data_type).code ==
        SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_greedy_argmax_low_precision(&token,
        view(device[2], logits_bytes), view(device[3], token_bytes),
        0, kVocabulary, kVocabulary, data_type).code ==
        SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_greedy_argmax_low_precision(&token,
        view(device[2], logits_bytes), view(device[3], token_bytes),
        kTokens, kVocabulary, kVocabulary + 1, data_type).code ==
        SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_greedy_argmax_low_precision(&token,
        view(device[2], logits_bytes), view(device[3], token_bytes),
        kTokens, kVocabulary, kVocabulary, SEEN_CUDA_F32).code ==
        SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_greedy_argmax_low_precision(&token,
        view(device[2], logits_bytes - sizeof(T)),
        view(device[3], token_bytes), kTokens, kVocabulary, kVocabulary,
        data_type).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_greedy_argmax_low_precision(&token,
        view(device[2], logits_bytes), view(device[2], token_bytes),
        kTokens, kVocabulary, kVocabulary, data_type).code ==
        SEEN_CUDA_INVALID_ARGUMENT);
    SeenCudaStreamLaunchToken incompatible = token;
    incompatible.generation = 0;
    CHECK(seen_qwen_greedy_argmax_low_precision(&incompatible,
        view(device[2], logits_bytes), view(device[3], token_bytes),
        kTokens, kVocabulary, kVocabulary, data_type).code ==
        SEEN_CUDA_INCOMPATIBLE);

    for (int iteration = 0; iteration < 1000; ++iteration) CHECK(enqueue());
    CHECK_STATUS(seen_cuda_stream_synchronize(stream));

    for (int index = 4; index >= 0; --index)
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
    std::printf("PASS: FEL-1442 resident F16/BF16 LM-head greedy ordering capture negatives 1000-session cleanup\n");
    return 0;
}
