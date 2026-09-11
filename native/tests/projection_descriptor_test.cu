#include "seen_cuda.h"

#include <cstdint>
#include <cstdio>
#include <limits>

#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); return 1; \
} } while (0)
#define CHECK_STATUS(expression) do { SeenCudaStatus status_ = (expression); \
    if (status_.code != SEEN_CUDA_OK) { \
        std::fprintf(stderr, "FAIL line %d: %s (%d: %s)\n", __LINE__, \
            #expression, status_.code, status_.message); return 1; \
    } \
} while (0)

static SeenCudaMatmulDesc descriptor(uint64_t tokens, uint64_t input,
                                     uint64_t output, int32_t type,
                                     uint64_t workspace) {
    return SeenCudaMatmulDesc{SEEN_CUDA_ABI_VERSION, type, 1, 0,
        output, tokens, input, static_cast<int64_t>(input),
        static_cast<int64_t>(input), static_cast<int64_t>(output), workspace};
}

static bool same_algorithm(const SeenCudaAlgorithm &left,
                           const SeenCudaAlgorithm &right) {
    return left.algorithm_id == right.algorithm_id &&
        left.tile_id == right.tile_id && left.split_k == right.split_k &&
        left.reduction_scheme == right.reduction_scheme &&
        left.workspace_bytes == right.workspace_bytes &&
        left.cache_identity == right.cache_identity;
}

int main() {
    SeenCudaHandle handle = 0;
    CHECK_STATUS(seen_cublaslt_create(0, &handle));
    CHECK(handle != 0);

    auto ffn = descriptor(16, 5120, 17408, SEEN_CUDA_BF16, 32ull << 20);
    SeenCudaAlgorithm first{}, repeated{};
    CHECK_STATUS(seen_cublaslt_select_algorithm(handle, &ffn, &first));
    CHECK_STATUS(seen_cublaslt_select_algorithm(handle, &ffn, &repeated));
    CHECK(first.cache_identity != 0);
    CHECK(first.workspace_bytes <= ffn.workspace_limit_bytes);
    CHECK(same_algorithm(first, repeated));

    auto gdn = descriptor(1, 6144, 5120, SEEN_CUDA_F16, 32ull << 20);
    SeenCudaAlgorithm different{};
    CHECK_STATUS(seen_cublaslt_select_algorithm(handle, &gdn, &different));
    CHECK(different.cache_identity != 0);
    CHECK(different.cache_identity != first.cache_identity);
    CHECK(different.workspace_bytes <= gdn.workspace_limit_bytes);

    auto no_workspace = descriptor(1, 5120, 5120, SEEN_CUDA_BF16, 0);
    SeenCudaAlgorithm bounded{};
    CHECK_STATUS(seen_cublaslt_select_algorithm(handle, &no_workspace, &bounded));
    CHECK(bounded.workspace_bytes == 0);

    SeenCudaAlgorithm ignored{};
    auto invalid = ffn;
    invalid.abi_version = 0;
    CHECK(seen_cublaslt_select_algorithm(handle, &invalid, &ignored).code ==
          SEEN_CUDA_INVALID_ARGUMENT);
    invalid = ffn;
    invalid.m = static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) + 1;
    CHECK(seen_cublaslt_select_algorithm(handle, &invalid, &ignored).code ==
          SEEN_CUDA_INVALID_ARGUMENT);
    invalid = ffn;
    invalid.data_type = SEEN_CUDA_F32;
    CHECK(seen_cublaslt_select_algorithm(handle, &invalid, &ignored).code ==
          SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_cublaslt_select_algorithm(0, &ffn, &ignored).code ==
          SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_cublaslt_select_algorithm(handle, &ffn, nullptr).code ==
          SEEN_CUDA_INVALID_ARGUMENT);

    CHECK_STATUS(seen_cublaslt_destroy(&handle));
    CHECK(handle == 0);
    CHECK_STATUS(seen_cublaslt_destroy(&handle));
    std::printf("PASS: FEL-1445 official Qwen cuBLASLt descriptor selection cleanup\n");
    return 0;
}
