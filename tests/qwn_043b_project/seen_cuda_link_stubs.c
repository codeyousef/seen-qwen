// CPU-only CI linker fixture. Every entry traps if reached: this supplies ABI
// symbols for pre-CUDA validation tests without emulating CUDA or falling back.

#define SEEN_LINK_ONLY_STUB(name) \
    __attribute__((noreturn, visibility("default"))) void name(void) { \
        __builtin_trap(); \
    }

SEEN_LINK_ONLY_STUB(seen_cuda_stream_create)
SEEN_LINK_ONLY_STUB(seen_cuda_stream_destroy)
SEEN_LINK_ONLY_STUB(seen_cublaslt_create)
SEEN_LINK_ONLY_STUB(seen_cublaslt_destroy)
SEEN_LINK_ONLY_STUB(seen_cublaslt_select_algorithm)
SEEN_LINK_ONLY_STUB(seen_cublaslt_matmul)
