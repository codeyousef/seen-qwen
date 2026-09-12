// CPU-only link fixture: reaching any CUDA operation is a hard failure.
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
