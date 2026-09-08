#include "seen_qwen_cuda.h"

#include <stddef.h>

_Static_assert(SEEN_QWEN_CUDA_REFERENCE_ABI_VERSION == 1u, "ABI version");
_Static_assert(sizeof(SeenQwenCudaBufferView) == 24u, "fixed-width view");
_Static_assert(offsetof(SeenQwenCudaBufferView, address) == 8u, "address offset");
_Static_assert(offsetof(SeenQwenCudaBufferView, byte_length) == 16u, "length offset");

int main(void) { return 0; }
