# Complete Qwen model initialization and residency contract

FEL-1447 / QWN-046A admits one exact experimental-hardware profile for the
complete 851-text plus 15-MTP tensor catalog. The derived SQW uses
`Q4_SYM_G64` for every BF16 source tensor, canonical UTF-8 tensor ordering,
and 64-byte component alignment. It contains 14,515,042,384 component bytes.
This lossy profile is not quality approved and cannot become a production or
default profile without the later correctness and quality gates. Fallback,
host offload, and tensor omission are prohibited.

The offline builder validates the immutable 18-shard official source and
tokenizer locks, quantizes bounded row batches serially, seals every component
and SQW section, independently reads back all 866 directory entries and all
file/section digests, and then atomically promotes the content-addressed
artifact. It imports neither model code nor PyTorch and keeps its staging and
output below the ignored `.seen` artifact root.

The 128-token, batch-one memory plan accounts for 14,515,042,384 weight bytes,
150,994,944 recurrent GDN bytes, 5,898,240 convolution-state bytes, 8,388,608
KV bytes, and 1,947,205,632 bounded activation, logit, cuBLASLt, scratch,
graph, and telemetry bytes. The allocation total is 16,627,529,808 bytes. It
also reserves the greater of 512 MiB or 3% of physical device memory and
rejects insufficient capacity before allocation with a stable typed error.
All arithmetic is checked 64-bit geometry.

Hardware verification opens and validates the actual sealed SQW, creates one
Seen-owned stream, allocates every planned region on device zero, uploads all
1,732 data and scale components to one contiguous weight allocation, and
initializes the remaining regions on that stream. It verifies SM89 RTX 4090
identity, allocation addresses and device ownership, resident VRAM delta,
serial transfer completion, reverse-order cleanup, idempotent teardown, and
post-cleanup VRAM recovery. This gate initializes the complete engine only;
it deliberately performs no model math, sampling, or quality inference.
