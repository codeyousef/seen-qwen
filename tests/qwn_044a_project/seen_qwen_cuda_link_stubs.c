// CPU-only link fixture: reaching a native Qwen kernel is a hard failure.
__attribute__((noreturn, visibility("default")))
void seen_qwen_swiglu_low_precision(void) { __builtin_trap(); }

__attribute__((noreturn, visibility("default")))
void seen_qwen_greedy_argmax_low_precision(void) { __builtin_trap(); }
