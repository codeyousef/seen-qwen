// CPU-only CI linker fixture. The native Qwen kernel must never execute here.

__attribute__((noreturn, visibility("default")))
void seen_qwen_swiglu_low_precision(void) {
    __builtin_trap();
}
