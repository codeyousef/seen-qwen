# Seen Qwen CLI contract

FEL-1457 / QWN-047A freezes the bounded command and configuration surface that
later command leaves execute. The executable name is `seen-qwen`; accepted
commands are `inspect`, `convert`, `verify`, `run`, and `bench`. `serve` is
explicitly unsupported until the CLI correctness gates are complete.

Parsing is fail-closed. It rejects positional arguments, unknown or
command-incompatible options, duplicates, missing or empty values, numeric
suffixes and overflow, multiple prompt sources, unknown suites, and more than
64 arguments, 4096 bytes per argument, or 16384 bytes in aggregate. Defaults
are explicit and versioned: `profiles/runtime.toml`,
`profiles/sampling.toml`, `configs/quantization.toml`, human diagnostics, info
logging, one job, and no selected device. A backend or fallback is never
selected by the CLI parser.

Exit codes are frozen as: success `0`, CLI/config `2`, artifact validation `3`,
unsupported `4`, host allocation `5`, device allocation `6`, CUDA/runtime `7`,
correctness `8`, benchmark qualification `9`, cancellation `10`, and internal
invariant `11`. Native CUDA status values never become process exit codes.

Machine diagnostics use canonical one-line JSON with schema
`seen-qwen-diagnostic-v1`, stable code, field, message, and numeric exit code.
The QWN-047A executable honestly reports recognized commands as not installed;
QWN-047B through QWN-047E replace that terminal dispatch one command at a time.
