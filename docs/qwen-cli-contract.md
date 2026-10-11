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

## `inspect` source audit

`seen-qwen inspect --model-lock PATH --output PATH` is the installed QWN-047B
read-only source inspection leaf. It strictly validates the pinned model lock,
the sibling source lock, `config.json`, `model.safetensors.index.json`, and the
metadata headers of all eighteen assigned safetensors shards. It reports exact
text, MTP, vision, and total source bytes plus the checked experimental Q4
memory plan. Tensor payloads are not mapped or converted.

The command does not initialize CUDA unless `--device ORDINAL` is explicitly
provided. With no device option its evidence records `device_probe` as
`not-requested`. With a requested device it reports compatibility and fit; it
does not allocate, fall back, offload, or alter policy.

The explicit `--output` destination is the command's only file mutation. It is
written atomically after every input and cleanup gate succeeds. Inputs, shard
contents, locks, and caches are never modified by inspection.

## `convert` and `verify`

`seen-qwen convert --model-lock PATH --quant-profile bf16-bringup-v1
--output-root PATH` installs QWN-047C's lossless offline conversion path. The
native Seen command creates an exact structured request for the tracked,
bounded streaming converter; it does not invoke a shell. The converter accepts
only sibling files under the frozen model-lock source root, the configured
quantization policy, serial execution, and an aggregate host-memory request no
greater than 64 GiB. Python is confined to this offline conversion tool and
never enters the production inference runtime. PyTorch, Transformers, remote
model code, pickle, implicit resume, and hidden policy fallback remain
prohibited. `--verify-only` performs the complete locked source, header,
payload-digest, policy, and layout plan without staging or promoting an SQW.

`seen-qwen verify --engine PATH --suite smoke [--output PATH]` strictly checks
the closed engine manifest, sibling model and source locks, conventional SQW
digest, SQW logical whole digest, all section digests, directory geometry, and
deterministic reader cleanup. Optional evidence is written atomically. The
`operators`, `logits`, `quality`, `long-context`, and `all` suites are parsed
but explicitly unsupported by this leaf; they never fall back to smoke or CPU.
Their owning later leaves replace that fail-closed boundary.
