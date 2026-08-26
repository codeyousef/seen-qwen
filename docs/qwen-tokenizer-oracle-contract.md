# Qwen tokenizer oracle contract

FEL-1396 generates small deterministic vectors from the official
`Qwen/Qwen3.8-27B` tokenizer assets at model revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` and Transformers commit
`562cfd944ee1f20702cfb0f4404014ee27c24813`.

`tools/generate_qwen_tokenizer_oracles.py` is offline-only in normal use. It
verifies the byte size and SHA-256 of the vocabulary, merges, tokenizer config,
chat template, generation config, model config, official Transformers archive,
and Qwen2 tokenizer source before loading with `local_files_only=true`,
`trust_remote_code=false`, and the non-fast `Qwen2Tokenizer` implementation.

The generated fixture covers ASCII, Arabic, combining Unicode, Seen code,
whitespace, literal special tokens, and a bounded 1,024-character input. Chat
vectors cover default xhigh reasoning, low reasoning with a system prompt,
disabled thinking, `preserve_thinking=false`, and a complete tool
definition/call/response exchange. Negative vectors lock the template errors
for empty messages, an invalid reasoning effort, and a late system message.

Each text and rendered-chat vector records exact token IDs, decoded text, token
count, and UTF-8 SHA-256. The fixture also records all special token IDs, exact
runtime versions, source and asset hashes, and a canonical payload SHA-256.
Generation is bounded to a 1 MiB output and uses atomic replacement.

The combining-Unicode vector deliberately keeps its original decomposed input
and the tokenizer's normalized decoded result. Rendered chat vectors are exact
round trips. `tests/test_qwen_tokenizer_oracles.py` validates the fixture using
only the Python standard library, so downstream checks do not need the oracle
environment.

The fixture is oracle evidence for later native Seen implementation. It does
not add Python, Transformers, Jinja, or remote-code execution to the production
runtime.
