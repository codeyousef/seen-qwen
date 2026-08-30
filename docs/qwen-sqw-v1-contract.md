# Seen Quantized Weights v1.0 contract

SQW is a deterministic, derived runtime format. Safetensors remains the canonical source
and exchange format. An SQW file cannot replace its immutable
model/source locks or authorize remote code.

Version 1.0 is little-endian and admits no optional reader features. Readers
reject an unknown major, minor, flag, identifier, codec, or compatibility
feature. A future compatible minor version must define explicit feature
negotiation; version numbers alone never authorize reinterpretation, repair, or
fallback.

## File order and alignment

All offsets are absolute from byte zero and all integer fields are fixed-width
little-endian. Although fields use unsigned storage, v1.0 deliberately caps
every offset, length, shape product, and manifest integer at
9,223,372,036,854,775,807 (`2^63 - 1`). Values with the high bit set are
non-canonical. This keeps all checked geometry within the project-wide
signed-safe bound and does not limit a practical SQW artifact. The canonical
order is:

1. 256-byte fixed header;
2. canonical UTF-8 JSON manifest;
3. 64-byte-aligned fixed-entry tensor directory;
4. 64-byte-aligned UTF-8 name table;
5. zero padding to a 4,096-byte boundary;
6. tensor payloads;
7. optional bounded calibration/evidence bytes;
8. a 64-byte-aligned footer checksum table ending exactly at EOF.

Every gap, unused shape slot, reserved field, and reserved byte is zero.
Tensor component offsets obey their directory entry's power-of-two alignment,
which is at most 2 MiB. An absent component is encoded only as offset zero and
length zero.

## Fixed header

| Offset | Bytes | Field |
|---:|---:|---|
| 0 | 4 | ASCII `SQW1` |
| 4 | 4 | endian marker `0x01020304` (bytes `04 03 02 01`) |
| 8 | 2 | major, `1` |
| 10 | 2 | minor, `0` |
| 12 | 4 | header bytes, `256` |
| 16 | 4 | flags; bit 0 requires the whole-file SHA-256 |
| 20 | 4 | reserved zero |
| 24 | 8 | manifest offset |
| 32 | 8 | manifest length |
| 40 | 8 | tensor-directory offset |
| 48 | 4 | directory-entry bytes, `256` |
| 52 | 4 | tensor count, `1..2048` |
| 56 | 8 | name-table offset |
| 64 | 8 | name-table length |
| 72 | 8 | payload offset |
| 80 | 8 | payload length |
| 88 | 8 | evidence offset, or zero |
| 96 | 8 | evidence length, or zero |
| 104 | 8 | footer offset |
| 112 | 8 | footer length |
| 120 | 8 | absolute whole-file digest offset |
| 128 | 32 | raw model-lock SHA-256 |
| 160 | 32 | raw conversion-policy SHA-256 |
| 192 | 64 | reserved zero |

The manifest starts at byte 256 and is at most 16 MiB. The name table is at
most 16 MiB and the optional evidence section at most 64 MiB. Production Qwen
conversion additionally requires the exact 866 text/MTP entries and the pinned
catalog fingerprint; the general parser cap remains 2,048.

## Tensor directory

Entries are 256 bytes and sorted strictly by canonical UTF-8 tensor-name bytes.
Names are unique, contain no NUL, are at most 1,024 bytes, and must classify as
required `model.language_model.*`, `lm_head.weight`, or `mtp.*`. Vision and
unknown namespaces are invalid.

| Offset | Bytes | Field |
|---:|---:|---|
| 0 | 8 | name offset relative to the name table |
| 8 | 4 | name byte length |
| 12 | 2 | semantic-role ID |
| 14 | 2 | source-dtype ID |
| 16 | 2 | runtime-codec ID |
| 18 | 2 | rank, `1..8` |
| 20 | 4 | flags, zero in v1.0 |
| 24 | 64 | eight `UInt64` shape slots; unused slots are zero |
| 88 | 8 | logical element count |
| 96 | 16 | data offset and length |
| 112 | 16 | scale offset and length |
| 128 | 16 | zero-point offset and length |
| 144 | 16 | metadata offset and length |
| 160 | 8 | logical row elements |
| 168 | 4 | group elements |
| 172 | 4 | required alignment |
| 176 | 32 | raw source-tensor SHA-256 |
| 208 | 32 | raw converted-bundle SHA-256 |
| 240 | 16 | reserved zero |

The converted-bundle digest covers the exact concatenation
`data || scale || zero || metadata`, excluding alignment padding. The logical
shape product is checked within the v1.0 64-bit geometry bound and must equal the
stored logical element count. Row geometry must divide the logical count.
Present components occur physically in that same order and never overlap;
alignment padding between them is zero.

Semantic-role IDs are: embedding 1, LM head 2, attention Q/K/V/O 3 through 6,
GDN projection 7, GDN state parameter 8, MLP gate/up/down 9 through 11, norm
12, bias/scalar 13, and MTP 14.

Source-dtype IDs are: BOOL 1, I8 2, U8 3, I16 4, U16 5, F16 6, BF16 7, I32 8,
U32 9, F32 10, I64 11, U64 12, and F64 13. Runtime-codec IDs are BF16 1, F16
2, F32 3, Q8_SYM_G64 4, and Q4_SYM_G64 5. Zero is invalid for every ID.

Plain codecs store no scale, zero-point, or metadata component. Q8/Q4 use
groups of 64 and one FP16 scale per row-major group. Q8 stores one byte per
logical value. Q4 stores signed two's-complement nibbles, low nibble first,
with each row's odd tail padded independently. Symmetric codecs have no
zero-point component.

## Manifest and checksums

The embedded manifest conforms to
`schemas/qwen-sqw-manifest.schema.json`. It is duplicate-free strict JSON and
must be byte-for-byte equal to Seen's RFC 8785-style canonical serialization:
no BOM, insignificant whitespace, or trailing newline. Stored 64-bit geometry
uses canonical decimal strings in the inclusive range 0 through
9,223,372,036,854,775,807, avoiding conversion through JSON floating-point
numbers. The schema enforces canonical 1-to-19-digit spelling and the reader
enforces the exact numerical ceiling. The manifest repeats all directory
semantics and ranges; any mismatch is corruption.

The footer is a 64-byte header followed by four mandatory 64-byte entries for
manifest, directory, names, and payload, plus an evidence entry when present.
Its header contains ASCII `SQWF`, version `1`, header bytes `64`, entry bytes
`64`, SHA-256 algorithm ID `1`, entry count, the raw whole-file digest at byte
16, and zero reserved bytes. Each entry contains a 32-bit section ID, zero
flags, a 64-bit offset, a 64-bit length, a raw SHA-256, and eight zero bytes.

The whole-file digest is SHA-256 of every file byte with its own 32-byte footer
slot replaced by zero. This non-recursive definition must match the external
`engine.json` weights digest. Directory and tensor hashes are verified before
any tensor window is exposed.

## Validation, ownership, and errors

Readers reject the high bit, then use checked 64-bit addition, multiplication,
alignment, and range arithmetic before allocation or mapping. They reject truncation,
overlap, aliasing, non-canonical order or padding, malformed UTF-8/JSON,
duplicate or unsorted names, unsupported IDs/codecs, invalid shapes/group
tails, digest mismatch, and model/source/policy/catalog incompatibility. The
first exact mismatch is reported with a stable `sqw.*` diagnostic; no repair or
retry is implicit.

The file owner outlives all borrowed manifest strings and mapped tensor views.
Closing invalidates those borrows, closes every window before the file, and is
deterministic and idempotent. QWN-030A freezes the byte contract and bounded
decoders only. QWN-030B owns mapped zero-copy reading, and QWN-030C owns unique
temporary creation, full reopen/validation, fsync, and atomic promotion.
