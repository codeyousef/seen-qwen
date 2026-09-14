#include "seen_qwen_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr uint64_t kContext = 128;
constexpr uint64_t kHidden = 48;
constexpr uint64_t kIntermediate = 160;
constexpr uint64_t kVocabulary = 256;
constexpr uint64_t kAttentionHeads = 6;
constexpr uint64_t kKvHeads = 1;
constexpr uint64_t kAttentionHeadDim = 24;
constexpr uint64_t kRotaryDim = 6;
constexpr uint64_t kGdnHeads = 6;
constexpr uint64_t kGdnKeyHeads = 2;
constexpr uint64_t kGdnHeadDim = 8;
constexpr uint64_t kGdnWidth = 80;
constexpr uint64_t kConvKernel = 4;
constexpr uint64_t kScratchSlotFloats = kContext * 288;
constexpr uint64_t kScratchSlots = 20;
constexpr uint64_t kConvStateFloats = 6 * 3 * kGdnWidth;
constexpr uint64_t kRecurrentStateFloats = 6 * kGdnHeads * kGdnHeadDim * kGdnHeadDim;
constexpr uint64_t kKvLayerFloats = kContext * kKvHeads * kAttentionHeadDim;
constexpr uint64_t kPersistentFloats = kConvStateFloats + kRecurrentStateFloats +
    4 * kKvLayerFloats;

#define REQUIRE(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "FAIL:%d: %s\n", __LINE__, #expr); return false; \
} } while (0)
#define CUDA_OK(expr) do { SeenCudaStatus s_ = (expr); \
    if (s_.code != SEEN_CUDA_OK) { \
        std::fprintf(stderr, "FAIL:%d: %s code=%d native=%d op=%s message=%s\n", \
            __LINE__, #expr, s_.code, s_.native_code, s_.operation, s_.message); \
        return false; \
    } \
} while (0)

std::vector<uint8_t> read_bytes(const char *path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return {};
    input.seekg(0, std::ios::end);
    const std::streamoff length = input.tellg();
    if (length <= 0 || length > 2 * 1024 * 1024) return {};
    input.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(static_cast<size_t>(length));
    input.read(reinterpret_cast<char *>(bytes.data()), length);
    return input ? bytes : std::vector<uint8_t>{};
}

uint64_t little_u64(const uint8_t *bytes) {
    uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index)
        value |= static_cast<uint64_t>(bytes[index]) << (index * 8);
    return value;
}

bool parse_u64(const std::string &text, size_t *cursor, uint64_t *value) {
    while (*cursor < text.size() &&
           (text[*cursor] < '0' || text[*cursor] > '9')) ++*cursor;
    if (*cursor == text.size()) return false;
    uint64_t result = 0;
    while (*cursor < text.size() && text[*cursor] >= '0' &&
           text[*cursor] <= '9') {
        const uint64_t digit = static_cast<uint64_t>(text[*cursor] - '0');
        if (result > (std::numeric_limits<uint64_t>::max() - digit) / 10)
            return false;
        result = result * 10 + digit;
        ++*cursor;
    }
    *value = result;
    return true;
}

struct ModelFile {
    std::vector<uint8_t> bytes;
    std::string header;
    uint64_t data_start = 0;

    bool openea(const char *path) {
        bytes = read_bytes(path);
        if (bytes.size() < 8) return false;
        const uint64_t header_length = little_u64(bytes.data());
        if (header_length == 0 || header_length > 1024 * 1024 ||
            header_length > bytes.size() - 8) return false;
        data_start = 8 + header_length;
        header.assign(reinterpret_cast<const char *>(bytes.data() + 8),
                      static_cast<size_t>(header_length));
        return true;
    }

    bool tensor(const std::string &name, uint64_t elements,
                uint64_t *absolute_offset) const {
        const std::string key = "\"" + name + "\":";
        const size_t begin = header.find(key);
        if (begin == std::string::npos) return false;
        const size_t end = header.find('}', begin + key.size());
        if (end == std::string::npos ||
            header.find("\"dtype\":\"F32\"", begin) > end) return false;
        const size_t offsets = header.find("\"data_offsets\":[", begin);
        if (offsets == std::string::npos || offsets > end) return false;
        size_t cursor = offsets + std::strlen("\"data_offsets\":[");
        uint64_t first = 0, last = 0;
        if (!parse_u64(header, &cursor, &first) ||
            !parse_u64(header, &cursor, &last) || last < first ||
            last - first != elements * sizeof(float) ||
            data_start > bytes.size() || first > bytes.size() - data_start ||
            last > bytes.size() - data_start) return false;
        *absolute_offset = data_start + first;
        return true;
    }
};

bool parse_numbers(const std::string &json, const char *field, size_t count,
                   std::vector<float> *values) {
    size_t cursor = json.find(std::string("\"") + field + "\"");
    if (cursor == std::string::npos) return false;
    cursor = json.find('[', cursor);
    if (cursor == std::string::npos) return false;
    ++cursor;
    values->clear();
    while (values->size() < count && cursor < json.size()) {
        while (cursor < json.size() && json[cursor] != '-' &&
               json[cursor] != '+' && json[cursor] != '.' &&
               (json[cursor] < '0' || json[cursor] > '9')) ++cursor;
        if (cursor == json.size()) return false;
        char *end = nullptr;
        const float value = std::strtof(json.c_str() + cursor, &end);
        if (end == json.c_str() + cursor || !std::isfinite(value)) return false;
        values->push_back(value);
        cursor = static_cast<size_t>(end - json.c_str());
    }
    return values->size() == count;
}

SeenQwenCudaBufferView view(void *address, uint64_t bytes) {
    return SeenQwenCudaBufferView{SEEN_QWEN_CUDA_REFERENCE_ABI_VERSION, 0,
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address)), bytes};
}

struct MiniEngine {
    ModelFile model;
    SeenCudaHandle stream = 0;
    SeenCudaHandle event = 0;
    SeenCudaHandle model_allocation = 0;
    SeenCudaHandle scratch_allocation = 0;
    SeenCudaHandle state_allocation = 0;
    SeenCudaHandle ids_allocation = 0;
    void *model_device = nullptr;
    float *scratch = nullptr;
    float *state = nullptr;
    int32_t *ids = nullptr;
    uint64_t position = 0;
    uint64_t allocation_attempts = 0;
    uint64_t fail_allocation_at = 0;
    int32_t last_token = -1;
    bool prefilled = false;
    bool cancelled = false;
    bool closed = false;

    SeenCudaStreamLaunchToken token() const {
        SeenCudaStreamLaunchToken result{};
        const SeenCudaStatus status =
            seen_cuda_stream_borrow_launch_token(stream, 0, &result);
        if (status.code != SEEN_CUDA_OK) std::abort();
        return result;
    }

    float *slot(uint64_t index) const {
        return scratch + index * kScratchSlotFloats;
    }

    SeenQwenCudaBufferView floats(float *address, uint64_t count) const {
        return view(address, count * sizeof(float));
    }

    SeenQwenCudaBufferView weight(const std::string &name,
                                  uint64_t elements) const {
        uint64_t offset = 0;
        if (!model.tensor(name, elements, &offset)) return view(nullptr, 0);
        return view(static_cast<uint8_t *>(model_device) + offset,
                    elements * sizeof(float));
    }

    bool allocate(uint64_t bytes, SeenCudaHandle *allocation) {
        ++allocation_attempts;
        if (fail_allocation_at == allocation_attempts) return false;
        return seen_cuda_malloc(0, bytes, allocation).code == SEEN_CUDA_OK;
    }

    bool reject_invalid_composition() const {
        auto launch = token();
        SeenCudaStatus status = seen_qwen_linear_f32(&launch,
            floats(slot(0), kHidden), weight("lm_head.weight",
                kVocabulary * kHidden), floats(slot(1), kVocabulary),
            0, kHidden, kVocabulary);
        REQUIRE(status.code == SEEN_CUDA_INVALID_ARGUMENT);

        launch = token();
        status = seen_qwen_linear_f32(&launch,
            floats(slot(0), kHidden), weight("lm_head.weight",
                kVocabulary * kHidden), floats(slot(0), kVocabulary),
            1, kHidden, kVocabulary);
        REQUIRE(status.code == SEEN_CUDA_INVALID_ARGUMENT);

        launch = token();
        status = seen_qwen_gdn_prepare_f32(&launch,
            floats(slot(0), kGdnWidth), floats(slot(1), kHidden),
            floats(slot(2), kHidden), floats(slot(3), kHidden), 1,
            kGdnHeads, kGdnKeyHeads, kGdnHeadDim, 0.0f);
        REQUIRE(status.code == SEEN_CUDA_INVALID_ARGUMENT);

        launch = token();
        status = seen_qwen_gdn_parameters_f32(&launch,
            floats(slot(0), kGdnHeads), floats(slot(1), kGdnHeads),
            floats(slot(2), kGdnHeads), floats(slot(3), kGdnHeads),
            floats(slot(4), kGdnHeads), floats(slot(4), kGdnHeads), 1,
            kGdnHeads);
        REQUIRE(status.code == SEEN_CUDA_INVALID_ARGUMENT);
        return true;
    }

    bool open(const char *model_path, uint64_t injected_failure = 0) {
        REQUIRE(model.openea(model_path));
        fail_allocation_at = injected_failure;
        CUDA_OK(seen_cuda_stream_create(0, &stream));
        CUDA_OK(seen_cuda_event_create(0, &event));
        if (!allocate(model.bytes.size(), &model_allocation) ||
            !allocate(kScratchSlots * kScratchSlotFloats * sizeof(float),
                &scratch_allocation) ||
            !allocate(kPersistentFloats * sizeof(float), &state_allocation) ||
            !allocate(kContext * sizeof(int32_t), &ids_allocation)) {
            close();
            return false;
        }
        uint64_t bytes = 0; int32_t ordinal = -1;
        CUDA_OK(seen_cuda_allocation_address(model_allocation, &model_device,
            &bytes, &ordinal));
        REQUIRE(bytes == model.bytes.size() && ordinal == 0);
        void *pointer = nullptr;
        CUDA_OK(seen_cuda_allocation_address(scratch_allocation, &pointer,
            &bytes, &ordinal)); scratch = static_cast<float *>(pointer);
        REQUIRE(bytes == kScratchSlots * kScratchSlotFloats * sizeof(float));
        CUDA_OK(seen_cuda_allocation_address(state_allocation, &pointer,
            &bytes, &ordinal)); state = static_cast<float *>(pointer);
        REQUIRE(bytes == kPersistentFloats * sizeof(float));
        CUDA_OK(seen_cuda_allocation_address(ids_allocation, &pointer,
            &bytes, &ordinal)); ids = static_cast<int32_t *>(pointer);
        REQUIRE(bytes == kContext * sizeof(int32_t));
        CUDA_OK(seen_cuda_memcpy_async(model_device, model.bytes.data(),
            model.bytes.size(), SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
        CUDA_OK(seen_cuda_memset_async(state, 0,
            kPersistentFloats * sizeof(float), stream));
        CUDA_OK(seen_cuda_event_record(event, stream));
        CUDA_OK(seen_cuda_event_synchronize(event));
        return true;
    }

    float *conv_state(uint64_t layer_state) const {
        return state + layer_state * 3 * kGdnWidth;
    }
    float *recurrent_state(uint64_t layer_state) const {
        return state + kConvStateFloats +
            layer_state * kGdnHeads * kGdnHeadDim * kGdnHeadDim;
    }
    float *key_cache(uint64_t attention_state) const {
        return state + kConvStateFloats + kRecurrentStateFloats +
            attention_state * 2 * kKvLayerFloats;
    }
    float *value_cache(uint64_t attention_state) const {
        return key_cache(attention_state) + kKvLayerFloats;
    }

    bool linear(float *input, const std::string &weight_name, float *output,
                uint64_t rows, uint64_t input_width,
                uint64_t output_width) const {
        auto launch = token();
        CUDA_OK(seen_qwen_linear_f32(&launch,
            floats(input, rows * input_width),
            weight(weight_name, input_width * output_width),
            floats(output, rows * output_width), rows, input_width,
            output_width));
        return true;
    }

    bool norm(float *input, const std::string &weight_name, float *output,
              uint64_t rows, uint64_t width = kHidden) const {
        auto launch = token();
        CUDA_OK(seen_qwen_rms_norm_f32(&launch, floats(input, rows * width),
            weight(weight_name, width), floats(output, rows * width), rows,
            width, 0.000001f));
        return true;
    }

    bool gdn(float *input, float *output, uint64_t rows,
             uint64_t start, uint64_t layer, uint64_t state_index,
             bool decode) const {
        const std::string base = "model.language_model.layers." +
            std::to_string(layer) + ".linear_attn";
        float *mixed = slot(2), *convolution = slot(3);
        float *query = slot(4), *key = slot(5), *value = slot(6);
        float *beta_projection = slot(7), *decay_projection = slot(8);
        float *beta = slot(9), *decay = slot(10), *core = slot(11);
        float *gate = slot(12), *normalized = slot(13);
        REQUIRE(linear(input, base + ".in_proj_qkv.weight", mixed, rows,
            kHidden, kGdnWidth));
        auto launch = token();
        CUDA_OK(seen_qwen_causal_conv_silu_f32(&launch,
            floats(mixed, rows * kGdnWidth),
            weight(base + ".conv1d.weight", kGdnWidth * kConvKernel),
            floats(conv_state(state_index), 3 * kGdnWidth),
            floats(convolution, rows * kGdnWidth), rows, kGdnWidth,
            kConvKernel, start, start));
        launch = token();
        CUDA_OK(seen_qwen_gdn_prepare_f32(&launch,
            floats(convolution, rows * kGdnWidth),
            floats(query, rows * kHidden), floats(key, rows * kHidden),
            floats(value, rows * kHidden), rows, kGdnHeads, kGdnKeyHeads,
            kGdnHeadDim, 0.000001f));
        REQUIRE(linear(input, base + ".in_proj_b.weight", beta_projection,
            rows, kHidden, kGdnHeads));
        REQUIRE(linear(input, base + ".in_proj_a.weight", decay_projection,
            rows, kHidden, kGdnHeads));
        launch = token();
        CUDA_OK(seen_qwen_gdn_parameters_f32(&launch,
            floats(beta_projection, rows * kGdnHeads),
            floats(decay_projection, rows * kGdnHeads),
            weight(base + ".A_log", kGdnHeads),
            weight(base + ".dt_bias", kGdnHeads),
            floats(beta, rows * kGdnHeads),
            floats(decay, rows * kGdnHeads), rows, kGdnHeads));
        launch = token();
        if (decode) {
            CUDA_OK(seen_qwen_gdn_recurrent_decode_f32(&launch,
                floats(query, kHidden), floats(key, kHidden),
                floats(value, kHidden), floats(beta, kGdnHeads),
                floats(decay, kGdnHeads),
                floats(recurrent_state(state_index),
                    kGdnHeads * kGdnHeadDim * kGdnHeadDim),
                floats(core, kHidden), kGdnHeads, kGdnHeadDim, kGdnHeadDim,
                start, start));
        } else {
            CUDA_OK(seen_qwen_gdn_recurrent_prefill_f32(&launch,
                floats(query, rows * kHidden), floats(key, rows * kHidden),
                floats(value, rows * kHidden),
                floats(beta, rows * kGdnHeads),
                floats(decay, rows * kGdnHeads),
                floats(recurrent_state(state_index),
                    kGdnHeads * kGdnHeadDim * kGdnHeadDim),
                floats(core, rows * kHidden), rows, kGdnHeads, kGdnHeadDim,
                kGdnHeadDim, start, start));
        }
        REQUIRE(linear(input, base + ".in_proj_z.weight", gate, rows,
            kHidden, kHidden));
        launch = token();
        CUDA_OK(seen_qwen_gdn_gated_rms_norm_f32(&launch,
            floats(core, rows * kHidden), floats(gate, rows * kHidden),
            weight(base + ".norm.weight", kGdnHeadDim),
            floats(normalized, rows * kHidden), rows * kGdnHeads,
            kGdnHeadDim, 0.000001f));
        return linear(normalized, base + ".out_proj.weight", output, rows,
            kHidden, kHidden);
    }

    bool attention(float *input, float *output, uint64_t rows,
                   uint64_t start, uint64_t layer,
                   uint64_t state_index, bool decode) const {
        const std::string base = "model.language_model.layers." +
            std::to_string(layer) + ".self_attn";
        float *query_gate = slot(2), *key_projection = slot(3);
        float *value = slot(4), *query = slot(5), *key = slot(6);
        float *gate = slot(7), *attended = slot(8), *gated = slot(9);
        REQUIRE(linear(input, base + ".q_proj.weight", query_gate, rows,
            kHidden, 288));
        REQUIRE(linear(input, base + ".k_proj.weight", key_projection, rows,
            kHidden, kAttentionHeadDim));
        REQUIRE(linear(input, base + ".v_proj.weight", value, rows,
            kHidden, kAttentionHeadDim));
        auto launch = token();
        CUDA_OK(seen_qwen_attention_qk_rope_f32(&launch,
            floats(query_gate, rows * 288),
            floats(key_projection, rows * kAttentionHeadDim),
            weight(base + ".q_norm.weight", kAttentionHeadDim),
            weight(base + ".k_norm.weight", kAttentionHeadDim),
            floats(query, rows * kAttentionHeads * kAttentionHeadDim),
            floats(key, rows * kAttentionHeadDim),
            floats(gate, rows * kAttentionHeads * kAttentionHeadDim), rows,
            kAttentionHeads, kKvHeads, kAttentionHeadDim, kRotaryDim, start,
            kContext, 10000000.0f, 0.000001f));
        if (decode) {
            launch = token();
            CUDA_OK(seen_qwen_kv_append_f32(&launch,
                floats(key, kAttentionHeadDim),
                floats(value, kAttentionHeadDim),
                floats(key_cache(state_index), kKvLayerFloats),
                floats(value_cache(state_index), kKvLayerFloats), 1,
                kKvHeads, kAttentionHeadDim, start, kContext));
            launch = token();
            CUDA_OK(seen_qwen_attention_decode_f32(&launch,
                floats(query, kAttentionHeads * kAttentionHeadDim),
                floats(key_cache(state_index), kKvLayerFloats),
                floats(value_cache(state_index), kKvLayerFloats),
                floats(attended, kAttentionHeads * kAttentionHeadDim),
                kAttentionHeads, kKvHeads, kAttentionHeadDim, start + 1,
                kContext));
        } else {
            launch = token();
            CUDA_OK(seen_qwen_attention_prefill_f32(&launch,
                floats(query, rows * kAttentionHeads * kAttentionHeadDim),
                floats(key, rows * kAttentionHeadDim),
                floats(value, rows * kAttentionHeadDim),
                floats(key_cache(state_index), kKvLayerFloats),
                floats(value_cache(state_index), kKvLayerFloats),
                floats(attended, rows * kAttentionHeads * kAttentionHeadDim),
                rows, kAttentionHeads, kKvHeads, kAttentionHeadDim, start,
                kContext));
        }
        launch = token();
        CUDA_OK(seen_qwen_sigmoid_gate_f32(&launch,
            floats(attended, rows * 144), floats(gate, rows * 144),
            floats(gated, rows * 144), rows * 144));
        return linear(gated, base + ".o_proj.weight", output, rows, 144,
            kHidden);
    }

    bool layer(float *hidden, float *output, uint64_t rows, uint64_t start,
               uint64_t layer_index, bool decode) const {
        const std::string base = "model.language_model.layers." +
            std::to_string(layer_index);
        float *mixer_input = slot(14), *mixer = slot(15);
        float *post_mixer = slot(16), *mlp_input = slot(17);
        float *gate = slot(2), *up = slot(3), *activated = slot(4);
        float *mlp = slot(5);
        REQUIRE(norm(hidden, base + ".input_layernorm.weight", mixer_input,
            rows));
        if (layer_index == 3 || layer_index == 7) {
            REQUIRE(attention(mixer_input, mixer, rows, start, layer_index,
                layer_index == 3 ? 0 : 1, decode));
        } else {
            const uint64_t state_index = layer_index < 3 ? layer_index :
                layer_index - 1;
            REQUIRE(gdn(mixer_input, mixer, rows, start, layer_index,
                state_index, decode));
        }
        auto launch = token();
        CUDA_OK(seen_qwen_add_f32(&launch, floats(hidden, rows * kHidden),
            floats(mixer, rows * kHidden),
            floats(post_mixer, rows * kHidden), rows * kHidden));
        REQUIRE(norm(post_mixer, base + ".post_attention_layernorm.weight",
            mlp_input, rows));
        REQUIRE(linear(mlp_input, base + ".mlp.gate_proj.weight", gate,
            rows, kHidden, kIntermediate));
        REQUIRE(linear(mlp_input, base + ".mlp.up_proj.weight", up,
            rows, kHidden, kIntermediate));
        launch = token();
        CUDA_OK(seen_qwen_swiglu_f32(&launch,
            floats(gate, rows * kIntermediate),
            floats(up, rows * kIntermediate),
            floats(activated, rows * kIntermediate), rows * kIntermediate));
        REQUIRE(linear(activated, base + ".mlp.down_proj.weight", mlp,
            rows, kIntermediate, kHidden));
        launch = token();
        CUDA_OK(seen_qwen_add_f32(&launch,
            floats(post_mixer, rows * kHidden),
            floats(mlp, rows * kHidden), floats(output, rows * kHidden),
            rows * kHidden));
        return true;
    }

    bool forward(const int32_t *host_ids, uint64_t rows,
                 std::vector<float> *last_logits, int32_t *sampled) {
        REQUIRE(!closed && !cancelled && rows > 0 && rows <= kContext - position);
        CUDA_OK(seen_cuda_memcpy_async(ids, host_ids,
            rows * sizeof(int32_t), SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
        auto launch = token();
        float *hidden = slot(0), *next = slot(1);
        CUDA_OK(seen_qwen_embedding_gather_f32(&launch,
            weight("model.language_model.embed_tokens.weight",
                kVocabulary * kHidden),
            view(ids, rows * sizeof(int32_t)), floats(hidden, rows * kHidden),
            rows, kVocabulary, kHidden));
        const bool decode = rows == 1 && position != 0;
        for (uint64_t layer_index = 0; layer_index < 8; ++layer_index) {
            REQUIRE(layer(hidden, next, rows, position, layer_index, decode));
            std::swap(hidden, next);
        }
        REQUIRE(norm(hidden, "model.language_model.norm.weight", next, rows));
        float *logits = slot(18);
        REQUIRE(linear(next, "lm_head.weight", logits, rows, kHidden,
            kVocabulary));
        launch = token();
        CUDA_OK(seen_qwen_greedy_argmax_f32(&launch,
            floats(logits, rows * kVocabulary),
            view(ids, rows * sizeof(int32_t)), rows, kVocabulary,
            kVocabulary));
        last_logits->assign(kVocabulary, 0.0f);
        CUDA_OK(seen_cuda_memcpy_async(last_logits->data(),
            logits + (rows - 1) * kVocabulary,
            kVocabulary * sizeof(float), SEEN_CUDA_COPY_DEVICE_TO_HOST,
            stream));
        CUDA_OK(seen_cuda_memcpy_async(sampled, ids + rows - 1,
            sizeof(int32_t), SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
        CUDA_OK(seen_cuda_event_record(event, stream));
        CUDA_OK(seen_cuda_event_synchronize(event));
        if (cancelled) return false;
        position += rows;
        last_token = *sampled;
        return true;
    }

    bool prefill(const std::vector<int32_t> &prompt,
                 std::vector<float> *logits, int32_t *sampled) {
        if (closed || cancelled || prefilled || prompt.empty() ||
            prompt.size() >= kContext) return false;
        if (!forward(prompt.data(), prompt.size(), logits, sampled)) return false;
        prefilled = true;
        return true;
    }

    bool decode(std::vector<float> *logits, int32_t *sampled) {
        if (closed || cancelled || !prefilled || last_token < 0 ||
            position >= kContext) return false;
        const int32_t input = last_token;
        return forward(&input, 1, logits, sampled);
    }

    bool reset() {
        if (closed) return false;
        CUDA_OK(seen_cuda_memset_async(state, 0,
            kPersistentFloats * sizeof(float), stream));
        CUDA_OK(seen_cuda_event_record(event, stream));
        CUDA_OK(seen_cuda_event_synchronize(event));
        position = 0; last_token = -1; prefilled = false; cancelled = false;
        return true;
    }

    bool cancel() {
        if (closed) return false;
        cancelled = true;
        return true;
    }

    bool close() {
        if (closed) return true;
        bool okay = true;
        auto release = [&](SeenCudaHandle *handle) {
            if (*handle == 0) return;
            const SeenCudaStatus status = seen_cuda_free(handle);
            okay = okay && status.code == SEEN_CUDA_OK && *handle == 0;
        };
        release(&ids_allocation); release(&state_allocation);
        release(&scratch_allocation); release(&model_allocation);
        if (event != 0) {
            const SeenCudaStatus status = seen_cuda_event_destroy(&event);
            okay = okay && status.code == SEEN_CUDA_OK && event == 0;
        }
        if (stream != 0) {
            const SeenCudaStatus status = seen_cuda_stream_destroy(&stream);
            okay = okay && status.code == SEEN_CUDA_OK && stream == 0;
        }
        model_device = nullptr; scratch = nullptr; state = nullptr; ids = nullptr;
        position = 0; last_token = -1; prefilled = false; cancelled = true;
        closed = true;
        return okay;
    }
};

bool compare_logits(const std::vector<float> &actual,
                    const std::vector<float> &expected, size_t row) {
    float maximum_error = 0.0f;
    for (size_t index = 0; index < kVocabulary; ++index) {
        REQUIRE(std::isfinite(actual[index]));
        maximum_error = std::max(maximum_error,
            std::fabs(actual[index] - expected[row * kVocabulary + index]));
    }
    std::printf("QWN-045B row=%zu max_abs_error=%.9g\n", row,
                maximum_error);
    REQUIRE(maximum_error <= 0.00008f);
    return true;
}

bool certify_allocation_failures(const char *model_path) {
    for (uint64_t failure = 1; failure <= 4; ++failure) {
        MiniEngine engine;
        REQUIRE(!engine.open(model_path, failure));
        REQUIRE(engine.closed && engine.stream == 0 && engine.event == 0 &&
            engine.model_allocation == 0 && engine.scratch_allocation == 0 &&
            engine.state_allocation == 0 && engine.ids_allocation == 0);
        REQUIRE(engine.close());
    }
    return true;
}

bool run(const char *model_path, const char *oracle_path) {
    REQUIRE(certify_allocation_failures(model_path));
    const auto oracle_bytes = read_bytes(oracle_path);
    REQUIRE(!oracle_bytes.empty());
    const std::string oracle(reinterpret_cast<const char *>(oracle_bytes.data()),
                             oracle_bytes.size());
    std::vector<float> prompt_values, token_values, expected_logits;
    REQUIRE(parse_numbers(oracle, "prompt_ids", 9, &prompt_values));
    REQUIRE(parse_numbers(oracle, "greedy_token_ids", 4, &token_values));
    REQUIRE(parse_numbers(oracle, "decode_logits", 4 * kVocabulary,
                          &expected_logits));
    std::vector<int32_t> prompt;
    for (float value : prompt_values) prompt.push_back(static_cast<int32_t>(value));

    MiniEngine engine;
    REQUIRE(engine.open(model_path));
    REQUIRE(engine.reject_invalid_composition());
    std::vector<float> logits;
    int32_t sampled = -1;
    REQUIRE(!engine.decode(&logits, &sampled));
    REQUIRE(engine.prefill(prompt, &logits, &sampled));
    REQUIRE(sampled == static_cast<int32_t>(token_values[0]));
    REQUIRE(compare_logits(logits, expected_logits, 0));
    for (size_t step = 1; step < 4; ++step) {
        REQUIRE(engine.decode(&logits, &sampled));
        REQUIRE(sampled == static_cast<int32_t>(token_values[step]));
        REQUIRE(compare_logits(logits, expected_logits, step));
    }
    REQUIRE(engine.position == 12);
    REQUIRE(engine.reset());
    REQUIRE(engine.cancel());
    REQUIRE(!engine.prefill(prompt, &logits, &sampled));
    REQUIRE(engine.reset());
    REQUIRE(engine.prefill(prompt, &logits, &sampled));
    REQUIRE(sampled == static_cast<int32_t>(token_values[0]));
    REQUIRE(compare_logits(logits, expected_logits, 0));
    REQUIRE(engine.close());
    REQUIRE(engine.close());
    REQUIRE(!engine.decode(&logits, &sampled));
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s MODEL_SAFETENSORS ORACLE_JSON\n", argv[0]);
        return 2;
    }
    if (!run(argv[1], argv[2])) return 1;
    std::puts("PASS: QWN-045B CUDA mini-model prefill/decode, CPU differential, reset, cancellation, and teardown");
    return 0;
}
