#include "seen_qwen_cuda.h"

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>

#define CHECK_STATUS(expr) do { SeenCudaStatus s_ = (expr); if (s_.code != SEEN_CUDA_OK) { \
    std::fprintf(stderr, "FAIL:%d: %s code=%d native=%d op=%s message=%s\n", \
        __LINE__, #expr, s_.code, s_.native_code, s_.operation, s_.message); return 1; } } while (0)
#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "FAIL:%d: %s\n", __LINE__, #expr); return 1; } } while (0)

namespace {

SeenQwenCudaBufferView view(void *address, uint64_t bytes, int32_t device = 0) {
    return SeenQwenCudaBufferView{SEEN_QWEN_CUDA_REFERENCE_ABI_VERSION,
                                  device,
                                  static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address)),
                                  bytes};
}

SeenCudaStreamLaunchToken borrow(SeenCudaHandle stream) {
    SeenCudaStreamLaunchToken token{};
    const SeenCudaStatus result = seen_cuda_stream_borrow_launch_token(stream, 0, &token);
    if (result.code != SEEN_CUDA_OK) {
        std::fprintf(stderr, "FAIL: token borrow code=%d message=%s\n", result.code, result.message);
        std::abort();
    }
    return token;
}

bool equal(float left, float right) { return std::fabs(left - right) <= 1.0e-6f; }

}  // namespace

int main() {
    constexpr uint64_t count = 24;
    constexpr uint64_t bytes = count * sizeof(float);
    constexpr uint64_t id_bytes = 3 * sizeof(int32_t);
    SeenCudaHandle stream = 0, event = 0, a_handle = 0, b_handle = 0;
    SeenCudaHandle c_handle = 0, d_handle = 0, ids_handle = 0, host_handle = 0;
    SeenCudaStreamLaunchToken rejected{};
    CHECK_STATUS(seen_cuda_stream_create(0, &stream));
    CHECK_STATUS(seen_cuda_event_create(0, &event));
    CHECK_STATUS(seen_cuda_malloc(0, bytes, &a_handle));
    CHECK_STATUS(seen_cuda_malloc(0, bytes, &b_handle));
    CHECK_STATUS(seen_cuda_malloc(0, bytes, &c_handle));
    CHECK_STATUS(seen_cuda_malloc(0, bytes, &d_handle));
    CHECK_STATUS(seen_cuda_malloc(0, id_bytes, &ids_handle));
    CHECK_STATUS(seen_cuda_host_alloc(bytes, &host_handle));

    void *a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr, *ids = nullptr, *host = nullptr;
    uint64_t actual = 0; int32_t device = -1;
    CHECK_STATUS(seen_cuda_allocation_address(a_handle, &a, &actual, &device)); CHECK(actual == bytes && device == 0);
    CHECK_STATUS(seen_cuda_allocation_address(b_handle, &b, &actual, &device));
    CHECK_STATUS(seen_cuda_allocation_address(c_handle, &c, &actual, &device));
    CHECK_STATUS(seen_cuda_allocation_address(d_handle, &d, &actual, &device));
    CHECK_STATUS(seen_cuda_allocation_address(ids_handle, &ids, &actual, &device)); CHECK(actual == id_bytes);
    CHECK_STATUS(seen_cuda_host_allocation_address(host_handle, &host, &actual)); CHECK(actual == bytes);
    auto *values = static_cast<float *>(host);

    for (uint64_t i = 0; i < count; ++i) values[i] = static_cast<float>(i);
    CHECK_STATUS(seen_cuda_memcpy_async(a, host, bytes, SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    SeenCudaStreamLaunchToken token = borrow(stream);
    CHECK_STATUS(seen_qwen_fill_f32(&token, view(b, bytes), count, 100.0f));
    token = borrow(stream);
    CHECK(token.flags & SEEN_CUDA_STREAM_LAUNCH_CAPTURE_COMPATIBLE);
    CHECK_STATUS(seen_qwen_add_f32(&token, view(a, bytes), view(b, bytes), view(c, bytes), count));
    token = borrow(stream);
    CHECK_STATUS(seen_qwen_copy_f32(&token, view(c, bytes), view(d, bytes), count));
    CHECK_STATUS(seen_cuda_memcpy_async(host, d, bytes, SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (uint64_t i = 0; i < count; ++i) CHECK(equal(values[i], 100.0f + i));

    token = borrow(stream);
    CHECK_STATUS(seen_qwen_fill_f32(&token, view(c, bytes), count, 3.25f));
    CHECK_STATUS(seen_cuda_memcpy_async(host, c, bytes, SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream)); CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (uint64_t i = 0; i < count; ++i) CHECK(equal(values[i], 3.25f));

    for (uint64_t i = 0; i < count; ++i) values[i] = static_cast<float>(i + 1);
    CHECK_STATUS(seen_cuda_memcpy_async(a, host, bytes, SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    token = borrow(stream);
    CHECK_STATUS(seen_qwen_transpose_2d_f32(&token, view(a, bytes), view(b, bytes), 4, 6));
    CHECK_STATUS(seen_cuda_memcpy_async(host, b, bytes, SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream)); CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (uint64_t column = 0; column < 6; ++column)
        for (uint64_t row = 0; row < 4; ++row)
            CHECK(equal(values[column * 4 + row], static_cast<float>(row * 6 + column + 1)));

    token = borrow(stream);
    CHECK_STATUS(seen_qwen_row_sum_f32(&token, view(a, bytes), view(c, bytes), 4, 6));
    CHECK_STATUS(seen_cuda_memcpy_async(host, c, 4 * sizeof(float), SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream)); CHECK_STATUS(seen_cuda_event_synchronize(event));
    CHECK(equal(values[0], 21.0f) && equal(values[1], 57.0f) && equal(values[2], 93.0f) && equal(values[3], 129.0f));

    int32_t host_ids[3] = {2, 0, 3};
    CHECK_STATUS(seen_cuda_memcpy_async(ids, host_ids, id_bytes, SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    token = borrow(stream);
    CHECK_STATUS(seen_qwen_embedding_gather_f32(&token, view(a, bytes), view(ids, id_bytes), view(c, bytes), 3, 4, 6));
    CHECK_STATUS(seen_cuda_memcpy_async(host, c, 18 * sizeof(float), SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream)); CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (uint64_t col = 0; col < 6; ++col) {
        CHECK(equal(values[col], static_cast<float>(13 + col)));
        CHECK(equal(values[6 + col], static_cast<float>(1 + col)));
        CHECK(equal(values[12 + col], static_cast<float>(19 + col)));
    }

    int32_t invalid_ids[3] = {0, -1, 4};
    CHECK_STATUS(seen_cuda_memcpy_async(ids, invalid_ids, id_bytes,
                                        SEEN_CUDA_COPY_HOST_TO_DEVICE, stream));
    token = borrow(stream);
    CHECK_STATUS(seen_qwen_embedding_gather_f32(
        &token, view(a, bytes), view(ids, id_bytes), view(c, bytes), 3, 4, 6));
    CHECK_STATUS(seen_cuda_memcpy_async(host, c, 18 * sizeof(float),
                                        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (uint64_t col = 0; col < 6; ++col) {
        CHECK(equal(values[col], static_cast<float>(1 + col)));
        CHECK(std::isnan(values[6 + col]));
        CHECK(std::isnan(values[12 + col]));
    }

    token = borrow(stream);
    CHECK_STATUS(seen_qwen_fill_f32(&token, view(c, bytes), 1, -2.5f));
    CHECK_STATUS(seen_cuda_memcpy_async(host, c, sizeof(float),
                                        SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream));
    CHECK_STATUS(seen_cuda_event_synchronize(event));
    CHECK(equal(values[0], -2.5f));

    SeenCudaStreamLaunchToken bad = token;
    bad.abi_version += 1;
    CHECK(seen_qwen_fill_f32(&bad, view(c, bytes), count, 0.0f).code == SEEN_CUDA_INCOMPATIBLE);
    bad = token; bad.reserved = 1;
    CHECK(seen_qwen_fill_f32(&bad, view(c, bytes), count, 0.0f).code == SEEN_CUDA_INCOMPATIBLE);
    bad = token; bad.flags |= 0x80000000u;
    CHECK(seen_qwen_fill_f32(&bad, view(c, bytes), count, 0.0f).code == SEEN_CUDA_INCOMPATIBLE);
    bad = token; bad.device_ordinal = 1;
    CHECK(seen_qwen_fill_f32(&bad, view(c, bytes, 1), count, 0.0f).code == SEEN_CUDA_INCOMPATIBLE);
    CHECK(seen_qwen_fill_f32(nullptr, view(c, bytes), count, 0.0f).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_cuda_stream_borrow_launch_token(0, 0, &rejected).code == SEEN_CUDA_CLOSED);
    CHECK(seen_cuda_stream_borrow_launch_token(UINT64_MAX, 0, &rejected).code == SEEN_CUDA_INVALID_ARGUMENT);
    SeenQwenCudaBufferView bad_view = view(c, bytes); bad_view.abi_version += 1;
    CHECK(seen_qwen_fill_f32(&token, bad_view, count, 0.0f).code == SEEN_CUDA_INVALID_ARGUMENT);
    bad_view = view(c, bytes); bad_view.device_ordinal = 1;
    CHECK(seen_qwen_fill_f32(&token, bad_view, count, 0.0f).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_fill_f32(&token, view(c, bytes - 1), count, 0.0f).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_fill_f32(&token, view(c, bytes + 4), count, 0.0f).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_fill_f32(&token, view(host, bytes), count, 0.0f).code != SEEN_CUDA_OK);
    CHECK(seen_qwen_add_f32(&token, view(a, bytes), view(b, bytes), view(c, bytes), UINT64_MAX).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_fill_f32(&token, view(c, bytes), 0, 0.0f).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_row_sum_f32(&token, view(a, bytes), view(c, bytes), 0, 6).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_transpose_2d_f32(&token, view(a, bytes), view(b, bytes), 4, 0).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_copy_f32(&token, view(a, bytes), view(b, bytes), 0).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_embedding_gather_f32(&token, view(a, bytes), view(ids, id_bytes), view(c, bytes), 0, 4, 6).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_copy_f32(&token, view(a, bytes), view(static_cast<char *>(a) + sizeof(float), bytes - sizeof(float)), count - 1).code == SEEN_CUDA_INVALID_ARGUMENT);
    CHECK(seen_qwen_transpose_2d_f32(&token, view(a, bytes), view(a, bytes), 4, 6).code == SEEN_CUDA_INVALID_ARGUMENT);

    CHECK_STATUS(seen_cuda_graph_begin_capture(stream));
    token = borrow(stream);
    CHECK(token.flags & SEEN_CUDA_STREAM_LAUNCH_CAPTURE_ACTIVE);
    CHECK_STATUS(seen_qwen_fill_f32(&token, view(c, bytes), count, 7.0f));
    SeenCudaHandle graph = 0, graph_exec = 0;
    CHECK_STATUS(seen_cuda_graph_end_capture(stream, &graph));
    CHECK_STATUS(seen_cuda_graph_instantiate(graph, &graph_exec));
    CHECK_STATUS(seen_cuda_graph_launch(graph_exec, stream));
    CHECK_STATUS(seen_cuda_memcpy_async(host, c, bytes, SEEN_CUDA_COPY_DEVICE_TO_HOST, stream));
    CHECK_STATUS(seen_cuda_event_record(event, stream)); CHECK_STATUS(seen_cuda_event_synchronize(event));
    for (uint64_t i = 0; i < count; ++i) CHECK(equal(values[i], 7.0f));
    CHECK_STATUS(seen_cuda_graph_exec_destroy(&graph_exec));
    CHECK_STATUS(seen_cuda_graph_destroy(&graph));

    SeenCudaHandle temporary = 0; CHECK_STATUS(seen_cuda_stream_create(0, &temporary));
    const SeenCudaHandle stale = temporary; token = borrow(temporary);
    CHECK_STATUS(seen_cuda_stream_destroy(&temporary));
    CHECK(seen_cuda_stream_borrow_launch_token(stale, 0, &rejected).code == SEEN_CUDA_CLOSED);
    CHECK(rejected.native_stream == 0);

    CHECK_STATUS(seen_cuda_host_free(&host_handle));
    CHECK_STATUS(seen_cuda_free(&ids_handle)); CHECK_STATUS(seen_cuda_free(&d_handle));
    CHECK_STATUS(seen_cuda_free(&c_handle)); CHECK_STATUS(seen_cuda_free(&b_handle)); CHECK_STATUS(seen_cuda_free(&a_handle));
    CHECK_STATUS(seen_cuda_event_destroy(&event));
    const SeenCudaHandle closed = stream; CHECK_STATUS(seen_cuda_stream_destroy(&stream));
    CHECK(seen_cuda_stream_borrow_launch_token(closed, 0, &rejected).code == SEEN_CUDA_CLOSED);
    CHECK(stream == 0 && event == 0 && a_handle == 0 && host_handle == 0);
    std::printf("PASS: FEL-1432 correctness ordering capture negatives teardown\n");
    return 0;
}
