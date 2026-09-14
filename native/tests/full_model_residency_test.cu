#include "seen_cuda.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr uint64_t kHeaderBytes = 256;
constexpr uint64_t kDirectoryEntryBytes = 256;
constexpr uint64_t kTensorCount = 866;
constexpr uint64_t kWeightBytes = 14515042384ULL;
constexpr uint64_t kGdnBytes = 150994944;
constexpr uint64_t kConvolutionBytes = 5898240;
constexpr uint64_t kKvBytes = 8388608;
constexpr uint64_t kActivationBytes = 536870912;
constexpr uint64_t kLogitsBytes = 1048576;
constexpr uint64_t kCublasLtBytes = 536870912;
constexpr uint64_t kScratchBytes = 536870912;
constexpr uint64_t kGraphBytes = 268435456;
constexpr uint64_t kTelemetryBytes = 67108864;
constexpr uint64_t kAllocationBytes = 16627529808ULL;
constexpr uint64_t kHostChunkBytes = 64ULL * 1024 * 1024;

#define REQUIRE(expression) do { if (!(expression)) { \
    std::fprintf(stderr, "FAIL:%d: %s\n", __LINE__, #expression); return false; \
} } while (0)
#define CUDA_OK(expression) do { SeenCudaStatus status_ = (expression); \
    if (status_.code != SEEN_CUDA_OK) { \
        std::fprintf(stderr, "FAIL:%d: %s code=%d native=%d op=%s message=%s\n", \
            __LINE__, #expression, status_.code, status_.native_code, \
            status_.operation, status_.message); return false; \
    } \
} while (0)

uint16_t u16(const uint8_t *value) {
    return static_cast<uint16_t>(value[0]) |
        static_cast<uint16_t>(value[1]) << 8;
}

uint32_t u32(const uint8_t *value) {
    uint32_t result = 0;
    for (unsigned index = 0; index < 4; ++index)
        result |= static_cast<uint32_t>(value[index]) << (index * 8);
    return result;
}

uint64_t u64(const uint8_t *value) {
    uint64_t result = 0;
    for (unsigned index = 0; index < 8; ++index)
        result |= static_cast<uint64_t>(value[index]) << (index * 8);
    return result;
}

bool add_checked(uint64_t left, uint64_t right, uint64_t *result) {
    if (right > std::numeric_limits<uint64_t>::max() - left) return false;
    *result = left + right;
    return true;
}

bool read_exact(int file, uint64_t offset, void *destination, uint64_t length) {
    auto *cursor = static_cast<uint8_t *>(destination);
    while (length != 0) {
        const size_t requested = static_cast<size_t>(
            std::min<uint64_t>(length, 1024 * 1024));
        const ssize_t count = pread(file, cursor, requested,
                                    static_cast<off_t>(offset));
        if (count <= 0) return false;
        cursor += count; offset += static_cast<uint64_t>(count);
        length -= static_cast<uint64_t>(count);
    }
    return true;
}

struct Component {
    uint64_t file_offset;
    uint64_t length;
};

struct SqwView {
    int file = -1;
    uint64_t file_bytes = 0;
    std::vector<Component> components;

    bool open_and_validate(const char *path) {
        file = ::open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (file < 0) return false;
        struct stat information {};
        if (fstat(file, &information) != 0 || !S_ISREG(information.st_mode) ||
            information.st_size <= 0) return false;
        file_bytes = static_cast<uint64_t>(information.st_size);
        uint8_t header[kHeaderBytes] {};
        REQUIRE(read_exact(file, 0, header, sizeof(header)));
        REQUIRE(std::memcmp(header, "SQW1", 4) == 0);
        REQUIRE(u32(header + 4) == 0x01020304 && u16(header + 8) == 1 &&
                u16(header + 10) == 0 && u32(header + 12) == kHeaderBytes);
        const uint64_t directory_offset = u64(header + 40);
        const uint64_t names_offset = u64(header + 56);
        const uint64_t names_length = u64(header + 64);
        REQUIRE(u32(header + 48) == kDirectoryEntryBytes &&
                u32(header + 52) == kTensorCount && names_length != 0);
        uint64_t directory_end = 0, names_end = 0;
        REQUIRE(add_checked(directory_offset,
            kTensorCount * kDirectoryEntryBytes, &directory_end));
        REQUIRE(add_checked(names_offset, names_length, &names_end));
        REQUIRE(directory_end <= file_bytes && names_end <= file_bytes &&
                names_length <= 1024 * 1024);
        std::vector<uint8_t> directory(kTensorCount * kDirectoryEntryBytes);
        std::vector<uint8_t> names(static_cast<size_t>(names_length));
        REQUIRE(read_exact(file, directory_offset, directory.data(), directory.size()));
        REQUIRE(read_exact(file, names_offset, names.data(), names.size()));

        uint64_t total = 0;
        std::string previous;
        components.reserve(kTensorCount * 2);
        for (uint64_t ordinal = 0; ordinal < kTensorCount; ++ordinal) {
            const uint8_t *entry = directory.data() + ordinal * kDirectoryEntryBytes;
            const uint64_t name_offset = u64(entry);
            const uint32_t name_length = u32(entry + 8);
            const uint64_t elements = u64(entry + 88);
            const uint64_t data_offset = u64(entry + 96);
            const uint64_t data_length = u64(entry + 104);
            const uint64_t scale_offset = u64(entry + 112);
            const uint64_t scale_length = u64(entry + 120);
            const uint64_t row_elements = u64(entry + 160);
            uint64_t name_end = 0, data_end = 0, scale_end = 0;
            REQUIRE(name_length != 0 && name_length <= 1024 &&
                    add_checked(name_offset, name_length, &name_end) &&
                    name_end <= names_length);
            const std::string name(reinterpret_cast<const char *>(
                names.data() + name_offset), name_length);
            REQUIRE((ordinal == 0 || previous < name) &&
                    u16(entry + 14) == 7 && u16(entry + 16) == 5 &&
                    elements != 0 && row_elements != 0 &&
                    elements % row_elements == 0 && u32(entry + 168) == 64 &&
                    u32(entry + 172) == 64 && data_length != 0 &&
                    scale_length != 0 && u64(entry + 128) == 0 &&
                    u64(entry + 136) == 0 && u64(entry + 144) == 0 &&
                    u64(entry + 152) == 0);
            REQUIRE(add_checked(data_offset, data_length, &data_end) &&
                    add_checked(scale_offset, scale_length, &scale_end) &&
                    data_end <= file_bytes && scale_end <= file_bytes &&
                    add_checked(total, data_length, &total) &&
                    add_checked(total, scale_length, &total));
            REQUIRE(std::any_of(entry + 176, entry + 208,
                                [](uint8_t byte) { return byte != 0; }) &&
                    std::any_of(entry + 208, entry + 240,
                                [](uint8_t byte) { return byte != 0; }));
            components.push_back(Component{data_offset, data_length});
            components.push_back(Component{scale_offset, scale_length});
            previous = name;
        }
        REQUIRE(total == kWeightBytes && components.size() == kTensorCount * 2);
        return true;
    }

    void close() {
        if (file >= 0) { ::close(file); file = -1; }
        components.clear(); file_bytes = 0;
    }
};

struct Allocation {
    SeenCudaHandle handle = 0;
    void *address = nullptr;
    uint64_t bytes = 0;
};

bool allocate(int32_t device, uint64_t bytes, Allocation *allocation) {
    allocation->bytes = bytes;
    CUDA_OK(seen_cuda_malloc(device, bytes, &allocation->handle));
    uint64_t actual = 0; int32_t actual_device = -1;
    CUDA_OK(seen_cuda_allocation_address(allocation->handle,
        &allocation->address, &actual, &actual_device));
    REQUIRE(allocation->address != nullptr && actual == bytes &&
            actual_device == device);
    return true;
}

struct Resources {
    SeenCudaHandle stream = 0;
    SeenCudaHandle host = 0;
    std::vector<Allocation> allocations = std::vector<Allocation>(10);

    bool close() {
        bool okay = true;
        for (size_t index = allocations.size(); index != 0; --index) {
            SeenCudaStatus status = seen_cuda_free(&allocations[index - 1].handle);
            okay = okay && status.code == SEEN_CUDA_OK;
            allocations[index - 1].address = nullptr;
            allocations[index - 1].bytes = 0;
        }
        SeenCudaStatus host_status = seen_cuda_host_free(&host);
        SeenCudaStatus stream_status = seen_cuda_stream_destroy(&stream);
        return okay && host_status.code == SEEN_CUDA_OK &&
            stream_status.code == SEEN_CUDA_OK;
    }

    ~Resources() { (void)close(); }
};

bool run(const char *path) {
    SqwView sqw;
    REQUIRE(sqw.open_and_validate(path));
    SeenCudaDeviceInfo before {};
    CUDA_OK(seen_cuda_device_get(0, &before));
    REQUIRE(before.compute_major == 8 && before.compute_minor == 9 &&
            std::string(before.name).find("RTX 4090") != std::string::npos &&
            before.total_memory_bytes >= kAllocationBytes + 536870912);
    const uint64_t reserve = std::max<uint64_t>(
        536870912, (before.total_memory_bytes / 100) * 3);
    REQUIRE(kAllocationBytes <= before.total_memory_bytes - reserve &&
            before.free_memory_bytes >= kAllocationBytes);

    Resources resources;
    void *host_address = nullptr;
    CUDA_OK(seen_cuda_stream_create(0, &resources.stream));
    CUDA_OK(seen_cuda_host_alloc(kHostChunkBytes, &resources.host));
    uint64_t host_bytes = 0;
    CUDA_OK(seen_cuda_host_allocation_address(resources.host,
        &host_address, &host_bytes));
    REQUIRE(host_address != nullptr && host_bytes == kHostChunkBytes);
    const uint64_t sizes[] = {kWeightBytes, kGdnBytes, kConvolutionBytes,
        kKvBytes, kActivationBytes, kLogitsBytes, kCublasLtBytes,
        kScratchBytes, kGraphBytes, kTelemetryBytes};
    for (size_t index = 0; index < resources.allocations.size(); ++index)
        REQUIRE(allocate(0, sizes[index], &resources.allocations[index]));

    uint64_t destination_offset = 0, transfer_count = 0;
    for (const Component &component : sqw.components) {
        uint64_t source_offset = component.file_offset;
        uint64_t remaining = component.length;
        while (remaining != 0) {
            const uint64_t chunk = std::min(remaining, kHostChunkBytes);
            REQUIRE(read_exact(sqw.file, source_offset, host_address, chunk));
            CUDA_OK(seen_cuda_memcpy_async(
                static_cast<uint8_t *>(resources.allocations[0].address) +
                    destination_offset,
                host_address, chunk, SEEN_CUDA_COPY_HOST_TO_DEVICE,
                resources.stream));
            CUDA_OK(seen_cuda_stream_synchronize(resources.stream));
            source_offset += chunk; destination_offset += chunk;
            remaining -= chunk; ++transfer_count;
        }
    }
    REQUIRE(destination_offset == kWeightBytes && transfer_count >= 1732);
    for (size_t index = 1; index < resources.allocations.size(); ++index)
        CUDA_OK(seen_cuda_memset_async(resources.allocations[index].address, 0,
            resources.allocations[index].bytes, resources.stream));
    CUDA_OK(seen_cuda_stream_synchronize(resources.stream));
    SeenCudaDeviceInfo resident {};
    CUDA_OK(seen_cuda_device_get(0, &resident));
    REQUIRE(before.free_memory_bytes >= resident.free_memory_bytes &&
            before.free_memory_bytes - resident.free_memory_bytes >= kAllocationBytes);
    std::printf("QWN-046A residency: tensors=%llu components=%zu transfers=%llu "
        "weights=%llu allocation=%llu reserve=%llu total_vram=%llu "
        "free_before=%llu free_resident=%llu\n",
        static_cast<unsigned long long>(kTensorCount), sqw.components.size(),
        static_cast<unsigned long long>(transfer_count),
        static_cast<unsigned long long>(kWeightBytes),
        static_cast<unsigned long long>(kAllocationBytes),
        static_cast<unsigned long long>(reserve),
        static_cast<unsigned long long>(before.total_memory_bytes),
        static_cast<unsigned long long>(before.free_memory_bytes),
        static_cast<unsigned long long>(resident.free_memory_bytes));

    REQUIRE(resources.close());
    REQUIRE(resources.close());
    SeenCudaDeviceInfo after {};
    CUDA_OK(seen_cuda_device_get(0, &after));
    REQUIRE(after.free_memory_bytes + 64 * 1024 * 1024 >= before.free_memory_bytes);
    sqw.close(); sqw.close();
    std::puts("PASS: QWN-046A complete Q4 model residency and deterministic cleanup");
    return true;
}

bool sanitizer_smoke(const char *path) {
    SqwView sqw;
    REQUIRE(sqw.open_and_validate(path));
    Resources resources;
    void *host_address = nullptr;
    uint64_t host_bytes = 0;
    CUDA_OK(seen_cuda_stream_create(0, &resources.stream));
    CUDA_OK(seen_cuda_host_alloc(1024 * 1024, &resources.host));
    CUDA_OK(seen_cuda_host_allocation_address(resources.host,
        &host_address, &host_bytes));
    REQUIRE(host_address != nullptr && host_bytes == 1024 * 1024);
    for (Allocation &allocation : resources.allocations) {
        REQUIRE(allocate(0, 1024 * 1024, &allocation));
        CUDA_OK(seen_cuda_memset_async(allocation.address, 0xA5,
            allocation.bytes, resources.stream));
    }
    CUDA_OK(seen_cuda_stream_synchronize(resources.stream));
    REQUIRE(resources.close()); REQUIRE(resources.close());
    sqw.close(); sqw.close();
    std::puts("PASS: QWN-046A sanitizer ownership and cleanup smoke");
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2 && argc != 3) {
        std::fprintf(stderr,
            "usage: qwn_046a_cuda_test WEIGHTS.sqw [--sanitizer-smoke]\n");
        return 64;
    }
    if (argc == 3) {
        if (std::strcmp(argv[2], "--sanitizer-smoke") != 0) return 64;
        return sanitizer_smoke(argv[1]) ? 0 : 1;
    }
    return run(argv[1]) ? 0 : 1;
}
