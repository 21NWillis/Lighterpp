#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>


struct Reader {
    const uint8_t* data;
    size_t pos;
    size_t size;
};

static bool read_bytes(Reader* r, void* out, size_t n) {
    if (r->pos + n > r->size) return false;
    memcpy(out, r->data + r->pos, n);
    r->pos += n;
    return true;
}

static bool read_u32(Reader* r, uint32_t* out) {
    return read_bytes(r, out, sizeof(uint32_t));
}

static bool read_u64(Reader* r, uint64_t* out) {
    return read_bytes(r, out, sizeof(uint64_t));
}

static bool read_f32(Reader* r, float* out) {
    return read_bytes(r, out, sizeof(float));
}

static char* read_string(Reader* r) {
    uint64_t len;
    if (!read_u64(r, &len)) return nullptr;
    if (r->pos + len > r->size) return nullptr;
    
    char* str = (char*)malloc(len + 1);
    memcpy(str, r->data + r->pos, len);
    str[len] = '\0';
    r->pos += len;
    return str;
}

static bool skip_metadata_value(Reader* r, GGUFMetadataType type);

static bool skip_metadata_value(Reader* r, GGUFMetadataType type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            r->pos += 1;
            return r->pos <= r->size;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            r->pos += 2;
            return r->pos <= r->size;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            r->pos += 4;
            return r->pos <= r->size;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            r->pos += 8;
            return r->pos <= r->size;
        case GGUF_TYPE_STRING: {
            uint64_t len;
            if (!read_u64(r, &len)) return false;
            r->pos += len;
            return r->pos <= r->size;
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t elem_type;
            uint64_t count;
            if (!read_u32(r, &elem_type)) return false;
            if (!read_u64(r, &count)) return false;
            for (uint64_t i = 0; i < count; i++) {
                if (!skip_metadata_value(r, (GGUFMetadataType)elem_type)) return false;
            }
            return true;
        }
        default:
            return false;
    }
}

// Helper: find metadata entry by key (linear search - fine for ~23 entries)
static GGUFMetadataEntry* find_metadata(GGUFFile* file, const char* key) {
    for (size_t i = 0; i < file->metadata_count; i++) {
        if (strcmp(file->metadata[i].key, key) == 0) {
            return &file->metadata[i];
        }
    }
    return nullptr;
}

static size_t ggml_type_size(GGMLType type) {
    switch (type) {
        case GGML_TYPE_F32:  return 4;
        case GGML_TYPE_F16:  return 2;

        case GGML_TYPE_Q8_0: return 1; 
        case GGML_TYPE_Q4_0: return 1; 
        case GGML_TYPE_Q4_1: return 1; 
        default: return 0;
    }
}

static size_t compute_tensor_size(GGUFTensorInfo* t) {
    size_t nelements = 1;
    for (uint32_t i = 0; i < t->n_dims; i++) {
        nelements *= t->dims[i];
    }
    return nelements * ggml_type_size(t->type);
}


GGUFFile* gguf_open(const char* path) {
    int fd = open(path, O_RDONLY);
    if (fd == -1) {
        fprintf(stderr, "GGUF: Failed to open file: %s\n", path);
        return nullptr;
    }
    
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        fprintf(stderr, "GGUF: Failed to get file size\n");
        close(fd);
        return nullptr;
    }
    size_t file_size = sb.st_size;
    
    void* mmap_addr = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);  // Can close fd after mmap
    
    if (mmap_addr == MAP_FAILED) {
        fprintf(stderr, "GGUF: mmap failed\n");
        return nullptr;
    }
    
    // Set up reader
    Reader r;
    r.data = (const uint8_t*)mmap_addr;
    r.pos = 0;
    r.size = file_size;
    
    // Allocate file handle
    GGUFFile* file = (GGUFFile*)calloc(1, sizeof(GGUFFile));
    file->mmap_addr = mmap_addr;
    file->mmap_size = file_size;
    
    // Read header
    if (!read_u32(&r, &file->header.magic) ||
        !read_u32(&r, &file->header.version) ||
        !read_u64(&r, &file->header.tensor_count) ||
        !read_u64(&r, &file->header.metadata_kv_count)) {
        fprintf(stderr, "GGUF: Failed to read header\n");
        gguf_close(file);
        return nullptr;
    }
    
    // Validate magic
    if (file->header.magic != GGUF_MAGIC) {
        fprintf(stderr, "GGUF: Invalid magic (got 0x%08X, expected 0x%08X)\n",
                file->header.magic, GGUF_MAGIC);
        gguf_close(file);
        return nullptr;
    }
    
    // Validate version
    if (file->header.version < 2 || file->header.version > 3) {
        fprintf(stderr, "GGUF: Unsupported version %u (need 2 or 3)\n", file->header.version);
        gguf_close(file);
        return nullptr;
    }
    
    printf("GGUF: version=%u, tensors=%lu, metadata=%lu\n",
           file->header.version,
           (unsigned long)file->header.tensor_count,
           (unsigned long)file->header.metadata_kv_count);
    
    // Allocate metadata index
    file->metadata_count = file->header.metadata_kv_count;
    file->metadata = (GGUFMetadataEntry*)calloc(file->metadata_count, sizeof(GGUFMetadataEntry));

    for (uint64_t i = 0; i < file->header.metadata_kv_count; i++) {
        GGUFMetadataEntry* entry = &file->metadata[i];
        
        entry->key = read_string(&r);
        if (!entry->key) {
            fprintf(stderr, "GGUF: Failed to read metadata key %lu\n", (unsigned long)i);
            gguf_close(file);
            return nullptr;
        }
        
        uint32_t type;
        if (!read_u32(&r, &type)) {
            fprintf(stderr, "GGUF: Failed to read metadata type\n");
            gguf_close(file);
            return nullptr;
        }
        entry->type = (GGUFMetadataType)type;
        
        // IMPORTANT: Store current position BEFORE skipping - this is where the value lives!
        entry->value_offset = r.pos;
        
        if (!skip_metadata_value(&r, entry->type)) {
            fprintf(stderr, "GGUF: Failed to skip metadata value for key: %s\n", entry->key);
            gguf_close(file);
            return nullptr;
        }
    }
    
    file->tensors = (GGUFTensorInfo*)calloc(file->header.tensor_count, sizeof(GGUFTensorInfo));
    
    for (uint64_t i = 0; i < file->header.tensor_count; i++) {
        GGUFTensorInfo* t = &file->tensors[i];
        
        t->name = read_string(&r);
        if (!t->name) {
            fprintf(stderr, "GGUF: Failed to read tensor name\n");
            gguf_close(file);
            return nullptr;
        }
        
        if (!read_u32(&r, &t->n_dims)) {
            fprintf(stderr, "GGUF: Failed to read tensor dims\n");
            gguf_close(file);
            return nullptr;
        }
        
        for (uint32_t d = 0; d < t->n_dims; d++) {
            if (!read_u64(&r, &t->dims[d])) {
                fprintf(stderr, "GGUF: Failed to read tensor dim %u\n", d);
                gguf_close(file);
                return nullptr;
            }
        }
        
        uint32_t type_raw;
        if (!read_u32(&r, &type_raw)) {
            fprintf(stderr, "GGUF: Failed to read tensor type\n");
            gguf_close(file);
            return nullptr;
        }
        t->type = (GGMLType)type_raw;
        
        if (!read_u64(&r, &t->offset)) {
            fprintf(stderr, "GGUF: Failed to read tensor offset\n");
            gguf_close(file);
            return nullptr;
        }
        
        t->size_bytes = compute_tensor_size(t);
    }
    
    size_t alignment = 32;
    size_t tensor_data_start = (r.pos + alignment - 1) & ~(alignment - 1);
    file->tensor_data = (uint8_t*)mmap_addr + tensor_data_start;
    
    printf("GGUF: Parsed %lu tensors, data starts at offset %zu\n",
           (unsigned long)file->header.tensor_count, tensor_data_start);
    
    return file;
}

void gguf_close(GGUFFile* file) {
    if (!file) return;
    
    // Free metadata keys
    if (file->metadata) {
        for (size_t i = 0; i < file->metadata_count; i++) {
            free(file->metadata[i].key);
        }
        free(file->metadata);
    }
    
    if (file->tensors) {
        for (uint64_t i = 0; i < file->header.tensor_count; i++) {
            free(file->tensors[i].name);
        }
        free(file->tensors);
    }
    
    if (file->mmap_addr && file->mmap_addr != MAP_FAILED) {
        munmap(file->mmap_addr, file->mmap_size);
    }
    
    free(file);
}

GGUFTensorInfo* gguf_find_tensor(GGUFFile* file, const char* name) {
    for (uint64_t i = 0; i < file->header.tensor_count; i++) {
        if (strcmp(file->tensors[i].name, name) == 0) {
            return &file->tensors[i];
        }
    }
    return nullptr;
}

void* gguf_tensor_data(GGUFFile* file, GGUFTensorInfo* tensor) {
    return (uint8_t*)file->tensor_data + tensor->offset;
}

// =============================================================================
// METADATA GETTERS
// The pattern here: find entry by key, check type matches, read directly from mmap
// This is efficient because mmap gives us random access - no need to re-parse!
// =============================================================================

bool gguf_get_uint32(GGUFFile* file, const char* key, uint32_t* out) {
    GGUFMetadataEntry* entry = find_metadata(file, key);
    if (!entry || entry->type != GGUF_TYPE_UINT32) return false;
    
    // Read directly from mmap'd file at stored offset
    memcpy(out, (uint8_t*)file->mmap_addr + entry->value_offset, sizeof(uint32_t));
    return true;
}

bool gguf_get_uint64(GGUFFile* file, const char* key, uint64_t* out) {
    GGUFMetadataEntry* entry = find_metadata(file, key);
    if (!entry || entry->type != GGUF_TYPE_UINT64) return false;
    
    memcpy(out, (uint8_t*)file->mmap_addr + entry->value_offset, sizeof(uint64_t));
    return true;
}

bool gguf_get_float32(GGUFFile* file, const char* key, float* out) {
    GGUFMetadataEntry* entry = find_metadata(file, key);
    if (!entry || entry->type != GGUF_TYPE_FLOAT32) return false;
    
    memcpy(out, (uint8_t*)file->mmap_addr + entry->value_offset, sizeof(float));
    return true;
}

bool gguf_get_string(GGUFFile* file, const char* key, char** out) {
    GGUFMetadataEntry* entry = find_metadata(file, key);
    if (!entry || entry->type != GGUF_TYPE_STRING) return false;
    
    // String format: u64 length, then chars (no null terminator)
    const uint8_t* ptr = (uint8_t*)file->mmap_addr + entry->value_offset;
    uint64_t len;
    memcpy(&len, ptr, sizeof(uint64_t));
    
    *out = (char*)malloc(len + 1);
    memcpy(*out, ptr + sizeof(uint64_t), len);
    (*out)[len] = '\0';
    return true;
}

size_t gguf_get_string_array(GGUFFile* file, const char* key, char*** out) {
    GGUFMetadataEntry* entry = find_metadata(file, key);
    if (!entry || entry->type != GGUF_TYPE_ARRAY) return 0;
    
    // Array format: u32 element_type, u64 count, then elements
    const uint8_t* ptr = (uint8_t*)file->mmap_addr + entry->value_offset;
    
    uint32_t elem_type;
    memcpy(&elem_type, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    
    if (elem_type != GGUF_TYPE_STRING) return 0;  // Only handle string arrays
    
    uint64_t count;
    memcpy(&count, ptr, sizeof(uint64_t));
    ptr += sizeof(uint64_t);
    
    // Allocate array of string pointers
    *out = (char**)malloc(count * sizeof(char*));
    
    // Read each string
    for (uint64_t i = 0; i < count; i++) {
        uint64_t len;
        memcpy(&len, ptr, sizeof(uint64_t));
        ptr += sizeof(uint64_t);
        
        (*out)[i] = (char*)malloc(len + 1);
        memcpy((*out)[i], ptr, len);
        (*out)[i][len] = '\0';
        ptr += len;
    }
    
    return (size_t)count;
}
