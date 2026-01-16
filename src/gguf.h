#ifndef GGUF_H
#define GGUF_H

#include <stdint.h>
#include <stddef.h>


#define GGUF_MAGIC 0x46554747  // "GGUF" in little-endian

enum GGMLType : uint32_t {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    // Quantized types (not yet supported)
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q8_0 = 8,
};

// Metadata value types
enum GGUFMetadataType : uint32_t {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

struct GGUFHeader {
    uint32_t magic;             
    uint32_t version;           
    uint64_t tensor_count;      
    uint64_t metadata_kv_count; 
};

struct GGUFTensorInfo {
    char* name;           // Tensor name (e.g., "blk.0.attn_q.weight")
    uint32_t n_dims;      // Number of dimensions (1-4)
    uint64_t dims[4];     // Dimension sizes
    GGMLType type;        // Data type (F32, F16, etc.)
    uint64_t offset;      // Offset from start of tensor data section
    
    size_t size_bytes;    // Total size in bytes
};

// Metadata entry: stores key and location of value in mmap'd file
// This is the "index" that lets us do O(n) lookup instead of re-parsing
struct GGUFMetadataEntry {
    char* key;
    GGUFMetadataType type;
    size_t value_offset;  // Offset in file where value starts
};

struct GGUFFile {
    // Memory-mapped file
    void* mmap_addr;
    size_t mmap_size;
    
    GGUFHeader header;
    
    // Metadata index (built during parsing)
    GGUFMetadataEntry* metadata;
    size_t metadata_count;
    
    GGUFTensorInfo* tensors;
    
    // Pointer to start of tensor data section
    void* tensor_data;
};


GGUFFile* gguf_open(const char* path);


void gguf_close(GGUFFile* file);


GGUFTensorInfo* gguf_find_tensor(GGUFFile* file, const char* name);

void* gguf_tensor_data(GGUFFile* file, GGUFTensorInfo* tensor);

bool gguf_get_uint32(GGUFFile* file, const char* key, uint32_t* out);
bool gguf_get_uint64(GGUFFile* file, const char* key, uint64_t* out);
bool gguf_get_float32(GGUFFile* file, const char* key, float* out);
bool gguf_get_string(GGUFFile* file, const char* key, char** out);


size_t gguf_get_string_array(GGUFFile* file, const char* key, char*** out);

#endif // GGUF_H
