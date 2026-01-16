#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

// Internal C++ helper for fast lookup
struct TokenizerInternals {
    std::map<std::string, int> params;
};

void load_tokenizer(Tokenizer* tokenizer, const char* path, int vocab_size) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        printf("Error: Could not open tokenizer file: %s\n", path);
        exit(1);
    }

    tokenizer->vocab_size = vocab_size;

    if (fread(&tokenizer->max_token_len, sizeof(int), 1, file) != 1) {
        printf("Error: Could not read max token length\n");
        exit(1);
    }

    tokenizer->vocab = (char**)malloc(vocab_size * sizeof(char*));
    tokenizer->vocab_scores = (float*)malloc(vocab_size * sizeof(float));

    for (int i = 0; i < vocab_size; i++) {
        if (fread(&tokenizer->vocab_scores[i], sizeof(float), 1, file) != 1) {
            printf("Error: Could not read vocab score %d\n", i);
            exit(1);
        }

        int len;
        if (fread(&len, sizeof(int), 1, file) != 1) {
            printf("Error: Could not read token length %d\n", i);
            exit(1);
        }

        tokenizer->vocab[i] = (char*)malloc(len + 1);
        if (fread(tokenizer->vocab[i], sizeof(char), len, file) != (size_t)len) {
            printf("Error: Could not read token %d\n", i);
            exit(1);
        }
        tokenizer->vocab[i][len] = '\0';
    }

    fclose(file);
    printf("Tokenizer loaded: %d tokens\n", vocab_size);
}

// Static buffer for decoded token (avoids allocation per call)
static char decoded_buffer[256];

char* decode_token(Tokenizer* tokenizer, int token_id) {
    if (token_id < 0 || token_id >= tokenizer->vocab_size) {
        printf("Error: Invalid token ID: %d\n", token_id);
        exit(1);
    }
    
    char* raw = tokenizer->vocab[token_id];
    
    // Handle special tokens (skip them in output)
    if (raw[0] == '<' && raw[strlen(raw)-1] == '>') {
        // Check for byte tokens like <0xC2>
        if (strlen(raw) == 6 && raw[1] == '0' && raw[2] == 'x') {
            int byte_val;
            if (sscanf(raw, "<0x%02X>", &byte_val) == 1) {
                decoded_buffer[0] = (char)byte_val;
                decoded_buffer[1] = '\0';
                return decoded_buffer;
            }
        }
        // Other special tokens like <s>, </s>, <unk> - return empty
        decoded_buffer[0] = '\0';
        return decoded_buffer;
    }
    
    // SentencePiece uses ▁ (U+2581, "Lower One Eighth Block") for spaces
    // It's encoded as 3 bytes: 0xE2 0x96 0x81
    char* src = raw;
    char* dst = decoded_buffer;
    char* dst_end = decoded_buffer + sizeof(decoded_buffer) - 1;
    
    while (*src && dst < dst_end) {
        // Check for SentencePiece space marker (▁ = 0xE2 0x96 0x81)
        if ((unsigned char)src[0] == 0xE2 && 
            (unsigned char)src[1] == 0x96 && 
            (unsigned char)src[2] == 0x81) {
            *dst++ = ' ';
            src += 3;
        } else {
            *dst++ = *src++;
        }
    }
    *dst = '\0';
    
    return decoded_buffer;
}

// =============================================================================
// ENCODING
// =============================================================================

static std::map<std::string, int> token_map;
static bool map_initialized = false;

static void build_token_map(Tokenizer* t) {
    if (map_initialized) return;
    
    for (int i = 0; i < t->vocab_size; ++i) {
        token_map[t->vocab[i]] = i;
    }
    map_initialized = true;
    printf("Tokenizer: Built lookup map for %d tokens\n", t->vocab_size);
}

static int str_lookup(const std::string& str, Tokenizer* t) {
    if (!map_initialized) build_token_map(t);
    auto it = token_map.find(str);
    if (it != token_map.end()) return it->second;
    return -1;
}

void encode(Tokenizer* t, const char* text, int* tokens, int* n_tokens, int max_tokens) {
    if (text == NULL) return;
    
    if (!map_initialized) build_token_map(t);
    

    std::vector<int> current_tokens;

    if (t->vocab_size > 1 && strcmp(t->vocab[1], "<s>") == 0) {
        current_tokens.push_back(1);
    }
    
    std::string processed_text = " ";
    processed_text += text;
    
    for (size_t i = 0; i < processed_text.length(); i++) {
        std::string ch_str;
        unsigned char ch = processed_text[i];
        
        if (ch == ' ') {
             ch_str = "\xE2\x96\x81";
        } else {
            ch_str = std::string(1, (char)ch);
        }
        
        int id = str_lookup(ch_str, t);
        if (id == -1) {
            char byte_token[8];
            snprintf(byte_token, sizeof(byte_token), "<0x%02X>", ch);
            id = str_lookup(byte_token, t);
        }
        
        if (id != -1) {
            current_tokens.push_back(id);
        } else {
            fprintf(stderr, "Warning: Tokenizer could not encode char '%c' (0x%02X)\n", ch, ch);
        }
    }
    
    while (true) {
        float best_score = -1e10;
        int best_idx = -1;
        int best_token = -1;
        
        for (size_t i = 0; i < current_tokens.size() - 1; i++) {
            std::string merged = std::string(t->vocab[current_tokens[i]]) + 
                                 std::string(t->vocab[current_tokens[i+1]]);
            
            int id = str_lookup(merged, t);
            if (id != -1) {
                float score = t->vocab_scores[id];
                if (score > best_score) {
                    best_score = score;
                    best_idx = i;
                    best_token = id;
                }
            }
        }
        
        if (best_idx == -1) break;
        
        current_tokens[best_idx] = best_token;
        current_tokens.erase(current_tokens.begin() + best_idx + 1);
    }
    
    *n_tokens = 0;
    for (int token : current_tokens) {
        if (*n_tokens < max_tokens) {
            tokens[(*n_tokens)++] = token;
        }
    }
}

void free_tokenizer(Tokenizer* tokenizer) {
    if (tokenizer->vocab) {
        for (int i = 0; i < tokenizer->vocab_size; i++) {
            free(tokenizer->vocab[i]);
        }
        free(tokenizer->vocab);
    }
    if (tokenizer->vocab_scores) free(tokenizer->vocab_scores);
}