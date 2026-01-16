#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

void free_tokenizer(Tokenizer* tokenizer) {
    for (int i = 0; i < tokenizer->vocab_size; i++) {
        free(tokenizer->vocab[i]);
    }
    free(tokenizer->vocab);
    free(tokenizer->vocab_scores);
}