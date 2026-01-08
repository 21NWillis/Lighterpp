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

char* decode_token(Tokenizer* tokenizer, int token_id) {
    if (token_id < 0 || token_id >= tokenizer->vocab_size) {
        printf("Error: Invalid token ID: %d\n", token_id);
        exit(1);
    }
    return tokenizer->vocab[token_id];
}

void free_tokenizer(Tokenizer* tokenizer) {
    for (int i = 0; i < tokenizer->vocab_size; i++) {
        free(tokenizer->vocab[i]);
    }
    free(tokenizer->vocab);
    free(tokenizer->vocab_scores);
}