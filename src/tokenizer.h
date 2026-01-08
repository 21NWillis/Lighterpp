#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

struct Tokenizer {
    char** vocab; // Array of strings (one per token)
    float* vocab_scores; // Token scores
    int vocab_size; // Number of tokens
    int max_token_len;
};

// Loads a tokenizer from a file
// Parameters:
//  tokenizer: Pointer to a Tokenizer struct to be initialized
//  path: Path to the tokenizer file
//  vocab_size: Number of tokens in the vocabulary
void load_tokenizer(Tokenizer* tokenizer, const char* path, int vocab_size);

// Decodes a token ID to a string
// Parameters:
//  tokenizer: Pointer to a Tokenizer struct
//  token_id: ID of the token to decode
// Returns:
//  A pointer to the decoded string
char* decode_token(Tokenizer* tokenizer, int token_id);

// Frees the tokenizer
// Parameters:
//  tokenizer: Pointer to a Tokenizer struct to be freed
void free_tokenizer(Tokenizer* tokenizer);

#endif