#ifndef TOKENIZER_H
#define TOKENIZER_H

typedef enum {
    TOKENIZER_SENTENCEPIECE,
    TOKENIZER_BPE
} TokenizerType;

typedef struct {
    char** vocab;
    float* vocab_scores;
    int vocab_size;
    int max_token_len;
    TokenizerType type;
} Tokenizer;

void load_tokenizer(Tokenizer* tokenizer, const char* path, int vocab_size);
void free_tokenizer(Tokenizer* tokenizer);

char* decode_token(Tokenizer* tokenizer, int token_id);
void encode(Tokenizer* tokenizer, const char* text, int* tokens, int* n_tokens, int max_tokens);

#endif