/* Inference for LLaMa 3.x transformer_t model in pure C */

#include <float.h>
#include <ctype.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_set_num_threads(a)
#endif

// ----------------------------------------------------------------------------
// Utilities

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// Return time in milliseconds, for benchmarking the model speed
long time_in_ms() {
  struct timespec time;
  clock_gettime(CLOCK_REALTIME, &time);
  return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// Xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
unsigned int random_u32(unsigned long long* state) {
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

// Random float32 in [0,1)
float random_f32(unsigned long long* state) {
  return (random_u32(state) >> 8) / 16777216.0f;
}

// Print a vector with some statistics for debug purpose
void vector_print(size_t size, float* vector, size_t sample_size, char* name) {
  if (vector == NULL || size == 0) {
    fprintf(stderr, "Empty or invalid vector.\n");
    return;
  }

  // Print the first sample_size elements
  size_t end = (sample_size < size) ? sample_size : size;

  fprintf(stderr, "%6s: ", name ? name : "vector");
  for (size_t i = 0; i < end; i++) {
    fprintf(stderr, "%7.3f ", vector[i]);
  }

  // Print ellipsis if middle elements are skipped
  if (sample_size * 2 < size) {
    fprintf(stderr, "... ");
  }

  // Print the last sample_size elements
  size_t start_tail = (sample_size < size) ? size - sample_size : end;
  for (size_t i = start_tail; i < size; i++) {
    fprintf(stderr, "%7.3f ", vector[i]);
  }

  fprintf(stderr, "-- ");

  // Compute statistics
  float min = FLT_MAX;
  float max = -FLT_MAX;
  float sum = 0.0f;

  for (size_t i = 0; i < size; i++) {
    if (vector[i] < min)
      min = vector[i];
    if (vector[i] > max)
      max = vector[i];
    sum += vector[i];
  }

  float mean = sum / size;

  // Print statistics
  fprintf(
      stderr,
      "Min: %7.3f, Max: %7.3f, Mean: %7.3f, Sum: %f\n",
      min,
      max,
      mean,
      sum
  );
}

// ----------------------------------------------------------------------------
// transformer_t model

#define SEQUENCE_CHUNK_MAX_LEN 512

typedef struct {
  int embedding_dim;  // Token representation (embedding) dimension
  int hidden_dim;     // Intermediate representation dimension in the FFN
  int layer_count;    // Number of decoder layers
  int q_head_count;   // Number of query heads
  int kv_head_count;  // Number of key/value heads
  int vocabulary_len; // Vocabulary size
  int context_len;    // Maximum sequence length
} configuration_t;

typedef struct {
  // Embedding parameter set
  float* embedding_weight; // [vocabulary_len][embedding_dim]
  // Decoder parameter set
  // - Multi-head attention
  float* mha_norm_weight; // [layer_count][embedding_dim]
  float* mha_q_weight;    // [layer_count][kv_head_count][
                          //  q_head_per_kv_head_count][head_dim][embedding_dim]
  float* mha_k_weight;    // [layer_count][kv_head_count][
                          //  head_dim][embedding_dim]
  float* mha_v_weight;    // [layer_count][kv_head_count][
                          //  head_dim][embedding_dim]
  float* mha_out_weight;  // [layer_count][embedding_dim][embedding_dim]
  // - Feed-forward network
  float* ffn_norm_weight; // [layer_count][embedding_dim]
  float* ffn_fc_weight;   // [layer_count][embedding_dim][hidden_dim]
  float* ffn_up_weight;   // [layer_count][embedding_dim][hidden_dim]
  float* ffn_out_weight;  // [layer_count][hidden_dim][embedding_dim]
  // Output parameter set
  float* out_norm_weight; // [embedding_dim]
  float* out_weight;      // [vocabulary_len][embedding_dim]
} parameter_set_t;

typedef struct {
  // Activations
  float* embedding;    // [chunk_len][embedding_dim]
  float* mha_norm;     // [chunk_len][embedding_dim]
  float* mha_q;        // [kv_head_count][q_head_per_kv_head_count][
                       //  chunk_len][head_dim]
  float* mha_score;    // [kv_head_count][q_head_per_kv_head_count][
                       //  chunk_len][context_len]
  float* mha_att;      // [chunk_len][kv_head_count][q_head_per_kv_head_count][
                       //  head_dim] also seen as [chunk_len][embedding_dim]
  float* mha_out;      // [chunk_len][embedding_dim]
  float* ffn_norm;     // [chunk_len][embedding_dim]
  float* ffn_fc;       // [chunk_len][hidden_dim]
  float* ffn_up;       // [chunk_len][hidden_dim]
  float* ffn_out;      // [chunk_len][embedding_dim]
  float* logits;       // [chunk_len][vocabulary_len]
  // KV-cache
  float* k_cache;      // [layer_count][kv_head_count][context_len][head_dim]
  float* v_cache;      // [layer_count][kv_head_count][context_len][head_dim]
  // Utility variables
  float* rope_cos_sin; // [context_len][head_dim]
  // Temporary, array-expanded variables (were loop-private variables)
  float* mha_ss;       // [chunk_len]
  float* mha_k0;       // [kv_head_count][chunk_len][head_dim]
  float* mha_q0;       // [kv_head_count][q_head_per_kv_head_count][
                       //  chunk_len][head_dim]
  float* mha_max;      // [kv_head_count][q_head_per_kv_head_count][chunk_len]
  float* mha_sum;      // [kv_head_count][q_head_per_kv_head_count][chunk_len]
  float* ffn_ss;       // [chunk_len]
  float* out_ss;       // [chunk_len]
} state_t;

typedef enum {
  IMPLEMENTATION_BASIC,  // Basic implementation
  IMPLEMENTATION_INLINED // Inlined implementation
} implementation_t;

typedef struct {
  configuration_t config; // Hyperparameters
  parameter_set_t params; // Weights
  state_t state;          // Activations
  int fd;                 // File descriptor for memory mapping
  float* data;            // Memory mapped data pointer
  ssize_t file_size;      // Size of the checkpoint file in bytes
  implementation_t impl;  // Implementation type
} transformer_t;

void state_malloc(state_t* s, configuration_t* p) {
  size_t head_dim = p->embedding_dim / p->q_head_count;
  size_t kv_dim = head_dim * p->kv_head_count;
  size_t embedding_len = SEQUENCE_CHUNK_MAX_LEN * p->embedding_dim;
  size_t hidden_len = SEQUENCE_CHUNK_MAX_LEN * p->hidden_dim;
  size_t score_len = p->q_head_count * SEQUENCE_CHUNK_MAX_LEN * p->context_len;
  size_t cache_len = p->context_len * p->layer_count * kv_dim;
  size_t logits_len = SEQUENCE_CHUNK_MAX_LEN * p->vocabulary_len;
  size_t rope_len = p->context_len * head_dim;
  size_t ss_len = SEQUENCE_CHUNK_MAX_LEN;
  size_t k0_len = p->kv_head_count * SEQUENCE_CHUNK_MAX_LEN * head_dim;
  size_t q0_len = SEQUENCE_CHUNK_MAX_LEN * p->embedding_dim;
  size_t max_len = p->q_head_count * SEQUENCE_CHUNK_MAX_LEN;
  size_t sum_len = p->q_head_count * SEQUENCE_CHUNK_MAX_LEN;

  s->embedding = calloc(embedding_len, sizeof(*s->embedding));
  s->mha_norm = calloc(embedding_len, sizeof(*s->mha_norm));
  s->mha_q = calloc(embedding_len, sizeof(*s->mha_q));
  s->mha_score = calloc(score_len, sizeof(*s->mha_score));
  s->mha_att = calloc(embedding_len, sizeof(*s->mha_att));
  s->mha_out = calloc(embedding_len, sizeof(*s->mha_out));
  s->ffn_norm = calloc(embedding_len, sizeof(*s->ffn_norm));
  s->ffn_fc = calloc(hidden_len, sizeof(*s->ffn_fc));
  s->ffn_up = calloc(hidden_len, sizeof(*s->ffn_up));
  s->ffn_out = calloc(embedding_len, sizeof(*s->ffn_out));
  s->logits = calloc(logits_len, sizeof(float));
  s->k_cache = calloc(cache_len, sizeof(*s->k_cache));
  s->v_cache = calloc(cache_len, sizeof(*s->v_cache));
  s->rope_cos_sin = calloc(rope_len, sizeof(*s->rope_cos_sin));
  s->mha_ss = calloc(ss_len, sizeof(*s->mha_ss));
  s->mha_k0 = calloc(k0_len, sizeof(*s->mha_k0));
  s->mha_q0 = calloc(q0_len, sizeof(*s->mha_q0));
  s->mha_max = calloc(max_len, sizeof(*s->mha_max));
  s->mha_sum = calloc(sum_len, sizeof(*s->mha_sum));
  s->ffn_ss = calloc(ss_len, sizeof(*s->ffn_ss));
  s->out_ss = calloc(ss_len, sizeof(*s->out_ss));

  // Ensure all mallocs went fine
  if (!s->embedding ||
      !s->mha_norm ||
      !s->mha_q ||
      !s->mha_score ||
      !s->mha_att ||
      !s->mha_out ||
      !s->ffn_norm ||
      !s->ffn_fc ||
      !s->ffn_up ||
      !s->ffn_out ||
      !s->logits ||
      !s->k_cache ||
      !s->v_cache ||
      !s->rope_cos_sin ||
      !s->mha_ss ||
      !s->mha_k0 ||
      !s->mha_q0 ||
      !s->mha_max ||
      !s->mha_sum ||
      !s->ffn_ss ||
      !s->out_ss) {
    fprintf(stderr, "malloc failed!\n");
    exit(EXIT_FAILURE);
  }

  // Initialize RoPE cosine and sine values
  for (size_t i = 0; i < p->context_len; i++) {
    for (size_t j = 0; j < head_dim; j += 2) {
      float freq = 1.0f / powf(500000.0f, j / (float)head_dim);
      float val = i * freq;
      s->rope_cos_sin[i * head_dim + j] = cosf(val);
      s->rope_cos_sin[i * head_dim + j + 1] = sinf(val);
    }
  }
}

void state_free(state_t* s) {
  free(s->embedding);
  free(s->mha_norm);
  free(s->mha_q);
  free(s->mha_score);
  free(s->mha_att);
  free(s->mha_out);
  free(s->ffn_norm);
  free(s->ffn_fc);
  free(s->ffn_up);
  free(s->ffn_out);
  free(s->logits);
  free(s->k_cache);
  free(s->v_cache);
  free(s->rope_cos_sin);
  free(s->mha_ss);
  free(s->mha_k0);
  free(s->mha_q0);
  free(s->mha_max);
  free(s->mha_sum);
  free(s->ffn_ss);
  free(s->out_ss);
}

void parameter_set_mmap(
    parameter_set_t* w, configuration_t* p, float* ptr, int shared_params
) {
  size_t head_dim = p->embedding_dim / p->q_head_count;
  size_t layer_count = p->layer_count;
  w->embedding_weight = ptr;
  ptr += p->vocabulary_len * p->embedding_dim;
  w->mha_norm_weight = ptr;
  ptr += layer_count * p->embedding_dim;
  w->mha_q_weight = ptr;
  ptr += layer_count * p->embedding_dim * (p->q_head_count * head_dim);
  w->mha_k_weight = ptr;
  ptr += layer_count * p->embedding_dim * (p->kv_head_count * head_dim);
  w->mha_v_weight = ptr;
  ptr += layer_count * p->embedding_dim * (p->kv_head_count * head_dim);
  w->mha_out_weight = ptr;
  ptr += layer_count * (p->q_head_count * head_dim) * p->embedding_dim;
  w->ffn_norm_weight = ptr;
  ptr += layer_count * p->embedding_dim;
  w->ffn_fc_weight = ptr;
  ptr += layer_count * p->embedding_dim * p->hidden_dim;
  w->ffn_out_weight = ptr;
  ptr += layer_count * p->hidden_dim * p->embedding_dim;
  w->ffn_up_weight = ptr;
  ptr += layer_count * p->embedding_dim * p->hidden_dim;
  w->out_norm_weight = ptr;
  ptr += p->embedding_dim;
  // Skip what used to be freq_cis_real (for RoPE)
  ptr += p->context_len * head_dim / 2;
  // Skip what used to be freq_cis_imag (for RoPE)
  ptr += p->context_len * head_dim / 2;
  w->out_weight = shared_params ? w->embedding_weight : ptr;
}

void transformer_read_checkpoint(
    char* checkpoint,
    configuration_t* config,
    parameter_set_t* params,
    int* fd,
    float** data,
    ssize_t* file_size
) {
  FILE* file = fopen(checkpoint, "rb");
  if (!file) {
    fprintf(stderr, "Couldn't open file %s\n", checkpoint);
    exit(EXIT_FAILURE);
  }
  // Read in the config header
  if (fread(config, sizeof(configuration_t), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }
  // Negative vocab size is hacky way of signaling unshared params. bit yikes.
  int shared_params = config->vocabulary_len > 0 ? 1 : 0;
  config->vocabulary_len = abs(config->vocabulary_len);
  // Figure out the file size
  fseek(file, 0, SEEK_END); // Move file pointer to end of file
  *file_size = ftell(file); // Get the file size, in bytes
  fclose(file);
  // Memory map the transformer_t params into the data pointer
  *fd = open(checkpoint, O_RDONLY); // Open in read only mode
  if (*fd == -1) {
    fprintf(stderr, "open failed!\n");
    exit(EXIT_FAILURE);
  }
  *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
  if (*data == MAP_FAILED) {
    fprintf(stderr, "mmap failed!\n");
    exit(EXIT_FAILURE);
  }
  float* params_ptr = *data + sizeof(configuration_t) / sizeof(float);
  parameter_set_mmap(params, config, params_ptr, shared_params);
}

void transformer_build(transformer_t* t, char* checkpoint_path) {
  // Read in the configuration_t and the params from the checkpoint
  transformer_read_checkpoint(
      checkpoint_path, &t->config, &t->params, &t->fd, &t->data, &t->file_size
  );
  // Allocate the state_t buffers
  state_malloc(&t->state, &t->config);
}

void transformer_free(transformer_t* t) {
  // Close the memory mapping
  if (t->data != MAP_FAILED) {
    munmap(t->data, t->file_size);
  }
  if (t->fd != -1) {
    close(t->fd);
  }
  // Free the state_t buffers
  state_free(&t->state);
}

// ----------------------------------------------------------------------------
// Neural net blocks; the dynamics of the transformer_t

void rmsnorm(
    int sequence_len,
    int embedding_dim,
    float y[sequence_len][embedding_dim],
    float x[sequence_len][embedding_dim],
    float w[embedding_dim],
    float epsilon
) {
  for (int i = 0; i < sequence_len; i++) {
    // Calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < embedding_dim; j++) {
      ss += x[i][j] * x[i][j];
    }
    ss /= embedding_dim;
    ss += epsilon;
    ss = 1.0f / sqrtf(ss);
    // Normalize and scale
    for (int j = 0; j < embedding_dim; j++) {
      y[i][j] = w[j] * (ss * x[i][j]);
    }
  }
}

void softmax(
    int sequence_len,
    int past,
    int context_len,
    float x[sequence_len][context_len]
) {
  for (int i = 0; i < sequence_len; i++) {
    // Find max value (for numerical stability)
    float max_val = x[i][0];
    for (int j = 1; j < past + i + 1; j++) {
      if (x[i][j] > max_val) {
        max_val = x[i][j];
      }
    }
    // Exp and sum
    float sum = 0.0f;
    for (int j = 0; j < past + i + 1; j++) {
      x[i][j] = expf(x[i][j] - max_val);
      sum += x[i][j];
    }
    // Normalize
    for (int j = 0; j < past + i + 1; j++) {
      x[i][j] /= sum;
    }
  }
}

void matmul(
    int row_count,
    int col_count,
    int red_count,
    float y[row_count][col_count],
    float x[row_count][red_count],
    float w[col_count][red_count]
) {
  for (int i = 0; i < row_count; i++) {
    for (int j = 0; j < col_count; j++) {
      y[i][j] = 0.0f;
      for (int k = 0; k < red_count; k++) {
        y[i][j] += x[i][k] * w[j][k];
      }
    }
  }
}

void rope(
    int context_len,
    int sequence_len,
    int head_dim,
    float x[sequence_len][head_dim],
    float rope_cos_sin[context_len][head_dim],
    int pos
) {
  for (int i = 0; i < sequence_len; i++) {
    for (int j = 0; j < head_dim; j += 2) {
      float fcr = rope_cos_sin[pos + i][j];
      float fci = rope_cos_sin[pos + i][j + 1];
      float v0 = x[i][j];
      float v1 = x[i][j + 1];
      x[i][j] = v0 * fcr - v1 * fci;
      x[i][j + 1] = v0 * fci + v1 * fcr;
    }
  }
}

// Basic transformer implementation using basic operators
void transformer_basic(
    int token_count,
    int* token,
    int vocabulary_len,
    int context_len,
    int layer_count,
    int q_head_count,
    int kv_head_count,
    int q_head_per_kv_head_count,
    int embedding_dim,
    int head_dim,
    int q_dim,
    int kv_dim,
    int hidden_dim,

    float epsilon,

    float embedding_weight[restrict vocabulary_len][embedding_dim],
    float mha_norm_weight[restrict layer_count][embedding_dim],
    float mha_q_weight[restrict layer_count][kv_head_count]
                      [q_head_per_kv_head_count][head_dim][embedding_dim],
    float mha_k_weight[restrict layer_count][kv_head_count][head_dim]
                      [embedding_dim],
    float mha_v_weight[restrict layer_count][kv_head_count][head_dim]
                      [embedding_dim],
    float mha_out_weight[restrict layer_count][embedding_dim][embedding_dim],
    float ffn_norm_weight[restrict layer_count][embedding_dim],
    float ffn_fc_weight[restrict layer_count][hidden_dim][embedding_dim],
    float ffn_up_weight[restrict layer_count][hidden_dim][embedding_dim],
    float ffn_out_weight[restrict layer_count][embedding_dim][hidden_dim],
    float out_norm_weight[restrict embedding_dim],
    float out_weight[restrict vocabulary_len][embedding_dim],

    float embedding[restrict SEQUENCE_CHUNK_MAX_LEN][embedding_dim],
    float mha_norm[restrict SEQUENCE_CHUNK_MAX_LEN][embedding_dim],
    float mha_q[restrict kv_head_count][q_head_per_kv_head_count]
               [SEQUENCE_CHUNK_MAX_LEN][head_dim],
    float k_cache[restrict layer_count][kv_head_count][context_len][head_dim],
    float v_cache[restrict layer_count][kv_head_count][context_len][head_dim],
    float rope_cos_sin[restrict context_len][head_dim],
    float mha_score[restrict kv_head_count][q_head_per_kv_head_count]
                   [SEQUENCE_CHUNK_MAX_LEN][context_len],
    float mha_att[restrict SEQUENCE_CHUNK_MAX_LEN][kv_head_count]
                 [q_head_per_kv_head_count][head_dim],
    float mha_out[restrict SEQUENCE_CHUNK_MAX_LEN][embedding_dim],
    float ffn_norm[restrict SEQUENCE_CHUNK_MAX_LEN][embedding_dim],
    float ffn_fc[restrict SEQUENCE_CHUNK_MAX_LEN][hidden_dim],
    float ffn_up[restrict SEQUENCE_CHUNK_MAX_LEN][hidden_dim],
    float ffn_out[restrict SEQUENCE_CHUNK_MAX_LEN][embedding_dim],
    float logits[restrict SEQUENCE_CHUNK_MAX_LEN][vocabulary_len],
    int past,
    int logits_count
) {
  // Convert token ids to embedding vector representation
  for (int t = 0; t < token_count; t++) {
    for (int e = 0; e < embedding_dim; e++) {
      embedding[t][e] = embedding_weight[token[t]][e];
    }
  }

  // Execute decoder layers
  for (int l = 0; l < layer_count; l++) {

    // Attention rmsnorm: normalize the embedding vectors for the current layer
    rmsnorm(
        token_count,
        embedding_dim,
        mha_norm,
        embedding,
        mha_norm_weight[l],
        epsilon
    );

    // Q matmul for all Q-heads
    for (int k = 0; k < kv_head_count; k++) {
      for (int q = 0; q < q_head_per_kv_head_count; q++) {
        matmul(
            token_count,
            head_dim,
            embedding_dim,
            mha_q[k][q],
            mha_norm,
            mha_q_weight[l][k][q]
        );
      }
    }

    // K matmul for all KV-heads, storing in the k_cache
    for (int k = 0; k < kv_head_count; k++) {
      matmul(
          token_count,
          head_dim,
          embedding_dim,
          k_cache[l][k] + past,
          mha_norm,
          mha_k_weight[l][k]
      );
    }

    // V matmul for all KV-heads, storing in the v_cache
    for (int k = 0; k < kv_head_count; k++) {
      matmul(
          token_count,
          head_dim,
          embedding_dim,
          v_cache[l][k] + past,
          mha_norm,
          mha_v_weight[l][k]
      );
    }

    // RoPE Q for all Q-heads: complex-valued rotate Q in each head
    for (int k = 0; k < kv_head_count; k++) {
      for (int q = 0; q < q_head_per_kv_head_count; q++) {
        rope(
          context_len,
          token_count,
          head_dim,
          mha_q[k][q],
          rope_cos_sin,
          past
        );
      }
    }

    // RoPE K for all KV-heads: complex-valued rotate K in each head
    for (int k = 0; k < kv_head_count; k++) {
      rope(
          context_len,
          token_count,
          head_dim,
          k_cache[l][k] + past,
          rope_cos_sin,
          past
      );
    }

    // Multihead attention. iterate over all Q-heads
    for (int k = 0; k < kv_head_count; k++) {
      for (int q = 0; q < q_head_per_kv_head_count; q++) {
        for (int t = 0; t < token_count; t++) {
          // Calculate the attention score: QKˆT / sqrt(head_dim)
          // Here we don't use mask but a triangular loop (no compute
          // for future tokens)
          for (int p = 0; p < past + t + 1; p++) {
            mha_score[k][q][t][p] = 0.0f;
            for (int h = 0; h < head_dim; h++) {
              mha_score[k][q][t][p] += mha_q[k][q][t][h] * k_cache[l][k][p][h];
            }
            mha_score[k][q][t][p] /= sqrtf(head_dim);
          }
        }

        // Softmax the scores to get attention weights
        softmax(token_count, past, context_len, mha_score[k][q]);

        for (int t = 0; t < token_count; t++) {
          // Weighted sum of the values, here the access function of
          // mha_att is to please the output matmul
          for (int h = 0; h < head_dim; h++) {
            mha_att[t][k][q][h] = 0.0f;
          }
          for (int p = 0; p <= past + t; p++) {
            for (int h = 0; h < head_dim; h++) {
              mha_att[t][k][q][h] +=
                  mha_score[k][q][t][p] * v_cache[l][k][p][h];
            }
          }
        }
      }
    }
  
    // Final matmul to get the output of the attention
    // Note mha_att[t][k][q][h] is replaced by mha_att[t][e] with
    // 0 < e < embedding_dim (consecutive data) 
    matmul(
        token_count,
        embedding_dim,
        embedding_dim,
        mha_out,
        (float (*)[embedding_dim])mha_att,
        mha_out_weight[l]
    );

    // Residual connection back into x
    for (int t = 0; t < token_count; t++) {
      for (int e = 0; e < embedding_dim; e++) {
        embedding[t][e] += mha_out[t][e];
      }
    }

    // Feed-forward network's rmsnorm
    rmsnorm(
        token_count,
        embedding_dim,
        ffn_norm,
        embedding,
        ffn_norm_weight[l],
        epsilon
    );

    // Feed-forward's fully-connected matmul (a.k.a. gate)
    matmul(
        token_count,
        hidden_dim,
        embedding_dim,
        ffn_fc,
        ffn_norm,
        ffn_fc_weight[l]
    );

    // Feed-forward's up matmul
    matmul(
        token_count,
        hidden_dim,
        embedding_dim,
        ffn_up,
        ffn_norm,
        ffn_up_weight[l]
    );

    // SwiGLU non-linearity
    for (int t = 0; t < token_count; t++) {
      for (int e = 0; e < hidden_dim; e++) {
        // SiLU(x)=x*σ(x), where σ(x) is the logistic sigmoid
        ffn_fc[t][e] *= (1.0f / (1.0f + expf(-ffn_fc[t][e])));
        // Elementwise multiply with ffn_up_weight(x)
        ffn_fc[t][e] *= ffn_up[t][e];
      }
    }

    // Final matmul to get the output of the feed-forward network
    matmul(
        token_count,
        embedding_dim,
        hidden_dim,
        ffn_out,
        ffn_fc,
        ffn_out_weight[l]
    );

    // Residual connection
    for (int t = 0; t < token_count; t++) {
      for (int e = 0; e < embedding_dim; e++) {
        embedding[t][e] += ffn_out[t][e];
      }
    }
  }

  // Final rmsnorm
  rmsnorm(
      token_count,
      embedding_dim,
      embedding,
      embedding,
      out_norm_weight,
      epsilon
  );

  // Classifier into logits
  matmul(
      logits_count,
      vocabulary_len,
      embedding_dim,
      logits + token_count - logits_count,
      embedding + token_count - logits_count,
      out_weight
  );
}

// Transformer function designed for (polyhedral) compiler optimization:
// operators have been inlined and loop-private variable have beed replaced
// with expanded arrays. Loop and array dimensions are adapted to expose
// parallelism and vectorization opportunities.
void transformer_inlined(
    int token_count,
    int* token,
    int vocabulary_len,
    int context_len,
    int layer_count,
    int q_head_count,
    int kv_head_count,
    int q_head_per_kv_head_count,
    int embedding_dim,
    int head_dim,
    int q_dim,
    int kv_dim,
    int hidden_dim,

    float epsilon,

    float embedding_weight[restrict vocabulary_len][embedding_dim],
    float mha_norm_weight[restrict layer_count][embedding_dim],
    float mha_q_weight[restrict layer_count][kv_head_count]
                      [q_head_per_kv_head_count][head_dim][embedding_dim],
    float mha_k_weight[restrict layer_count][kv_head_count][head_dim]
                      [embedding_dim],
    float mha_v_weight[restrict layer_count][kv_head_count][head_dim]
                      [embedding_dim],
    float mha_out_weight[restrict layer_count][embedding_dim][kv_head_count]
                      [q_head_per_kv_head_count][head_dim],
    float ffn_norm_weight[restrict layer_count][embedding_dim],
    float ffn_fc_weight[restrict layer_count][hidden_dim][embedding_dim],
    float ffn_up_weight[restrict layer_count][hidden_dim][embedding_dim],
    float ffn_out_weight[restrict layer_count][embedding_dim][hidden_dim],
    float out_norm_weight[restrict embedding_dim],
    float out_weight[restrict vocabulary_len][embedding_dim],

    float embedding[restrict SEQUENCE_CHUNK_MAX_LEN][embedding_dim],
    float mha_norm[restrict SEQUENCE_CHUNK_MAX_LEN][embedding_dim],
    float mha_q[restrict kv_head_count][q_head_per_kv_head_count]
               [SEQUENCE_CHUNK_MAX_LEN][head_dim],
    float k_cache[restrict layer_count][kv_head_count][context_len][head_dim],
    float v_cache[restrict layer_count][kv_head_count][context_len][head_dim],
    float rope_cos_sin[restrict context_len][head_dim],
    float mha_score[restrict kv_head_count][q_head_per_kv_head_count]
                   [SEQUENCE_CHUNK_MAX_LEN][context_len],
    float mha_att[restrict SEQUENCE_CHUNK_MAX_LEN][kv_head_count]
                 [q_head_per_kv_head_count][head_dim],
    float mha_out[restrict SEQUENCE_CHUNK_MAX_LEN][embedding_dim],
    float ffn_norm[restrict SEQUENCE_CHUNK_MAX_LEN][embedding_dim],
    float ffn_fc[restrict SEQUENCE_CHUNK_MAX_LEN][hidden_dim],
    float ffn_up[restrict SEQUENCE_CHUNK_MAX_LEN][hidden_dim],
    float ffn_out[restrict SEQUENCE_CHUNK_MAX_LEN][embedding_dim],
    float logits[restrict SEQUENCE_CHUNK_MAX_LEN][vocabulary_len],
    
    float mha_ss[restrict SEQUENCE_CHUNK_MAX_LEN],
    float mha_k0[restrict kv_head_count][SEQUENCE_CHUNK_MAX_LEN][head_dim],
    float mha_q0[restrict kv_head_count][q_head_per_kv_head_count]
                [SEQUENCE_CHUNK_MAX_LEN][head_dim],
    float mha_max[restrict kv_head_count][q_head_per_kv_head_count]
                 [SEQUENCE_CHUNK_MAX_LEN],
    float mha_sum[restrict kv_head_count][q_head_per_kv_head_count]
                 [SEQUENCE_CHUNK_MAX_LEN],
    float ffn_ss[restrict SEQUENCE_CHUNK_MAX_LEN],
    float out_ss[restrict SEQUENCE_CHUNK_MAX_LEN],

    int past,
    int logits_count
) {
  // Convert token ids to embedding vector representation
  #pragma omp for
  for (int t = 0; t < token_count; t++) {
    for (int e = 0; e < embedding_dim; e++) {
      embedding[t][e] = embedding_weight[token[t]][e];
    }
  }

  // Execute decoder layers
  for (int l = 0; l < layer_count; l++) {

    // Attention rmsnorm: normalize the embedding vectors for the current layer
    #pragma omp for nowait
    for (int t = 0; t < token_count; t++) {
      // Calculate sum of squares
      mha_ss[t] = 0.0f;
      for (int e = 0; e < embedding_dim; e++) {
        mha_ss[t] += embedding[t][e] * embedding[t][e];
      }
      mha_ss[t] /= embedding_dim;
      mha_ss[t] += epsilon;
      mha_ss[t] = 1.0f / sqrtf(mha_ss[t]);
      // Normalize and scale
      for (int e = 0; e < embedding_dim; e++) {
        mha_norm[t][e] = mha_norm_weight[l][e] * (mha_ss[t] * embedding[t][e]);
      }
    }

    #pragma omp barrier

    // K matmul for all KV-heads, storing in the k_cache
    #pragma omp for collapse(2) schedule(static) nowait
    for (int k = 0; k < kv_head_count; k++) {
      for (int t = 0; t < token_count; t++) {
        for (int h = 0; h < head_dim; h++) {
          k_cache[l][k][t + past][h] = 0.0f;
          for (int e = 0; e < embedding_dim; e++) {
            k_cache[l][k][t + past][h] +=
                mha_norm[t][e] * mha_k_weight[l][k][h][e];
          }
        }
      }
    }

    // RoPE K for all KV-heads: complex-valued rotate K in each head
    #pragma omp for collapse(2) schedule(static) nowait
    for (int k = 0; k < kv_head_count; k++) {
      for (int t = 0; t < token_count; t++) {
        for (int h = 0; h < head_dim; h += 2) {
          mha_k0[k][t][h] = k_cache[l][k][t + past][h + 0];
          k_cache[l][k][t + past][h + 0] =
              rope_cos_sin[past + t][h + 0] * k_cache[l][k][t + past][h + 0] -
              rope_cos_sin[past + t][h + 1] * k_cache[l][k][t + past][h + 1];
          k_cache[l][k][t + past][h + 1] =
              rope_cos_sin[past + t][h + 1] * mha_k0[k][t][h] +
              rope_cos_sin[past + t][h + 0] * k_cache[l][k][t + past][h + 1];
        }
      }
    }

    // V matmul for all KV-heads, storing in the v_cache
    #pragma omp for collapse(2) schedule(static) nowait
    for (int k = 0; k < kv_head_count; k++) {
      for (int t = 0; t < token_count; t++) {
        for (int h = 0; h < head_dim; h++) {
          v_cache[l][k][t + past][h] = 0.0f;
          for (int e = 0; e < embedding_dim; e++) {
            v_cache[l][k][t + past][h] +=
                mha_norm[t][e] * mha_v_weight[l][k][h][e];
          }
        }
      }
    }

    #pragma omp barrier

    // Q matmul for all Q-heads
    #pragma omp for collapse(3) schedule(static) nowait
    for (int k = 0; k < kv_head_count; k++) {
      for (int q = 0; q < q_head_per_kv_head_count; q++) {
        for (int t = 0; t < token_count; t++) {
          for (int h = 0; h < head_dim; h++) {
            mha_q[k][q][t][h] = 0.0f;
            for (int e = 0; e < embedding_dim; e++) {
              mha_q[k][q][t][h] += mha_norm[t][e] * mha_q_weight[l][k][q][h][e];
            }
          }
        }
      }
    }

    // RoPE Q for all Q-heads: complex-valued rotate Q in each head
    #pragma omp for collapse(3) schedule(static) nowait
    for (int k = 0; k < kv_head_count; k++) {
      for (int q = 0; q < q_head_per_kv_head_count; q++) {
        for (int t = 0; t < token_count; t++) {
          for (int h = 0; h < head_dim; h += 2) {
            mha_q0[k][q][t][h] = mha_q[k][q][t][h + 0];
            mha_q[k][q][t][h + 0] =
                rope_cos_sin[past + t][h + 0] * mha_q[k][q][t][h + 0] -
                rope_cos_sin[past + t][h + 1] * mha_q[k][q][t][h + 1];
            mha_q[k][q][t][h + 1] =
                rope_cos_sin[past + t][h + 1] * mha_q0[k][q][t][h] +
                rope_cos_sin[past + t][h + 0] * mha_q[k][q][t][h + 1];
          }
        }
      }
    }

    // Multihead attention. iterate over all Q-heads
    #pragma omp for collapse(3) schedule(static) nowait
    for (int k = 0; k < kv_head_count; k++) {
      for (int q = 0; q < q_head_per_kv_head_count; q++) {
        for (int t = 0; t < token_count; t++) {
          // Calculate the attention score: QKˆT / sqrt(head_dim)
          // Here we don't use mask but a triangular loop (no compute
          // for future tokens)
          for (int s = 0; s < past + t + 1; s++) {
            mha_score[k][q][t][s] = 0.0f;
            for (int h = 0; h < head_dim; h++) {
              mha_score[k][q][t][s] += mha_q[k][q][t][h] * k_cache[l][k][s][h];
            }
            mha_score[k][q][t][s] /= sqrtf(head_dim);
          }

          // Softmax the scores to get attention weights
          // - Find max value (for numerical stability)
          mha_max[k][q][t] = mha_score[k][q][t][0];
          for (int s = 1; s < past + t + 1; s++) {
            mha_max[k][q][t] = (mha_score[k][q][t][s] > mha_max[k][q][t])
                                   ? mha_score[k][q][t][s]
                                   : mha_max[k][q][t];
          }
          // - Exp and sum
          mha_sum[k][q][t] = 0.0f;
          for (int s = 0; s < past + t + 1; s++) {
            mha_score[k][q][t][s] =
                expf(mha_score[k][q][t][s] - mha_max[k][q][t]);
            mha_sum[k][q][t] += mha_score[k][q][t][s];
          }
          // - Normalize
          for (int s = 0; s < past + t + 1; s++) {
            mha_score[k][q][t][s] /= mha_sum[k][q][t];
          }

          // Weighted sum of the values, here the access function of
          // mha_att is to please the output matmul
          for (int h = 0; h < head_dim; h++) {
            mha_att[t][k][q][h] = 0.0f;
          }
          for (int s = 0; s < past + t + 1; s++) {
            for (int h = 0; h < head_dim; h++) {
              mha_att[t][k][q][h] +=
                  mha_score[k][q][t][s] * v_cache[l][k][s][h];
            }
          }
        }
      }
    }

    #pragma omp barrier

    // Final matmul to get the output of the attention
    // Note k/q/h loops could be a single loop 0 < h < embedding_dim
    // with accesses in [k][q][h] replaced by [h] (consecutive data) 
    #pragma omp for collapse(2) schedule(static) nowait
    for (int t = 0; t < token_count; t++) {
      for (int e = 0; e < embedding_dim; e++) {
        mha_out[t][e] = 0.0f;
        for (int k = 0; k < kv_head_count; k++) {
          for (int q = 0; q < q_head_per_kv_head_count; q++) {
            for (int h = 0; h < head_dim; h++) {
              mha_out[t][e] +=
                  mha_att[t][k][q][h] * mha_out_weight[l][e][k][q][h];
            }
          }
        }
      }
    }

    // Residual connection back into x
    #pragma omp for collapse(2) schedule(static) nowait
    for (int t = 0; t < token_count; t++) {
      for (int e = 0; e < embedding_dim; e++) {
        embedding[t][e] += mha_out[t][e];
      }
    }

    #pragma omp barrier

    // Feed-forward network's rmsnorm
    #pragma omp for nowait
    for (int i = 0; i < token_count; i++) {
      // Calculate sum of squares
      ffn_ss[i] = 0.0f;
      for (int j = 0; j < embedding_dim; j++) {
        ffn_ss[i] += embedding[i][j] * embedding[i][j];
      }
      ffn_ss[i] /= embedding_dim;
      ffn_ss[i] += epsilon;
      ffn_ss[i] = 1.0f / sqrtf(ffn_ss[i]);
      // Normalize and scale
      for (int j = 0; j < embedding_dim; j++) {
        ffn_norm[i][j] = ffn_norm_weight[l][j] * (ffn_ss[i] * embedding[i][j]);
      }
    }

    #pragma omp barrier

    // Feed-forward's fully-connected matmul (a.k.a. gate)
    #pragma omp for collapse(2) schedule(static) nowait
    for (int t = 0; t < token_count; t++) {
      for (int h = 0; h < hidden_dim; h++) {
        ffn_fc[t][h] = 0.0f;
        for (int e = 0; e < embedding_dim; e++) {
          ffn_fc[t][h] += ffn_norm[t][e] * ffn_fc_weight[l][h][e];
        }
      }
    }

    // Feed-forward's up matmul
    #pragma omp for collapse(2) schedule(static) nowait
    for (int t = 0; t < token_count; t++) {
      for (int h = 0; h < hidden_dim; h++) {
        ffn_up[t][h] = 0.0f;
        for (int e = 0; e < embedding_dim; e++) {
          ffn_up[t][h] += ffn_norm[t][e] * ffn_up_weight[l][h][e];
        }
      }
    }

    // SwiGLU non-linearity
    #pragma omp for collapse(2) schedule(static) nowait
    for (int t = 0; t < token_count; t++) {
      for (int h = 0; h < hidden_dim; h++) {
        // SiLU(x)=x*σ(x), where σ(x) is the logistic sigmoid
        ffn_fc[t][h] *= (1.0f / (1.0f + expf(-ffn_fc[t][h])));
        // Elementwise multiply with ffn_up_weight(x)
        ffn_fc[t][h] *= ffn_up[t][h];
      }
    }

    #pragma omp barrier

    // Final matmul to get the output of the feed-forward network
    #pragma omp for collapse(2) schedule(static) nowait
    for (int t = 0; t < token_count; t++) {
      for (int e = 0; e < embedding_dim; e++) {
        ffn_out[t][e] = 0.0f;
        for (int h = 0; h < hidden_dim; h++) {
          ffn_out[t][e] += ffn_fc[t][h] * ffn_out_weight[l][e][h];
        }
      }
    }

    // Residual connection
    #pragma omp for collapse(2) schedule(static) nowait
    for (int t = 0; t < token_count; t++) {
      for (int e = 0; e < embedding_dim; e++) {
        embedding[t][e] += ffn_out[t][e];
      }
    }

    #pragma omp barrier
  }

  // Final rmsnorm
  #pragma omp for
  for (int t = 0; t < token_count; t++) {
    // Calculate sum of squares
    out_ss[t] = 0.0f;
    for (int e = 0; e < embedding_dim; e++) {
      out_ss[t] += embedding[t][e] * embedding[t][e];
    }
    out_ss[t] /= embedding_dim;
    out_ss[t] += epsilon;
    out_ss[t] = 1.0f / sqrtf(out_ss[t]);
    // Normalize and scale
    for (int e = 0; e < embedding_dim; e++) {
      embedding[t][e] = out_norm_weight[e] * (out_ss[t] * embedding[t][e]);
    }
  }

  // Classifier into logits
  #pragma omp for collapse(2)
  for (int l = 0; l < logits_count; l++) {
    for (int v = 0; v < vocabulary_len; v++) {
      logits[l + token_count - logits_count][v] = 0.0f;
      for (int e = 0; e < embedding_dim; e++) {
        logits[l + token_count - logits_count][v] +=
            embedding[l + token_count - logits_count][e] * out_weight[v][e];
      }
    }
  }
}

void transformer_driver(
    transformer_t* transformer,
    int sequence_len,
    int* sequence,
    int past,
    int logits_count
) {
  configuration_t* c = &transformer->config;
  parameter_set_t* p = &transformer->params;
  state_t* s = &transformer->state;
  int vocabulary_len = c->vocabulary_len;
  int context_len = c->context_len;
  int layer_count = c->layer_count;
  int q_head_count = c->q_head_count;
  int kv_head_count = c->kv_head_count;
  int q_head_per_kv_head_count = q_head_count / kv_head_count;
  int embedding_dim = c->embedding_dim;
  int head_dim = embedding_dim / q_head_count;
  int q_dim = head_dim * q_head_count;
  int kv_dim = head_dim * kv_head_count;
  int hidden_dim = c->hidden_dim;

  if (transformer->impl == IMPLEMENTATION_INLINED) {
    transformer_inlined(
        sequence_len,
        sequence,
        vocabulary_len,
        context_len,
        layer_count,
        q_head_count,
        kv_head_count,
        q_head_per_kv_head_count,
        embedding_dim,
        head_dim,
        q_dim,
        kv_dim,
        hidden_dim,

        1e-5f,

        (float (*)[embedding_dim])p->embedding_weight,
        (float (*)[embedding_dim])p->mha_norm_weight,
        (float (*)[kv_head_count][q_head_per_kv_head_count][head_dim]
                  [embedding_dim])p->mha_q_weight,
        (float (*)[kv_head_count][head_dim][embedding_dim])p->mha_k_weight,
        (float (*)[kv_head_count][head_dim][embedding_dim])p->mha_v_weight,
        (float (*)[embedding_dim][kv_head_count][q_head_per_kv_head_count]
                  [head_dim])p->mha_out_weight,
        (float (*)[embedding_dim])p->ffn_norm_weight,
        (float (*)[hidden_dim][embedding_dim])p->ffn_fc_weight,
        (float (*)[hidden_dim][embedding_dim])p->ffn_up_weight,
        (float (*)[embedding_dim][hidden_dim])p->ffn_out_weight,
        (float(*))p->out_norm_weight,
        (float (*)[embedding_dim])p->out_weight,

        (float (*)[embedding_dim])s->embedding,
        (float (*)[embedding_dim])s->mha_norm,
        (float (*)[q_head_per_kv_head_count][SEQUENCE_CHUNK_MAX_LEN]
                  [head_dim])s->mha_q,
        (float (*)[kv_head_count][context_len][head_dim])s->k_cache,
        (float (*)[kv_head_count][context_len][head_dim])s->v_cache,
        (float (*)[head_dim])s->rope_cos_sin,
        (float (*)[q_head_per_kv_head_count][context_len]
                  [context_len])s->mha_score,
        (float (*)[kv_head_count][q_head_per_kv_head_count]
                  [head_dim])s->mha_att,
        (float (*)[embedding_dim])s->mha_out,
        (float (*)[embedding_dim])s->ffn_norm,
        (float (*)[hidden_dim])s->ffn_fc,
        (float (*)[hidden_dim])s->ffn_up,
        (float (*)[embedding_dim])s->ffn_out,
        (float (*)[vocabulary_len])s->logits,

        (float(*))s->mha_ss,
        (float (*)[SEQUENCE_CHUNK_MAX_LEN][head_dim])s->mha_k0,
        (float (*)[q_head_per_kv_head_count][SEQUENCE_CHUNK_MAX_LEN]
                  [head_dim])s->mha_q0,
        (float (*)[q_head_per_kv_head_count][SEQUENCE_CHUNK_MAX_LEN])s->mha_max,
        (float (*)[q_head_per_kv_head_count][SEQUENCE_CHUNK_MAX_LEN])s->mha_sum,
        (float(*))s->ffn_ss,
        (float(*))s->out_ss,

        past,
        logits_count
    );
  } else {
    transformer_basic(
        sequence_len,
        sequence,
        vocabulary_len,
        context_len,
        layer_count,
        q_head_count,
        kv_head_count,
        q_head_per_kv_head_count,
        embedding_dim,
        head_dim,
        q_dim,
        kv_dim,
        hidden_dim,

        1e-5f,

        (float (*)[embedding_dim])p->embedding_weight,
        (float (*)[embedding_dim])p->mha_norm_weight,
        (float (*)[kv_head_count][q_head_per_kv_head_count][head_dim]
                  [embedding_dim])p->mha_q_weight,
        (float (*)[kv_head_count][head_dim][embedding_dim])p->mha_k_weight,
        (float (*)[kv_head_count][head_dim][embedding_dim])p->mha_v_weight,
        (float (*)[embedding_dim][embedding_dim])p->mha_out_weight,
        (float (*)[embedding_dim])p->ffn_norm_weight,
        (float (*)[hidden_dim][embedding_dim])p->ffn_fc_weight,
        (float (*)[hidden_dim][embedding_dim])p->ffn_up_weight,
        (float (*)[embedding_dim][hidden_dim])p->ffn_out_weight,
        (float(*))p->out_norm_weight,
        (float (*)[embedding_dim])p->out_weight,

        (float (*)[embedding_dim])s->embedding,
        (float (*)[embedding_dim])s->mha_norm,
        (float (*)[q_head_per_kv_head_count][SEQUENCE_CHUNK_MAX_LEN]
                  [head_dim])s->mha_q,
        (float (*)[kv_head_count][context_len][head_dim])s->k_cache,
        (float (*)[kv_head_count][context_len][head_dim])s->v_cache,
        (float (*)[head_dim])s->rope_cos_sin,
        (float (*)[q_head_per_kv_head_count][context_len]
                  [context_len])s->mha_score,
        (float (*)[kv_head_count][q_head_per_kv_head_count]
                  [head_dim])s->mha_att,
        (float (*)[embedding_dim])s->mha_out,
        (float (*)[embedding_dim])s->ffn_norm,
        (float (*)[hidden_dim])s->ffn_fc,
        (float (*)[hidden_dim])s->ffn_up,
        (float (*)[embedding_dim])s->ffn_out,
        (float (*)[vocabulary_len])s->logits,

        past,
        logits_count
    );
  }
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

#define MAX_TOKEN_STRING 512 // Must be multiple of 2
#define TOKEN_BOS 128000 // Beginning of sequence token
#define TOKEN_EOS 128001 // End of sequence token

typedef struct {
  char* str;
  int id;
} token_index_t;

typedef struct {
  char** vocab;
  float* vocab_scores;
  token_index_t* sorted_vocab;
  int vocab_size;
  unsigned int max_token_length;
  unsigned char byte_pieces[MAX_TOKEN_STRING]; // Stores all single-byte strings
} tokenizer_t;

int compare_tokens(const void* a, const void* b) {
  return strcmp(((token_index_t*)a)->str, ((token_index_t*)b)->str);
}

void tokenizer_build(tokenizer_t* t, char* tokenizer_path, int vocab_size) {
  // I should have written the vocab_size into the tokenizer file... sigh
  t->vocab_size = vocab_size;
  // Malloc space to hold the scores and the strings
  t->vocab = malloc(vocab_size * sizeof(*t->vocab));
  t->vocab_scores = malloc(vocab_size * sizeof(*t->vocab_scores));
  t->sorted_vocab = NULL; // Initialized lazily
  for (int i = 0; i < MAX_TOKEN_STRING / 2; i++) {
    t->byte_pieces[i * 2] = (unsigned char)i;
    t->byte_pieces[i * 2 + 1] = '\0';
  }
  // Read in the file
  FILE* file = fopen(tokenizer_path, "rb");
  if (!file) {
    fprintf(stderr, "couldn't load %s\n", tokenizer_path);
    exit(EXIT_FAILURE);
  }
  if (fread(&t->max_token_length, sizeof(t->max_token_length), 1, file) != 1) {
    fprintf(stderr, "failed read\n");
    exit(EXIT_FAILURE);
  }
  int len;
  for (int i = 0; i < vocab_size; i++) {
    if (fread(t->vocab_scores + i, sizeof(*t->vocab_scores), 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    if (fread(&len, sizeof(len), 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    t->vocab[i] = malloc(len + 1);
    if (fread(t->vocab[i], len, 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    t->vocab[i][len] = '\0'; // Add the string terminating token
  }
  fclose(file);
}

void tokenizer_free(tokenizer_t* t) {
  for (int i = 0; i < t->vocab_size; i++) {
    free(t->vocab[i]);
  }
  free(t->vocab);
  free(t->vocab_scores);
  free(t->sorted_vocab);
}

char* tokenizer_decode(tokenizer_t* t, int token) {
  char* piece = t->vocab[token];

  // cCreful, some tokens designate raw bytes, and look like e.g. '<0x01>'
  // parse this and convert and return the actual byte
  unsigned char byte_val;
  if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
    piece = (char*)t->byte_pieces + byte_val * 2;
  }
  return piece;
}

void safe_printf(char* piece) {
  // Piece might be a raw byte token, and we only want to print printable chars
  // or whitespace because some of the other bytes can be various control codes,
  // backspace, etc.
  if (piece == NULL) {
    return;
  }
  if (piece[0] == '\0') {
    return;
  }
  if (piece[1] == '\0') {
    unsigned char byte_val = piece[0];
    if (!(isprint(byte_val) || isspace(byte_val))) {
      return; // Bad byte, don't print it
    }
  }
  printf("%s", piece);
}

int str_lookup(char* str, token_index_t* sorted_vocab, int vocab_size) {
  // Efficiently find the perfect match for str in vocab, return its index or -1
  // if not found
  token_index_t tok = {.str = str}; // Acts as the key to search for
  token_index_t* res = bsearch(
      &tok, sorted_vocab, vocab_size, sizeof(*sorted_vocab), compare_tokens
  );
  return res != NULL ? res->id : -1;
}

void tokenizer_encode(
    tokenizer_t* t,
    char* text,
    int8_t bos,
    int8_t eos,
    int* tokens,
    int* n_tokens
) {
  // Encode the string text (input) into an upper-bound preallocated tokens[]
  // array bos != 0 means prepend the BOS token, eos != 0 means append the
  // EOS token
  if (text == NULL) {
    fprintf(stderr, "cannot encode NULL text\n");
    exit(EXIT_FAILURE);
  }

  if (t->sorted_vocab == NULL) {
    // Lazily malloc and sort the vocabulary
    t->sorted_vocab = malloc(t->vocab_size * sizeof(*t->sorted_vocab));
    for (int i = 0; i < t->vocab_size; i++) {
      t->sorted_vocab[i].str = t->vocab[i];
      t->sorted_vocab[i].id = i;
    }
    qsort(
        t->sorted_vocab, t->vocab_size, sizeof(*t->sorted_vocab), compare_tokens
    );
  }

  // Create a temporary buffer that will store merge candidates of always two
  // consecutive tokens *2 for concat, +1 for null terminator +2 for UTF8 (in
  // case max_token_length is 1)
  size_t str_buffer_size = (t->max_token_length * 2 + 1 + 2) * sizeof(char);
  char* str_buffer = malloc(str_buffer_size);
  size_t str_len = 0;

  // Start at 0 tokens
  *n_tokens = 0;

  // Add optional BOS token, if desired
  if (bos) {
    tokens[(*n_tokens)++] = TOKEN_BOS;
  }

  // add_dummy_prefix is true by default
  // so prepend a dummy prefix token to the input string, but only if text != ""
  // TODO: pretty sure this isn't correct in the general case but I don't have
  // the energy to read more of the sentencepiece code to figure out what it's
  // doing

  // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
  // Code point ↔ UTF-8 conversion
  // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
  // U+0000  U+007F    0xxxxxxx
  // U+0080  U+07FF    110xxxxx 10xxxxxx
  // U+0800  U+FFFF    1110xxxx 10xxxxxx 10xxxxxx
  // U+10000 U+10FFFF  11110xxx 10xxxxxx 10xxxxxx 10xxxxxx

  // Process the raw (UTF-8) byte sequence of the input string
  for (char* c = text; *c != '\0'; c++) {

    // Reset buffer if the current byte is ASCII or a leading byte
    // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the
    // rest 0x80 is 10000000 in UTF-8, all continuation bytes start with "10" in
    // first two bits so in English this is: "if this byte is not a continuation
    // byte"
    if ((*c & 0xC0) != 0x80) {
      // This byte must be either a leading byte (11...) or an ASCII char
      // (0x...)
      // => reset our location, as we're starting a new UTF-8 codepoint
      str_len = 0;
    }

    // Append the current byte to the buffer
    // note: ++ is post-increment, incremented after this line
    str_buffer[str_len++] = *c;
    str_buffer[str_len] = '\0';

    // While the next character is a continuation byte, continue appending
    // but if there are too many of them, just stop to avoid overruning
    // str_buffer size.
    if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
      continue;
    }

    // OK c+1 is not a continuation byte, so we've read in a full codepoint
    int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

    if (id != -1) {
      // We found this codepoint in vocab, add it as a token
      tokens[(*n_tokens)++] = id;
    } else {
      // byte_fallback encoding: just encode each byte as a token
      // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
      // so the individual bytes only start at index 3
      for (int i = 0; i < str_len; i++) {
        tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
      }
    }
    str_len = 0; // Protect against a sequence of stray UTF8 continuation bytes
  }

  // Merge the best consecutive pair or triple each iteration, according to the
  // scores in vocab_scores
  while (1) {
    float best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;
    // Length of the best merge sequence (2 for pair, 3 for triple)
    int best_len = 2;

    // First, try to find the best pair to merge
    for (int i = 0; i < (*n_tokens - 1); i++) {
      // Check if we can merge the pair (tokens[i], tokens[i+1])
      snprintf(
          str_buffer,
          str_buffer_size,
          "%s%s",
          t->vocab[tokens[i]],
          t->vocab[tokens[i + 1]]
      );
      int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
      if (id != -1 && t->vocab_scores[id] > best_score) {
        // This merge pair exists in vocab! record its score and position
        best_score = t->vocab_scores[id];
        best_id = id;
        best_idx = i;
      }
    }

    // If no pair was found, try to find the best triple to merge
    if (best_idx == -1) {
      for (int i = 0; i < (*n_tokens - 2); i++) {
        // Check if we can merge the triple (tokens[i], tokens[i+1],
        // tokens[i+2])
        snprintf(
            str_buffer,
            str_buffer_size,
            "%s%s%s",
            t->vocab[tokens[i]],
            t->vocab[tokens[i + 1]],
            t->vocab[tokens[i + 2]]
        );
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1 && t->vocab_scores[id] > best_score) {
          // This merge triple exists in vocab! record its score and position
          best_score = t->vocab_scores[id];
          best_id = id;
          best_idx = i;
          best_len = 3;
        }
      }
    }

    if (best_idx == -1) {
      // We couldn't find any more pairs or triples to merge, so we're done
      break;
    }

    // Merge the consecutive pair or triple (best_idx, best_idx+1[, best_idx+2])
    // into new token best_id
    tokens[best_idx] = best_id;
    // Delete token(s) at position best_idx+1 (and optionally best_idx+2), shift
    // the entire sequence back
    for (int i = best_idx + 1; i < (*n_tokens - best_len + 1); i++) {
      tokens[i] = tokens[i + best_len - 1];
    }
    // Token length decreased by the number of merged tokens minus one
    (*n_tokens) -= (best_len - 1);
  }

  // Add optional EOS token, if desired
  if (eos) {
    tokens[(*n_tokens)++] = TOKEN_EOS;
  }

  free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

// Struct used when sorting probabilities during top-p sampling
typedef struct {
  float probability;
  int index;
} probability_index_t;

typedef struct {
  int vocabulary_len;
  probability_index_t* probindex; // Buffer used in top-p sampling
  float temperature;
  float topp;
  unsigned long long rng_state;
} sampler_t;

int sample_argmax(float* probability, int n) {
  // Return the index that has the highest probability
  int max_i = 0;
  float max_p = probability[0];
  for (int i = 1; i < n; i++) {
    if (probability[i] > max_p) {
      max_i = i;
      max_p = probability[i];
    }
  }
  return max_i;
}

int sample_mult(float* probability, int n, float coin) {
  // Sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probability[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1; // In case of rounding errors
}

int compare(const void* a, const void* b) {
  probability_index_t* a_ = (probability_index_t*)a;
  probability_index_t* b_ = (probability_index_t*)b;
  if (a_->probability > b_->probability)
    return -1;
  if (a_->probability < b_->probability)
    return 1;
  return 0;
}

int sample_topp(
    float* probability,
    int n,
    float topp,
    probability_index_t* probindex,
    float coin
) {
  // Top-p sampling (or "nucleus sampling") samples from the smallest set of
  // tokens that exceed probability topp. This way we never sample tokens that
  // have very low probabilities and are less likely to go "off the rails".
  // coin is a random number in [0, 1), usually from random_f32()

  int n0 = 0;
  // Quicksort indices in descending order of probabilities
  // values smaller than (1 - topp) / (n - 1) cannot be part of the result
  // so for efficiency we crop these out as candidates before sorting
  const float cutoff = (1.0f - topp) / (n - 1);
  for (int i = 0; i < n; i++) {
    if (probability[i] >= cutoff) {
      probindex[n0].index = i;
      probindex[n0].probability = probability[i];
      n0++;
    }
  }
  qsort(probindex, n0, sizeof(*probindex), compare);

  // Truncate the list where cumulative probability exceeds topp
  float cumulative_prob = 0.0f;
  int last_idx = n0 - 1; // In case of rounding errors consider all elements
  for (int i = 0; i < n0; i++) {
    cumulative_prob += probindex[i].probability;
    if (cumulative_prob > topp) {
      last_idx = i;
      break; // We've exceeded topp by including last_idx
    }
  }

  // Sample from the truncated list
  float r = coin * cumulative_prob;
  float cdf = 0.0f;
  for (int i = 0; i <= last_idx; i++) {
    cdf += probindex[i].probability;
    if (r < cdf) {
      return probindex[i].index;
    }
  }
  return probindex[last_idx].index; // In case of rounding errors
}

void sampler_build(
    sampler_t* sampler,
    int vocabulary_len,
    float temperature,
    float topp,
    unsigned long long rng_seed
) {
  sampler->vocabulary_len = vocabulary_len;
  sampler->temperature = temperature;
  sampler->topp = topp;
  sampler->rng_state = rng_seed;
  // Buffer only used with nucleus sampling; may not need but it's ~small
  sampler->probindex =
      malloc(sampler->vocabulary_len * sizeof(*sampler->probindex));
}

void sampler_free(sampler_t* sampler) {
  free(sampler->probindex);
}

int sampler_sample(sampler_t* sampler, float* logits) {
  // Sample the token given the logits and some hyperparameters
  int next;
  if (sampler->temperature == 0.0f) {
    // Greedy argmax sampling: take the token with the highest probability
    next = sample_argmax(logits, sampler->vocabulary_len);
  } else {
    // Apply the temperature to the logits
    for (int q = 0; q < sampler->vocabulary_len; q++) {
      logits[q] /= sampler->temperature;
    }
    // Apply softmax to the logits to get the probabilities for next token
    softmax(
        1,
        sampler->vocabulary_len,
        sampler->vocabulary_len,
        (float (*)[sampler->vocabulary_len])logits
    );
    // Flip a (float) coin (this is our source of entropy for sampling)
    float coin = random_f32(&sampler->rng_state);
    // We sample from this distribution to get the next token
    if (sampler->topp <= 0 || sampler->topp >= 1) {
      // Simply sample from the predicted probability distribution
      next = sample_mult(logits, sampler->vocabulary_len, coin);
    } else {
      // Top-p (nucleus) sampling, clamping the least likely tokens to zero
      next = sample_topp(
          logits,
          sampler->vocabulary_len,
          sampler->topp,
          sampler->probindex,
          coin
      );
    }
  }
  return next;
}

// ----------------------------------------------------------------------------
// Generation loop

void generate(
    transformer_t* transformer,
    tokenizer_t* tokenizer,
    sampler_t* sampler,
    char* prompt,
    int steps
) {
  char* empty_prompt = "";
  if (prompt == NULL) {
    prompt = empty_prompt;
  }

  // Encode the (string) prompt into tokens sequence
  int sequence_len = 0;
  // Allocate +3 for '\0', ?BOS, ?EOS
  int* sequence = malloc((strlen(prompt) + 3) * sizeof(*sequence));
  tokenizer_encode(tokenizer, prompt, 1, 0, sequence, &sequence_len);
  if (sequence_len < 1) {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  // Print the prompt tokens
  fprintf(stderr, "Sequence (%d tokens):\n", sequence_len);
  for (int i = 0; i < sequence_len; i++) {
    fprintf(stderr, "%d (\"", sequence[i]);
    char* piece = tokenizer_decode(tokenizer, sequence[i]);
    safe_printf(piece);
    fflush(stdout);
    fprintf(stderr, "\") ");
  }
  fprintf(stderr, "\n");

  int next; // Will store the next token in the sequence

  // Cache warmup
  transformer_driver(transformer, 1, sequence, 0, 1);

  // Print the prompt string
  fprintf(stderr, "\033[1;34m");
  while (*prompt) {
    fprintf(stderr, "%c", *prompt);
    prompt++;
  }
  fprintf(stderr, "\033[0m");

  size_t generated_count = 0; // Number of tokens generated so far
  size_t past = 0;            // Number of tokens already processed
  size_t vocabulary_len = transformer->config.vocabulary_len;
  size_t logits_offset = (sequence_len - 1) * vocabulary_len;

  // Input token sequence must not be empty
  if (sequence_len == 0) {
    fprintf(stderr, "prompt must not be empty\n");
    exit(EXIT_FAILURE);
  }

  // Timing
  long start = 0;
  long end = 0;
  double prefill_time = 0.0;
  double decode_time = 0.0;

  // First process the input token sequence by chunks
  int* chunk = sequence;
  size_t remaining_sequence_len = sequence_len;
  size_t chunk_len = MIN(remaining_sequence_len, SEQUENCE_CHUNK_MAX_LEN);
  start = time_in_ms();

  #pragma omp parallel if(transformer->impl == IMPLEMENTATION_INLINED)
  {
    while (chunk_len != 0) {
      // We only compute the very last logits
      int logits_count = (remaining_sequence_len - chunk_len == 0) ? 1 : 0;

      // Run the transformer to fill the KV-cache and get final logits
      transformer_driver(transformer, chunk_len, chunk, past, logits_count);

      #pragma omp single
      {
        remaining_sequence_len -= chunk_len;
        chunk += chunk_len;
        past += chunk_len;
        chunk_len = MIN(remaining_sequence_len, SEQUENCE_CHUNK_MAX_LEN);

        // First token generation after the prompt has been processed
        if (chunk_len == 0) {
          end = time_in_ms();
          prefill_time = (end - start) / 1000.0;

          float* logits = transformer->state.logits + logits_offset;

          next = sampler_sample(sampler, logits);
          // print the token as string, decode it with the tokenizer_t object
          char* piece = tokenizer_decode(tokenizer, next);
          safe_printf(piece); // Safe printf("%s", piece)
          fflush(stdout);
          generated_count++;
        }
      }
    }

    // Then generate tokens one by one
    while (generated_count < steps) {
      // Forward the transformer to get logits for the next token
      int current = next;

      #pragma omp single
      {
        start = time_in_ms();
      }

      transformer_driver(transformer, 1, &current, past, 1);

      #pragma omp single
      {
        end = time_in_ms();
        decode_time += (end - start) / 1000.0;

        next = sampler_sample(sampler, transformer->state.logits);
        generated_count++;
        past++;

        if (next != TOKEN_EOS) {
          // Print the token as string, decode it with the tokenizer_t object
          char* piece = tokenizer_decode(tokenizer, next);
          safe_printf(piece);
          fflush(stdout);
        }
      }

      // Data-dependent terminating condition: the EOS token delimits sequences
      if (next == TOKEN_EOS) {
        break;
      }
    }
  }
  printf("\n");

  // Report achieved tok/s
  if (past > 2) {
    fprintf(
        stderr,
        "Prompt processing (prefill): %d tokens in %6.3f s (%f tok/s)\n",
        sequence_len,
        prefill_time,
        sequence_len / prefill_time
    );
    fprintf(
        stderr,
        "Token generation  (decode):  %zu tokens in %6.3f s (%f tok/s)\n",
        (generated_count - 1),
        decode_time,
        (generated_count - 1) / decode_time
    );
  }

  free(sequence);
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage(char* argv[]) {
  fprintf(stderr, "Usage: %s [options]\n", argv[0]);
  fprintf(stderr, "Example: %s -m model.bin -n 32 -p \"42 is the\"\n", argv[0]);
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -f <path>       read prompt from file\n");
  fprintf(stderr, "  -h, --help      print usage and exit\n");
  fprintf(stderr, "  --impl <impl>   implementation of the transformer:\n");
  fprintf(stderr, "                  \"basic\" or \"inlined\" (default)\n");
  fprintf(stderr, "  -m <path>       model file (default model.bin)\n");
  fprintf(stderr, "  -n <int>        num of tokens to predict (default 256)\n");
  fprintf(stderr, "  -p <string>     input prompt\n");
  fprintf(stderr, "  -s <int>        random seed (default time(NULL))\n");
  fprintf(stderr, "  -t <int>        set the number of threads (default 8)\n");
  fprintf(stderr, "  --temp <float>  temperature in [0, inf] (default 1.0)\n");
  fprintf(stderr, "  --top-p <float> top-p sampling in [0, 1] (default 0.9)\n");
  fprintf(stderr, "  -z <path>       tokenizer file (default tokenizer.bin)\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {
  // Default parameters
  // Path to the model file, e.g. out/model.bin
  char* checkpoint_path = "model.bin";
  char* tokenizer_path = "tokenizer.bin";
  // Temperature: 0.0 = greedy deterministic. 1.0 = original. don't set higher
  float temperature = 1.0f;
  // Top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
  float topp = 0.9f;
  // Number of steps (number of predicted tokens) to run for
  int steps = 256;
  // Prompt string or file
  char* prompt = "Once upon a time, ";
  char* prompt_file = NULL;
  // Seed rng with time by default
  unsigned long long rng_seed = 0;
  // Set the number of threads to use for OpenMP
  int num_threads = 8;
  // Implementation of the transformer, either basic or inlined
  implementation_t impl = IMPLEMENTATION_INLINED;

  // Poor man's C argparse
  for (int i = 1; i < argc; i += 2) {
    if (strcmp(argv[i], "-f") == 0) {
      if (i + 1 < argc) {
        prompt_file = argv[i + 1];
      } else {
        error_usage(argv);
      }
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      error_usage(argv);
    } else if (strcmp(argv[i], "--impl") == 0) {
      if (i + 1 < argc) {
        if (strcmp(argv[i + 1], "basic") == 0) {
          impl = IMPLEMENTATION_BASIC;
        } else if (strcmp(argv[i + 1], "inlined") == 0) {
          impl = IMPLEMENTATION_INLINED;
        } else {
          fprintf(stderr, "Unknown implementation: %s\n", argv[i + 1]);
          exit(EXIT_FAILURE);
        }
      } else {
        error_usage(argv);
      }
    } else if (strcmp(argv[i], "-m") == 0) {
      if (i + 1 < argc) {
        checkpoint_path = argv[i + 1];
      } else {
        error_usage(argv);
      }
    } else if (strcmp(argv[i], "-n") == 0) {
      if (i + 1 < argc) {
        steps = atoi(argv[i + 1]);
      } else {
        error_usage(argv);
      }
    } else if (strcmp(argv[i], "-p") == 0) {
      if (i + 1 < argc) {
        prompt = argv[i + 1];
      } else {
        error_usage(argv);
      }
    } else if (strcmp(argv[i], "-s") == 0) {
      if (i + 1 < argc) {
        rng_seed = strtoull(argv[i + 1], NULL, 10);
      } else {
        error_usage(argv);
      }
    } else if (strcmp(argv[i], "-t") == 0) {
      if (i + 1 < argc) {
        int thread_count = atoi(argv[i + 1]);
        if (thread_count <= 0) {
          fprintf(stderr, "Invalid number of threads: %d\n", thread_count);
          exit(EXIT_FAILURE);
        }
        num_threads = thread_count;
      } else {
        error_usage(argv);
      }
    } else if (strcmp(argv[i], "--temp") == 0) {
      if (i + 1 < argc) {
        temperature = atof(argv[i + 1]);
      } else {
        error_usage(argv);
      }
    } else if (strcmp(argv[i], "--top-p") == 0) {
      if (i + 1 < argc) {
        topp = atof(argv[i + 1]);
      } else {
        error_usage(argv);
      }
    } else if (strcmp(argv[i], "-z") == 0) {
      if (i + 1 < argc) {
        tokenizer_path = argv[i + 1];
      } else {
        error_usage(argv);
      }
    } else {
      error_usage(argv);
    }
  }

  // Parameter validation/overrides
  if (rng_seed <= 0) {
    rng_seed = (unsigned int)time(NULL);
  }
  if (temperature < 0.0) {
    temperature = 0.0;
  }
  if (topp < 0.0 || 1.0 < topp) {
    topp = 0.9;
  }
  if (steps < 0) {
    steps = 0;
  }
  if (prompt_file != NULL) {
    FILE* pf = fopen(prompt_file, "rb");
    if (!pf) {
      fprintf(stderr, "Could not open prompt file: %s\n", prompt_file);
      exit(EXIT_FAILURE);
    }
    fseek(pf, 0, SEEK_END);
    long fsize = ftell(pf);
    fseek(pf, 0, SEEK_SET);
    char* file_prompt = malloc(fsize + 1);
    if (!file_prompt) {
      fprintf(stderr, "Failed to allocate memory for prompt file\n");
      exit(EXIT_FAILURE);
    }
    fread(file_prompt, 1, fsize, pf);
    fclose(pf);
    file_prompt[fsize] = '\0';
    // Strip trailing newline if present
    if (fsize > 0 && file_prompt[fsize - 1] == '\n') {
      file_prompt[fsize - 1] = '\0';
    }
    prompt = file_prompt;
  }

  // Set OpenMP parameters
  setenv("OMP_WAIT_POLICY", "ACTIVE", 1);
  omp_set_num_threads(num_threads);

  // Build the transformer_t via the model .bin file
  transformer_t transformer;
  transformer_build(&transformer, checkpoint_path);
  transformer.impl = impl;

  // Print configuration_t data
  fprintf(stderr, "transformer_t configuration:\n");
  fprintf(stderr, "- embedding_dim:  %d\n", transformer.config.embedding_dim);
  fprintf(stderr, "- hidden_dim:     %d\n", transformer.config.hidden_dim);
  fprintf(stderr, "- layer_count:    %d\n", transformer.config.layer_count);
  fprintf(stderr, "- q_head_count:   %d\n", transformer.config.q_head_count);
  fprintf(stderr, "- kv_head_count:  %d\n", transformer.config.kv_head_count);
  fprintf(stderr, "- vocabulary_len: %d\n", transformer.config.vocabulary_len);
  fprintf(stderr, "- context_len:    %d\n", transformer.config.context_len);
  fprintf(stderr, "runtime setup:\n");
  fprintf(stderr, "- seed:           %llu\n", rng_seed);
  fprintf(stderr, "- num_threads:    %d\n", num_threads);
  fprintf(
      stderr,
      "- implementation: %s\n",
      (transformer.impl == IMPLEMENTATION_BASIC) ? "basic" : "inlined"
  );
  int vocabulary_len = transformer.config.vocabulary_len; 

  // Build the tokenizer_t via the tokenizer .bin file
  tokenizer_t tokenizer;
  tokenizer_build(&tokenizer, tokenizer_path, vocabulary_len);

  // Build the sampler_t
  sampler_t sampler;
  sampler_build(&sampler, vocabulary_len, temperature, topp, rng_seed);

  // Run! Override to ~max length if necessary
  if (steps == 0 || steps > transformer.config.context_len) {
    steps = transformer.config.context_len;
  }
  generate(&transformer, &tokenizer, &sampler, prompt, steps);

  // Memory and file handles cleanup
  sampler_free(&sampler);
  tokenizer_free(&tokenizer);
  transformer_free(&transformer);
  if (prompt_file != NULL) {
    free(prompt);
  }
  return EXIT_SUCCESS;
}
#endif
