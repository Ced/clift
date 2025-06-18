#ifndef INSTRUMENTOR_H
#define INSTRUMENTOR_H

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

static inline double get_time() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return (t.tv_sec + t.tv_usec * 1e-6) * 1e3;
}

static double* tab_values = NULL;
static double* tab_compt_values = NULL;
static double** tab_values_2 = NULL;
static double*** tab_values_3 = NULL;

static int size_tab_values = 0;
static int size_tab_compt_values = 0;
static int size_tab_values_2_dim1 = 0;
static int size_tab_values_2_dim2 = 0;
static int size_tab_values_3_dim1 = 0;
static int size_tab_values_3_dim2 = 0;
static int size_tab_values_3_dim3 = 0;

#ifdef INSTR

static inline void let_tab_instr(int size) {
  size_tab_values = size;
  tab_values = calloc(size, sizeof(double));
}

static inline void let_tab_compt_instr(int size) {
  size_tab_compt_values = size;
  tab_compt_values = calloc(size, sizeof(double));
}

static inline void let_tab_instr_2(int size1, int size2) {
  size_tab_values_2_dim1 = size1;
  size_tab_values_2_dim2 = size2;
  tab_values_2 = calloc(size1, sizeof(double*));
  for (int i = 0; i < size1; ++i)
    tab_values_2[i] = calloc(size2, sizeof(double));
}

static inline void let_tab_instr_3(int size1, int size2, int size3) {
  size_tab_values_3_dim1 = size1;
  size_tab_values_3_dim2 = size2;
  size_tab_values_3_dim3 = size3;
  tab_values_3 = calloc(size1, sizeof(double**));
  for (int i = 0; i < size1; ++i) {
    tab_values_3[i] = calloc(size2, sizeof(double*));
    for (int j = 0; j < size2; ++j) {
      tab_values_3[i][j] = calloc(size3, sizeof(double));
    }
  }
}

static inline void free_instr_data() {
  free(tab_values);
  tab_values = NULL;
  free(tab_compt_values);
  tab_compt_values = NULL;

  if (tab_values_2) {
    for (int i = 0; i < size_tab_values_2_dim1; ++i) {
      free(tab_values_2[i]);
    }
    free(tab_values_2);
    tab_values_2 = NULL;
  }

  if (tab_values_3) {
    for (int i = 0; i < size_tab_values_3_dim1; ++i) {
      for (int j = 0; j < size_tab_values_3_dim2; ++j) {
        free(tab_values_3[i][j]);
      }
      free(tab_values_3[i]);
    }
    free(tab_values_3);
    tab_values_3 = NULL;
  }
}

#define LET_TAB(X)             X

#define START(st)              st = get_time()
#define STOP_1(st, index)      tab_values[index] += get_time() - st
#define ADD_1(compt, index)    tab_compt_values[index] += compt
#define STOP_2(st, i1, i2)     tab_values_2[i1][i2] += get_time() - st
#define STOP_3(st, i1, i2, i3) tab_values_3[i1][i2][i3] += get_time() - st

#define PRINT_JSON( \
    string_values_1_add, string_values_1, string_values_2, string_values_3 \
) \
  print_json( \
      string_values_1_add, string_values_1, string_values_2, string_values_3 \
  )

#else

#define LET_TAB(X)
#define START(st)
#define STOP_1(st, index)
#define ADD_1(compt, index)
#define STOP_2(st, i1, i2)
#define STOP_3(st, i1, i2, i3)
#define PRINT_JSON( \
    string_values_1_add, string_values_1, string_values_2, string_values_3 \
)

#endif

static void print_json_1_content(
    const char* json_name, char** tab_str, double* values, int size
) {
  printf("  \"%s\" : {\n", json_name);
  for (int i = 0; i < size; ++i) {
    printf(
        "    \"%s\": %.4f%s\n", tab_str[i], values[i], (i < size - 1 ? "," : "")
    );
  }
  printf("  }\n");
}

static void print_json_2_content(const char* json_name, char** tab_str) {
  printf("  \"%s\" : {\n", json_name);
  for (int i = 0; i < size_tab_values_2_dim1; ++i) {
    printf("    \"%s\": [", tab_str[i]);
    for (int j = 0; j < size_tab_values_2_dim2; ++j) {
      printf(
          "%.4f%s",
          tab_values_2[i][j],
          (j < size_tab_values_2_dim2 - 1 ? ", " : "")
      );
    }
    printf("]%s\n", (i < size_tab_values_2_dim1 - 1 ? "," : ""));
  }
  printf("  }\n");
}

static void print_json_3_content(const char* json_name, char** tab_str) {
  printf("  \"%s\" : {\n", json_name);
  for (int i = 0; i < size_tab_values_3_dim1; ++i) {
    printf("    \"%s\": [\n", tab_str[i]);
    for (int j = 0; j < size_tab_values_3_dim2; ++j) {
      printf("      [");
      for (int k = 0; k < size_tab_values_3_dim3; ++k) {
        printf(
            "%.4f%s",
            tab_values_3[i][j][k],
            (k < size_tab_values_3_dim3 - 1 ? ", " : "")
        );
      }
      printf("]%s\n", (j < size_tab_values_3_dim2 - 1 ? "," : ""));
    }
    printf("    ]%s\n", (i < size_tab_values_3_dim1 - 1 ? "," : ""));
  }
  printf("  }\n");
}

void print_json(char* string_values_1_add[],char* string_values_1[],char* string_values_2[],char* string_values_3[]) {
  printf("{\n");
  print_json_1_content(
      "JSON_1_COMPT", string_values_1, tab_compt_values, size_tab_compt_values
  );
  printf(",\n");
  print_json_1_content("JSON_1", string_values_1, tab_values, size_tab_values);
  printf(",\n");
  print_json_2_content("JSON_2", string_values_2);
  printf(",\n");
  print_json_3_content("JSON_3", string_values_3);
  printf("}\n");
}

int count_words(const char* str) {
  int count = 0;
  int in_word = 0;

  while (*str) {
    if (isspace((unsigned char)*str)) {
      in_word = 0;
    } else if (!in_word) {
      count++;
      in_word = 1;
    }
    str++;
  }

  return count;
}

#endif
