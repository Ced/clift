# choose your compiler, e.g. gcc/clang
# example override to clang: make clift CC=clang
CC = gcc

HOMEBREW_CLANG = /opt/homebrew/opt/llvm/bin/clang
HOMEBREW_FLAGS = -L/opt/homebrew/opt/llvm/lib -L/opt/homebrew/opt/libomp/lib

# the most basic way of building that is most likely to work on most systems
.PHONY: clift
clift: clift.c
	$(CC) -O3 -ffast-math -fno-omit-frame-pointer -Wall -o clift clift.c -lm

.PHONY: cliftinstr
cliftinstr: clift.c
	$(CC) -O3 -ffast-math -fno-omit-frame-pointer -Wall -o clift clift.c -lm -DINSTR
# Useful to debug memory issues, e.g. with Clang's address sanitizer.
.PHONY: asan
asan: clift.c
	$(CC) -fsanitize=address -g -O1 -o clift clift.c -lm

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./clift out/model.bin -n 3
debug: clift.c
	$(CC) -g -o clift clift.c -lm

# additionally compiles with OpenMP, allowing multithreaded clifts
# make sure to also enable multiple threads when cliftning, e.g.:
# OMP_NUM_THREADS=4 ./clift out/model.bin
.PHONY: omp
omp: clift.c
	$(CC) -O3 -ffast-math -fno-omit-frame-pointer -Wall -fopenmp -march=native clift.c  -lm  -o clift

.PHONY: ompinstr
ompinstr: clift.c
	$(CC) -O3 -DINSTR -ffast-math -fno-omit-frame-pointer -Wall -fopenmp -march=native clift.c  -lm  -o cliftinstr
# compiles with gnu99 standard flags for amazon linux, coreos, etc. compatibility
.PHONY: cliftgnu
gnu:
	$(CC) -Ofast -std=gnu11 -o clift clift.c -lm

.PHONY: cliftompgnu
ompgnu:
	$(CC) -Ofast -fopenmp -std=gnu11 clift.c  -lm  -o clift

.PHONY: clean
clean:
	rm -f clift
