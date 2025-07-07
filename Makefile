# choose your compiler, e.g. gcc/clang
# example override to clang: make clift CC=clang
CC = gcc

HOMEBREW_CLANG = /opt/homebrew/opt/llvm/bin/clang
HOMEBREW_FLAGS = -L/opt/homebrew/opt/llvm/lib -L/opt/homebrew/opt/libomp/lib

# the most basic way of building that is most likely to work on most systems
.PHONY: clift
clift: clift.c
	$(CC) -O3 -ffast-math -fno-omit-frame-pointer -Wall -o clift clift.c -lm

# Useful to debug memory issues, e.g. with Clang's address sanitizer.
.PHONY: asan
asan: clift.c
	$(CC) -fsanitize=address -g -O1 -o clift clift.c -lm

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./clift out/model.bin -n 3
debug: clift.c
	$(CC) -g -o clift clift.c -lm

# additionally compiles with OpenMP, allowing multithreaded clift
.PHONY: brewmp
brewmp: clift.c
	$(HOMEBREW_CLANG) $(HOMEBREW_FLAGS) -O3 -ffast-math -fno-omit-frame-pointer -Wall -fopenmp -march=native clift.c -lm -o clift

# compiles with gnu99 standard flags for amazon linux, coreos, etc. compatibility
.PHONY: omp
omp: clift.c
	$(CC) -O3 -ffast-math -fno-omit-frame-pointer -Wall -fopenmp -march=native clift.c  -lm  -o clift

.PHONY: clean
clean:
	rm -f clift
