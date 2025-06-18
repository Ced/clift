# choose your compiler, e.g. gcc/clang
# example override to clang: make clift CC=clang
CC = gcc

# the most basic way of building that is most likely to work on most systems
.PHONY: clift
clift: clift.c
	$(CC) -O3 -ffast-math -fno-omit-frame-pointer -Wall -o clift clift.c -lm

cliftinstr : clift.c
	$(CC) -DINSTR -O3 -ffast-math -fno-omit-frame-pointer -Wall -o cliftinstr clift.c -lm
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
.PHONY: cliftomp
omp: clift.c
	$(CC) -Ofast -fopenmp -march=native clift.c  -lm  -o cliftomp

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
	rm -f cliftinstr
