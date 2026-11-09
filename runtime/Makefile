# VAULT v3 runtime build
#   make            -> hosted harness (vault3_host) + scalar & AVX2 freestanding object checks
#   make freestanding

CC ?= gcc
CFLAGS_FS_SCALAR = -std=c99 -O2 -ffreestanding -nostdlib -Wall -Wextra -Wframe-larger-than=2048
CFLAGS_FS_AVX2   = -std=c99 -O2 -ffreestanding -nostdlib -mavx2 -mfma -mf16c -Wall -Wextra -Wframe-larger-than=2048
CFLAGS_HOST      = -std=c11 -O3 -mavx2 -mfma -mf16c -fno-math-errno -pthread -Wall -Wextra

all: freestanding vault3_host

freestanding: vault3.c vault3.h vault3_nfc_tables.h
	$(CC) $(CFLAGS_FS_SCALAR) -DV3_FORCE_SCALAR_MATH -c vault3.c -o vault3_freestanding_scalar.o
	@if nm vault3_freestanding_scalar.o | grep -q " U "; then echo "undefined symbols in scalar:"; nm vault3_freestanding_scalar.o | grep " U "; exit 1; else echo "[ok] vault3_freestanding_scalar.o (pure scalar C99, no -fno-math-errno): 0 external undefined symbols"; fi
	$(CC) $(CFLAGS_FS_AVX2) -c vault3.c -o vault3_freestanding.o
	@if nm vault3_freestanding.o | grep -q " U "; then echo "undefined symbols in avx2:"; nm vault3_freestanding.o | grep " U "; exit 1; else echo "[ok] vault3_freestanding.o (AVX2+FMA, no -fno-math-errno): 0 external undefined symbols"; fi

vault3_host: vault3.c vault3.h vault3_nfc_tables.h vault3_host.c
	$(CC) $(CFLAGS_HOST) vault3.c vault3_host.c -o vault3_host

clean:
	rm -f vault3_host vault3_freestanding.o vault3_freestanding_scalar.o
