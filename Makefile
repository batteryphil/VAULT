CC ?= gcc
CFLAGS ?= -O3 -march=native -Wall -Wextra -std=c99
LDFLAGS ?= -lm
PYTHON ?= $(shell if [ -x ../venv/bin/python ]; then echo ../venv/bin/python; else echo python3; fi)

PRIME_MOMENT_SRC ?= ../PRIME-Moment-Attention/c/prime_moment.c

.PHONY: all bench torture test test-flash bench-flash sft-flash upgrade-flash deep2k-sft test-v2 data-v2 train-v2 eval-v2 clean

all: c/bench_vault c/torture_vault c/libvault.so

c/bench_vault: c/bench_vault.c c/vault.c c/vault.h $(PRIME_MOMENT_SRC)
	$(CC) $(CFLAGS) c/bench_vault.c c/vault.c $(PRIME_MOMENT_SRC) $(LDFLAGS) -o c/bench_vault

c/torture_vault: c/torture_vault.c c/vault.c c/vault.h
	$(CC) $(CFLAGS) c/torture_vault.c c/vault.c $(LDFLAGS) -o c/torture_vault

c/libvault.so: c/vault.c c/vault.h
	$(CC) $(CFLAGS) -fPIC -shared c/vault.c $(LDFLAGS) -o c/libvault.so

bench: c/bench_vault
	./c/bench_vault

torture: c/torture_vault
	./c/torture_vault
	$(PYTHON) benchmarks/torture_test_vault.py

test:
	$(PYTHON) tests/test_vault_rigorous.py

test-flash:
	$(PYTHON) tests/test_flash_vault_grad.py

bench-flash:
	$(PYTHON) benchmarks/bench_flash_vault.py

sft-flash:
	$(PYTHON) sft_vault_150m.py

upgrade-flash:
	$(PYTHON) run_flash_2stage_upgrade.py

deep2k-sft:
	$(PYTHON) run_flash_deep2k_and_sft.py

test-v2:
	$(PYTHON) -u tests/test_vault_v2.py

data-v2:
	$(PYTHON) -u prepare_vault_v2_data.py

train-v2:
	$(PYTHON) -u train_vault_v2.py

eval-v2:
	$(PYTHON) -u benchmarks/eval_vault_v2_standalone.py

clean:
	rm -f c/bench_vault c/torture_vault c/libvault.so
