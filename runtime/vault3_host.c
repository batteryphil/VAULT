/* SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
 * Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
 * Copyright (c) 2026 Philip Bartsch (batteryphil)
 *
 * vault3_host.c — HOSTED test/benchmark/CLI harness for the freestanding VAULT v3 runtime.
 * Uses POSIX mmap + pthreads (+ Linux futex adaptive barrier) on the host while keeping
 * runtime/vault3.c 100% freestanding.
 *
 *   vault3_host model.vlt3 --prompt "Bonjour" [--chat] [--system "..."] [--n 128]
 *                          [--temp 0] [--topk 40] [--rep-penalty 1.1] [--threads 12]
 *                          [--seg-budget 1024] [--noseg] [--novault] [--pin N]
 *   vault3_host model.vlt3 --tokenize-file in.txt [--ordinary]
 *   vault3_host model.vlt3 --ids-file ids.txt --dump out.bin
 *   vault3_host model.vlt3 [--bench-prefill 256] --bench 128 [--threads 12]
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#ifdef __linux__
#include <linux/futex.h>
#include <sys/syscall.h>
#endif

#include "vault3.h"

/* ============================================================================
 * Adaptive spin + futex worker pool implementing v3_parallel_for_fn
 * ========================================================================== */
#define MAX_WORKERS 64

typedef struct {
    int n_threads;           /* total threads including main thread */
    uint32_t spin_limit;     /* adaptive spin budget (4096 <= phys cores, 32 on SMT/oversubscription) */
    pthread_t th[MAX_WORKERS];
    v3_task_fn fn;
    void *arg;
    uint32_t n, grain;
    atomic_uint gen;
    atomic_uint next;
    atomic_uint done;
    atomic_uint sleeping;
    atomic_uint main_waiting;
    atomic_int stop;
} thread_pool;

typedef struct { thread_pool *p; int id; } worker_arg;

static inline void cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

static inline void futex_wait_u32(atomic_uint *addr, uint32_t expected) {
#ifdef __linux__
    syscall(SYS_futex, (uint32_t *)addr, FUTEX_WAIT_PRIVATE, expected, NULL, NULL, 0);
#else
    (void)addr; (void)expected;
    struct timespec ts = {0, 10000};
    nanosleep(&ts, NULL);
#endif
}

static inline void futex_wake_u32(atomic_uint *addr, int count) {
#ifdef __linux__
    syscall(SYS_futex, (uint32_t *)addr, FUTEX_WAKE_PRIVATE, count, NULL, NULL, 0);
#else
    (void)addr; (void)count;
#endif
}

static void run_work(thread_pool *p) {
    uint32_t g = p->grain, n = p->n;
    for (;;) {
        uint32_t i0 = atomic_fetch_add_explicit(&p->next, g, memory_order_relaxed);
        if (i0 >= n) break;
        uint32_t i1 = i0 + g < n ? i0 + g : n;
        p->fn(p->arg, i0, i1);
    }
}

static void *worker_main(void *raw) {
    worker_arg wa = *(worker_arg *)raw;
    free(raw);
    thread_pool *p = wa.p;
    uint32_t seen = 0;
    uint32_t want = (uint32_t)(p->n_threads - 1);
    uint32_t slim = p->spin_limit;
    for (;;) {
        uint32_t spins = 0, g;
        while ((g = atomic_load_explicit(&p->gen, memory_order_acquire)) == seen) {
            if (atomic_load_explicit(&p->stop, memory_order_acquire)) return NULL;
            if (spins < slim) {
                spins++;
                cpu_relax();
            } else {
                atomic_fetch_add_explicit(&p->sleeping, 1u, memory_order_seq_cst);
                if (atomic_load_explicit(&p->gen, memory_order_seq_cst) == seen &&
                    !atomic_load_explicit(&p->stop, memory_order_seq_cst)) {
                    futex_wait_u32(&p->gen, seen);
                }
                atomic_fetch_sub_explicit(&p->sleeping, 1u, memory_order_relaxed);
            }
        }
        if (atomic_load_explicit(&p->stop, memory_order_acquire)) return NULL;
        seen = g;
        run_work(p);
        uint32_t prev = atomic_fetch_add_explicit(&p->done, 1u, memory_order_seq_cst);
        if (prev + 1u == want && atomic_load_explicit(&p->main_waiting, memory_order_seq_cst))
            futex_wake_u32(&p->done, 1);
    }
}

static void pool_parallel_for(void *ctx, v3_task_fn fn, void *arg, uint32_t n) {
    thread_pool *p = (thread_pool *)ctx;
    if (p->n_threads <= 1 || n == 0) { fn(arg, 0, n); return; }
    uint32_t g = n / ((uint32_t)p->n_threads * 4u);
    uint32_t min_g = (n >= 256u) ? 16u : 1u;
    if (g < min_g) g = min_g;
    p->fn = fn;
    p->arg = arg;
    p->n = n;
    p->grain = g;
    atomic_store_explicit(&p->next, 0u, memory_order_relaxed);
    atomic_store_explicit(&p->done, 0u, memory_order_relaxed);
    atomic_fetch_add_explicit(&p->gen, 1u, memory_order_seq_cst);
    if (atomic_load_explicit(&p->sleeping, memory_order_seq_cst) > 0u)
        futex_wake_u32(&p->gen, INT_MAX);

    run_work(p);

    uint32_t want = (uint32_t)(p->n_threads - 1), wspins = 0, slim = p->spin_limit, dcur;
    while ((dcur = atomic_load_explicit(&p->done, memory_order_acquire)) < want) {
        if (wspins < slim) {
            wspins++;
            cpu_relax();
        } else {
            atomic_store_explicit(&p->main_waiting, 1u, memory_order_seq_cst);
            dcur = atomic_load_explicit(&p->done, memory_order_seq_cst);
            if (dcur < want) futex_wait_u32(&p->done, dcur);
            atomic_store_explicit(&p->main_waiting, 0u, memory_order_relaxed);
        }
    }
}

static void pool_init(thread_pool *p, int n_threads) {
    memset(p, 0, sizeof(*p));
    if (n_threads > MAX_WORKERS) n_threads = MAX_WORKERS;
    if (n_threads < 1) n_threads = 1;
    long onln = sysconf(_SC_NPROCESSORS_ONLN);
    long phys = (onln >= 4) ? (onln / 2) : (onln > 0 ? onln : 1);
    p->n_threads = n_threads;
    p->spin_limit = (n_threads <= (int)phys) ? 4096u : 32u;
    for (int i = 1; i < n_threads; i++) {
        worker_arg *wa = malloc(sizeof(*wa));
        if (!wa) { fprintf(stderr, "oom creating worker\n"); exit(1); }
        wa->p = p; wa->id = i;
        if (pthread_create(&p->th[i], NULL, worker_main, wa) != 0) {
            fprintf(stderr, "pthread_create failed\n");
            exit(1);
        }
    }
}

static void pool_destroy(thread_pool *p) {
    atomic_store_explicit(&p->stop, 1, memory_order_seq_cst);
    atomic_fetch_add_explicit(&p->gen, 1u, memory_order_seq_cst);
    futex_wake_u32(&p->gen, INT_MAX);
    for (int i = 1; i < p->n_threads; i++) pthread_join(p->th[i], NULL);
}

static int default_threads(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n <= 1) return 1;
    int t = (n >= 4) ? (int)(n / 2) : (int)n;
    return t > 16 ? 16 : t;
}

/* ============================================================================
 * Helpers
 * ========================================================================== */
static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double peak_rss_mb(void) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0.0;
    return ru.ru_maxrss / 1024.0;
}

static char *read_text(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    char *b = malloc((size_t)n + 1u);
    if (!b) { fclose(f); return NULL; }
    if (n > 0 && fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    b[n] = 0;
    fclose(f);
    *len = (size_t)n;
    return b;
}

typedef struct {
    uint8_t rem[4];
    uint32_t len;
} utf8_stream;

static inline uint32_t utf8_expected_len(uint8_t b0) {
    if (b0 < 0x80u) return 1u;
    if ((b0 & 0xE0u) == 0xC0u) return (b0 >= 0xC2u) ? 2u : 1u;
    if ((b0 & 0xF0u) == 0xE0u) return 3u;
    if ((b0 & 0xF8u) == 0xF0u) return (b0 <= 0xF4u) ? 4u : 1u;
    return 1u;
}

static void print_token_utf8(utf8_stream *u, const v3_model *m, uint32_t id) {
    const uint8_t *b; uint32_t l;
    if (v3_token_bytes(m, id, &b, &l) != V3_OK || l == 0) return;
    uint32_t pos = 0;
    if (u->len > 0) {
        uint32_t need = utf8_expected_len(u->rem[0]);
        while (u->len < need && pos < l) {
            uint8_t c = b[pos];
            if ((c & 0xC0u) != 0x80u) {
                fwrite(u->rem, 1, u->len, stdout);
                u->len = 0;
                break;
            }
            u->rem[u->len++] = c;
            pos++;
        }
        if (u->len > 0 && u->len == need) {
            fwrite(u->rem, 1, u->len, stdout);
            u->len = 0;
        }
        if (pos >= l) return;
    }
    uint32_t tail = l;
    for (uint32_t back = 1; back <= 3 && back <= l - pos; back++) {
        uint8_t c = b[l - back];
        if ((c & 0xC0u) != 0x80u) {
            uint32_t need = utf8_expected_len(c);
            if (need > back) tail = l - back;
            break;
        }
    }
    if (tail > pos) fwrite(b + pos, 1, tail - pos, stdout);
    if (tail < l) {
        u->len = l - tail;
        memcpy(u->rem, b + tail, u->len);
    }
}

static void flush_utf8_stream(utf8_stream *u) {
    if (u->len > 0) {
        fwrite(u->rem, 1, u->len, stdout);
        u->len = 0;
    }
}

static void apply_rep_penalty(float *dst, const float *src, uint32_t vocab, const uint32_t *hist, size_t n_hist, float pen) {
    memcpy(dst, src, vocab * sizeof(float));
    if (pen <= 1.0001f || n_hist == 0) return;
    size_t start = n_hist > 128 ? n_hist - 128 : 0;
    for (size_t i = start; i < n_hist; i++) {
        uint32_t id = hist[i];
        if (id < vocab) dst[id] = (src[id] > 0.0f) ? (src[id] / pen) : (src[id] * pen);
    }
}

typedef struct {
    uint32_t *data;
    size_t n, cap;
} id_vec;

static void id_vec_push(id_vec *v, uint32_t id) {
    if (v->n >= v->cap) {
        size_t ncap = v->cap ? v->cap * 2u : 256u;
        uint32_t *nd = realloc(v->data, ncap * sizeof(uint32_t));
        if (!nd) { fprintf(stderr, "oom growing token buffer\n"); exit(1); }
        v->data = nd;
        v->cap = ncap;
    }
    v->data[v->n++] = id;
}

static int id_vec_encode(id_vec *v, v3_tokenizer *tk, const char *s, size_t len, int allow_special) {
    size_t need = len + 32u;
    for (;;) {
        if (v->n + need > v->cap) {
            size_t ncap = v->cap ? v->cap : 256u;
            while (ncap < v->n + need) ncap *= 2u;
            uint32_t *nd = realloc(v->data, ncap * sizeof(uint32_t));
            if (!nd) { fprintf(stderr, "oom growing encode buffer\n"); exit(1); }
            v->data = nd;
            v->cap = ncap;
        }
        size_t got = 0;
        int rc = v3_encode_ex(tk, (const uint8_t *)s, len, v->data + v->n, v->cap - v->n, &got, allow_special);
        if (rc == V3_OK) { v->n += got; return V3_OK; }
        if (rc != V3_E_SIZE) return rc;
        need *= 2u;
    }
}

static int parse_u32_line(FILE *f, id_vec *out) {
    int c, have = 0;
    uint64_t v = 0;
    while ((c = fgetc(f)) != EOF && c != '\n') {
        if (c >= '0' && c <= '9') {
            v = v * 10ull + (uint64_t)(c - '0');
            if (v > 0xFFFFFFFFull) return -1;
            have = 1;
        } else if (have) {
            id_vec_push(out, (uint32_t)v);
            v = 0;
            have = 0;
        }
    }
    if (have) id_vec_push(out, (uint32_t)v);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.vlt3 [options]\n", argv[0]); return 2; }
    const char *prompt = NULL, *prompt_file = NULL, *sys_prompt = "You are a helpful assistant.";
    const char *tok_file = NULL, *ids_file = NULL, *dump = NULL;
    const char *forbid_pats[V3_GOV_MAX_RULES];
    int n_forbid = 0, no_gov = 0;
    int n_gen = 64, chat = 0, novault = 0, noseg = 0, ordinary = 0;
    int seg_override = -1, pin_override = -1, no_autopin = 0;
    int bench = 0, bench_pf = 0, verify = 0, n_threads = default_threads();
    float temp = 0.0f, rep_pen = 1.0f;
    unsigned topk = 40;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--prompt-file") && i + 1 < argc) prompt_file = argv[++i];
        else if (!strcmp(argv[i], "--system") && i + 1 < argc) sys_prompt = argv[++i];
        else if (!strcmp(argv[i], "--chat")) {
            chat = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-') prompt = argv[++i];
        }
        else if (!strcmp(argv[i], "--n") && i + 1 < argc) n_gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp") && i + 1 < argc) temp = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--topk") && i + 1 < argc) { int k = atoi(argv[++i]); topk = k > 0 ? (unsigned)k : 1u; }
        else if (!strcmp(argv[i], "--rep-penalty") && i + 1 < argc) rep_pen = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--novault")) novault = 1;
        else if (!strcmp(argv[i], "--noseg")) noseg = 1;
        else if (!strcmp(argv[i], "--seg-budget") && i + 1 < argc) seg_override = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pin") && i + 1 < argc) pin_override = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-autopin")) no_autopin = 1;
        else if (!strcmp(argv[i], "--no-governor")) no_gov = 1;
        else if (!strcmp(argv[i], "--forbid") && i + 1 < argc) {
            const char *fp = argv[++i];
            if (n_forbid < V3_GOV_MAX_RULES) forbid_pats[n_forbid++] = fp;
        }
        else if (!strcmp(argv[i], "--ordinary")) ordinary = 1;
        else if (!strcmp(argv[i], "--verify")) verify = 1;
        else if (!strcmp(argv[i], "--tokenize-file") && i + 1 < argc) tok_file = argv[++i];
        else if (!strcmp(argv[i], "--ids-file") && i + 1 < argc) ids_file = argv[++i];
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump = argv[++i];
        else if (!strcmp(argv[i], "--bench-prefill") && i + 1 < argc) bench_pf = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bench") && i + 1 < argc) bench = atoi(argv[++i]);
    }
    if (n_gen < 0) n_gen = 0;
    if (bench < 0) bench = 0;
    if (bench_pf < 0) bench_pf = 0;

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { fprintf(stderr, "empty/stat error\n"); close(fd); return 1; }
    const uint8_t *buf = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (buf == MAP_FAILED) { perror("mmap"); close(fd); return 1; }
#ifdef MADV_SEQUENTIAL
    (void)madvise((void *)buf, (size_t)st.st_size, MADV_SEQUENTIAL);
#endif

    v3_model m;
    double t0 = now_s();
    int rc = v3_load(&m, buf, (uint64_t)st.st_size, verify);
    if (rc != V3_OK) { fprintf(stderr, "v3_load failed: %d\n", rc); return 2; }
    fprintf(stderr, "[load] %.1f MB, %u layers, d=%u, heads=%u/%u, window=%u, sinks=%u, seg_budget=%u, cells=%u, vocab=%u, fmt=%u (%.2fs%s)\n",
            st.st_size / 1e6, m.n_layers, m.d_model, m.n_heads, m.n_kv_heads, m.window, m.n_sink, m.seg_budget,
            m.n_cells, m.vocab_size, m.weight_format, now_s() - t0, verify ? ", CRC+NaN verified" : ", NaN verified");

    size_t tok_sz = v3_tok_arena_size(&m);
    if (tok_sz == 0) { fprintf(stderr, "invalid tok arena size\n"); return 1; }
    void *tok_arena = aligned_alloc(64, (tok_sz + 63u) / 64u * 64u);
    if (!tok_arena) { fprintf(stderr, "oom tok_arena\n"); return 1; }
    v3_tokenizer tk;
    if ((rc = v3_tok_init(&tk, &m, tok_arena, tok_sz)) != V3_OK) { fprintf(stderr, "tok init %d\n", rc); return 1; }

    if (tok_file) {
        size_t len = 0;
        char *txt = read_text(tok_file, &len);
        if (!txt) { fprintf(stderr, "cannot read %s\n", tok_file); free(tok_arena); return 1; }
        id_vec ids = {0, 0, 0};
        rc = id_vec_encode(&ids, &tk, txt, len, !ordinary);
        free(txt);
        if (rc) { fprintf(stderr, "encode %d\n", rc); free(ids.data); free(tok_arena); return 1; }
        for (size_t i = 0; i < ids.n; i++) printf("%u%c", ids.data[i], i + 1 < ids.n ? ' ' : '\n');
        free(ids.data);
        free(tok_arena);
        munmap((void *)buf, (size_t)st.st_size);
        close(fd);
        return 0;
    }

    uint32_t active_seg_budget = m.seg_budget;
    if (noseg || m.seg_n_feat == 0) active_seg_budget = 0;
    else if (seg_override >= 0) {
        if ((uint32_t)seg_override > V3_MAX_SEG_BUDGET) {
            fprintf(stderr, "seg-budget %d exceeds max %u\n", seg_override, (unsigned)V3_MAX_SEG_BUDGET);
            free(tok_arena);
            return 2;
        }
        active_seg_budget = (uint32_t)seg_override;
    }

    size_t st_sz = v3_state_arena_size_ex(&m, active_seg_budget);
    if (st_sz == 0) { fprintf(stderr, "invalid state arena size\n"); free(tok_arena); return 1; }
    void *arena = aligned_alloc(64, (st_sz + 63u) / 64u * 64u);
    if (!arena) { fprintf(stderr, "oom state arena\n"); free(tok_arena); return 1; }
    v3_state s;
    if ((rc = v3_state_init_ex(&s, &m, arena, st_sz, active_seg_budget)) != V3_OK) {
        fprintf(stderr, "state init %d\n", rc);
        free(arena); free(tok_arena);
        return 1;
    }
    s.vault_on = !novault;
    if (noseg) v3_set_seg_budget(&s, 0u);
    else if (seg_override >= 0 && m.seg_n_feat > 0) v3_set_seg_budget(&s, active_seg_budget);
    if (pin_override >= 0) {
        if (v3_set_pinned(&s, (uint32_t)pin_override) != V3_OK) {
            fprintf(stderr, "invalid --pin %d (max %u)\n", pin_override, s.sink_cap);
            free(arena); free(tok_arena);
            return 2;
        }
    } else if (no_autopin) {
        v3_set_pinned(&s, m.n_sink);
    }

    thread_pool pool;
    pool_init(&pool, n_threads);
    if (pool.n_threads > 1) { s.par_for = pool_parallel_for; s.par_ctx = &pool; }
    fprintf(stderr, "[mem] state arena %.1f MB (fixed, seg_budget=%u, sink_cap=%u), tokenizer arena %.1f MB | threads=%d\n",
            st_sz / 1e6, s.seg_budget, s.sink_cap, tok_sz / 1e6, pool.n_threads);

    if (ids_file) {
        if (!dump) { fprintf(stderr, "--ids-file requires --dump <out.bin>\n"); pool_destroy(&pool); free(arena); free(tok_arena); return 2; }
        FILE *f = fopen(ids_file, "r");
        if (!f) { perror("ids-file"); pool_destroy(&pool); free(arena); free(tok_arena); return 1; }
        id_vec ids = {0, 0, 0}, dpos = {0, 0, 0};
        if (parse_u32_line(f, &ids) != 0 || parse_u32_line(f, &dpos) != 0) {
            fprintf(stderr, "malformed integer in %s\n", ids_file);
            fclose(f); free(ids.data); free(dpos.data); pool_destroy(&pool); free(arena); free(tok_arena); return 2;
        }
        fclose(f);
        FILE *o = fopen(dump, "wb");
        if (!o) { perror("dump"); free(ids.data); free(dpos.data); pool_destroy(&pool); free(arena); free(tok_arena); return 1; }
        size_t k = 0;
        double ts = now_s();
        for (size_t i = 0; i < ids.n; i++) {
            const float *lg = v3_step(&s, ids.data[i]);
            if (k < dpos.n && dpos.data[k] == i) { fwrite(lg, sizeof(float), m.vocab_size, o); k++; }
        }
        fclose(o);
        double dt = now_s() - ts;
        fprintf(stderr, "[dump] %zu tokens, %zu positions dumped, %.2f tok/s (n_pin=%u, mem_cnt=%u)\n",
                ids.n, k, dt > 0 ? ids.n / dt : 0.0, s.n_pin, s.mem_cnt);
        free(ids.data); free(dpos.data);
        pool_destroy(&pool);
        free(arena);
        free(tok_arena);
        munmap((void *)buf, (size_t)st.st_size);
        close(fd);
        return 0;
    }

    if (bench > 0) {
        /* Touch first token once so OS mmap page-ins don't skew steady-state decode timing. */
        v3_step(&s, 100u);
        v3_state_reset(&s);
        if (pin_override >= 0) v3_set_pinned(&s, (uint32_t)pin_override);
        else if (no_autopin) v3_set_pinned(&s, m.n_sink);
        double t_pf = 0.0;
        if (bench_pf > 0) {
            double tp0 = now_s();
            for (int i = 0; i < bench_pf; i++) v3_step(&s, (uint32_t)(500 + i * 13) % m.vocab_size);
            t_pf = now_s() - tp0;
        }
        double ts = now_s();
        for (int i = 0; i < bench; i++) v3_step(&s, (uint32_t)(1000 + i * 7) % m.vocab_size);
        double dt = now_s() - ts;
        printf("{\"threads\":%d,\"prefill_tokens\":%d,\"prefill_s\":%.4f,\"prefill_tok_s\":%.2f,"
               "\"decode_tokens\":%d,\"decode_s\":%.4f,\"decode_tok_s\":%.2f,\"ms_per_tok\":%.2f,"
               "\"seg_budget\":%u,\"mem_cnt\":%u,\"state_arena_mb\":%.2f,\"tok_arena_mb\":%.2f,"
               "\"file_mb\":%.2f,\"peak_rss_mb\":%.2f}\n",
               pool.n_threads, bench_pf, t_pf, bench_pf > 0 ? bench_pf / (t_pf > 0 ? t_pf : 1e-9) : 0.0,
               bench, dt, bench / (dt > 0 ? dt : 1e-9), 1000.0 * dt / bench,
               s.seg_budget, s.mem_cnt, st_sz / 1e6, tok_sz / 1e6, st.st_size / 1e6, peak_rss_mb());
        pool_destroy(&pool);
        free(arena);
        free(tok_arena);
        munmap((void *)buf, (size_t)st.st_size);
        close(fd);
        return 0;
    }

    char *pbuf = NULL;
    size_t prompt_len = 0;
    if (prompt_file) {
        pbuf = read_text(prompt_file, &prompt_len);
        if (!pbuf) { fprintf(stderr, "cannot read %s\n", prompt_file); pool_destroy(&pool); free(arena); free(tok_arena); return 1; }
        prompt = pbuf;
    } else if (prompt) {
        prompt_len = strlen(prompt);
    } else {
        prompt = "Bonjour ! Peux-tu te présenter en deux phrases ?";
        prompt_len = strlen(prompt);
    }

    v3_governor gov;
    v3_gov_init(&gov);
    if (!no_gov) {
        if (chat && sys_prompt) v3_gov_arm_system(&gov, (const uint8_t *)sys_prompt, strlen(sys_prompt));
        for (int k = 0; k < n_forbid; k++)
            v3_gov_add_rule(&gov, (const uint8_t *)forbid_pats[k], strlen(forbid_pats[k]));
        v3_gov_reset_stream(&gov);
    }

    id_vec ids = {0, 0, 0};
    if (chat) {
        /* Delimiter-safe ChatML encoding: untrusted system/user text is encoded with allow_special=0
         * so embedded <|im_start|>/<|im_end|> strings cannot hijack role turns. */
        id_vec_push(&ids, m.im_start_id);
        id_vec_encode(&ids, &tk, "system\n", 7, 0);
        id_vec_encode(&ids, &tk, sys_prompt, strlen(sys_prompt), 0);
        id_vec_push(&ids, m.im_end_id);
        id_vec_encode(&ids, &tk, "\n", 1, 0);

        id_vec_push(&ids, m.im_start_id);
        id_vec_encode(&ids, &tk, "user\n", 5, 0);
        id_vec_encode(&ids, &tk, prompt, prompt_len, 0);
        id_vec_push(&ids, m.im_end_id);
        id_vec_encode(&ids, &tk, "\n", 1, 0);

        id_vec_push(&ids, m.im_start_id);
        id_vec_encode(&ids, &tk, "assistant\n", 10, 0);
    } else {
        id_vec_encode(&ids, &tk, prompt, prompt_len, 1);
    }

    size_t n = ids.n;
    double ts = now_s();
    const float *lg = NULL;
    for (size_t i = 0; i < n; i++) lg = v3_step(&s, ids.data[i]);
    if (!lg && n_gen > 0) {
        id_vec_push(&ids, m.bos_id);
        n = ids.n;
        lg = v3_step(&s, m.bos_id);
    }
    double t_pf = now_s() - ts;
    if (!chat && prompt_len > 0) fwrite(prompt, 1, prompt_len, stdout);

    uint64_t rng = 0x1234567ull;
    float *lbuf = (rep_pen > 1.0001f || gov.active) ? malloc(m.vocab_size * sizeof(float)) : NULL;
    uint32_t gov_guard = (temp <= 0.0f || topk < 16u) ? 16u : (topk > 64u ? 64u : topk);
    utf8_stream utf8 = {{0, 0, 0, 0}, 0u};
    ts = now_s();
    int produced = 0;
    for (int i = 0; i < n_gen && lg != NULL; i++) {
        const float *use_lg = lg;
        if (lbuf) {
            apply_rep_penalty(lbuf, lg, m.vocab_size, ids.data, ids.n, rep_pen);
            if (gov.active) v3_govern_logits(&gov, &m, lbuf, gov_guard);
            use_lg = lbuf;
        }
        uint32_t nx = v3_sample_topk(use_lg, m.vocab_size, topk, temp, &rng);
        if (nx == m.im_end_id || nx == m.bos_id) break;
        if (gov.active) v3_gov_commit(&gov, &m, nx);
        id_vec_push(&ids, nx);
        print_token_utf8(&utf8, &m, nx);
        fflush(stdout);
        produced++;
        lg = v3_step(&s, nx);
    }
    flush_utf8_stream(&utf8);
    double t_dec = now_s() - ts;
    printf("\n");
    fprintf(stderr, "[perf] prompt %zu tok (n_pin=%u, mem_cnt=%u, gov_rules=%u) in %.2fs (%.2f tok/s) | generated %d tok in %.2fs (%.2f tok/s, %.1f ms/tok)\n",
            n, s.n_pin, s.mem_cnt, gov.n_rules, t_pf, n / (t_pf > 0 ? t_pf : 1e-9),
            produced, t_dec, produced / (t_dec > 0 ? t_dec : 1e-9),
            produced > 0 ? 1000.0 * t_dec / produced : 0.0);
    free(lbuf);
    free(ids.data);
    free(pbuf);
    pool_destroy(&pool);
    free(arena);
    free(tok_arena);
    munmap((void *)buf, (size_t)st.st_size);
    close(fd);
    return 0;
}
