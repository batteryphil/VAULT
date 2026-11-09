/* SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
 * Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
 * Copyright (c) 2026 Philip Bartsch (batteryphil)
 *
 * vault3.h — freestanding C99 inference runtime for VAULT v3 (.vlt3 files).
 *
 *   - No libc, no malloc, no threads, no external floating-point library calls.
 *     Only <stdint.h> and <stddef.h> (both available in freestanding mode).
 *   - The model file is used in place (pointer + size). The buffer must be 64-byte aligned.
 *   - All mutable memory comes from caller-provided arenas whose sizes are known up front:
 *       v3_state_arena_size(model)  -> inference state (fixed; does NOT grow with context)
 *       v3_tok_arena_size(model)    -> tokenizer merge table + O(n log n) BPE heap scratch
 *   - Every header field, 64-bit shape product, offset, size, CRC32, and NaN/Inf bound is
 *     validated in v3_load().
 *   - Supports dynamic ChatML system-prompt pinning (up to V3_MAX_SINK tokens), Kenyon Vault
 *     associative memory, and fixed-budget Segment-Evicted KV Memory.
 */
#ifndef VAULT3_H
#define VAULT3_H

#include <stddef.h>
#include <stdint.h>

#define V3_MAX_LAYERS 64
#define V3_MAX_HEADS 256
#define V3_MAX_BANKS 8
#define V3_MAX_HEAD_DIM 256
#define V3_MAX_SINK 512
#define V3_MAX_SEG_FEAT 4
#define V3_MAX_SEG_HIDDEN 1024
#define V3_MAX_SEG_BUDGET 16384
#define V3_TOK_MAX_PIECE 32768

enum {
    V3_OK = 0,
    V3_E_SIZE = -1,      /* buffer too small / size mismatch / output token buffer full */
    V3_E_MAGIC = -2,     /* bad magic, version, header size or endianness tag */
    V3_E_CRC = -3,       /* header or tensor CRC mismatch */
    V3_E_BOUNDS = -4,    /* an offset or size points outside the file / region */
    V3_E_DTYPE = -5,     /* unsupported dtype for a tensor */
    V3_E_MISSING = -6,   /* a required tensor is absent or duplicated */
    V3_E_SHAPE = -7,     /* tensor shape inconsistent with hyper-parameters */
    V3_E_ARENA = -8,     /* arena too small or misaligned */
    V3_E_TOKENIZER = -9, /* malformed tokenizer blob */
    V3_E_LIMIT = -10,    /* hyper-parameter exceeds compiled limits */
    V3_E_ALIGN = -11,    /* model buffer not 64-byte aligned */
    V3_E_NAN = -12       /* non-finite (NaN / Inf) weight or scale detected */
};

enum { V3_F32 = 0, V3_F16 = 1, V3_BF16 = 2, V3_Q8_0 = 8, V3_Q4_0 = 9 };

typedef struct {
    const uint8_t *data;
    uint32_t dtype, ndim;
    uint32_t dims[4];
    uint64_t nbytes;
} v3_tensor;

typedef struct {
    v3_tensor ln1, wq, bq, wk, bk, wv, bv, wo, ln2, wg, wu, wd;
    v3_tensor a_k, a_q, beta_w, beta_b, alpha; /* Kenyon Vault */
} v3_layer;

typedef struct {
    const uint8_t *base;
    uint64_t size;
    uint32_t vocab_size, d_model, n_layers, n_heads, n_kv_heads, head_dim, d_ff, window;
    uint32_t n_banks, primes[V3_MAX_BANKS], n_cells;
    float rms_eps, rope_theta, vault_eps;
    uint32_t bos_id, eos_id, im_start_id, im_end_id, pad_id, weight_format, flags;
    uint32_t n_sink;            /* base attention-sink tokens (auto-expands on ChatML system turn) */
    /* Optional Segment-Evicted KV Memory (seg_n_feat == 0 in Kenyon-only files) */
    uint32_t seg_budget, seg_hidden, seg_min, seg_max, seg_n_feat;
    uint32_t seg_layers[V3_MAX_SEG_FEAT];
    v3_tensor embed, norm, route;
    v3_tensor seg_ln_w, seg_ln_b, seg_w1, seg_b1, seg_w2, seg_b2;
    v3_layer layers[V3_MAX_LAYERS];
    /* tokenizer views (all point into the model buffer) */
    uint32_t tok_n_vocab, tok_n_merges, tok_n_special, tok_n_ranges;
    const uint8_t *tok_table;   /* n_vocab x (u32 offset, u32 len) */
    const uint8_t *tok_pool;
    uint64_t tok_pool_size;
    const uint8_t *tok_merges;  /* n_merges x (u32 a, u32 b, u32 result), rank = index */
    const uint8_t *tok_special; /* n_special x u32 id */
    const uint8_t *tok_ranges;  /* n_ranges x (u32 lo, u32 hi, u32 cat) sorted, cat 1 = letter, 2 = number */
    const uint8_t *tok_bytes;   /* 256 x u32: token id of each raw byte */
} v3_model;

/* Parse and validate a .vlt3 buffer. verify_crc=1 also checks every tensor CRC32 & full NaN/Inf scan. */
int v3_load(v3_model *m, const uint8_t *buf, uint64_t size, int verify_crc);

/* ---------------- inference state (fixed size) ---------------- */

typedef struct {
    float *k_prev, *k_cur;       /* [G][W][D] post-RoPE keys (positions re-based to [0, 2W)) */
    float *v_prev, *v_cur;       /* [G][W][D] values */
    float *kraw_prev, *kraw_cur; /* [G][W][D] pre-RoPE raw keys (for Segment Memory bank admission) */
    float *kt_prev, *kt_cur;     /* [G][W][D] adapted Vault write keys */
    float *b_prev, *b_cur;       /* [G][W] write strength */
    float *M;                    /* [G][C][D][D] Kenyon cells */
    float *z;                    /* [G][C][D] normalisers */
    float *k_sink_raw, *k_sink;  /* [G][S_cap][D] pre-RoPE and rotated sink keys */
    float *v_sink;               /* [G][S_cap][D] sink values */
    float *k_seg_raw, *k_seg;    /* [G][B_cap][D] pre-RoPE and rotated Segment Memory keys */
    float *v_seg;                /* [G][B_cap][D] Segment Memory values */
} v3_layer_state;

typedef void (*v3_task_fn)(void *arg, uint32_t i0, uint32_t i1);
/* Optional parallelism hook supplied by the host OS: run fn(arg, i0, i1) over a partition of [0, n),
 * possibly on several cores, and return only when every range has finished. NULL = single core. */
typedef void (*v3_parallel_for_fn)(void *ctx, v3_task_fn fn, void *arg, uint32_t n);

typedef struct {
    const v3_model *m;
    uint64_t pos;            /* absolute token count */
    int vault_on;
    int seg_on;
    int auto_pin;            /* 1 = automatically pin ChatML <|im_start|>system...<|im_end|> prefix */
    int pin_locked;
    int saw_im_start;
    int awaiting_sys_nl;
    int ema_ready;
    int seg_bias_active;
    uint32_t sink_cap;
    uint32_t n_pin;          /* active pinned prefix tokens (<= sink_cap) */
    uint32_t seg_cap;
    uint32_t seg_budget;     /* active Segment Memory KV slots (<= seg_cap) */
    uint32_t mem_cnt;        /* currently occupied Segment Memory slots (<= seg_budget) */
    uint32_t n_uniq_segs;    /* distinct segments currently in Segment Memory */
    uint32_t sh_cap;         /* power-of-2 capacity of shingle hash index */
    uint32_t post_cap;       /* max (shingle, segment) posting entries */
    uint32_t seg_id_offset;
    uint32_t cur_seg_id;
    uint32_t cur_seg_len;
    v3_parallel_for_fn par_for;
    void *par_ctx;
    v3_layer_state ls[V3_MAX_LAYERS];
    float *x, *xb, *q, *qraw, *k, *kraw, *v, *att, *ao, *hb, *hb2, *logits, *tmp;
    float *inv_freq, *cos_t, *sin_t, *cos_w, *sin_w;
    float *w_phk, *w_delta;  /* chunk write scratch */
    uint32_t *w_cells;
    /* Segment Memory state, APL causal EMA habituation & shingle resonance scratch */
    uint32_t *tok_prev, *tok_cur;
    float *sc_prev, *sc_cur, *seg_mx;
    uint32_t *seg_prev, *seg_cur;
    uint64_t *mem_pos, *cand_pos;
    float *mem_score, *cand_score;
    uint32_t *mem_tok, *cand_tok;
    uint32_t *mem_seg, *cand_seg;
    uint32_t *seg_inv;
    float *seg_bias_arch, *seg_bias_sup, *seg_bias_hop1;
    uint32_t *cand_order, *keep_old, *add_cand;
    float *seg_feat, *seg_feat_ema, *seg_feat_hab, *seg_norm, *seg_hid;
    uint64_t *sh_keys, *q_mask;
    uint32_t *sh_df, *sh_head, *sh_last_sid;
    uint32_t *post_sid, *post_w, *post_next;
    uint32_t *q_scores, *q_hits, *q_matched, *q_shingles;
    uint8_t *q_norm_buf;
} v3_state;

size_t v3_state_arena_size_ex(const v3_model *m, uint32_t max_seg_budget);
size_t v3_state_arena_size(const v3_model *m);
int v3_state_init_ex(v3_state *s, const v3_model *m, void *arena, size_t arena_size, uint32_t max_seg_budget);
int v3_state_init(v3_state *s, const v3_model *m, void *arena, size_t arena_size);
void v3_state_reset(v3_state *s);
int v3_set_pinned(v3_state *s, uint32_t n_pin);
int v3_set_seg_budget(v3_state *s, uint32_t budget);
/* Feed one token, return pointer to vocab_size logits (valid until the next call). */
const float *v3_step(v3_state *s, uint32_t token);

/* ---------------- tokenizer (byte-level BPE, Qwen2 pre-tokenizer + UAX #15 NFC) ---------------- */

#define V3_NFC_BUF_CAP 8192
#define V3_NFC_CPS_CAP 2048

typedef struct {
    const v3_model *m;
    uint64_t *keys;     /* hash: (a << 32) | b */
    uint32_t *vals;     /* (rank, result) pairs */
    uint32_t mask;
    uint32_t *piece;    /* V3_TOK_MAX_PIECE symbol scratch */
    int32_t *prev;      /* V3_TOK_MAX_PIECE doubly-linked list */
    int32_t *next;      /* V3_TOK_MAX_PIECE doubly-linked list */
    uint64_t *heap;     /* 3 * V3_TOK_MAX_PIECE min-heap: (rank << 32) | left_pos */
    uint32_t *win_toks; /* V3_TOK_MAX_PIECE sliding-window BPE token scratch (>32KB pieces) */
    uint8_t *nfc_buf;   /* V3_NFC_BUF_CAP UTF-8 NFC normalization scratch */
    uint32_t *nfc_cps;  /* V3_NFC_CPS_CAP codepoint decomposition/composition scratch */
} v3_tokenizer;

size_t v3_tok_arena_size(const v3_model *m);
int v3_tok_init(v3_tokenizer *t, const v3_model *m, void *arena, size_t arena_size);
/* Automatically applies UAX #15 NFC normalization when bytes >= 0xCC are present (zero-copy fast path otherwise).
 * v3_encode_ex(..., allow_special=0) treats all special-token strings (<|im_start|>, etc.) as literal text
 * so untrusted user inputs or log files can never inject ChatML control delimiters. */
int v3_encode_ex(v3_tokenizer *t, const uint8_t *text, size_t len, uint32_t *out, size_t max_out, size_t *n_out, int allow_special);
int v3_encode(v3_tokenizer *t, const uint8_t *text, size_t len, uint32_t *out, size_t max_out, size_t *n_out);
int v3_token_bytes(const v3_model *m, uint32_t id, const uint8_t **bytes, uint32_t *len);

/* ---------------- CAAI / OPI constitutional output governor ---------------- */

#define V3_GOV_MAX_RULES 8
#define V3_GOV_RULE_CAP 32
#define V3_GOV_HIST_CAP 64
#define V3_GOV_RAW_CAP 96

typedef struct {
    int active;
    uint32_t n_rules;
    uint8_t rules[V3_GOV_MAX_RULES][V3_GOV_RULE_CAP];
    uint32_t rule_len[V3_GOV_MAX_RULES];
    uint8_t hist[V3_GOV_HIST_CAP];
    uint32_t hist_len;
    uint8_t utf8_rem[4];
    uint32_t utf8_len;
    uint8_t raw_tail[V3_GOV_RAW_CAP];
    uint32_t raw_len;
} v3_governor;

void v3_gov_init(v3_governor *g);
void v3_gov_reset_stream(v3_governor *g);
int v3_gov_add_rule(v3_governor *g, const uint8_t *pat, size_t len);
void v3_gov_arm_system(v3_governor *g, const uint8_t *sys_text, size_t len);
void v3_govern_logits(const v3_governor *g, const v3_model *m, float *logits, uint32_t top_k_guard);
void v3_gov_commit(v3_governor *g, const v3_model *m, uint32_t id);

/* ---------------- sampling ---------------- */
uint32_t v3_argmax(const float *logits, uint32_t n);
/* Top-k (k <= 64) + temperature sampling with a caller-held xorshift state. */
uint32_t v3_sample_topk(const float *logits, uint32_t n, uint32_t k, float temp, uint64_t *rng);

#endif /* VAULT3_H */
