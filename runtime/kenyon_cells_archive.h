/* SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
 * Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
 * Copyright (c) 2026 Philip Bartsch (batteryphil)
 *
 * kenyon_cells_archive.h — Freestanding C99 reference archive of Co-Prime Kenyon Cell
 * Associative Memory (extracted from runtime/vault3.c for future repurposing).
 *
 *   - Zero libc, zero malloc, deterministic caller-supplied scratch buffers.
 *   - Co-prime ternary LSH routing (P = {13, 17, 19} -> 4,199 combinatorial triplets).
 *   - Chunkwise delta-rule associative write + homogeneous cell rescaling (|z| <= 65,536)
 *     so FP32 accumulators remain bounded across infinite token streams.
 *   - Bank-averaged normalized associative read.
 */
#ifndef VAULT3_KENYON_CELLS_ARCHIVE_H
#define VAULT3_KENYON_CELLS_ARCHIVE_H

#include <stdint.h>

#define KC_MAX_BANKS 8

typedef struct {
    uint32_t head_dim;
    uint32_t window;
    uint32_t n_banks;
    uint32_t primes[KC_MAX_BANKS];
    uint32_t n_cells;
    float vault_eps;
    const float *route; /* [head_dim][n_cells] ternary {-1, 0, +1} */
} kc_config;

static inline float kc_dot_f32(const float *a, const float *b, uint32_t n) {
    float s = 0.0f;
    for (uint32_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static inline void kc_axpy_f32(float *y, float a, const float *x, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) y[i] += a * x[i];
}

static inline float kc_sqrtf(float x) {
    if (x <= 0.0f) return 0.0f;
    union { float f; uint32_t u; } v = {x};
    v.u = (v.u >> 1) + 0x1fc00000u;
    float y = v.f;
    for (int i = 0; i < 4; i++) y = 0.5f * (y + x / y);
    return y;
}

static inline float kc_expf(float x) {
    if (x < -87.0f) return 0.0f;
    if (x > 88.0f) x = 88.0f;
    float z = x * 1.4426950408889634f;
    int32_t k = (z >= 0.0f) ? (int32_t)(z + 0.5f) : (int32_t)(z - 0.5f);
    float r = x - (float)k * 0.6931471805599453f;
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f + r * (0.041666668f + r * 0.008333334f))));
    union { uint32_t u; float f; } s;
    s.u = (uint32_t)(k + 127) << 23;
    return p * s.f;
}

static inline float kc_tanhf(float x) {
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return -1.0f;
    float e2 = kc_expf(2.0f * x);
    return (e2 - 1.0f) / (e2 + 1.0f);
}

static inline float kc_phi(float x) {
    return x > 0.0f ? x + 1.0f : kc_expf(x); /* elu(x) + 1 > 0 */
}

static inline void kc_unit_rms(float *out, const float *x, uint32_t n) {
    float inv = 1.0f / kc_sqrtf(kc_dot_f32(x, x, n) / (float)n + 1e-6f);
    for (uint32_t i = 0; i < n; i++) out[i] = x[i] * inv;
}

static inline void kc_vecmat(float *out, const float *in, const float *A, uint32_t D) {
    for (uint32_t e = 0; e < D; e++) out[e] = 0.0f;
    for (uint32_t d0 = 0; d0 < D; d0++) kc_axpy_f32(out, in[d0], A + (uint64_t)d0 * D, D);
}

static inline void kc_route_cells(const kc_config *cfg, const float *x, uint32_t *cells) {
    const float *R = cfg->route;
    uint32_t D = cfg->head_dim, C = cfg->n_cells, off = 0;
    for (uint32_t b = 0; b < cfg->n_banks; b++) {
        uint32_t best = 0;
        float bestv = 0.0f;
        for (uint32_t j = 0; j < cfg->primes[b]; j++) {
            float s = 0.0f;
            for (uint32_t d0 = 0; d0 < D; d0++) s += x[d0] * R[(uint64_t)d0 * C + off + j];
            if (j == 0 || s > bestv) { bestv = s; best = j; }
        }
        cells[b] = off + best;
        off += cfg->primes[b];
    }
}

/* Chunkwise delta-rule write into Kenyon cells M [C][D][D] and normalizers z [C][D].
 * Caller provides scratch buffers: w_phk [W*D], w_delta [W*NB*D], w_cells [W*NB]. */
static inline void kc_write_chunk(
    const kc_config *cfg,
    float *M,
    float *z,
    const float *kt,
    const float *v,
    const float *beta,
    float *w_phk,
    float *w_delta,
    uint32_t *w_cells
) {
    uint32_t W = cfg->window, D = cfg->head_dim, C = cfg->n_cells, NB = cfg->n_banks;
    float eps = cfg->vault_eps;
    for (uint32_t t = 0; t < W; t++) {
        const float *kk = kt + (uint64_t)t * D;
        float *ph = w_phk + (uint64_t)t * D;
        for (uint32_t d0 = 0; d0 < D; d0++) ph[d0] = kc_phi(kk[d0]);
        uint32_t *cells = w_cells + (uint64_t)t * NB;
        kc_route_cells(cfg, kk, cells);
        for (uint32_t b = 0; b < NB; b++) {
            const float *Mc = M + (uint64_t)cells[b] * D * D, *zc = z + (uint64_t)cells[b] * D;
            float den = kc_dot_f32(ph, zc, D);
            float *dl = w_delta + ((uint64_t)t * NB + b) * D;
            for (uint32_t e = 0; e < D; e++) dl[e] = 0.0f;
            if (den > eps) {
                for (uint32_t d0 = 0; d0 < D; d0++) kc_axpy_f32(dl, ph[d0], Mc + (uint64_t)d0 * D, D);
                float inv = 1.0f / den;
                for (uint32_t e = 0; e < D; e++) dl[e] *= inv;
            }
            const float *vv = v + (uint64_t)t * D;
            for (uint32_t e = 0; e < D; e++) dl[e] = beta[t] * (vv[e] - dl[e]);
        }
    }
    for (uint32_t t = 0; t < W; t++) {
        const float *ph = w_phk + (uint64_t)t * D;
        const uint32_t *cells = w_cells + (uint64_t)t * NB;
        for (uint32_t b = 0; b < NB; b++) {
            float *Mc = M + (uint64_t)cells[b] * D * D, *zc = z + (uint64_t)cells[b] * D;
            const float *dl = w_delta + ((uint64_t)t * NB + b) * D;
            for (uint32_t d0 = 0; d0 < D; d0++) {
                float a = ph[d0];
                kc_axpy_f32(Mc + (uint64_t)d0 * D, a, dl, D);
                zc[d0] += beta[t] * a;
            }
        }
    }
    for (uint32_t cell = 0; cell < C; cell++) {
        float *zc = z + (uint64_t)cell * D;
        float zmax = 0.0f;
        for (uint32_t d0 = 0; d0 < D; d0++) if (zc[d0] > zmax) zmax = zc[d0];
        if (zmax > 65536.0f) {
            float *Mc = M + (uint64_t)cell * D * D;
            for (uint32_t i = 0; i < D * D; i++) Mc[i] *= 0.5f;
            for (uint32_t d0 = 0; d0 < D; d0++) zc[d0] *= 0.5f;
        }
    }
}

static inline void kc_read(
    const kc_config *cfg,
    const float *M,
    const float *z,
    const float *qt,
    float *out,
    float *phq
) {
    uint32_t D = cfg->head_dim, NB = cfg->n_banks;
    uint32_t cells[KC_MAX_BANKS];
    for (uint32_t d0 = 0; d0 < D; d0++) phq[d0] = kc_phi(qt[d0]);
    kc_route_cells(cfg, qt, cells);
    for (uint32_t e = 0; e < D; e++) out[e] = 0.0f;
    for (uint32_t b = 0; b < NB; b++) {
        const float *Mc = M + (uint64_t)cells[b] * D * D, *zc = z + (uint64_t)cells[b] * D;
        float den = kc_dot_f32(phq, zc, D);
        if (!(den > cfg->vault_eps)) continue;
        float inv = 1.0f / den;
        for (uint32_t d0 = 0; d0 < D; d0++) kc_axpy_f32(out, phq[d0] * inv, Mc + (uint64_t)d0 * D, D);
    }
    float invnb = 1.0f / (float)NB;
    for (uint32_t e = 0; e < D; e++) out[e] *= invnb;
}

#endif /* VAULT3_KENYON_CELLS_ARCHIVE_H */
