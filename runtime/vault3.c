/* SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
 * Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
 * Copyright (c) 2026 Philip Bartsch (batteryphil)
 *
 * vault3.c — freestanding C99 inference runtime for VAULT v3. See vault3.h and docs/VAULT_v3_SPEC.md.
 *   - Zero libc, zero malloc, zero external math symbols (even without -fno-math-errno).
 *   - 64-bit overflow-checked header/shape/arena validation + NaN/Inf weight & scale detection.
 *   - Dynamic ChatML system-prompt pinning, Kenyon Vault with homogeneous cell rescaling,
 *     single-pass Segment-Evicted KV Memory, and O(n log n) min-heap byte-level BPE.
 */
#include "vault3.h"
#include "vault3_nfc_tables.h"
#ifdef __AVX2__
#include <immintrin.h>
#elif defined(__SSE__)
#include <xmmintrin.h>
#endif

#define HDR_SIZE 4096u
#define ENTRY_SIZE 128u
#define FILE_ALIGN 64u
#define QK 32u

/* ============================================================================
 * Little-endian readers & IEEE-754 bit helpers (no unaligned loads)
 * ========================================================================== */
static uint16_t rd16(const uint8_t *p) { return (uint16_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8)); }
static uint32_t rd32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t rd64(const uint8_t *p) { return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32); }
static float u2f(uint32_t u) { union { uint32_t u; float f; } c; c.u = u; return c.f; }
static uint32_t f2u(float f) { union { uint32_t u; float f; } c; c.f = f; return c.u; }
static float rdf32(const uint8_t *p) { return u2f(rd32(p)); }

static int is_finite_f32_bits(uint32_t u) { return (u & 0x7F800000u) != 0x7F800000u; }
static int is_finite_f32(float x) { return is_finite_f32_bits(f2u(x)); }
static int is_finite_f16_bits(uint16_t h) { return (h & 0x7C00u) != 0x7C00u; }
static int is_finite_bf16_bits(uint16_t b) { return (b & 0x7F80u) != 0x7F80u; }

/* ============================================================================
 * CRC32 (IEEE 802.3, identical to zlib.crc32) and fp16 table
 * ========================================================================== */
static uint32_t g_crc_table[256];
static float g_f16[65536];
static int g_tables_ready = 0;

static float f16_to_f32_slow(uint16_t h) {
    uint32_t s = (uint32_t)(h >> 15) & 1u, e = (uint32_t)(h >> 10) & 0x1Fu, mt = (uint32_t)h & 0x3FFu, u;
    if (e == 0) {
        if (mt == 0) {
            u = s << 31;
        } else {
            int ee = -1;
            do { ee++; mt <<= 1; } while (!(mt & 0x400u));
            mt &= 0x3FFu;
            u = (s << 31) | ((uint32_t)(112 - ee) << 23) | (mt << 13);
        }
    } else if (e == 31) {
        u = (s << 31) | 0x7F800000u | (mt << 13);
    } else {
        u = (s << 31) | ((e - 15u + 127u) << 23) | (mt << 13);
    }
    return u2f(u);
}

static void init_tables(void) {
    if (g_tables_ready) return;
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        g_crc_table[i] = c;
    }
    for (uint32_t i = 0; i < 65536; i++)
        g_f16[i] = is_finite_f16_bits((uint16_t)i) ? f16_to_f32_slow((uint16_t)i) : 0.0f;
    g_tables_ready = 1;
}

static uint32_t crc32_buf(const uint8_t *p, uint64_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (uint64_t i = 0; i < n; i++) c = g_crc_table[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

/* ============================================================================
 * Freestanding math (zero libm calls even without -fno-math-errno)
 * ========================================================================== */
static float v3_sqrtf(float x) {
    if (!(x > 0.0f)) return 0.0f;
#if !defined(V3_FORCE_SCALAR_MATH) && (defined(__AVX2__) || defined(__SSE__))
    return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)));
#else
    uint32_t u = f2u(x);
    float y = u2f((u >> 1) + 0x1FC00000u);
    y = 0.5f * (y + x / y);
    y = 0.5f * (y + x / y);
    y = 0.5f * (y + x / y);
    y = 0.5f * (y + x / y);
    return y;
#endif
}

static float v3_expf(float x) {
    if (x > 88.72f) return u2f(0x7F800000u);
    if (x < -103.9f) return 0.0f;
    float kf = x * 1.44269504088896341f;
    int k = (int)(kf + (kf >= 0.0f ? 0.5f : -0.5f));
    float r = (x - (float)k * 0.693145751953125f) - (float)k * 1.428606765330187e-06f;
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (1.6666667e-1f + r * (4.1666668e-2f + r * (8.3333338e-3f + r * (1.3888889e-3f + r * 1.9841270e-4f))))));
    if (k > 127) return p * u2f(254u << 23) * 2.0f;
    if (k < -126) return p * u2f((uint32_t)(k + 126 + 127) << 23) * u2f(1u << 23);
    return p * u2f((uint32_t)(k + 127) << 23);
}

static float v3_sigmoid(float x) { return 1.0f / (1.0f + v3_expf(-x)); }
static float v3_tanhf(float x) {
    if (x > 9.0f) return 1.0f;
    if (x < -9.0f) return -1.0f;
    return 1.0f - 2.0f / (v3_expf(2.0f * x) + 1.0f);
}
static float v3_phi(float x) { return x > 0.0f ? x + 1.0f : v3_expf(x); } /* elu(x) + 1 */

static float v3_erff(float x) { /* Abramowitz & Stegun 7.1.26, |err| <= 1.5e-7 */
    float z = x < 0.0f ? -x : x;
    if (z > 4.0f) return x < 0.0f ? -1.0f : 1.0f;
    float t = 1.0f / (1.0f + 0.3275911f * z);
    float p = (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    float y = 1.0f - p * v3_expf(-z * z);
    return x < 0.0f ? -y : y;
}

static float v3_geluf(float x) {
    return 0.5f * x * (1.0f + v3_erff(x * 0.70710678118654752f));
}

static double v3_exp_d(double x) {
    double kd = x * 1.4426950408889634;
    long long k = (long long)(kd + (kd >= 0 ? 0.5 : -0.5));
    double r = (x - (double)k * 6.93147180369123816490e-01) - (double)k * 1.90821492927058770002e-10;
    double t = 1.0, s = 1.0;
    for (int i = 1; i <= 22; i++) { t *= r / (double)i; s += t; }
    union { uint64_t u; double d; } c;
    c.u = (uint64_t)(k + 1023) << 52;
    return s * c.d;
}

static double v3_log_d(double x) { /* x > 0, normal */
    union { uint64_t u; double d; } c;
    c.d = x;
    int e = (int)((c.u >> 52) & 0x7FF) - 1023;
    c.u = (c.u & 0x000FFFFFFFFFFFFFull) | 0x3FF0000000000000ull; /* mantissa in [1, 2) */
    double mval = c.d;
    if (mval > 1.4142135623730951) { mval *= 0.5; e += 1; }
    double s = (mval - 1.0) / (mval + 1.0), s2 = s * s, term = s, sum = 0.0;
    for (int i = 1; i < 60; i += 2) { sum += term / (double)i; term *= s2; }
    return 2.0 * sum + (double)e * 0.69314718055994530942;
}

static double sin_k(double x) { double x2 = x * x; return x * (1 - x2 / 6 * (1 - x2 / 20 * (1 - x2 / 42 * (1 - x2 / 72 * (1 - x2 / 110 * (1 - x2 / 156 * (1 - x2 / 210))))))); }
static double cos_k(double x) { double x2 = x * x; return 1 - x2 / 2 * (1 - x2 / 12 * (1 - x2 / 30 * (1 - x2 / 56 * (1 - x2 / 90 * (1 - x2 / 132 * (1 - x2 / 182 * (1 - x2 / 240))))))); }
static void v3_sincos(double a, double *s, double *c) {
    double kd = a / 1.5707963267948966;
    long long k = (long long)(kd + (kd >= 0 ? 0.5 : -0.5));
    double r = (a - (double)k * 1.5707963267948966) - (double)k * 6.123233995736766e-17;
    double sr = sin_k(r), cr = cos_k(r);
    switch ((int)(k & 3)) {
    case 0: *s = sr; *c = cr; break;
    case 1: *s = cr; *c = -sr; break;
    case 2: *s = -sr; *c = -cr; break;
    default: *s = -cr; *c = sr; break;
    }
}

/* ============================================================================
 * Loading / validation
 * ========================================================================== */
static uint64_t dtype_nbytes(uint32_t dt, uint64_t n, int *ok) {
    *ok = 1;
    switch (dt) {
    case V3_F32:
        if (n > (~0ull) / 4ull) { *ok = 0; return 0; }
        return 4ull * n;
    case V3_F16:
    case V3_BF16:
        if (n > (~0ull) / 2ull) { *ok = 0; return 0; }
        return 2ull * n;
    case V3_Q8_0: {
        uint64_t blks = n / QK;
        if ((n % QK) != 0 || blks > (~0ull) / 34ull) { *ok = 0; return 0; }
        return blks * 34ull;
    }
    case V3_Q4_0: {
        uint64_t blks = n / QK;
        if ((n % QK) != 0 || blks > (~0ull) / 18ull) { *ok = 0; return 0; }
        return blks * 18ull;
    }
    default:
        *ok = 0;
        return 0;
    }
}

static int region_ok(uint64_t off, uint64_t n, uint64_t size) {
    return off >= HDR_SIZE && off <= size && n <= size - off;
}

static int expect_shape(const v3_tensor *t, uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2) {
    if (!t->data || t->ndim != ndim) return 0;
    if ((uint64_t)t->dims[0] != d0) return 0;
    if (ndim > 1 && (uint64_t)t->dims[1] != d1) return 0;
    if (ndim > 2 && (uint64_t)t->dims[2] != d2) return 0;
    return 1;
}

static int is_f32(const v3_tensor *t) { return t->dtype == V3_F32; }
static int is_matmul_dtype(const v3_tensor *t) {
    return t->dtype == V3_F32 || t->dtype == V3_F16 || t->dtype == V3_BF16 || t->dtype == V3_Q8_0 || t->dtype == V3_Q4_0;
}

static int check_tensor_finite(const uint8_t *p, uint32_t dt, uint64_t n, int full_scan) {
    (void)full_scan;
    switch (dt) {
    case V3_F32: {
        uint64_t i = 0;
#ifdef __AVX2__
        const __m256i exp_m = _mm256_set1_epi32(0x7F800000);
        for (; i + 8ull <= n; i += 8ull) {
            __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)(p + 4ull * i));
            __m256i c = _mm256_cmpeq_epi32(_mm256_and_si256(v, exp_m), exp_m);
            if (!_mm256_testz_si256(c, c)) return 0;
        }
#endif
        for (; i < n; i++)
            if (!is_finite_f32_bits(rd32(p + 4ull * i))) return 0;
        return 1;
    }
    case V3_F16: {
        uint64_t i = 0;
#ifdef __AVX2__
        const __m256i exp_m = _mm256_set1_epi16(0x7C00);
        for (; i + 16ull <= n; i += 16ull) {
            __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)(p + 2ull * i));
            __m256i c = _mm256_cmpeq_epi16(_mm256_and_si256(v, exp_m), exp_m);
            if (!_mm256_testz_si256(c, c)) return 0;
        }
#endif
        for (; i < n; i++)
            if (!is_finite_f16_bits(rd16(p + 2ull * i))) return 0;
        return 1;
    }
    case V3_BF16: {
        uint64_t i = 0;
#ifdef __AVX2__
        const __m256i exp_m = _mm256_set1_epi16(0x7F80);
        for (; i + 16ull <= n; i += 16ull) {
            __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)(p + 2ull * i));
            __m256i c = _mm256_cmpeq_epi16(_mm256_and_si256(v, exp_m), exp_m);
            if (!_mm256_testz_si256(c, c)) return 0;
        }
#endif
        for (; i < n; i++)
            if (!is_finite_bf16_bits(rd16(p + 2ull * i))) return 0;
        return 1;
    }
    case V3_Q8_0:
    case V3_Q4_0: {
        uint64_t blks = n / QK;
        uint64_t stride = (dt == V3_Q8_0) ? 34ull : 18ull;
        for (uint64_t b = 0; b < blks; b++)
            if (!is_finite_f16_bits(rd16(p + b * stride))) return 0;
        return 1;
    }
    default:
        return 0;
    }
}

int v3_load(v3_model *m, const uint8_t *buf, uint64_t size, int verify_crc) {
    init_tables();
    if (!m) return V3_E_SIZE;
    uint8_t *mz = (uint8_t *)m;
    for (size_t i = 0; i < sizeof(*m); i++) mz[i] = 0;
    if (!buf || size < HDR_SIZE) return V3_E_SIZE;
    if (((uintptr_t)buf) % FILE_ALIGN) return V3_E_ALIGN;
    if (buf[0] != 'V' || buf[1] != 'L' || buf[2] != 'T' || buf[3] != '3') return V3_E_MAGIC;
    if (rd32(buf + 4) != 1 || rd32(buf + 8) != HDR_SIZE || rd32(buf + 12) != 0x01020304u) return V3_E_MAGIC;
    if (crc32_buf(buf, HDR_SIZE - 4) != rd32(buf + HDR_SIZE - 4)) return V3_E_CRC;
    if (rd64(buf + 16) != size) return V3_E_SIZE;
    if (rd32(buf + 24) != FILE_ALIGN) return V3_E_MAGIC;

    uint32_t n_tensors = rd32(buf + 28);
    uint64_t dir_off = rd64(buf + 32), dir_size = rd64(buf + 40);
    uint64_t tok_off = rd64(buf + 48), tok_size = rd64(buf + 56);
    uint64_t data_off = rd64(buf + 64), data_size = rd64(buf + 72);
    if (!region_ok(dir_off, dir_size, size) || !region_ok(tok_off, tok_size, size) || !region_ok(data_off, data_size, size))
        return V3_E_BOUNDS;
    if (n_tensors > 8192u || dir_size != (uint64_t)n_tensors * ENTRY_SIZE) return V3_E_BOUNDS;

    m->base = buf;
    m->size = size;
    m->vocab_size = rd32(buf + 80); m->d_model = rd32(buf + 84); m->n_layers = rd32(buf + 88);
    m->n_heads = rd32(buf + 92); m->n_kv_heads = rd32(buf + 96); m->head_dim = rd32(buf + 100);
    m->d_ff = rd32(buf + 104); m->window = rd32(buf + 108); m->n_banks = rd32(buf + 112);
    for (int i = 0; i < V3_MAX_BANKS; i++) m->primes[i] = rd32(buf + 116 + 4 * i);
    m->n_cells = rd32(buf + 148);
    m->rms_eps = rdf32(buf + 152); m->rope_theta = rdf32(buf + 156); m->vault_eps = rdf32(buf + 160);
    m->bos_id = rd32(buf + 164); m->eos_id = rd32(buf + 168); m->im_start_id = rd32(buf + 172);
    m->im_end_id = rd32(buf + 176); m->pad_id = rd32(buf + 180); m->weight_format = rd32(buf + 184); m->flags = rd32(buf + 188);
    m->n_sink = rd32(buf + 256);
    m->seg_budget = rd32(buf + 260);
    m->seg_hidden = rd32(buf + 264);
    m->seg_min = rd32(buf + 268);
    m->seg_max = rd32(buf + 272);
    m->seg_n_feat = rd32(buf + 276);
    for (int i = 0; i < V3_MAX_SEG_FEAT; i++) m->seg_layers[i] = rd32(buf + 280 + 4 * i);

    if (m->vocab_size == 0 || m->vocab_size > 524288u || m->d_model == 0 || m->d_model > 16384u ||
        m->n_layers == 0 || m->n_layers > V3_MAX_LAYERS || m->n_heads == 0 || m->n_heads > V3_MAX_HEADS ||
        m->head_dim == 0 || m->head_dim > V3_MAX_HEAD_DIM || (m->head_dim & 1u) ||
        m->n_kv_heads == 0 || m->n_kv_heads > m->n_heads || (m->n_heads % m->n_kv_heads) != 0 ||
        m->d_ff == 0 || m->d_ff > 65536u || m->window == 0 || m->window > 65536u ||
        m->n_banks == 0 || m->n_banks > V3_MAX_BANKS || m->n_sink > V3_MAX_SINK || m->n_sink > m->window)
        return V3_E_LIMIT;
    if ((uint64_t)m->n_heads * (uint64_t)m->head_dim != (uint64_t)m->d_model) return V3_E_SHAPE;

    uint64_t csum = 0;
    for (uint32_t b = 0; b < m->n_banks; b++) {
        if (m->primes[b] == 0 || m->primes[b] > 4096u) return V3_E_LIMIT;
        csum += m->primes[b];
    }
    if (csum != (uint64_t)m->n_cells || m->n_cells == 0 || m->n_cells > 4096u) return V3_E_LIMIT;
    if (!is_finite_f32(m->rope_theta) || !is_finite_f32(m->rms_eps) || !is_finite_f32(m->vault_eps)) return V3_E_NAN;
    if (!(m->rope_theta > 1.0f) || !(m->rms_eps > 0.0f) || !(m->vault_eps > 0.0f)) return V3_E_LIMIT;
    if (m->bos_id >= m->vocab_size || m->eos_id >= m->vocab_size || m->im_start_id >= m->vocab_size ||
        m->im_end_id >= m->vocab_size || m->pad_id >= m->vocab_size)
        return V3_E_LIMIT;

    if (m->seg_n_feat > V3_MAX_SEG_FEAT) return V3_E_LIMIT;
    if (m->seg_n_feat == 0) {
        if (m->seg_budget != 0 || m->seg_hidden != 0 || m->seg_min != 0 || m->seg_max != 0) return V3_E_LIMIT;
    } else {
        if (m->seg_budget > V3_MAX_SEG_BUDGET || m->seg_hidden == 0 || m->seg_hidden > V3_MAX_SEG_HIDDEN ||
            m->seg_min == 0 || m->seg_min > m->seg_max || m->seg_max > m->window)
            return V3_E_LIMIT;
        for (uint32_t i = 0; i < m->seg_n_feat; i++)
            if (m->seg_layers[i] >= m->n_layers) return V3_E_LIMIT;
    }

    /* tensor directory */
    uint64_t prev_end = 0;
    for (uint32_t i = 0; i < n_tensors; i++) {
        const uint8_t *e = buf + dir_off + (uint64_t)i * ENTRY_SIZE;
        uint32_t id = rd32(e), dt = rd32(e + 4), ndim = rd32(e + 8);
        uint32_t dims[4] = {rd32(e + 12), rd32(e + 16), rd32(e + 20), rd32(e + 24)};
        uint64_t off = rd64(e + 32), nb = rd64(e + 40);
        uint32_t crc = rd32(e + 48);
        if (rd32(e + 28) != 0) return V3_E_BOUNDS;
        if (ndim < 1 || ndim > 4) return V3_E_SHAPE;
        uint64_t n = 1;
        for (uint32_t d0 = 0; d0 < ndim; d0++) {
            if (dims[d0] == 0 || n > (~0ull) / (uint64_t)dims[d0]) return V3_E_SHAPE;
            n *= (uint64_t)dims[d0];
        }
        int ok;
        uint64_t want = dtype_nbytes(dt, n, &ok);
        if (!ok) return V3_E_DTYPE;
        if ((dt == V3_Q8_0 || dt == V3_Q4_0) && (dims[ndim - 1] % QK != 0)) return V3_E_SHAPE;
        if (want != nb) return V3_E_SHAPE;
        if (off % FILE_ALIGN || off < data_off || nb > data_off + data_size - off) return V3_E_BOUNDS;
        if (off < prev_end) return V3_E_BOUNDS;
        prev_end = off + nb;
        if (verify_crc && crc32_buf(buf + off, nb) != crc) return V3_E_CRC;
        if (!check_tensor_finite(buf + off, dt, n, verify_crc)) return V3_E_NAN;

        v3_tensor *dst = 0;
        if (id < 256) {
            if (id == 1) dst = &m->embed;
            else if (id == 2) dst = &m->norm;
            else if (id == 3) dst = &m->route;
            else if (m->seg_n_feat > 0) {
                if (id == 4) dst = &m->seg_ln_w;
                else if (id == 5) dst = &m->seg_ln_b;
                else if (id == 6) dst = &m->seg_w1;
                else if (id == 7) dst = &m->seg_b1;
                else if (id == 8) dst = &m->seg_w2;
                else if (id == 9) dst = &m->seg_b2;
            }
        } else {
            uint32_t l = (id >> 8) - 1, kind = id & 0xFFu;
            if (l >= m->n_layers) return V3_E_MISSING;
            v3_layer *L = &m->layers[l];
            switch (kind) {
            case 1: dst = &L->ln1; break; case 2: dst = &L->wq; break; case 3: dst = &L->bq; break;
            case 4: dst = &L->wk; break;  case 5: dst = &L->bk; break; case 6: dst = &L->wv; break;
            case 7: dst = &L->bv; break;  case 8: dst = &L->wo; break; case 9: dst = &L->ln2; break;
            case 10: dst = &L->wg; break; case 11: dst = &L->wu; break; case 12: dst = &L->wd; break;
            case 13: dst = &L->a_k; break; case 14: dst = &L->a_q; break; case 15: dst = &L->beta_w; break;
            case 16: dst = &L->beta_b; break; case 17: dst = &L->alpha; break;
            default: dst = 0; break;
            }
        }
        if (!dst || dst->data) return V3_E_MISSING;
        dst->data = buf + off;
        dst->dtype = dt;
        dst->ndim = ndim;
        dst->nbytes = nb;
        for (int d0 = 0; d0 < 4; d0++) dst->dims[d0] = dims[d0];
    }

    /* shapes and dtypes (64-bit arithmetic) */
    uint64_t d = m->d_model, H = m->n_heads, G = m->n_kv_heads, D = m->head_dim, F = m->d_ff, C = m->n_cells;
    if (!expect_shape(&m->embed, 2, m->vocab_size, d, 0) || !is_matmul_dtype(&m->embed)) return V3_E_SHAPE;
    if (!expect_shape(&m->norm, 1, d, 0, 0) || !is_f32(&m->norm)) return V3_E_SHAPE;
    if (!expect_shape(&m->route, 2, D, C, 0) || !is_f32(&m->route)) return V3_E_SHAPE;
    if (m->seg_n_feat > 0) {
        uint64_t fd = (uint64_t)m->seg_n_feat * d, hd = m->seg_hidden;
        if (!expect_shape(&m->seg_ln_w, 1, fd, 0, 0) || !is_f32(&m->seg_ln_w)) return V3_E_SHAPE;
        if (!expect_shape(&m->seg_ln_b, 1, fd, 0, 0) || !is_f32(&m->seg_ln_b)) return V3_E_SHAPE;
        if (!expect_shape(&m->seg_w1, 2, hd, fd, 0) || !is_matmul_dtype(&m->seg_w1)) return V3_E_SHAPE;
        if (!expect_shape(&m->seg_b1, 1, hd, 0, 0) || !is_f32(&m->seg_b1)) return V3_E_SHAPE;
        if (!expect_shape(&m->seg_w2, 2, 1, hd, 0) || !is_matmul_dtype(&m->seg_w2)) return V3_E_SHAPE;
        if (!expect_shape(&m->seg_b2, 1, 1, 0, 0) || !is_f32(&m->seg_b2)) return V3_E_SHAPE;
    }
    for (uint32_t l = 0; l < m->n_layers; l++) {
        const v3_layer *L = &m->layers[l];
        if (!expect_shape(&L->ln1, 1, d, 0, 0) || !is_f32(&L->ln1)) return V3_E_SHAPE;
        if (!expect_shape(&L->ln2, 1, d, 0, 0) || !is_f32(&L->ln2)) return V3_E_SHAPE;
        if (!expect_shape(&L->wq, 2, H * D, d, 0) || !is_matmul_dtype(&L->wq)) return V3_E_SHAPE;
        if (!expect_shape(&L->wk, 2, G * D, d, 0) || !is_matmul_dtype(&L->wk)) return V3_E_SHAPE;
        if (!expect_shape(&L->wv, 2, G * D, d, 0) || !is_matmul_dtype(&L->wv)) return V3_E_SHAPE;
        if (!expect_shape(&L->wo, 2, d, H * D, 0) || !is_matmul_dtype(&L->wo)) return V3_E_SHAPE;
        if (!expect_shape(&L->wg, 2, F, d, 0) || !is_matmul_dtype(&L->wg)) return V3_E_SHAPE;
        if (!expect_shape(&L->wu, 2, F, d, 0) || !is_matmul_dtype(&L->wu)) return V3_E_SHAPE;
        if (!expect_shape(&L->wd, 2, d, F, 0) || !is_matmul_dtype(&L->wd)) return V3_E_SHAPE;
        if (!expect_shape(&L->bq, 1, H * D, 0, 0) || !is_f32(&L->bq)) return V3_E_SHAPE;
        if (!expect_shape(&L->bk, 1, G * D, 0, 0) || !is_f32(&L->bk)) return V3_E_SHAPE;
        if (!expect_shape(&L->bv, 1, G * D, 0, 0) || !is_f32(&L->bv)) return V3_E_SHAPE;
        if (!expect_shape(&L->a_k, 3, G, D, D) || !is_f32(&L->a_k)) return V3_E_SHAPE;
        if (!expect_shape(&L->a_q, 3, G, D, D) || !is_f32(&L->a_q)) return V3_E_SHAPE;
        if (!expect_shape(&L->beta_w, 2, G, d, 0) || !is_f32(&L->beta_w)) return V3_E_SHAPE;
        if (!expect_shape(&L->beta_b, 1, G, 0, 0) || !is_f32(&L->beta_b)) return V3_E_SHAPE;
        if (!expect_shape(&L->alpha, 1, H, 0, 0) || !is_f32(&L->alpha)) return V3_E_SHAPE;
    }

    /* tokenizer blob */
    const uint8_t *tb = buf + tok_off;
    if (tok_size < 128 || tb[0] != 'V' || tb[1] != 'T' || tb[2] != 'O' || tb[3] != 'K' || rd32(tb + 4) != 1) return V3_E_TOKENIZER;
    m->tok_n_vocab = rd32(tb + 8); m->tok_n_merges = rd32(tb + 12); m->tok_n_special = rd32(tb + 16); m->tok_n_ranges = rd32(tb + 20);
    if (m->tok_n_vocab == 0 || m->tok_n_vocab > m->vocab_size || m->tok_n_merges > 2097152u ||
        m->tok_n_special > m->tok_n_vocab || m->tok_n_ranges > 65536u)
        return V3_E_TOKENIZER;
    uint64_t o_tab = rd64(tb + 28), o_pool = rd64(tb + 36), o_mer = rd64(tb + 44), o_spc = rd64(tb + 52), o_rng = rd64(tb + 60), o_byt = rd64(tb + 68);
    uint64_t s_tab = (uint64_t)m->tok_n_vocab * 8ull, s_mer = (uint64_t)m->tok_n_merges * 12ull, s_spc = (uint64_t)m->tok_n_special * 4ull,
             s_rng = (uint64_t)m->tok_n_ranges * 12ull, s_byt = 256ull * 4ull;
#define TOK_REGION(o, s) ((o) % 4 == 0 && (o) <= tok_size && (s) <= tok_size - (o))
    if (!TOK_REGION(o_tab, s_tab) || !TOK_REGION(o_mer, s_mer) || !TOK_REGION(o_spc, s_spc) || !TOK_REGION(o_rng, s_rng) ||
        !TOK_REGION(o_byt, s_byt) || o_pool > o_mer || o_pool < o_tab + s_tab)
        return V3_E_TOKENIZER;
#undef TOK_REGION
    m->tok_table = tb + o_tab; m->tok_pool = tb + o_pool; m->tok_pool_size = o_mer - o_pool;
    m->tok_merges = tb + o_mer; m->tok_special = tb + o_spc; m->tok_ranges = tb + o_rng; m->tok_bytes = tb + o_byt;
    for (uint32_t i = 0; i < m->tok_n_vocab; i++) {
        uint64_t so = rd32(m->tok_table + 8ull * i), sl = rd32(m->tok_table + 8ull * i + 4);
        if (so > m->tok_pool_size || sl > m->tok_pool_size - so) return V3_E_TOKENIZER;
    }
    for (uint32_t i = 0; i < m->tok_n_merges; i++) {
        const uint8_t *e = m->tok_merges + 12ull * i;
        if (rd32(e) >= m->tok_n_vocab || rd32(e + 4) >= m->tok_n_vocab || rd32(e + 8) >= m->tok_n_vocab) return V3_E_TOKENIZER;
    }
    for (uint32_t i = 0; i < m->tok_n_special; i++) if (rd32(m->tok_special + 4ull * i) >= m->tok_n_vocab) return V3_E_TOKENIZER;
    for (uint32_t i = 0; i < 256; i++) if (rd32(m->tok_bytes + 4ull * i) >= m->tok_n_vocab) return V3_E_TOKENIZER;
    return V3_OK;
}

/* ============================================================================
 * Kernels (portable scalar C99 + optional freestanding AVX2/FMA fast path)
 * ========================================================================== */
#ifdef __AVX2__
#ifdef __FMA__
#define V3_FMADD(a, b, c) _mm256_fmadd_ps((a), (b), (c))
#else
#define V3_FMADD(a, b, c) _mm256_add_ps(_mm256_mul_ps((a), (b)), (c))
#endif

static float hsum256(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}
#endif

static float dot_f32(const float *a, const float *b, uint32_t n) {
    uint32_t i = 0;
    float acc = 0.0f;
#ifdef __AVX2__
    __m256 sum = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) sum = V3_FMADD(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum);
    acc = hsum256(sum);
#endif
    for (; i < n; i++) acc += a[i] * b[i];
    return acc;
}

static void axpy_f32(float *dst, float a, const float *src, uint32_t n) {
    uint32_t i = 0;
#ifdef __AVX2__
    __m256 va = _mm256_set1_ps(a);
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(dst + i, V3_FMADD(va, _mm256_loadu_ps(src + i), _mm256_loadu_ps(dst + i)));
#endif
    for (; i < n; i++) dst[i] += a * src[i];
}

static float row_dot(const v3_tensor *w, uint32_t r, const float *x, uint32_t cols) {
    float acc = 0.0f;
    switch (w->dtype) {
    case V3_F32: {
        const float *p = (const float *)(const void *)w->data + (uint64_t)r * cols;
        acc = dot_f32(p, x, cols);
        break;
    }
    case V3_BF16: {
        const uint16_t *p = (const uint16_t *)(const void *)w->data + (uint64_t)r * cols;
        uint32_t c = 0;
#ifdef __AVX2__
        __m256 sum = _mm256_setzero_ps();
        for (; c + 8 <= cols; c += 8) {
            __m128i h = _mm_loadu_si128((const __m128i *)(const void *)(p + c));
            __m256 wf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
            sum = V3_FMADD(wf, _mm256_loadu_ps(x + c), sum);
        }
        acc = hsum256(sum);
#endif
        for (; c < cols; c++) acc += u2f((uint32_t)p[c] << 16) * x[c];
        break;
    }
    case V3_F16: {
        const uint16_t *p = (const uint16_t *)(const void *)w->data + (uint64_t)r * cols;
        uint32_t c = 0;
#if defined(__AVX2__) && defined(__F16C__)
        __m256 sum = _mm256_setzero_ps();
        for (; c + 8 <= cols; c += 8) {
            __m128i h = _mm_loadu_si128((const __m128i *)(const void *)(p + c));
            sum = V3_FMADD(_mm256_cvtph_ps(h), _mm256_loadu_ps(x + c), sum);
        }
        acc = hsum256(sum);
#endif
        for (; c < cols; c++) acc += g_f16[p[c]] * x[c];
        break;
    }
    case V3_Q8_0: {
        const uint8_t *p = w->data + (uint64_t)r * (cols / QK) * 34u;
#ifdef __AVX2__
        __m256 sum = _mm256_setzero_ps();
        for (uint32_t b = 0; b < cols / QK; b++, p += 34) {
            const float *xb = x + b * QK;
            __m256 q0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(const void *)(p + 2))));
            __m256 q1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(const void *)(p + 10))));
            __m256 q2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(const void *)(p + 18))));
            __m256 q3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(const void *)(p + 26))));
            __m256 bs = _mm256_mul_ps(q0, _mm256_loadu_ps(xb + 0));
            bs = V3_FMADD(q1, _mm256_loadu_ps(xb + 8), bs);
            bs = V3_FMADD(q2, _mm256_loadu_ps(xb + 16), bs);
            bs = V3_FMADD(q3, _mm256_loadu_ps(xb + 24), bs);
            sum = V3_FMADD(_mm256_set1_ps(g_f16[rd16(p)]), bs, sum);
        }
        acc = hsum256(sum);
#else
        for (uint32_t b = 0; b < cols / QK; b++, p += 34) {
            const int8_t *q = (const int8_t *)(p + 2);
            const float *xb = x + b * QK;
            float s = 0.0f;
            for (uint32_t j = 0; j < QK; j++) s += (float)q[j] * xb[j];
            acc += g_f16[rd16(p)] * s;
        }
#endif
        break;
    }
    case V3_Q4_0: {
        const uint8_t *p = w->data + (uint64_t)r * (cols / QK) * 18u;
#ifdef __AVX2__
        __m256 sum = _mm256_setzero_ps();
        const __m128i m0f = _mm_set1_epi8(0x0F), m8 = _mm_set1_epi8(8);
        for (uint32_t b = 0; b < cols / QK; b++, p += 18) {
            const float *xb = x + b * QK;
            __m128i raw = _mm_loadu_si128((const __m128i *)(const void *)(p + 2));
            __m128i lo = _mm_sub_epi8(_mm_and_si128(raw, m0f), m8);
            __m128i hi = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw, 4), m0f), m8);
            __m256 q0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
            __m256 q1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
            __m256 q2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
            __m256 q3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
            __m256 bs = _mm256_mul_ps(q0, _mm256_loadu_ps(xb + 0));
            bs = V3_FMADD(q1, _mm256_loadu_ps(xb + 8), bs);
            bs = V3_FMADD(q2, _mm256_loadu_ps(xb + 16), bs);
            bs = V3_FMADD(q3, _mm256_loadu_ps(xb + 24), bs);
            sum = V3_FMADD(_mm256_set1_ps(g_f16[rd16(p)]), bs, sum);
        }
        acc = hsum256(sum);
#else
        for (uint32_t b = 0; b < cols / QK; b++, p += 18) {
            const uint8_t *q = p + 2;
            const float *xb = x + b * QK;
            float s = 0.0f;
            for (uint32_t j = 0; j < QK / 2; j++) {
                s += (float)((int)(q[j] & 0x0F) - 8) * xb[j];
                s += (float)((int)(q[j] >> 4) - 8) * xb[j + QK / 2];
            }
            acc += g_f16[rd16(p)] * s;
        }
#endif
        break;
    }
    default: break;
    }
    return acc;
}

typedef struct {
    float *out;
    const v3_tensor *w;
    const float *x, *bias;
    uint32_t cols;
} mv_task;

static void mv_run(void *arg, uint32_t r0, uint32_t r1) {
    const mv_task *t = (const mv_task *)arg;
    for (uint32_t r = r0; r < r1; r++)
        t->out[r] = row_dot(t->w, r, t->x, t->cols) + (t->bias ? t->bias[r] : 0.0f);
}

static void matvec(v3_state *s, float *out, const v3_tensor *w, const float *x, const v3_tensor *bias) {
    mv_task t;
    t.out = out; t.w = w; t.x = x; t.cols = w->dims[1];
    t.bias = bias ? (const float *)(const void *)bias->data : 0;
    uint32_t rows = w->dims[0];
    if (s->par_for && rows >= 256) s->par_for(s->par_ctx, mv_run, &t, rows);
    else mv_run(&t, 0, rows);
}

static void embed_row(float *out, const v3_tensor *w, uint32_t r) {
    uint32_t cols = w->dims[1];
    switch (w->dtype) {
    case V3_F32: { const float *p = (const float *)(const void *)w->data + (uint64_t)r * cols; for (uint32_t c = 0; c < cols; c++) out[c] = p[c]; break; }
    case V3_BF16: { const uint16_t *p = (const uint16_t *)(const void *)w->data + (uint64_t)r * cols; for (uint32_t c = 0; c < cols; c++) out[c] = u2f((uint32_t)p[c] << 16); break; }
    case V3_F16: { const uint16_t *p = (const uint16_t *)(const void *)w->data + (uint64_t)r * cols; for (uint32_t c = 0; c < cols; c++) out[c] = g_f16[p[c]]; break; }
    case V3_Q8_0: {
        const uint8_t *p = w->data + (uint64_t)r * (cols / QK) * 34u;
        for (uint32_t b = 0; b < cols / QK; b++, p += 34) {
            float dsc = g_f16[rd16(p)];
            const int8_t *q = (const int8_t *)(p + 2);
            for (uint32_t j = 0; j < QK; j++) out[b * QK + j] = dsc * (float)q[j];
        }
        break;
    }
    case V3_Q4_0: {
        const uint8_t *p = w->data + (uint64_t)r * (cols / QK) * 18u;
        for (uint32_t b = 0; b < cols / QK; b++, p += 18) {
            float dsc = g_f16[rd16(p)];
            for (uint32_t j = 0; j < QK / 2; j++) {
                out[b * QK + j] = dsc * (float)((int)(p[2 + j] & 0x0F) - 8);
                out[b * QK + j + QK / 2] = dsc * (float)((int)(p[2 + j] >> 4) - 8);
            }
        }
        break;
    }
    default: break;
    }
}

static void rmsnorm(float *out, const float *x, const v3_tensor *w, uint32_t n, float eps) {
    float inv = 1.0f / v3_sqrtf(dot_f32(x, x, n) / (float)n + eps);
    const float *g = (const float *)(const void *)w->data;
    for (uint32_t i = 0; i < n; i++) out[i] = g[i] * (x[i] * inv);
}

static void unit_rms(float *out, const float *x, uint32_t n) {
    float inv = 1.0f / v3_sqrtf(dot_f32(x, x, n) / (float)n + 1e-6f);
    for (uint32_t i = 0; i < n; i++) out[i] = x[i] * inv;
}

static void vecmat(float *out, const float *in, const float *A, uint32_t D) {
    for (uint32_t e = 0; e < D; e++) out[e] = 0.0f;
    for (uint32_t d0 = 0; d0 < D; d0++) axpy_f32(out, in[d0], A + (uint64_t)d0 * D, D);
}

static void rope_rotate(float *x, const float *cs, const float *sn, uint32_t D) {
    uint32_t h = D / 2;
    for (uint32_t i = 0; i < h; i++) {
        float a = x[i], b = x[i + h];
        x[i] = a * cs[i] - b * sn[i];
        x[i + h] = b * cs[i] + a * sn[i];
    }
}

static void rope_copy_rotate(float *dst, const float *src, const float *cs, const float *sn, uint32_t D) {
    uint32_t h = D / 2;
    for (uint32_t i = 0; i < h; i++) {
        float a = src[i], b = src[i + h];
        dst[i] = a * cs[i] - b * sn[i];
        dst[i + h] = b * cs[i] + a * sn[i];
    }
}

static void route_cells(const v3_model *m, const float *x, uint32_t *cells) {
    const float *R = (const float *)(const void *)m->route.data; /* [D][C] */
    uint32_t D = m->head_dim, C = m->n_cells, off = 0;
    for (uint32_t b = 0; b < m->n_banks; b++) {
        uint32_t best = 0;
        float bestv = 0.0f;
        for (uint32_t j = 0; j < m->primes[b]; j++) {
            float s = 0.0f;
            for (uint32_t d0 = 0; d0 < D; d0++) s += x[d0] * R[(uint64_t)d0 * C + off + j];
            if (j == 0 || s > bestv) { bestv = s; best = j; }
        }
        cells[b] = off + best;
        off += m->primes[b];
    }
}

/* ============================================================================
 * State & overflow-checked arena bump allocator
 * ========================================================================== */
typedef struct { uint8_t *p; size_t left; size_t used; int overflow; } bump_t;

static void *bump(bump_t *b, uint64_t count, size_t elem_size) {
    if (b->overflow) return 0;
    if (elem_size != 0 && count > ((uint64_t)(~(size_t)0)) / (uint64_t)elem_size) {
        b->overflow = 1;
        return 0;
    }
    size_t n = (size_t)(count * (uint64_t)elem_size);
    size_t pad = (size_t)((64u - ((uintptr_t)b->p % 64u)) % 64u);
    if (pad > (~(size_t)0) - n || b->used > (~(size_t)0) - (pad + n)) {
        b->overflow = 1;
        return 0;
    }
    size_t need = pad + n;
    b->used += need;
    if (need > b->left) { b->p += need; b->left = 0; return 0; }
    void *r = b->p + pad;
    b->p += need;
    b->left -= need;
    return r;
}

static uint32_t compute_sink_cap(const v3_model *m) {
    uint32_t cap = m->window < V3_MAX_SINK ? m->window : V3_MAX_SINK;
    if (cap < m->n_sink) cap = m->n_sink;
    return cap > 0 ? cap : 1u;
}

static int state_layout(v3_state *s, const v3_model *m, bump_t *b, uint32_t max_seg_budget) {
    uint64_t G = m->n_kv_heads, W = m->window, D = m->head_dim, C = m->n_cells, H = m->n_heads, NB = m->n_banks;
    uint32_t sink_cap = compute_sink_cap(m);
    uint32_t seg_cap = (m->seg_n_feat > 0 && max_seg_budget > 0) ? max_seg_budget : 1u;
    uint64_t feat_dim = m->seg_n_feat > 0 ? (uint64_t)m->seg_n_feat * m->d_model : 1ull;
    uint64_t hid_dim = m->seg_hidden > 0 ? (uint64_t)m->seg_hidden : 1ull;
    uint32_t want_sh = 16u * seg_cap + 4096u;
    uint32_t sh_cap = 16384u;
    while (sh_cap < want_sh && sh_cap < (1u << 22)) sh_cap <<= 1;
    uint32_t post_cap = 24u * seg_cap + 1024u;
    if (s) {
        s->sink_cap = sink_cap;
        s->seg_cap = (m->seg_n_feat > 0) ? max_seg_budget : 0u;
        s->sh_cap = sh_cap;
        s->post_cap = post_cap;
    }
    int ok = 1;
#define AL(idx, fld, typ, cnt) do { \
    typ *ap_ = (typ *)bump(b, (uint64_t)(cnt), sizeof(typ)); \
    if (s) s->ls[idx].fld = ap_; \
    if (!ap_) ok = 0; \
} while (0)
#define A(fld, typ, cnt) do { \
    typ *ap_ = (typ *)bump(b, (uint64_t)(cnt), sizeof(typ)); \
    if (s) s->fld = ap_; \
    if (!ap_) ok = 0; \
} while (0)
    for (uint32_t l = 0; l < m->n_layers; l++) {
        AL(l, k_prev, float, G * W * D); AL(l, k_cur, float, G * W * D);
        AL(l, v_prev, float, G * W * D); AL(l, v_cur, float, G * W * D);
        AL(l, kraw_prev, float, G * W * D); AL(l, kraw_cur, float, G * W * D);
        AL(l, kt_prev, float, G * W * D); AL(l, kt_cur, float, G * W * D);
        AL(l, b_prev, float, G * W); AL(l, b_cur, float, G * W);
        AL(l, M, float, G * C * D * D); AL(l, z, float, G * C * D);
        AL(l, k_sink_raw, float, G * sink_cap * D); AL(l, k_sink, float, G * sink_cap * D); AL(l, v_sink, float, G * sink_cap * D);
        AL(l, k_seg_raw, float, G * seg_cap * D); AL(l, k_seg, float, G * seg_cap * D); AL(l, v_seg, float, G * seg_cap * D);
    }
    uint64_t att_stride = (uint64_t)sink_cap + seg_cap + 2ull * W + 64ull;
    uint64_t tmp_stride = 4ull * D + 2ull * V3_MAX_BANKS;
    A(x, float, m->d_model); A(xb, float, m->d_model); A(q, float, H * D); A(qraw, float, H * D);
    A(k, float, G * D); A(kraw, float, G * D); A(v, float, G * D);
    A(att, float, H * att_stride);
    A(ao, float, H * D); A(hb, float, m->d_ff); A(hb2, float, m->d_ff); A(logits, float, m->vocab_size);
    A(tmp, float, H * tmp_stride);
    A(inv_freq, float, D / 2); A(cos_t, float, D / 2); A(sin_t, float, D / 2); A(cos_w, float, D / 2); A(sin_w, float, D / 2);
    A(w_phk, float, W * D); A(w_delta, float, W * NB * D); A(w_cells, uint32_t, W * NB);
    A(tok_prev, uint32_t, W); A(tok_cur, uint32_t, W);
    A(sc_prev, float, W); A(sc_cur, float, W); A(seg_mx, float, W + 1ull);
    A(seg_prev, uint32_t, W); A(seg_cur, uint32_t, W);
    A(mem_pos, uint64_t, seg_cap); A(cand_pos, uint64_t, (uint64_t)seg_cap + W);
    A(mem_score, float, seg_cap); A(cand_score, float, (uint64_t)seg_cap + W);
    A(mem_tok, uint32_t, seg_cap); A(cand_tok, uint32_t, (uint64_t)seg_cap + W);
    A(mem_seg, uint32_t, seg_cap); A(cand_seg, uint32_t, (uint64_t)seg_cap + W);
    A(seg_inv, uint32_t, seg_cap);
    A(seg_bias_arch, float, (uint64_t)seg_cap + 1ull);
    A(seg_bias_sup, float, (uint64_t)seg_cap + 1ull);
    A(seg_bias_hop1, float, (uint64_t)seg_cap + 1ull);
    A(cand_order, uint32_t, (uint64_t)seg_cap + W); A(keep_old, uint32_t, seg_cap); A(add_cand, uint32_t, W);
    A(seg_feat, float, feat_dim); A(seg_feat_ema, float, feat_dim); A(seg_feat_hab, float, feat_dim);
    A(seg_norm, float, feat_dim); A(seg_hid, float, hid_dim);
    A(sh_keys, uint64_t, sh_cap); A(q_mask, uint64_t, (uint64_t)seg_cap + 1ull);
    A(sh_df, uint32_t, sh_cap); A(sh_head, uint32_t, sh_cap); A(sh_last_sid, uint32_t, sh_cap);
    A(post_sid, uint32_t, post_cap); A(post_w, uint32_t, post_cap); A(post_next, uint32_t, post_cap);
    A(q_scores, uint32_t, (uint64_t)seg_cap + 1ull); A(q_hits, uint32_t, (uint64_t)seg_cap + 1ull);
    A(q_matched, uint32_t, (uint64_t)seg_cap + 1ull); A(q_shingles, uint32_t, 2048ull);
    A(q_norm_buf, uint8_t, 8192ull);
#undef A
#undef AL
    return ok && !b->overflow;
}

size_t v3_state_arena_size_ex(const v3_model *m, uint32_t max_seg_budget) {
    if (!m || max_seg_budget > V3_MAX_SEG_BUDGET) return 0;
    bump_t b = {(uint8_t *)0 + 64, 0, 0, 0};
    state_layout(0, m, &b, max_seg_budget);
    if (b.overflow || b.used > (~(size_t)0) - 64u) return 0;
    return b.used + 64u;
}

size_t v3_state_arena_size(const v3_model *m) {
    if (!m) return 0;
    return v3_state_arena_size_ex(m, m->seg_budget);
}

static void zero_f(float *p, uint64_t n) { for (uint64_t i = 0; i < n; i++) p[i] = 0.0f; }
static void zero_u32(uint32_t *p, uint64_t n) { for (uint64_t i = 0; i < n; i++) p[i] = 0u; }

void v3_state_reset(v3_state *s) {
    const v3_model *m = s->m;
    uint64_t G = m->n_kv_heads, W = m->window, D = m->head_dim, C = m->n_cells;
    uint64_t scap = s->sink_cap > 0 ? s->sink_cap : 1ull;
    uint64_t bcap = s->seg_cap > 0 ? s->seg_cap : 1ull;
    uint64_t feat_dim = m->seg_n_feat > 0 ? (uint64_t)m->seg_n_feat * m->d_model : 1ull;
    for (uint32_t l = 0; l < m->n_layers; l++) {
        v3_layer_state *L = &s->ls[l];
        zero_f(L->k_prev, G * W * D); zero_f(L->k_cur, G * W * D);
        zero_f(L->v_prev, G * W * D); zero_f(L->v_cur, G * W * D);
        zero_f(L->kraw_prev, G * W * D); zero_f(L->kraw_cur, G * W * D);
        zero_f(L->kt_prev, G * W * D); zero_f(L->kt_cur, G * W * D);
        zero_f(L->b_prev, G * W); zero_f(L->b_cur, G * W);
        zero_f(L->M, G * C * D * D); zero_f(L->z, G * C * D);
        zero_f(L->k_sink_raw, G * scap * D); zero_f(L->k_sink, G * scap * D); zero_f(L->v_sink, G * scap * D);
        zero_f(L->k_seg_raw, G * bcap * D); zero_f(L->k_seg, G * bcap * D); zero_f(L->v_seg, G * bcap * D);
    }
    zero_u32(s->tok_prev, W); zero_u32(s->tok_cur, W);
    zero_f(s->sc_prev, W); zero_f(s->sc_cur, W);
    zero_u32(s->seg_prev, W); zero_u32(s->seg_cur, W);
    zero_f(s->seg_feat_ema, feat_dim);
    zero_f(s->seg_bias_arch, bcap + 1ull);
    zero_f(s->seg_bias_sup, bcap + 1ull);
    zero_f(s->seg_bias_hop1, bcap + 1ull);
    s->pos = 0;
    s->n_pin = m->n_sink <= s->sink_cap ? m->n_sink : s->sink_cap;
    s->pin_locked = s->auto_pin ? 0 : 1;
    s->saw_im_start = 0;
    s->awaiting_sys_nl = 0;
    s->ema_ready = 0;
    s->seg_bias_active = 0;
    s->mem_cnt = 0;
    s->n_uniq_segs = 0;
    s->seg_id_offset = 0;
    s->cur_seg_id = 0;
    s->cur_seg_len = 0;
}

int v3_state_init_ex(v3_state *s, const v3_model *m, void *arena, size_t arena_size, uint32_t max_seg_budget) {
    if (!s || !m || !arena || ((uintptr_t)arena % 64u)) return V3_E_ARENA;
    if (max_seg_budget > V3_MAX_SEG_BUDGET) return V3_E_LIMIT;
    uint8_t *sz = (uint8_t *)s;
    for (size_t i = 0; i < sizeof(*s); i++) sz[i] = 0;
    s->m = m;
    s->vault_on = 1;
    s->seg_on = (m->seg_n_feat > 0 && max_seg_budget > 0) ? 1 : 0;
    s->auto_pin = 1;
    s->seg_budget = (m->seg_n_feat > 0) ? (m->seg_budget <= max_seg_budget ? m->seg_budget : max_seg_budget) : 0u;
    if (m->seg_n_feat > 0 && m->seg_budget == 0 && max_seg_budget > 0) s->seg_budget = max_seg_budget;
    bump_t b = {(uint8_t *)arena, arena_size, 0, 0};
    if (!state_layout(s, m, &b, max_seg_budget)) return V3_E_ARENA;
    uint32_t D = m->head_dim, h = D / 2;
    double lt = v3_log_d((double)m->rope_theta);
    for (uint32_t i = 0; i < h; i++) {
        float invf = (float)(1.0 / v3_exp_d(lt * (double)(2 * i) / (double)D));
        s->inv_freq[i] = invf;
        double sn, cs;
        v3_sincos(-(double)m->window * (double)invf, &sn, &cs);
        s->cos_w[i] = (float)cs;
        s->sin_w[i] = (float)sn;
    }
    v3_state_reset(s);
    return V3_OK;
}

int v3_state_init(v3_state *s, const v3_model *m, void *arena, size_t arena_size) {
    if (!m) return V3_E_ARENA;
    return v3_state_init_ex(s, m, arena, arena_size, m->seg_budget);
}

static void refresh_prefix_rotations(v3_state *s, int seg_active) {
    const v3_model *m = s->m;
    uint32_t G = m->n_kv_heads, D = m->head_dim, hD = D / 2;
    uint32_t M_cnt = seg_active ? s->mem_cnt : 0u;
    uint32_t S = s->n_pin;
    float cs[V3_MAX_HEAD_DIM / 2], sn[V3_MAX_HEAD_DIM / 2];

    for (uint32_t j = 0; j < M_cnt; j++) {
        double p = (double)j - (double)M_cnt;
        for (uint32_t i = 0; i < hD; i++) {
            double dsn, dcs;
            v3_sincos(p * (double)s->inv_freq[i], &dsn, &dcs);
            cs[i] = (float)dcs; sn[i] = (float)dsn;
        }
        for (uint32_t l = 0; l < m->n_layers; l++) {
            v3_layer_state *L = &s->ls[l];
            for (uint32_t g = 0; g < G; g++) {
                uint64_t off = ((uint64_t)g * s->seg_cap + j) * D;
                rope_copy_rotate(L->k_seg + off, L->k_seg_raw + off, cs, sn, D);
            }
        }
    }
    for (uint32_t j = 0; j < S; j++) {
        double p = (double)j - (double)(M_cnt + S);
        for (uint32_t i = 0; i < hD; i++) {
            double dsn, dcs;
            v3_sincos(p * (double)s->inv_freq[i], &dsn, &dcs);
            cs[i] = (float)dcs; sn[i] = (float)dsn;
        }
        for (uint32_t l = 0; l < m->n_layers; l++) {
            v3_layer_state *L = &s->ls[l];
            for (uint32_t g = 0; g < G; g++) {
                uint64_t off = ((uint64_t)g * s->sink_cap + j) * D;
                rope_copy_rotate(L->k_sink + off, L->k_sink_raw + off, cs, sn, D);
            }
        }
    }
}

int v3_set_pinned(v3_state *s, uint32_t n_pin) {
    if (!s || !s->m) return V3_E_ARENA;
    if (n_pin > s->sink_cap || n_pin > s->m->window) return V3_E_LIMIT;
    s->n_pin = n_pin;
    s->auto_pin = 0;
    s->pin_locked = 1;
    if (s->pos >= 2ull * s->m->window) {
        int seg_active = (s->seg_on && s->m->seg_n_feat > 0 && s->seg_budget > 0);
        refresh_prefix_rotations(s, seg_active);
    }
    return V3_OK;
}

static void rebuild_shingle_index(v3_state *s);

int v3_set_seg_budget(v3_state *s, uint32_t budget) {
    if (!s || !s->m) return V3_E_ARENA;
    if (budget > s->seg_cap) return V3_E_LIMIT;
    s->seg_budget = budget;
    s->seg_on = (s->m->seg_n_feat > 0 && budget > 0) ? 1 : 0;
    if (!s->seg_on) s->seg_bias_active = 0;
    if (s->mem_cnt > budget) {
        s->mem_cnt = budget;
        rebuild_shingle_index(s);
    }
    if (s->pos >= 2ull * s->m->window) {
        int seg_active = (s->seg_on && s->m->seg_n_feat > 0 && s->seg_budget > 0);
        refresh_prefix_rotations(s, seg_active);
    }
    return V3_OK;
}

/* ============================================================================
 * Kenyon Vault + Segment-Evicted KV Memory
 * ========================================================================== */
static void vault_write_chunk(v3_state *s, v3_layer_state *L, uint32_t g, const float *kt, const float *v, const float *beta) {
    const v3_model *m = s->m;
    uint32_t W = m->window, D = m->head_dim, C = m->n_cells, NB = m->n_banks;
    float *M = L->M + (uint64_t)g * C * D * D, *z = L->z + (uint64_t)g * C * D;
    float eps = m->vault_eps;
    for (uint32_t t = 0; t < W; t++) {
        const float *kk = kt + (uint64_t)t * D;
        float *ph = s->w_phk + (uint64_t)t * D;
        for (uint32_t d0 = 0; d0 < D; d0++) ph[d0] = v3_phi(kk[d0]);
        uint32_t *cells = s->w_cells + (uint64_t)t * NB;
        route_cells(m, kk, cells);
        for (uint32_t b = 0; b < NB; b++) {
            const float *Mc = M + (uint64_t)cells[b] * D * D, *zc = z + (uint64_t)cells[b] * D;
            float den = dot_f32(ph, zc, D);
            float *dl = s->w_delta + ((uint64_t)t * NB + b) * D;
            for (uint32_t e = 0; e < D; e++) dl[e] = 0.0f;
            if (den > eps) {
                for (uint32_t d0 = 0; d0 < D; d0++) axpy_f32(dl, ph[d0], Mc + (uint64_t)d0 * D, D);
                float inv = 1.0f / den;
                for (uint32_t e = 0; e < D; e++) dl[e] *= inv;
            }
            const float *vv = v + (uint64_t)t * D;
            for (uint32_t e = 0; e < D; e++) dl[e] = beta[t] * (vv[e] - dl[e]);
        }
    }
    for (uint32_t t = 0; t < W; t++) {
        const float *ph = s->w_phk + (uint64_t)t * D;
        const uint32_t *cells = s->w_cells + (uint64_t)t * NB;
        for (uint32_t b = 0; b < NB; b++) {
            float *Mc = M + (uint64_t)cells[b] * D * D, *zc = z + (uint64_t)cells[b] * D;
            const float *dl = s->w_delta + ((uint64_t)t * NB + b) * D;
            for (uint32_t d0 = 0; d0 < D; d0++) {
                float a = ph[d0];
                axpy_f32(Mc + (uint64_t)d0 * D, a, dl, D);
                zc[d0] += beta[t] * a;
            }
        }
    }
    /* Homogeneous cell rescaling: M_c / z_c is scale-invariant, preventing fp32 saturation on >100M-token streams. */
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

static void vault_read(const v3_state *s, const v3_layer_state *L, uint32_t g, const float *qt, float *out, float *phq) {
    const v3_model *m = s->m;
    uint32_t D = m->head_dim, C = m->n_cells, NB = m->n_banks;
    const float *M = L->M + (uint64_t)g * C * D * D, *z = L->z + (uint64_t)g * C * D;
    uint32_t cells[V3_MAX_BANKS];
    for (uint32_t d0 = 0; d0 < D; d0++) phq[d0] = v3_phi(qt[d0]);
    route_cells(m, qt, cells);
    for (uint32_t e = 0; e < D; e++) out[e] = 0.0f;
    for (uint32_t b = 0; b < NB; b++) {
        const float *Mc = M + (uint64_t)cells[b] * D * D, *zc = z + (uint64_t)cells[b] * D;
        float den = dot_f32(phq, zc, D);
        if (!(den > m->vault_eps)) continue;
        float inv = 1.0f / den;
        for (uint32_t d0 = 0; d0 < D; d0++) axpy_f32(out, phq[d0] * inv, Mc + (uint64_t)d0 * D, D);
    }
    float invnb = 1.0f / (float)NB;
    for (uint32_t e = 0; e < D; e++) out[e] *= invnb;
}

/* 3-tier byte-level structural + JSON/pipe/TSV + CSV/query delimiter segmenter:
 *   Tier 1 (break at >= seg_min): '\n' anywhere, or trailing '.!?;' (ignoring trailing ASCII ws)
 *   Tier 2 (break at >= max(seg_min, 2*seg_max/3)): '}', ']', '|', '\t', or '",' (minified JSON / pipe / TSV)
 *   Tier 3 (break at >= max(seg_min, seg_max - 8)): ',' or '&' (CSV / URL query soft fallback before hard seg_max) */
static int seg_break_tier(const v3_model *m, uint32_t id) {
    const uint8_t *b = 0;
    uint32_t bl = 0;
    if (v3_token_bytes(m, id, &b, &bl) != V3_OK || bl == 0) return 0;
    int has_t2 = 0, has_t3 = 0;
    for (uint32_t i = 0; i < bl; i++) {
        uint8_t ch = b[i];
        if (ch == '\n') return 1;
        if (ch == '}' || ch == ']' || ch == '|' || ch == '\t') has_t2 = 1;
        else if (i + 1u < bl && ch == '"' && b[i + 1u] == ',') has_t2 = 1;
        else if (ch == ',' || ch == '&') has_t3 = 1;
    }
    uint32_t end = bl;
    while (end > 0) {
        uint8_t ch = b[end - 1u];
        if (ch == ' ' || ch == '\t' || ch == '\r' || ch == 0x0Bu || ch == 0x0Cu) end--;
        else break;
    }
    uint8_t last = (end > 0) ? b[end - 1u] : 0u;
    if (last == '.' || last == '!' || last == '?' || last == ';') return 1;
    if (has_t2) return 2;
    return has_t3 ? 3 : 0;
}

static float seg_scorer_eval_vec(v3_state *s, const float *x) {
    const v3_model *m = s->m;
    uint32_t fd = m->seg_n_feat * m->d_model, hd = m->seg_hidden;
    float mean = 0.0f;
    for (uint32_t i = 0; i < fd; i++) mean += x[i];
    mean /= (float)fd;
    float var = 0.0f;
    for (uint32_t i = 0; i < fd; i++) { float df = x[i] - mean; var += df * df; }
    float inv = 1.0f / v3_sqrtf(var / (float)fd + 1e-5f);
    const float *lw = (const float *)(const void *)m->seg_ln_w.data;
    const float *lb = (const float *)(const void *)m->seg_ln_b.data;
    for (uint32_t i = 0; i < fd; i++) s->seg_norm[i] = lw[i] * ((x[i] - mean) * inv) + lb[i];
    matvec(s, s->seg_hid, &m->seg_w1, s->seg_norm, &m->seg_b1);
    for (uint32_t i = 0; i < hd; i++) s->seg_hid[i] = v3_geluf(s->seg_hid[i]);
    const float *b2 = (const float *)(const void *)m->seg_b2.data;
    return row_dot(&m->seg_w2, 0, s->seg_hid, hd) + b2[0];
}

/* True iff candidate a is strictly WORSE than candidate b under (score DESC, idx ASC). */
static int cand_worse(const float *sc, uint32_t a, uint32_t b) {
    float sa = sc[a], sb = sc[b];
    return (sa < sb) || (sa == sb && a > b);
}

static void sift_worse(uint32_t *ord, const float *sc, uint32_t root, uint32_t n) {
    for (;;) {
        uint32_t l = 2u * root + 1u;
        if (l >= n) break;
        uint32_t best = l;
        if (l + 1u < n && cand_worse(sc, ord[l + 1u], ord[l])) best = l + 1u;
        if (!cand_worse(sc, ord[best], ord[root])) break;
        uint32_t tmp = ord[root]; ord[root] = ord[best]; ord[best] = tmp;
        root = best;
    }
}

static void sort_best_first(uint32_t *ord, const float *sc, uint32_t n) {
    if (n <= 1u) return;
    for (uint32_t i = n / 2u; i > 0u; i--) sift_worse(ord, sc, i - 1u, n);
    for (uint32_t end = n - 1u; end > 0u; end--) {
        uint32_t tmp = ord[0]; ord[0] = ord[end]; ord[end] = tmp;
        sift_worse(ord, sc, 0u, end);
    }
}

static void sift_u32_max(uint32_t *a, uint32_t root, uint32_t n) {
    for (;;) {
        uint32_t l = 2u * root + 1u;
        if (l >= n) break;
        uint32_t best = (l + 1u < n && a[l + 1u] > a[l]) ? (l + 1u) : l;
        if (a[best] <= a[root]) break;
        uint32_t tmp = a[root]; a[root] = a[best]; a[best] = tmp;
        root = best;
    }
}

static void sort_u32_asc(uint32_t *a, uint32_t n) {
    if (n <= 1u) return;
    for (uint32_t i = n / 2u; i > 0u; i--) sift_u32_max(a, i - 1u, n);
    for (uint32_t end = n - 1u; end > 0u; end--) {
        uint32_t tmp = a[0]; a[0] = a[end]; a[end] = tmp;
        sift_u32_max(a, 0u, end);
    }
}

static void copy_f32(float *dst, const float *src, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) dst[i] = src[i];
}

/* ============================================================================
 * 2-Tier Identifier + Alphabetic Shingle Resonance, Hop-2 Spreading Activation,
 * Rollback Pointer Resolution & MultiScale Supersession
 * ========================================================================== */
static uint32_t shingle_slot_hash(uint32_t k) {
    k ^= k >> 16; k *= 0x85ebca6bu; k ^= k >> 13; k *= 0xc2b2ae35u; k ^= k >> 16;
    return k;
}

static void norm_append_raw_bytes(const uint8_t *b, uint32_t bl, uint8_t *norm, uint32_t *nl, uint32_t cap) {
    uint32_t len = *nl;
    for (uint32_t i = 0; i < bl; i++) {
        uint8_t c = b[i];
        if (c >= 'A' && c <= 'Z') c = (uint8_t)(c + 32u);
        if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || c == '_' || c == '-') {
            if (len + 2u < cap) norm[len++] = c;
        } else if (norm[len - 1u] != ' ') {
            if (len + 2u < cap) norm[len++] = ' ';
        }
    }
    *nl = len;
}

static void norm_append_tok_id(const v3_model *m, uint32_t tid, uint8_t *norm, uint32_t *nl, uint32_t cap) {
    const uint8_t *b = 0;
    uint32_t bl = 0;
    if (v3_token_bytes(m, tid, &b, &bl) != V3_OK) return;
    norm_append_raw_bytes(b, bl, norm, nl, cap);
}

static int is_id_char_byte(uint8_t b) { return (b >= '0' && b <= '9') || b == '_' || b == '-'; }

static uint32_t fnv1a_word_framed(const uint8_t *p, uint32_t len) {
    uint32_t h = (2166136261u ^ 32u) * 16777619u;
    for (uint32_t i = 0; i < len; i++) h = (h ^ (uint32_t)p[i]) * 16777619u;
    return (h ^ 32u) * 16777619u;
}

static int eq_lit(const uint8_t *p, uint32_t len, const char *s, uint32_t slen) {
    if (len != slen) return 0;
    for (uint32_t i = 0; i < len; i++) if (p[i] != (uint8_t)s[i]) return 0;
    return 1;
}

static int is_alpha_stopword(const uint8_t *p, uint32_t len) {
    switch (len) {
    case 4u:
        return eq_lit(p, 4u, "what", 4u) || eq_lit(p, 4u, "with", 4u) || eq_lit(p, 4u, "only", 4u) ||
               eq_lit(p, 4u, "from", 4u) || eq_lit(p, 4u, "that", 4u) || eq_lit(p, 4u, "this", 4u) ||
               eq_lit(p, 4u, "into", 4u) || eq_lit(p, 4u, "find", 4u);
    case 5u:
        return eq_lit(p, 5u, "which", 5u) || eq_lit(p, 5u, "where", 5u) || eq_lit(p, 5u, "there", 5u) ||
               eq_lit(p, 5u, "their", 5u) || eq_lit(p, 5u, "about", 5u) || eq_lit(p, 5u, "after", 5u) ||
               eq_lit(p, 5u, "under", 5u) || eq_lit(p, 5u, "above", 5u) || eq_lit(p, 5u, "state", 5u) ||
               eq_lit(p, 5u, "reply", 5u) || eq_lit(p, 5u, "exact", 5u) || eq_lit(p, 5u, "value", 5u) ||
               eq_lit(p, 5u, "query", 5u) || eq_lit(p, 5u, "using", 5u) || eq_lit(p, 5u, "token", 5u) ||
               eq_lit(p, 5u, "first", 5u) || eq_lit(p, 5u, "other", 5u) || eq_lit(p, 5u, "these", 5u) ||
               eq_lit(p, 5u, "those", 5u) || eq_lit(p, 5u, "would", 5u) || eq_lit(p, 5u, "could", 5u);
    case 6u:
        return eq_lit(p, 6u, "before", 6u) || eq_lit(p, 6u, "string", 6u) || eq_lit(p, 6u, "single", 6u) ||
               eq_lit(p, 6u, "should", 6u) || eq_lit(p, 6u, "active", 6u);
    case 7u:
        return eq_lit(p, 7u, "current", 7u) || eq_lit(p, 7u, "through", 7u) || eq_lit(p, 7u, "between", 7u);
    case 8u:
        return eq_lit(p, 8u, "original", 8u) || eq_lit(p, 8u, "assigned", 8u);
    default:
        return 0;
    }
}

/* 2-bit tag encoding in 32-bit FNV-1a shingle hashes:
 *   (h & 3u) == 1u : Tier-A 4-byte strict-ID sub-shingle
 *   (h & 3u) == 3u : Tier-A whole-identifier / [t=N] tag hash (eligible for Hop-2 / Rollback bridging)
 *   (h & 3u) == 0u : Tier-B pure-alphabetic content whole-word hash (fallback when query has no ID hits) */
static uint32_t extract_norm_shingles(const uint8_t *norm, uint32_t nl, uint32_t *out, uint32_t max_out) {
    if (nl < 4u || max_out == 0u) return 0u;
    uint32_t cnt = 0u, i = 0u;
    while (i < nl && cnt < max_out) {
        if (norm[i] == ' ') { i++; continue; }
        uint32_t tok_l = i;
        int has_alpha = 0, has_num_us = 0, has_hyph = 0;
        while (i < nl && norm[i] != ' ') {
            uint8_t c = norm[i];
            if (c >= 'a' && c <= 'z') has_alpha = 1;
            else if ((c >= '0' && c <= '9') || c == '_') has_num_us = 1;
            else if (c == '-') has_hyph = 1;
            i++;
        }
        uint32_t tok_r = i;
        uint32_t tlen = tok_r - tok_l;
        if (tlen >= 3u && has_alpha && has_num_us) {
            int has_trail = (tok_r < nl);
            if (has_trail || tlen >= 4u) {
                uint32_t k_end = has_trail ? (tok_r - 3u) : (tok_r - 4u);
                for (uint32_t k = tok_l; k <= k_end && cnt < max_out; k++) {
                    uint8_t b0 = norm[k], b1 = norm[k + 1u], b2 = norm[k + 2u], b3 = norm[k + 3u];
                    if (!is_id_char_byte(b0) && !is_id_char_byte(b1) && !is_id_char_byte(b2) && !is_id_char_byte(b3))
                        continue;
                    uint32_t h = 2166136261u;
                    h = (h ^ (uint32_t)b0) * 16777619u;
                    h = (h ^ (uint32_t)b1) * 16777619u;
                    h = (h ^ (uint32_t)b2) * 16777619u;
                    h = (h ^ (uint32_t)b3) * 16777619u;
                    out[cnt++] = (h | 1u) & ~2u;
                }
            }
            if (cnt < max_out) out[cnt++] = fnv1a_word_framed(norm + tok_l, tlen) | 3u;
        } else if (tlen == 1u && has_alpha && !has_num_us && tok_r + 1u < nl) {
            uint32_t n_l = tok_r + 1u;
            while (n_l < nl && norm[n_l] == ' ') n_l++;
            uint32_t n_r = n_l;
            int all_dig = 1;
            while (n_r < nl && norm[n_r] != ' ') {
                if (!(norm[n_r] >= '0' && norm[n_r] <= '9')) all_dig = 0;
                n_r++;
            }
            if (all_dig && (n_r - n_l) >= 1u && (n_r - n_l) <= 3u && (n_l == tok_r + 1u) && cnt < max_out)
                out[cnt++] = fnv1a_word_framed(norm + tok_l, n_r - tok_l) | 3u;
        } else if (!has_num_us && has_alpha && tlen >= 5u) {
            if (!has_hyph) {
                if (!is_alpha_stopword(norm + tok_l, tlen) && cnt < max_out)
                    out[cnt++] = fnv1a_word_framed(norm + tok_l, tlen) & ~3u;
            } else {
                uint32_t sub_l = tok_l, n_sub = 0u;
                int all_sub_ok = 1;
                for (uint32_t k = tok_l; k <= tok_r; k++) {
                    if (k == tok_r || norm[k] == '-') {
                        uint32_t slen = k - sub_l;
                        n_sub++;
                        if (slen < 4u || is_alpha_stopword(norm + sub_l, slen)) all_sub_ok = 0;
                        if (slen >= 5u && !is_alpha_stopword(norm + sub_l, slen) && cnt < max_out)
                            out[cnt++] = fnv1a_word_framed(norm + sub_l, slen) & ~3u;
                        sub_l = k + 1u;
                    }
                }
                if (n_sub >= 2u && all_sub_ok && cnt < max_out)
                    out[cnt++] = fnv1a_word_framed(norm + tok_l, tlen) & ~3u;
            }
        }
    }
    if (cnt <= 1u) return cnt;
    sort_u32_asc(out, cnt);
    uint32_t u = 1u;
    for (uint32_t j = 1u; j < cnt; j++)
        if (out[j] != out[u - 1u]) out[u++] = out[j];
    return u;
}

static void rebuild_shingle_index(v3_state *s) {
    const v3_model *m = s->m;
    uint32_t M_cnt = s->mem_cnt;
    s->seg_bias_active = 0;
    if (M_cnt == 0u || s->sh_cap == 0u) { s->n_uniq_segs = 0u; return; }

    uint32_t N_s = 0u;
    for (uint32_t j = 0; j < M_cnt; j++) {
        if (j > 0u && s->mem_seg[j] != s->mem_seg[j - 1u]) N_s++;
        s->seg_inv[j] = N_s;
    }
    N_s++;
    s->n_uniq_segs = N_s;
    if (N_s <= 2u) return;

    uint32_t mask = s->sh_cap - 1u;
    for (uint32_t i = 0; i < s->sh_cap; i++) s->sh_keys[i] = ~0ull;
    uint32_t used = 0u;

    /* Pass 1: document frequency df[h] across distinct bank segments */
    uint32_t j = 0u;
    for (uint32_t sid = 0; sid < N_s; sid++) {
        uint32_t nl = 0u;
        s->q_norm_buf[nl++] = ' ';
        while (j < M_cnt && s->seg_inv[j] == sid) {
            norm_append_tok_id(m, s->mem_tok[j], s->q_norm_buf, &nl, 8192u);
            j++;
        }
        if (s->q_norm_buf[nl - 1u] != ' ' && nl < 8192u) s->q_norm_buf[nl++] = ' ';
        uint32_t n_sh = extract_norm_shingles(s->q_norm_buf, nl, s->q_shingles, 2048u);
        for (uint32_t k = 0; k < n_sh; k++) {
            uint64_t key = (uint64_t)s->q_shingles[k];
            uint32_t slot = shingle_slot_hash((uint32_t)key) & mask;
            while (s->sh_keys[slot] != ~0ull && s->sh_keys[slot] != key) slot = (slot + 1u) & mask;
            if (s->sh_keys[slot] == ~0ull) {
                if (used + 1u >= s->sh_cap) break;
                s->sh_keys[slot] = key;
                s->sh_df[slot] = 1u;
                s->sh_head[slot] = 0xFFFFFFFFu;
                s->sh_last_sid[slot] = sid;
                used++;
            } else if (s->sh_last_sid[slot] != sid) {
                s->sh_df[slot]++;
                s->sh_last_sid[slot] = sid;
            }
        }
    }

    /* Pass 2: store postings for selective Tier-A IDs and Tier-B content words */
    uint32_t n_post = 0u;
    j = 0u;
    for (uint32_t sid = 0; sid < N_s; sid++) {
        uint32_t nl = 0u;
        s->q_norm_buf[nl++] = ' ';
        while (j < M_cnt && s->seg_inv[j] == sid) {
            norm_append_tok_id(m, s->mem_tok[j], s->q_norm_buf, &nl, 8192u);
            j++;
        }
        if (s->q_norm_buf[nl - 1u] != ' ' && nl < 8192u) s->q_norm_buf[nl++] = ' ';
        uint32_t n_sh = extract_norm_shingles(s->q_norm_buf, nl, s->q_shingles, 2048u);
        for (uint32_t k = 0; k < n_sh; k++) {
            uint64_t key = (uint64_t)s->q_shingles[k];
            uint32_t slot = shingle_slot_hash((uint32_t)key) & mask;
            while (s->sh_keys[slot] != ~0ull && s->sh_keys[slot] != key) slot = (slot + 1u) & mask;
            if (s->sh_keys[slot] == ~0ull || n_post >= s->post_cap) continue;
            uint32_t df = s->sh_df[slot];
            int is_id = ((key & 1ull) != 0ull);
            int keep = is_id
                ? ((((key & 3ull) == 3ull) && df == 2u) || ((uint64_t)df * 4ull <= (uint64_t)N_s + 6ull))
                : (df <= 16u && ((uint64_t)df * 5ull <= (uint64_t)N_s * 2ull + 4ull));
            if (keep) {
                double lg = v3_log_d(((double)N_s + 1.0) / ((double)df + 0.5));
                double mult = ((key & 3ull) == 3ull) ? 2048.0 : (is_id ? 1024.0 : 768.0);
                uint32_t w_int = (uint32_t)(mult * lg * lg + 0.5);
                s->post_sid[n_post] = sid;
                s->post_w[n_post] = w_int;
                s->post_next[n_post] = s->sh_head[slot];
                s->sh_head[slot] = n_post;
                n_post++;
            }
        }
    }
}

static int token_has_nl(const v3_model *m, uint32_t tid) {
    const uint8_t *b = 0;
    uint32_t bl = 0;
    if (v3_token_bytes(m, tid, &b, &bl) != V3_OK) return 0;
    for (uint32_t i = 0; i < bl; i++) if (b[i] == '\n') return 1;
    return 0;
}

static uint32_t ring_tok_at(const v3_state *s, uint32_t W, uint32_t idx) {
    return (idx < W) ? s->tok_prev[idx] : s->tok_cur[idx - W];
}

static int sh_in_sorted(const uint32_t *a, uint32_t n, uint32_t x) {
    uint32_t lo = 0u, hi = n;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2u;
        if (a[mid] == x) return 1;
        if (a[mid] < x) lo = mid + 1u;
        else hi = mid;
    }
    return 0;
}

static int32_t find_ascii_sub(const uint8_t *hay, uint32_t hlen, const char *needle, uint32_t nlen) {
    if (hlen < nlen || nlen == 0u) return -1;
    for (uint32_t i = 0; i + nlen <= hlen; i++) {
        uint32_t k = 0;
        while (k < nlen && hay[i + k] == (uint8_t)needle[k]) k++;
        if (k == nlen) return (int32_t)i;
    }
    return -1;
}

static int32_t find_rollback_cue_pos(const uint8_t *raw_lc, uint32_t rlen) {
    int32_t p = find_ascii_sub(raw_lc, rlen, "rollback", 8u);
    if (p >= 0) return p;
    p = find_ascii_sub(raw_lc, rlen, "revert", 6u);
    if (p >= 0) return p;
    p = find_ascii_sub(raw_lc, rlen, "restore", 7u);
    if (p >= 0) return p;
    p = find_ascii_sub(raw_lc, rlen, "roll ", 5u);
    if (p >= 0 && find_ascii_sub(raw_lc + (uint32_t)p, rlen - (uint32_t)p, " back", 5u) >= 0) return p;
    return -1;
}

static void refine_matched_subclauses(
    v3_state *s, uint32_t M_cnt, uint32_t n_m, uint32_t n_q, int used_tier_a, float sib_val_atten
) {
    const v3_model *m = s->m;
    uint32_t mask = s->sh_cap - 1u;
    int has_active_q = 0;
    for (uint32_t qi = 0; qi < n_q; qi++) {
        uint32_t qh = s->q_shingles[qi];
        int in_tier = used_tier_a ? ((qh & 1u) != 0u) : ((qh & 3u) == 0u);
        if (!in_tier) continue;
        uint32_t slot = shingle_slot_hash(qh) & mask;
        while (s->sh_keys[slot] != ~0ull && s->sh_keys[slot] != (uint64_t)qh) slot = (slot + 1u) & mask;
        if (s->sh_keys[slot] != ~0ull && s->sh_head[slot] != 0xFFFFFFFFu) { has_active_q = 1; break; }
    }
    if (!has_active_q) return;

    uint32_t cl_beg[32], cl_end[32], cl_sc[32];
    uint8_t cl_ok[32];
    uint32_t *csh = s->q_shingles + 1024u;
    uint8_t *raw_cl = s->q_norm_buf + 2048u;

    for (uint32_t mi = 0; mi < n_m; mi++) {
        uint32_t sid = s->q_matched[mi];
        uint32_t k_beg = M_cnt, k_end = 0u;
        for (uint32_t k = 0; k < M_cnt; k++) {
            if (s->seg_inv[k] == sid) {
                if (k < k_beg) k_beg = k;
                k_end = k + 1u;
            }
        }
        if (k_end <= k_beg || (k_end - k_beg) < 8u) continue;

        uint32_t n_cl = 0u, c0 = k_beg;
        for (uint32_t k = k_beg; k < k_end; k++) {
            const uint8_t *b = 0; uint32_t bl = 0;
            int is_delim = 0;
            if (v3_token_bytes(m, s->mem_tok[k], &b, &bl) == V3_OK) {
                for (uint32_t bi = 0; bi < bl; bi++) {
                    uint8_t ch = b[bi];
                    if (ch == '\n' || ch == '|' || ch == '\t' || ch == ';') { is_delim = 1; break; }
                }
            }
            if (is_delim && (k - c0 + 1u) >= 4u && (k_end - (k + 1u)) >= 4u && n_cl + 1u < 32u) {
                cl_beg[n_cl] = c0;
                cl_end[n_cl] = k + 1u;
                n_cl++;
                c0 = k + 1u;
            }
        }
        if (c0 < k_end && n_cl < 32u) {
            cl_beg[n_cl] = c0;
            cl_end[n_cl] = k_end;
            n_cl++;
        }
        if (n_cl < 2u) continue;

        uint32_t mx_c = 0u;
        int any_sub_ok = 0;
        for (uint32_t ci = 0; ci < n_cl; ci++) {
            uint32_t cnl = 0u;
            s->q_norm_buf[cnl++] = ' ';
            for (uint32_t k = cl_beg[ci]; k < cl_end[ci]; k++)
                norm_append_tok_id(m, s->mem_tok[k], s->q_norm_buf, &cnl, 2048u);
            if (s->q_norm_buf[cnl - 1u] != ' ' && cnl < 2048u) s->q_norm_buf[cnl++] = ' ';
            uint32_t n_csh = extract_norm_shingles(s->q_norm_buf, cnl, csh, 512u);
            uint32_t n_tier_sh = 0u, sc = 0u;
            for (uint32_t si = 0; si < n_csh; si++) {
                uint32_t sh = csh[si];
                int in_tier = used_tier_a ? ((sh & 1u) != 0u) : ((sh & 3u) == 0u);
                if (!in_tier) continue;
                n_tier_sh++;
                if (!sh_in_sorted(s->q_shingles, n_q, sh)) continue;
                uint32_t slot = shingle_slot_hash(sh) & mask;
                while (s->sh_keys[slot] != ~0ull && s->sh_keys[slot] != (uint64_t)sh) slot = (slot + 1u) & mask;
                if (s->sh_keys[slot] == ~0ull) continue;
                for (uint32_t p = s->sh_head[slot]; p != 0xFFFFFFFFu; p = s->post_next[p]) {
                    if (s->post_sid[p] == sid) { sc += s->post_w[p]; break; }
                }
            }
            uint32_t rcl = 0u;
            for (uint32_t k = cl_beg[ci]; k < cl_end[ci]; k++) {
                const uint8_t *b = 0; uint32_t bl = 0;
                if (v3_token_bytes(m, s->mem_tok[k], &b, &bl) == V3_OK) {
                    for (uint32_t bi = 0; bi < bl && rcl < 2048u; bi++) raw_cl[rcl++] = b[bi];
                }
            }
            while (rcl > 0u) {
                uint8_t ch = raw_cl[rcl - 1u];
                if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n' || ch == '|' || ch == ',' || ch == ';') rcl--;
                else break;
            }
            int has_kv = 0;
            if (rcl > 2u) {
                for (uint32_t bi = 0; bi + 1u < rcl; bi++) {
                    if (raw_cl[bi] == '=' || raw_cl[bi] == ':') { has_kv = 1; break; }
                }
            }
            uint32_t clen = cl_end[ci] - cl_beg[ci];
            cl_sc[ci] = sc;
            cl_ok[ci] = (uint8_t)(((clen >= 9u || has_kv) && n_tier_sh >= 2u) ? 1u : 0u);
            if (sc > mx_c) mx_c = sc;
        }
        if (mx_c > 0u) {
            for (uint32_t ci = 0; ci < n_cl; ci++)
                if (cl_sc[ci] < mx_c && cl_ok[ci]) any_sub_ok = 1;
            if (any_sub_ok) {
                for (uint32_t ci = 0; ci < n_cl; ci++) {
                    if ((uint64_t)cl_sc[ci] * 4ull < (uint64_t)mx_c * 3ull && cl_ok[ci]) {
                        for (uint32_t k = cl_beg[ci]; k < cl_end[ci]; k++) {
                            s->seg_bias_arch[k] = sib_val_atten;
                            s->seg_bias_sup[k] = sib_val_atten;
                        }
                    }
                }
            }
        }
    }
}

static void update_query_resonance(v3_state *s, uint32_t c) {
    const v3_model *m = s->m;
    uint32_t M_cnt = s->mem_cnt, N_s = s->n_uniq_segs, W = m->window;
    s->seg_bias_active = 0;
    if (M_cnt == 0u || N_s <= 2u || s->sh_cap == 0u) return;

    uint32_t hi = W + c;
    uint32_t search_lo = (hi >= 127u) ? (hi - 127u) : 0u;
    int32_t p_end = -1;
    for (int32_t p = (int32_t)hi; p >= (int32_t)search_lo; p--) {
        if (ring_tok_at(s, W, (uint32_t)p) == m->im_end_id) {
            p_end = p;
            break;
        }
    }
    if (p_end <= 0) return;
    uint32_t gen_dist = hi - (uint32_t)p_end;
    if (gen_dist >= 38u) return;
    if (s->pos - (uint64_t)gen_dist < (uint64_t)s->n_pin) return;
    float atten = (gen_dist <= 18u) ? 1.0f : ((float)(38u - gen_dist) * 0.05f);

    uint32_t scan_from = (uint32_t)p_end - 1u;
    uint32_t q_stop = (scan_from >= 48u) ? (scan_from - 48u) : 0u;
    uint32_t lo_t = q_stop;
    for (int32_t p = (int32_t)scan_from; p >= (int32_t)q_stop; p--) {
        uint32_t tid = ring_tok_at(s, W, (uint32_t)p);
        if (tid == m->im_start_id) {
            lo_t = (uint32_t)p + 1u;
            break;
        }
        if ((uint32_t)p + 4u <= scan_from && token_has_nl(m, tid)) {
            lo_t = (uint32_t)p + 1u;
            break;
        }
    }
    if (lo_t > scan_from) return;

    uint32_t nl = 0u;
    s->q_norm_buf[nl++] = ' ';
    for (uint32_t t = lo_t; t <= scan_from; t++)
        norm_append_tok_id(m, ring_tok_at(s, W, t), s->q_norm_buf, &nl, 4096u);
    if (s->q_norm_buf[nl - 1u] != ' ' && nl < 4096u) s->q_norm_buf[nl++] = ' ';

    uint32_t n_q = extract_norm_shingles(s->q_norm_buf, nl, s->q_shingles, 1024u);
    if (n_q < 2u) return;

    uint32_t n_q_id = 0u, n_q_alpha = 0u;
    for (uint32_t qi = 0; qi < n_q; qi++) {
        if (s->q_shingles[qi] & 1u) n_q_id++;
        else n_q_alpha++;
    }

    uint32_t mask = s->sh_cap - 1u;
    uint32_t mx = 0u, n_m = 0u, max_hits = 0u;
    int used_tier_a = 0;

    /* Tier A: strict identifier shingles take priority whenever the query contains ID tokens */
    const uint64_t whole_flag = 1ull << 63;
    const uint64_t qbits_mask = whole_flag - 1ull;
    if (n_q_id >= 2u) {
        for (uint32_t sid = 0; sid < N_s; sid++) { s->q_scores[sid] = 0u; s->q_hits[sid] = 0u; s->q_mask[sid] = 0ull; }
        uint32_t bit_i = 0u;
        for (uint32_t qi = 0; qi < n_q; qi++) {
            uint32_t kh = s->q_shingles[qi];
            if ((kh & 1u) == 0u) continue;
            uint64_t bit = (1ull << (bit_i % 63u)) | (((kh & 3u) == 3u) ? whole_flag : 0ull);
            bit_i++;
            uint64_t key = (uint64_t)kh;
            uint32_t slot = shingle_slot_hash(kh) & mask;
            while (s->sh_keys[slot] != ~0ull && s->sh_keys[slot] != key) slot = (slot + 1u) & mask;
            if (s->sh_keys[slot] == ~0ull) continue;
            for (uint32_t p = s->sh_head[slot]; p != 0xFFFFFFFFu; p = s->post_next[p]) {
                uint32_t sid = s->post_sid[p];
                uint32_t ns = s->q_scores[sid] + s->post_w[p];
                s->q_scores[sid] = ns;
                s->q_hits[sid]++;
                s->q_mask[sid] |= bit;
                if (ns > mx) mx = ns;
            }
        }
        if (mx >= 2048u) {
            uint32_t n_at_mx = 0u;
            uint64_t best_whole = 0ull, best_qbits = 0ull;
            for (uint32_t sid = 0; sid < N_s; sid++) {
                if (s->q_hits[sid] >= 2u && s->q_scores[sid] == mx) {
                    n_at_mx++;
                    best_whole |= (s->q_mask[sid] & whole_flag);
                    best_qbits |= (s->q_mask[sid] & qbits_mask);
                }
            }
            for (uint32_t sid = 0; sid < N_s; sid++) {
                if (s->q_hits[sid] >= 2u && (uint64_t)s->q_scores[sid] * 2ull >= (uint64_t)mx) {
                    uint64_t ms = s->q_mask[sid];
                    if (n_at_mx == 1u && best_whole != 0ull && (ms & whole_flag) == 0ull &&
                        ((ms & qbits_mask) & ~best_qbits) == 0ull)
                        continue;
                    s->q_matched[n_m++] = sid;
                }
            }
            if (n_m > 0u) used_tier_a = 1;
        }
    }

    /* Tier B: pure-alphabetic content whole-word fallback when Tier A yields zero matches */
    if (n_m == 0u && n_q_alpha >= 2u) {
        for (uint32_t sid = 0; sid < N_s; sid++) { s->q_scores[sid] = 0u; s->q_hits[sid] = 0u; s->q_mask[sid] = 0ull; }
        mx = 0u;
        uint32_t bit_i = 0u;
        for (uint32_t qi = 0; qi < n_q; qi++) {
            uint32_t kh = s->q_shingles[qi];
            if ((kh & 3u) != 0u) continue;
            uint64_t bit = 1ull << (bit_i % 63u);
            bit_i++;
            uint64_t key = (uint64_t)kh;
            uint32_t slot = shingle_slot_hash(kh) & mask;
            while (s->sh_keys[slot] != ~0ull && s->sh_keys[slot] != key) slot = (slot + 1u) & mask;
            if (s->sh_keys[slot] == ~0ull) continue;
            for (uint32_t p = s->sh_head[slot]; p != 0xFFFFFFFFu; p = s->post_next[p]) {
                uint32_t sid = s->post_sid[p];
                uint32_t ns = s->q_scores[sid] + s->post_w[p];
                s->q_scores[sid] = ns;
                s->q_hits[sid]++;
                s->q_mask[sid] |= bit;
                if (ns > mx) mx = ns;
            }
        }
        if (mx >= 2500u) {
            uint32_t n_at_mx = 0u, hits_at_mx = 0u;
            uint64_t best_qbits = 0ull;
            for (uint32_t sid = 0; sid < N_s; sid++) {
                if (s->q_hits[sid] >= 2u && s->q_scores[sid] == mx) {
                    n_at_mx++;
                    best_qbits |= (s->q_mask[sid] & qbits_mask);
                    if (s->q_hits[sid] > hits_at_mx) hits_at_mx = s->q_hits[sid];
                }
            }
            for (uint32_t sid = 0; sid < N_s; sid++) {
                int cond = (hits_at_mx >= 3u)
                    ? ((uint64_t)s->q_scores[sid] * 4ull >= (uint64_t)mx * 3ull)
                    : ((uint64_t)s->q_scores[sid] * 2ull >= (uint64_t)mx);
                if (s->q_hits[sid] >= 2u && cond) {
                    uint64_t ms = s->q_mask[sid] & qbits_mask;
                    if (n_at_mx == 1u && s->q_scores[sid] < mx && (ms & ~best_qbits) == 0ull)
                        continue;
                    s->q_matched[n_m++] = sid;
                    if (s->q_hits[sid] > max_hits) max_hits = s->q_hits[sid];
                }
            }
        }
    }
    if (n_m == 0u) return;

    float bg = 0.0f;
    if (!used_tier_a && max_hits >= 3u) {
        float div_shift = (M_cnt > 1024u) ? (float)v3_log_d((double)M_cnt * (1.0 / 1024.0)) : 0.0f;
        bg = -(1.0f + div_shift);
    }
    for (uint32_t sid = 0; sid < N_s; sid++) {
        s->seg_bias_arch[sid] = bg;
        s->seg_bias_sup[sid] = bg;
        s->seg_bias_hop1[sid] = 0.0f;
    }

    int mode = 1;
    uint32_t s_pivot = s->q_matched[n_m - 1u];
    uint8_t *raw_lc = s->q_norm_buf + 4096u;
    uint32_t *piv_sh = s->q_shingles + 1024u;
    uint32_t rlen = 0u;
    for (uint32_t j = 0; j < M_cnt; j++) {
        if (s->seg_inv[j] != s_pivot) continue;
        const uint8_t *b = 0; uint32_t bl = 0;
        if (v3_token_bytes(m, s->mem_tok[j], &b, &bl) == V3_OK) {
            for (uint32_t bi = 0; bi < bl && rlen < 4096u; bi++) {
                uint8_t ch = b[bi];
                raw_lc[rlen++] = (ch >= 'A' && ch <= 'Z') ? (uint8_t)(ch + 32u) : ch;
            }
        }
    }

    /* Case 1: Rollback pointer resolution ("rollback / revert / restore / roll ... back" -> [t=K]) */
    int32_t cue_pos = find_rollback_cue_pos(raw_lc, rlen);
    int32_t s_rb = -1;
    if (cue_pos >= 0 && s_pivot > 0u) {
        int32_t to_rel = find_ascii_sub(raw_lc + (uint32_t)cue_pos, rlen - (uint32_t)cue_pos, " to ", 4u);
        uint32_t starts[2];
        uint32_t n_st = 0u;
        if (to_rel >= 0) starts[n_st++] = (uint32_t)(cue_pos + to_rel);
        starts[n_st++] = (uint32_t)cue_pos;
        for (uint32_t si = 0; si < n_st && s_rb < 0; si++) {
            uint32_t pnl = 0u;
            s->q_norm_buf[pnl++] = ' ';
            norm_append_raw_bytes(raw_lc + starts[si], rlen - starts[si], s->q_norm_buf, &pnl, 4096u);
            if (s->q_norm_buf[pnl - 1u] != ' ' && pnl < 4096u) s->q_norm_buf[pnl++] = ' ';
            uint32_t n_piv = extract_norm_shingles(s->q_norm_buf, pnl, piv_sh, 1024u);
            for (uint32_t pi = 0; pi < n_piv; pi++) {
                uint32_t kh = piv_sh[pi];
                if ((kh & 3u) != 3u || sh_in_sorted(s->q_shingles, n_q, kh)) continue;
                uint64_t key = (uint64_t)kh;
                uint32_t slot = shingle_slot_hash(kh) & mask;
                while (s->sh_keys[slot] != ~0ull && s->sh_keys[slot] != key) slot = (slot + 1u) & mask;
                if (s->sh_keys[slot] == ~0ull || s->sh_df[slot] > 3u) continue;
                for (uint32_t p = s->sh_head[slot]; p != 0xFFFFFFFFu; p = s->post_next[p]) {
                    uint32_t csid = s->post_sid[p];
                    if (csid < s_pivot && (int32_t)csid > s_rb) s_rb = (int32_t)csid;
                }
            }
        }
    }

    /* Case 2: Bounded 1-step Hop-2 relational join when Tier-A Hop-1 isolates s0 and s0 has a df==2 bridge ID */
    uint32_t hop2[2] = {0u, 0u};
    uint32_t n_hop2 = 0u;
    if (s_rb < 0 && n_m == 1u && used_tier_a) {
        uint32_t s0 = s->q_matched[0];
        uint32_t pnl = 0u;
        s->q_norm_buf[pnl++] = ' ';
        norm_append_raw_bytes(raw_lc, rlen, s->q_norm_buf, &pnl, 4096u);
        if (s->q_norm_buf[pnl - 1u] != ' ' && pnl < 4096u) s->q_norm_buf[pnl++] = ' ';
        uint32_t n_piv = extract_norm_shingles(s->q_norm_buf, pnl, piv_sh, 1024u);
        int overflow_h2 = 0;
        for (uint32_t pi = 0; pi < n_piv; pi++) {
            uint32_t kh = piv_sh[pi];
            if ((kh & 3u) != 3u || sh_in_sorted(s->q_shingles, n_q, kh)) continue;
            uint64_t key = (uint64_t)kh;
            uint32_t slot = shingle_slot_hash(kh) & mask;
            while (s->sh_keys[slot] != ~0ull && s->sh_keys[slot] != key) slot = (slot + 1u) & mask;
            if (s->sh_keys[slot] == ~0ull || s->sh_df[slot] != 2u) continue;
            for (uint32_t p = s->sh_head[slot]; p != 0xFFFFFFFFu; p = s->post_next[p]) {
                uint32_t csid = s->post_sid[p];
                if (csid == s0) continue;
                int dup = 0;
                for (uint32_t hi2 = 0; hi2 < n_hop2; hi2++) if (hop2[hi2] == csid) dup = 1;
                if (!dup) {
                    if (n_hop2 < 2u) hop2[n_hop2++] = csid;
                    else overflow_h2 = 1;
                }
            }
        }
        if (overflow_h2) n_hop2 = 0u;
    }

    uint32_t pin_thr = (m->n_sink > 16u) ? m->n_sink : 16u;
    int has_sys_pin = (s->n_pin > pin_thr);
    if (s_rb >= 0) {
        mode = 2;
        for (uint32_t sid = 0; sid < N_s; sid++) { s->seg_bias_arch[sid] = 0.0f; s->seg_bias_sup[sid] = 0.0f; }
        uint32_t rb = (uint32_t)s_rb;
        for (uint32_t mi = 0; mi < n_m; mi++) {
            uint32_t sid = s->q_matched[mi];
            s->seg_bias_hop1[sid] = -0.3f;
            s->seg_bias_arch[sid] = 0.4f;
            s->seg_bias_sup[sid] = -0.5f;
        }
        s->seg_bias_hop1[s_pivot] = 2.0f; s->seg_bias_hop1[rb] = 2.0f;
        s->seg_bias_arch[s_pivot] = 1.2f; s->seg_bias_arch[rb] = 2.2f;
        s->seg_bias_sup[s_pivot] = 0.8f;  s->seg_bias_sup[rb] = 2.8f;
    } else if (n_hop2 > 0u) {
        mode = 2;
        for (uint32_t sid = 0; sid < N_s; sid++) { s->seg_bias_arch[sid] = 0.0f; s->seg_bias_sup[sid] = 0.0f; }
        uint32_t s0 = s->q_matched[0];
        s->seg_bias_hop1[s0] = 2.4f;
        s->seg_bias_arch[s0] = 1.0f;
        s->seg_bias_sup[s0] = 0.6f;
        for (uint32_t hi2 = 0; hi2 < n_hop2; hi2++) {
            uint32_t s2 = hop2[hi2];
            s->seg_bias_hop1[s2] = 1.0f;
            s->seg_bias_arch[s2] = 2.4f;
            s->seg_bias_sup[s2] = 2.8f;
        }
    } else if (n_m == 1u) {
        uint32_t s0 = s->q_matched[0];
        if (!used_tier_a) {
            float ba = (max_hits >= 3u) ? 2.2f : 0.5f;
            float bs = (max_hits >= 3u) ? 2.5f : 0.7f;
            if (has_sys_pin) {
                if (ba > 1.5f) ba = 1.5f;
                if (bs > 1.9f) bs = 1.9f;
            }
            s->seg_bias_arch[s0] = ba;
            s->seg_bias_sup[s0] = bs;
        } else {
            s->seg_bias_arch[s0] = has_sys_pin ? 1.5f : 2.2f;
            s->seg_bias_sup[s0] = has_sys_pin ? 1.9f : 2.5f;
        }
    } else {
        uint32_t s_last = s->q_matched[n_m - 1u];
        uint64_t last_mask = s->q_mask[s_last] & qbits_mask;
        float inv_mx = 1.0f / (float)mx;
        float scale_tb = (used_tier_a || max_hits >= 3u) ? 1.0f : 0.5f;
        for (uint32_t mi = 0; mi < n_m; mi++) {
            uint32_t sid = s->q_matched[mi];
            float rel = (float)s->q_scores[sid] * inv_mx * scale_tb;
            if (sid == s_last) {
                s->seg_bias_arch[sid] = 1.5f * rel;
                s->seg_bias_sup[sid] = 2.5f * rel;
            } else if ((s->q_mask[sid] & last_mask) != 0ull) {
                s->seg_bias_arch[sid] = 1.5f * rel;
                s->seg_bias_sup[sid] = -0.4f;
            } else {
                s->seg_bias_arch[sid] = 1.8f * rel;
                s->seg_bias_sup[sid] = 2.0f * rel;
            }
        }
    }

    /* Expand segment biases to token slots in-place (seg_inv[k] <= k always holds) with post-retrieval annealing */
    for (uint32_t k = M_cnt; k > 0u; k--) {
        uint32_t sid = s->seg_inv[k - 1u];
        s->seg_bias_arch[k - 1u] = s->seg_bias_arch[sid] * atten;
        s->seg_bias_sup[k - 1u] = s->seg_bias_sup[sid] * atten;
        if (mode == 2) s->seg_bias_hop1[k - 1u] = s->seg_bias_hop1[sid] * atten;
    }
    if (mode == 1 && n_m <= 2u) {
        float sib_val = (bg < 0.0f) ? (bg * 0.5f) : 0.0f;
        refine_matched_subclauses(s, M_cnt, n_m, n_q, used_tier_a, sib_val * atten);
    }
    /* O(1) Orphan-Value Continuation Halo across newline segment split */
    if (mode == 1 && n_m == 1u) {
        uint32_t s0 = s->q_matched[0];
        if (s0 + 1u < N_s) {
            uint32_t k = 0u;
            while (k < M_cnt && s->seg_inv[k] < s0) k++;
            uint32_t ks0_beg = k;
            while (k < M_cnt && s->seg_inv[k] == s0) k++;
            uint32_t ks0_end = k;
            uint32_t ks1_beg = k;
            while (k < M_cnt && s->seg_inv[k] == s0 + 1u) k++;
            uint32_t ks1_end = k;
            if (ks0_end > ks0_beg && ks1_end > ks1_beg && s->mem_seg[ks1_beg] == s->mem_seg[ks0_beg] + 1u) {
                uint32_t ks0_last = ks0_end - 1u;
                if (s->seg_bias_arch[ks0_last] > 0.0f && token_has_nl(m, s->mem_tok[ks0_last])) {
                    uint32_t lim1 = ks1_end - ks1_beg;
                    if (lim1 > 10u) lim1 = 10u;
                    int32_t nl_idx = -1;
                    for (uint32_t j = 0u; j < lim1; j++) {
                        if (token_has_nl(m, s->mem_tok[ks1_beg + j])) { nl_idx = (int32_t)j; break; }
                    }
                    if (nl_idx >= 0) {
                        uint32_t pnl = 0u;
                        s->q_norm_buf[pnl++] = ' ';
                        for (uint32_t j = 0u; j <= (uint32_t)nl_idx; j++)
                            norm_append_tok_id(m, s->mem_tok[ks1_beg + j], s->q_norm_buf, &pnl, 512u);
                        if (s->q_norm_buf[pnl - 1u] != ' ' && pnl < 512u) s->q_norm_buf[pnl++] = ' ';
                        uint32_t n_psh = extract_norm_shingles(s->q_norm_buf, pnl, piv_sh, 64u);
                        uint32_t n_aw = 0u;
                        for (uint32_t pi = 0u; pi < n_psh; pi++) if ((piv_sh[pi] & 3u) == 0u) n_aw++;
                        if (n_aw < 2u) {
                            float ba0 = s->seg_bias_arch[ks0_last];
                            float bs0 = s->seg_bias_sup[ks0_last];
                            for (uint32_t j = 0u; j <= (uint32_t)nl_idx; j++) {
                                s->seg_bias_arch[ks1_beg + j] = ba0;
                                s->seg_bias_sup[ks1_beg + j] = bs0;
                            }
                        }
                    }
                }
            }
        }
    }
    s->seg_bias_active = mode;
}

static int buf_contains_ascii_lc(const uint8_t *hay, uint32_t hlen, const char *needle, uint32_t nlen) {
    if (hlen < nlen || nlen == 0u) return 0;
    for (uint32_t i = 0; i + nlen <= hlen; i++) {
        uint32_t k = 0;
        while (k < nlen && hay[i + k] == (uint8_t)needle[k]) k++;
        if (k == nlen) return 1;
    }
    return 0;
}

static int check_spoof_quarantine(v3_state *s, uint64_t pos, uint32_t c) {
    const v3_model *m = s->m;
    if (s->n_pin <= m->n_sink || pos < (uint64_t)s->n_pin) return 0;
    uint32_t W = m->window;
    uint64_t post_pin = pos - (uint64_t)s->n_pin + 1ull;
    uint32_t lb = post_pin < 24ull ? (uint32_t)post_pin : 24u;
    uint32_t avail_cur = c + 1u;
    uint32_t avail_tot = (pos >= W) ? (W + avail_cur) : avail_cur;
    if (lb > avail_tot) lb = avail_tot;
    uint32_t need_prev = (avail_cur < lb) ? (lb - avail_cur) : 0u;
    uint32_t cur_start = (avail_cur > lb) ? (avail_cur - lb) : 0u;

    uint32_t nl = 0u;
    for (uint32_t t = W - need_prev; t < W; t++) {
        const uint8_t *b = 0; uint32_t bl = 0;
        if (v3_token_bytes(m, s->tok_prev[t], &b, &bl) == V3_OK) {
            for (uint32_t i = 0; i < bl && nl < 4096u; i++) {
                uint8_t ch = b[i];
                s->q_norm_buf[nl++] = (ch >= 'A' && ch <= 'Z') ? (uint8_t)(ch + 32u) : ch;
            }
        }
    }
    for (uint32_t t = cur_start; t <= c; t++) {
        const uint8_t *b = 0; uint32_t bl = 0;
        if (v3_token_bytes(m, s->tok_cur[t], &b, &bl) == V3_OK) {
            for (uint32_t i = 0; i < bl && nl < 4096u; i++) {
                uint8_t ch = b[i];
                s->q_norm_buf[nl++] = (ch >= 'A' && ch <= 'Z') ? (uint8_t)(ch + 32u) : ch;
            }
        }
    }
    return buf_contains_ascii_lc(s->q_norm_buf, nl, "opi_signed_directive", 20u) ||
           buf_contains_ascii_lc(s->q_norm_buf, nl, "urgent opi delegation", 21u);
}

static void update_segment_bank(v3_state *s, uint64_t ev_chunk_idx) {
    const v3_model *m = s->m;
    uint32_t W = m->window, G = m->n_kv_heads, D = m->head_dim;
    uint32_t n_loc = 0;
    for (uint32_t t = 0; t < W; t++)
        if (s->seg_prev[t] + 1u > n_loc) n_loc = s->seg_prev[t] + 1u;
    for (uint32_t sg = 0; sg < n_loc; sg++) s->seg_mx[sg] = -3.4e38f;
    for (uint32_t t = 0; t < W; t++) {
        uint32_t sg = s->seg_prev[t];
        if (s->sc_prev[t] > s->seg_mx[sg]) s->seg_mx[sg] = s->sc_prev[t];
    }

    uint32_t t_start = (ev_chunk_idx == 0ull) ? s->n_pin : 0u;
    uint32_t base_off = s->seg_id_offset;
    s->seg_id_offset += n_loc;
    if (t_start >= W) return;

    uint32_t M_old = s->mem_cnt;
    uint32_t tot = M_old;
    for (uint32_t i = 0; i < M_old; i++) {
        s->cand_pos[i] = s->mem_pos[i];
        s->cand_score[i] = s->mem_score[i];
        s->cand_tok[i] = s->mem_tok[i];
        s->cand_seg[i] = s->mem_seg[i];
        s->cand_order[i] = i;
    }
    for (uint32_t t = t_start; t < W; t++) {
        float smx = s->seg_mx[s->seg_prev[t]];
        if (smx <= -1.0e29f) continue;
        uint32_t full_seg = s->seg_prev[t] + base_off;
        uint32_t g_seg = full_seg & 16383u;
        uint32_t idx = tot++;
        s->cand_pos[idx] = ev_chunk_idx * (uint64_t)W + (uint64_t)t;
        s->cand_score[idx] = smx - 1.0e-6f * (float)g_seg;
        s->cand_tok[idx] = s->tok_prev[t];
        s->cand_seg[idx] = full_seg;
        s->cand_order[idx] = idx;
    }

    uint32_t K = tot < s->seg_budget ? tot : s->seg_budget;
    if (tot > s->seg_budget) {
        sort_best_first(s->cand_order, s->cand_score, tot);
        sort_u32_asc(s->cand_order, K);
    }

    uint32_t n_keep = 0, n_add = 0;
    uint64_t ev_base = ev_chunk_idx * (uint64_t)W;
    for (uint32_t j = 0; j < K; j++) {
        uint32_t idx = s->cand_order[j];
        s->mem_pos[j] = s->cand_pos[idx];
        s->mem_score[j] = s->cand_score[idx];
        s->mem_tok[j] = s->cand_tok[idx];
        s->mem_seg[j] = s->cand_seg[idx];
        if (idx < M_old) s->keep_old[n_keep++] = idx;
        else s->add_cand[n_add++] = (uint32_t)(s->cand_pos[idx] - ev_base);
    }
    s->mem_cnt = K;

    /* In-place chronological KV compaction (j <= keep_old[j] always holds) + append newly admitted chunk entries */
    for (uint32_t l = 0; l < m->n_layers; l++) {
        v3_layer_state *L = &s->ls[l];
        for (uint32_t g = 0; g < G; g++) {
            float *ks = L->k_seg_raw + (uint64_t)g * s->seg_cap * D;
            float *vs = L->v_seg + (uint64_t)g * s->seg_cap * D;
            const float *kp = L->kraw_prev + (uint64_t)g * W * D;
            const float *vp = L->v_prev + (uint64_t)g * W * D;
            for (uint32_t j = 0; j < n_keep; j++) {
                uint32_t src = s->keep_old[j];
                if (src != j) {
                    copy_f32(ks + (uint64_t)j * D, ks + (uint64_t)src * D, D);
                    copy_f32(vs + (uint64_t)j * D, vs + (uint64_t)src * D, D);
                }
            }
            for (uint32_t j = 0; j < n_add; j++) {
                uint32_t src_t = s->add_cand[j];
                uint32_t dst_j = n_keep + j;
                copy_f32(ks + (uint64_t)dst_j * D, kp + (uint64_t)src_t * D, D);
                copy_f32(vs + (uint64_t)dst_j * D, vp + (uint64_t)src_t * D, D);
            }
        }
    }
    rebuild_shingle_index(s);
}

static void swapp_f(float **a, float **b) { float *t = *a; *a = *b; *b = t; }
static void swapp_u32(uint32_t **a, uint32_t **b) { uint32_t *t = *a; *a = *b; *b = t; }

static int is_system_role_tok(const v3_model *m, uint32_t id) {
    const uint8_t *b = 0;
    uint32_t bl = 0;
    if (!m->tok_table) return id == 8948u;
    if (v3_token_bytes(m, id, &b, &bl) != V3_OK) return 0;
    if (bl == 6u || (bl == 7u && b[6] == '\n'))
        return b[0] == 's' && b[1] == 'y' && b[2] == 's' && b[3] == 't' && b[4] == 'e' && b[5] == 'm';
    return 0;
}

static void kprev_rot_run(void *arg, uint32_t l0, uint32_t l1) {
    v3_state *s = (v3_state *)arg;
    const v3_model *m = s->m;
    uint32_t G = m->n_kv_heads, W = m->window, D = m->head_dim;
    for (uint32_t l = l0; l < l1; l++) {
        v3_layer_state *L = &s->ls[l];
        for (uint32_t g = 0; g < G; g++)
            for (uint32_t t = 0; t < W; t++)
                rope_rotate(L->k_prev + ((uint64_t)g * W + t) * D, s->cos_w, s->sin_w, D);
    }
}

typedef struct {
    float *hb;
    const v3_tensor *wg, *wu;
    const float *xb;
    uint32_t cols;
} mlp_gu_task;

static void mlp_gu_run(void *arg, uint32_t r0, uint32_t r1) {
    const mlp_gu_task *t = (const mlp_gu_task *)arg;
    for (uint32_t r = r0; r < r1; r++) {
        float g = row_dot(t->wg, r, t->xb, t->cols);
        float u = row_dot(t->wu, r, t->xb, t->cols);
        t->hb[r] = g * v3_sigmoid(g) * u;
    }
}

typedef struct {
    v3_state *s;
    const v3_layer *Lw;
    v3_layer_state *L;
    uint64_t pos;
    uint32_t layer_idx;
    uint32_t n_sk, n_seg, n_prev, n_cur, n_tot;
    uint64_t att_stride, tmp_stride;
    float scale;
} attn_task;

static void attn_run(void *arg, uint32_t h0, uint32_t h1) {
    const attn_task *t = (const attn_task *)arg;
    v3_state *s = t->s;
    const v3_model *m = s->m;
    const v3_layer *Lw = t->Lw;
    v3_layer_state *L = t->L;
    uint32_t H = m->n_heads, G = m->n_kv_heads, D = m->head_dim, W = m->window;
    uint32_t grp = H / G;
    uint32_t free_h = grp / 3u;
    uint32_t sup_h = (grp - free_h) / 2u;
    uint32_t arch_end = grp - sup_h;
    const float *alpha = (const float *)(const void *)Lw->alpha.data;
    uint32_t n_sk = t->n_sk, n_seg = t->n_seg, n_prev = t->n_prev, n_cur = t->n_cur, n_tot = t->n_tot;
    uint32_t l_lo = m->n_layers / 4u, l_mid = m->n_layers / 2u, l_hi = m->n_layers - (m->n_layers / 14u);
    int use_seg_bias = (s->seg_bias_active && n_seg > 0u && t->layer_idx >= l_lo && t->layer_idx < l_hi);
    float scale = t->scale;

    for (uint32_t h = h0; h < h1; h++) {
        uint32_t g = h / grp, hr = h % grp;
        const float *b_seg = 0;
        if (use_seg_bias && hr >= free_h) {
            if (s->seg_bias_active == 2 && t->layer_idx < l_mid) b_seg = s->seg_bias_hop1;
            else b_seg = (hr < arch_end) ? s->seg_bias_arch : s->seg_bias_sup;
        }
        float *att_h = s->att + (uint64_t)h * t->att_stride;
        float *tmp_h = s->tmp + (uint64_t)h * t->tmp_stride;
        float *qn = tmp_h, *qt = tmp_h + D, *mem = tmp_h + 2u * D, *phq = tmp_h + 3u * D;
        float *qh = s->q + (uint64_t)h * D;
        copy_f32(qh, s->qraw + (uint64_t)h * D, D);
        rope_rotate(qh, s->cos_t, s->sin_t, D);

        float mx = -3.0e38f;
        uint32_t idx = 0;
        const float *ks_g = L->k_sink + (uint64_t)g * s->sink_cap * D;
        const float *km_g = L->k_seg + (uint64_t)g * s->seg_cap * D;
        const float *kp_g = L->k_prev + (uint64_t)g * W * D;
        const float *kc_g = L->k_cur + (uint64_t)g * W * D;
        for (uint32_t j = 0; j < n_sk; j++, idx++) {
            float sc = dot_f32(qh, ks_g + (uint64_t)j * D, D) * scale;
            att_h[idx] = sc; if (sc > mx) mx = sc;
        }
        for (uint32_t j = 0; j < n_seg; j++, idx++) {
            float sc = dot_f32(qh, km_g + (uint64_t)j * D, D) * scale + (b_seg ? b_seg[j] : 0.0f);
            att_h[idx] = sc; if (sc > mx) mx = sc;
        }
        for (uint32_t j = 0; j < n_prev; j++, idx++) {
            float sc = dot_f32(qh, kp_g + (uint64_t)j * D, D) * scale;
            att_h[idx] = sc; if (sc > mx) mx = sc;
        }
        for (uint32_t j = 0; j < n_cur; j++, idx++) {
            float sc = dot_f32(qh, kc_g + (uint64_t)j * D, D) * scale;
            att_h[idx] = sc; if (sc > mx) mx = sc;
        }

        float sum = 0.0f;
        for (uint32_t j = 0; j < n_tot; j++) { att_h[j] = v3_expf(att_h[j] - mx); sum += att_h[j]; }
        float inv = 1.0f / sum;
        float *oh = s->ao + (uint64_t)h * D;
        for (uint32_t e = 0; e < D; e++) oh[e] = 0.0f;

        idx = 0;
        const float *vs_g = L->v_sink + (uint64_t)g * s->sink_cap * D;
        const float *vm_g = L->v_seg + (uint64_t)g * s->seg_cap * D;
        const float *vp_g = L->v_prev + (uint64_t)g * W * D;
        const float *vc_g = L->v_cur + (uint64_t)g * W * D;
        for (uint32_t j = 0; j < n_sk; j++, idx++) axpy_f32(oh, att_h[idx] * inv, vs_g + (uint64_t)j * D, D);
        for (uint32_t j = 0; j < n_seg; j++, idx++) axpy_f32(oh, att_h[idx] * inv, vm_g + (uint64_t)j * D, D);
        for (uint32_t j = 0; j < n_prev; j++, idx++) axpy_f32(oh, att_h[idx] * inv, vp_g + (uint64_t)j * D, D);
        for (uint32_t j = 0; j < n_cur; j++, idx++) axpy_f32(oh, att_h[idx] * inv, vc_g + (uint64_t)j * D, D);

        if (s->vault_on && t->pos >= 2ull * W) {
            float gate = v3_tanhf(alpha[h]);
            if (gate != 0.0f) {
                unit_rms(qn, s->qraw + (uint64_t)h * D, D);
                vecmat(qt, qn, (const float *)(const void *)Lw->a_q.data + (uint64_t)g * D * D, D);
                vault_read(s, L, g, qt, mem, phq);
                axpy_f32(oh, gate, mem, D);
            }
        }
    }
}

const float *v3_step(v3_state *s, uint32_t token) {
    const v3_model *m = s->m;
    uint32_t d = m->d_model, H = m->n_heads, G = m->n_kv_heads, D = m->head_dim, W = m->window, F = m->d_ff;
    uint32_t hD = D / 2;
    uint64_t pos = s->pos;
    uint32_t c = (uint32_t)(pos % W);
    int boundary = (c == 0 && pos > 0);
    if (token >= m->vocab_size) token = m->pad_id;
    int seg_active = (s->seg_on && m->seg_n_feat > 0 && s->seg_budget > 0);

    /* Online ChatML system-prompt auto-pinning (clamps oversized >max_sink system prompts to limit) */
    if (!s->pin_locked) {
        uint32_t limit = s->sink_cap < W ? s->sink_cap : W;
        uint32_t nl_id = m->tok_bytes ? rd32(m->tok_bytes + 4u * (uint32_t)'\n') : 198u;
        if (pos == 0) {
            if (token == m->im_start_id && limit >= 2u) s->saw_im_start = 1;
            else s->pin_locked = 1;
        } else if (pos == 1 && s->saw_im_start) {
            if (!is_system_role_tok(m, token)) {
                s->saw_im_start = 0;
                s->pin_locked = 1;
            }
        } else if (s->saw_im_start && pos < (uint64_t)limit) {
            if (s->awaiting_sys_nl) {
                uint32_t cand = (token == nl_id) ? ((uint32_t)pos + 1u) : (uint32_t)pos;
                if (cand > s->n_pin) s->n_pin = cand;
                s->pin_locked = 1;
                s->awaiting_sys_nl = 0;
            } else if (token == m->im_end_id) {
                uint32_t cand = (uint32_t)pos + 1u;
                if (cand > s->n_pin) s->n_pin = cand;
                if (cand < limit) s->awaiting_sys_nl = 1;
                else s->pin_locked = 1;
            }
        } else if (pos >= (uint64_t)limit) {
            if (s->saw_im_start && !s->awaiting_sys_nl && limit > s->n_pin)
                s->n_pin = limit;
            s->pin_locked = 1;
            s->awaiting_sys_nl = 0;
        }
    }

    if (boundary) {
        if (pos >= 2ull * W) {
            uint64_t ev_chunk_idx = pos / W - 2ull;
            if (seg_active) {
                int any_bad = 0;
                for (uint32_t t = 0; t < W; t++) if (s->sc_prev[t] <= -1.0e29f) { any_bad = 1; break; }
                if (any_bad) {
                    for (uint32_t t = 0; t < W; t++) s->seg_mx[t] = 0.0f;
                    for (uint32_t t = 0; t < W; t++) if (s->sc_prev[t] <= -1.0e29f) s->seg_mx[s->seg_prev[t]] = 1.0f;
                    for (uint32_t t = 0; t < W; t++) {
                        if (s->seg_mx[s->seg_prev[t]] > 0.5f) {
                            s->sc_prev[t] = -1.0e30f;
                            for (uint32_t l = 0; l < m->n_layers; l++)
                                for (uint32_t g = 0; g < G; g++)
                                    s->ls[l].b_prev[(uint64_t)g * W + t] = 0.0f;
                        }
                    }
                }
            }
            if (s->vault_on) {
                for (uint32_t l = 0; l < m->n_layers; l++) {
                    v3_layer_state *L = &s->ls[l];
                    for (uint32_t g = 0; g < G; g++)
                        vault_write_chunk(s, L, g, L->kt_prev + (uint64_t)g * W * D, L->v_prev + (uint64_t)g * W * D, L->b_prev + (uint64_t)g * W);
                }
            }
            if (seg_active) update_segment_bank(s, ev_chunk_idx);
        }
        for (uint32_t l = 0; l < m->n_layers; l++) {
            v3_layer_state *L = &s->ls[l];
            swapp_f(&L->k_prev, &L->k_cur); swapp_f(&L->v_prev, &L->v_cur);
            swapp_f(&L->kraw_prev, &L->kraw_cur);
            swapp_f(&L->kt_prev, &L->kt_cur); swapp_f(&L->b_prev, &L->b_cur);
        }
        if (pos >= 2ull * W) {
            if (s->par_for && m->n_layers > 1u) s->par_for(s->par_ctx, kprev_rot_run, s, m->n_layers);
            else kprev_rot_run(s, 0, m->n_layers);
        }
        swapp_u32(&s->tok_prev, &s->tok_cur);
        swapp_f(&s->sc_prev, &s->sc_cur);
        swapp_u32(&s->seg_prev, &s->seg_cur);
        s->cur_seg_id = 0;
        s->cur_seg_len = 0;
        if (pos >= 2ull * W) refresh_prefix_rotations(s, seg_active);
    }

    s->tok_cur[c] = token;

    if (seg_active) {
        int tier = seg_break_tier(m, token);
        uint32_t t2 = (m->seg_max * 2u) / 3u;
        if (t2 < m->seg_min) t2 = m->seg_min;
        uint32_t t3 = (m->seg_max > 8u) ? (m->seg_max - 8u) : m->seg_min;
        if (t3 < m->seg_min) t3 = m->seg_min;
        s->seg_cur[c] = s->cur_seg_id;
        s->cur_seg_len++;
        int brk = (tier == 1 && s->cur_seg_len >= m->seg_min) ||
                  (tier == 2 && s->cur_seg_len >= t2) ||
                  (tier == 3 && s->cur_seg_len >= t3);
        if (c == W - 1u || brk || s->cur_seg_len >= m->seg_max) {
            s->cur_seg_id++;
            s->cur_seg_len = 0;
        }
    }

    /* RoPE angles for this token, positions re-based so they never exceed 2W */
    uint32_t eff = (pos >= W) ? (W + c) : c;
    for (uint32_t i = 0; i < hD; i++) {
        double sn, cs;
        v3_sincos((double)eff * (double)s->inv_freq[i], &sn, &cs);
        s->cos_t[i] = (float)cs;
        s->sin_t[i] = (float)sn;
    }

    embed_row(s->x, &m->embed, token);
    float *qn0 = s->tmp;
    uint32_t seg_cap_alloc = (m->seg_n_feat > 0 && s->seg_cap > 0) ? s->seg_cap : 1u;
    uint64_t att_stride = (uint64_t)s->sink_cap + seg_cap_alloc + 2ull * W + 64ull;
    uint64_t tmp_stride = 4ull * D + 2ull * V3_MAX_BANKS;
    float scale = 1.0f / v3_sqrtf((float)D);
    uint32_t n_sk = (pos >= 2ull * W) ? s->n_pin : 0u;
    uint32_t n_seg = (pos >= 2ull * W && seg_active) ? s->mem_cnt : 0u;
    uint32_t n_prev = (pos >= W) ? W : 0u;
    uint32_t n_cur = c + 1u;
    uint32_t n_tot = n_sk + n_seg + n_prev + n_cur;

    if (n_seg > 0u) update_query_resonance(s, c);
    else s->seg_bias_active = 0;

    for (uint32_t l = 0; l < m->n_layers; l++) {
        if (seg_active) {
            for (uint32_t fi = 0; fi < m->seg_n_feat; fi++) {
                if (m->seg_layers[fi] == l) copy_f32(s->seg_feat + (uint64_t)fi * d, s->x, d);
            }
        }
        const v3_layer *Lw = &m->layers[l];
        v3_layer_state *L = &s->ls[l];

        rmsnorm(s->xb, s->x, &Lw->ln1, d, m->rms_eps);
        matvec(s, s->qraw, &Lw->wq, s->xb, &Lw->bq);
        matvec(s, s->kraw, &Lw->wk, s->xb, &Lw->bk);
        matvec(s, s->v, &Lw->wv, s->xb, &Lw->bv);

        const float *bw = (const float *)(const void *)Lw->beta_w.data, *bb = (const float *)(const void *)Lw->beta_b.data;
        for (uint32_t g = 0; g < G; g++) {
            float bl = bb[g] + dot_f32(bw + (uint64_t)g * d, s->xb, d);
            L->b_cur[(uint64_t)g * W + c] = v3_sigmoid(bl);
            unit_rms(qn0, s->kraw + (uint64_t)g * D, D);
            vecmat(L->kt_cur + ((uint64_t)g * W + c) * D, qn0, (const float *)(const void *)Lw->a_k.data + (uint64_t)g * D * D, D);
            float *kr_dst = L->kraw_cur + ((uint64_t)g * W + c) * D;
            float *kc = L->k_cur + ((uint64_t)g * W + c) * D;
            float *vc = L->v_cur + ((uint64_t)g * W + c) * D;
            for (uint32_t e = 0; e < D; e++) {
                float kv = s->kraw[(uint64_t)g * D + e];
                kr_dst[e] = kv;
                kc[e] = kv;
                vc[e] = s->v[(uint64_t)g * D + e];
            }
            rope_rotate(kc, s->cos_t, s->sin_t, D);
            if (pos < (uint64_t)s->sink_cap) {
                float *ksr = L->k_sink_raw + ((uint64_t)g * s->sink_cap + pos) * D;
                float *vs = L->v_sink + ((uint64_t)g * s->sink_cap + pos) * D;
                copy_f32(ksr, s->kraw + (uint64_t)g * D, D);
                copy_f32(vs, s->v + (uint64_t)g * D, D);
            }
        }

        attn_task at;
        at.s = s; at.Lw = Lw; at.L = L; at.pos = pos; at.layer_idx = l;
        at.n_sk = n_sk; at.n_seg = n_seg; at.n_prev = n_prev; at.n_cur = n_cur; at.n_tot = n_tot;
        at.att_stride = att_stride; at.tmp_stride = tmp_stride; at.scale = scale;
        if (s->par_for && H > 1u && (n_tot >= 96u || (s->vault_on && pos >= 2ull * W)))
            s->par_for(s->par_ctx, attn_run, &at, H);
        else
            attn_run(&at, 0, H);

        matvec(s, s->xb, &Lw->wo, s->ao, 0);
        for (uint32_t i = 0; i < d; i++) s->x[i] += s->xb[i];

        rmsnorm(s->xb, s->x, &Lw->ln2, d, m->rms_eps);
        mlp_gu_task mgt = {s->hb, &Lw->wg, &Lw->wu, s->xb, d};
        if (s->par_for && F >= 256u) s->par_for(s->par_ctx, mlp_gu_run, &mgt, F);
        else mlp_gu_run(&mgt, 0, F);
        matvec(s, s->xb, &Lw->wd, s->hb, 0);
        for (uint32_t i = 0; i < d; i++) s->x[i] += s->xb[i];
    }

    if (seg_active) {
        uint32_t fd = m->seg_n_feat * d;
        float raw_sc = seg_scorer_eval_vec(s, s->seg_feat);
        float sc = raw_sc;
        if (!s->ema_ready) {
            copy_f32(s->seg_feat_ema, s->seg_feat, fd);
            s->ema_ready = 1;
        } else {
            for (uint32_t i = 0; i < fd; i++) s->seg_feat_hab[i] = s->seg_feat[i] - 0.65f * s->seg_feat_ema[i];
            float hab_sc = seg_scorer_eval_vec(s, s->seg_feat_hab);
            for (uint32_t i = 0; i < fd; i++) s->seg_feat_ema[i] = 0.985f * s->seg_feat_ema[i] + 0.015f * s->seg_feat[i];
            if (hab_sc > raw_sc) sc = raw_sc + 0.35f * (hab_sc - raw_sc);
        }
        s->sc_cur[c] = sc;
    }

    if (check_spoof_quarantine(s, pos, c)) {
        if (seg_active) s->sc_cur[c] = -1.0e30f;
        for (uint32_t l = 0; l < m->n_layers; l++)
            for (uint32_t g = 0; g < G; g++)
                s->ls[l].b_cur[(uint64_t)g * W + c] = 0.0f;
    }

    rmsnorm(s->xb, s->x, &m->norm, d, m->rms_eps);
    matvec(s, s->logits, &m->embed, s->xb, 0);
    s->pos = pos + 1;
    return s->logits;
}

/* ============================================================================
 * Tokenizer (O(n log n) min-heap + doubly-linked list byte-level BPE)
 * ========================================================================== */
static uint32_t hash64(uint64_t k) {
    k ^= k >> 33; k *= 0xff51afd7ed558ccdull; k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ull; k ^= k >> 33;
    return (uint32_t)k;
}

static uint64_t tok_hash_cap(const v3_model *m) {
    uint64_t want = 2ull * (uint64_t)m->tok_n_merges + 16ull;
    uint64_t cap = 16ull;
    while (cap < want) {
        if (cap > (1ull << 30)) return 0;
        cap <<= 1;
    }
    return cap;
}

size_t v3_tok_arena_size(const v3_model *m) {
    if (!m) return 0;
    uint64_t cap = tok_hash_cap(m);
    if (cap == 0) return 0;
    bump_t b = {(uint8_t *)0 + 64, 0, 0, 0};
    bump(&b, cap, sizeof(uint64_t));
    bump(&b, cap * 2ull, sizeof(uint32_t));
    bump(&b, V3_TOK_MAX_PIECE, sizeof(uint32_t));
    bump(&b, V3_TOK_MAX_PIECE, sizeof(int32_t));
    bump(&b, V3_TOK_MAX_PIECE, sizeof(int32_t));
    bump(&b, 3ull * V3_TOK_MAX_PIECE, sizeof(uint64_t));
    bump(&b, V3_TOK_MAX_PIECE, sizeof(uint32_t));
    bump(&b, V3_NFC_BUF_CAP, sizeof(uint8_t));
    bump(&b, V3_NFC_CPS_CAP, sizeof(uint32_t));
    if (b.overflow || b.used > (~(size_t)0) - 64u) return 0;
    return b.used + 64u;
}

int v3_tok_init(v3_tokenizer *t, const v3_model *m, void *arena, size_t arena_size) {
    if (!t || !m || !arena || ((uintptr_t)arena % 64u)) return V3_E_ARENA;
    uint64_t cap = tok_hash_cap(m);
    if (cap == 0) return V3_E_LIMIT;
    bump_t b = {(uint8_t *)arena, arena_size, 0, 0};
    t->m = m;
    t->keys = bump(&b, cap, sizeof(uint64_t));
    t->vals = bump(&b, cap * 2ull, sizeof(uint32_t));
    t->piece = bump(&b, V3_TOK_MAX_PIECE, sizeof(uint32_t));
    t->prev = bump(&b, V3_TOK_MAX_PIECE, sizeof(int32_t));
    t->next = bump(&b, V3_TOK_MAX_PIECE, sizeof(int32_t));
    t->heap = bump(&b, 3ull * V3_TOK_MAX_PIECE, sizeof(uint64_t));
    t->win_toks = bump(&b, V3_TOK_MAX_PIECE, sizeof(uint32_t));
    t->nfc_buf = bump(&b, V3_NFC_BUF_CAP, sizeof(uint8_t));
    t->nfc_cps = bump(&b, V3_NFC_CPS_CAP, sizeof(uint32_t));
    if (!t->keys || !t->vals || !t->piece || !t->prev || !t->next || !t->heap ||
        !t->win_toks || !t->nfc_buf || !t->nfc_cps || b.overflow)
        return V3_E_ARENA;
    t->mask = (uint32_t)(cap - 1);
    for (uint64_t i = 0; i < cap; i++) t->keys[i] = ~0ull;
    for (uint32_t r = 0; r < m->tok_n_merges; r++) {
        const uint8_t *e = m->tok_merges + 12ull * r;
        uint64_t key = ((uint64_t)rd32(e) << 32) | rd32(e + 4);
        uint32_t h = hash64(key) & t->mask;
        while (t->keys[h] != ~0ull && t->keys[h] != key) h = (h + 1) & t->mask;
        if (t->keys[h] == key) continue;
        t->keys[h] = key;
        t->vals[2 * h] = r;
        t->vals[2 * h + 1] = rd32(e + 8);
    }
    return V3_OK;
}

static int merge_lookup(const v3_tokenizer *t, uint32_t a, uint32_t b, uint32_t *rank, uint32_t *res) {
    uint64_t key = ((uint64_t)a << 32) | b;
    uint32_t h = hash64(key) & t->mask;
    while (t->keys[h] != ~0ull) {
        if (t->keys[h] == key) { *rank = t->vals[2 * h]; *res = t->vals[2 * h + 1]; return 1; }
        h = (h + 1) & t->mask;
    }
    return 0;
}

int v3_token_bytes(const v3_model *m, uint32_t id, const uint8_t **bytes, uint32_t *len) {
    if (!m || id >= m->tok_n_vocab) { if (bytes) *bytes = 0; if (len) *len = 0; return V3_E_TOKENIZER; }
    *bytes = m->tok_pool + rd32(m->tok_table + 8ull * id);
    *len = rd32(m->tok_table + 8ull * id + 4);
    return V3_OK;
}

/* UTF-8 decode one code point; invalid bytes decode as 0x110000 marker of length 1 */
static uint32_t utf8_next(const uint8_t *s, size_t n, size_t i, uint32_t *len) {
    uint8_t c = s[i];
    if (c < 0x80) { *len = 1; return c; }
    uint32_t need, cp;
    if ((c & 0xE0) == 0xC0) { need = 1; cp = c & 0x1F; }
    else if ((c & 0xF0) == 0xE0) { need = 2; cp = c & 0x0F; }
    else if ((c & 0xF8) == 0xF0) { need = 3; cp = c & 0x07; }
    else { *len = 1; return 0x110000u; }
    if (need > n - 1 - i) { *len = 1; return 0x110000u; }
    for (uint32_t k = 1; k <= need; k++) {
        if ((s[i + k] & 0xC0) != 0x80) { *len = 1; return 0x110000u; }
        cp = (cp << 6) | (s[i + k] & 0x3F);
    }
    *len = need + 1;
    return cp;
}

/* Strict RFC-3629 UTF-8 decoder for lossless NFC; invalid raw bytes map to 0x110000 | byte */
static uint32_t utf8_next_strict(const uint8_t *s, size_t n, size_t i, uint32_t *len) {
    uint8_t c = s[i];
    if (c < 0x80u) { *len = 1u; return c; }
    uint32_t need, cp, min_cp;
    if ((c & 0xE0u) == 0xC0u) { need = 1u; cp = c & 0x1Fu; min_cp = 0x80u; }
    else if ((c & 0xF0u) == 0xE0u) { need = 2u; cp = c & 0x0Fu; min_cp = 0x800u; }
    else if ((c & 0xF8u) == 0xF0u) { need = 3u; cp = c & 0x07u; min_cp = 0x10000u; }
    else { *len = 1u; return 0x110000u | (uint32_t)c; }
    if (need > n - 1u - i) { *len = 1u; return 0x110000u | (uint32_t)c; }
    for (uint32_t k = 1u; k <= need; k++) {
        if ((s[i + k] & 0xC0u) != 0x80u) { *len = 1u; return 0x110000u | (uint32_t)c; }
        cp = (cp << 6) | (uint32_t)(s[i + k] & 0x3Fu);
    }
    if (cp < min_cp || cp > 0x10FFFFu || (cp >= 0xD800u && cp <= 0xDFFFu)) {
        *len = 1u;
        return 0x110000u | (uint32_t)c;
    }
    *len = need + 1u;
    return cp;
}

static uint32_t nfc_ccc(uint32_t cp) {
    if (cp < 768u || cp >= 0x110000u) return 0u;
    uint32_t lo = 0u, hi = V3_NFC_N_CCC;
    while (lo < hi) {
        uint32_t mid = (lo + hi) >> 1;
        uint32_t rlo = g_v3_ccc[mid].lo;
        uint32_t rhi = rlo + (uint32_t)g_v3_ccc[mid].len;
        if (cp < rlo) hi = mid;
        else if (cp > rhi) lo = mid + 1u;
        else return (uint32_t)g_v3_ccc[mid].ccc;
    }
    return 0u;
}

static void nfc_decompose_cp(uint32_t cp, uint32_t *cps, size_t *ncps, size_t cap) {
    if (cp >= 0xAC00u && cp <= 0xD7A3u) {
        uint32_t s_idx = cp - 0xAC00u;
        uint32_t l_cp = 0x1100u + s_idx / 588u;
        uint32_t v_cp = 0x1161u + (s_idx % 588u) / 28u;
        uint32_t t_off = s_idx % 28u;
        if (*ncps < cap) cps[(*ncps)++] = l_cp;
        if (*ncps < cap) cps[(*ncps)++] = v_cp;
        if (t_off > 0u && *ncps < cap) cps[(*ncps)++] = 0x11A7u + t_off;
        return;
    }
    if (cp >= 192u && cp < 0x110000u) {
        uint32_t lo = 0u, hi = V3_NFC_N_DECOMP;
        while (lo < hi) {
            uint32_t mid = (lo + hi) >> 1;
            uint32_t mcp = g_v3_nfd[mid].cp;
            if (cp < mcp) hi = mid;
            else if (cp > mcp) lo = mid + 1u;
            else {
                nfc_decompose_cp(g_v3_nfd[mid].a, cps, ncps, cap);
                if (g_v3_nfd[mid].b != 0u) nfc_decompose_cp(g_v3_nfd[mid].b, cps, ncps, cap);
                return;
            }
        }
    }
    if (*ncps < cap) cps[(*ncps)++] = cp;
}

static uint32_t nfc_compose_pair(uint32_t a, uint32_t b) {
    if (a >= 0x1100u && a <= 0x1112u && b >= 0x1161u && b <= 0x1175u)
        return 0xAC00u + ((a - 0x1100u) * 21u + (b - 0x1161u)) * 28u;
    if (a >= 0xAC00u && a <= 0xD7A3u && ((a - 0xAC00u) % 28u) == 0u && b >= 0x11A8u && b <= 0x11C2u)
        return a + (b - 0x11A7u);
    uint32_t lo = 0u, hi = V3_NFC_N_COMP;
    while (lo < hi) {
        uint32_t mid = (lo + hi) >> 1;
        uint32_t ma = g_v3_nfc[mid].a, mb = g_v3_nfc[mid].b;
        if (a < ma || (a == ma && b < mb)) hi = mid;
        else if (a > ma || (a == ma && b > mb)) lo = mid + 1u;
        else return g_v3_nfc[mid].comp;
    }
    return 0u;
}

static size_t nfc_normalize_span(v3_tokenizer *t, const uint8_t *src, size_t len) {
    uint32_t *cps = t->nfc_cps;
    size_t ncps = 0;
    size_t i = 0;
    while (i < len && ncps + 4u <= V3_NFC_CPS_CAP) {
        uint32_t cl = 1u;
        uint32_t cp = utf8_next_strict(src, len, i, &cl);
        nfc_decompose_cp(cp, cps, &ncps, V3_NFC_CPS_CAP);
        i += cl;
    }
    /* Canonical ordering (stable insertion sort over non-starter runs) */
    for (size_t k = 1; k < ncps; k++) {
        uint32_t ck = nfc_ccc(cps[k]);
        if (ck == 0u) continue;
        uint32_t v = cps[k];
        size_t j = k;
        while (j > 0 && nfc_ccc(cps[j - 1]) > ck) {
            cps[j] = cps[j - 1];
            j--;
        }
        cps[j] = v;
    }
    /* Canonical composition (UAX #15) */
    int32_t starter_idx = -1;
    uint32_t last_ccc = 0u;
    for (size_t k = 0; k < ncps; k++) {
        uint32_t cp = cps[k];
        uint32_t cc = nfc_ccc(cp);
        if (starter_idx >= 0 && (last_ccc < cc || last_ccc == 0u)) {
            uint32_t comp = nfc_compose_pair(cps[starter_idx], cp);
            if (comp != 0u) {
                cps[starter_idx] = comp;
                cps[k] = 0xFFFFFFFFu;
                continue;
            }
        }
        if (cc == 0u) { starter_idx = (int32_t)k; last_ccc = 0u; }
        else last_ccc = cc;
    }
    /* Re-encode UTF-8 into t->nfc_buf */
    uint8_t *dst = t->nfc_buf;
    size_t o = 0;
    for (size_t k = 0; k < ncps; k++) {
        uint32_t cp = cps[k];
        if (cp == 0xFFFFFFFFu) continue;
        if (cp >= 0x110000u) {
            if (o < V3_NFC_BUF_CAP) dst[o++] = (uint8_t)(cp & 0xFFu);
        } else if (cp < 0x80u) {
            if (o < V3_NFC_BUF_CAP) dst[o++] = (uint8_t)cp;
        } else if (cp < 0x800u) {
            if (o + 2u <= V3_NFC_BUF_CAP) {
                dst[o++] = (uint8_t)(0xC0u | (cp >> 6));
                dst[o++] = (uint8_t)(0x80u | (cp & 0x3Fu));
            }
        } else if (cp < 0x10000u) {
            if (o + 3u <= V3_NFC_BUF_CAP) {
                dst[o++] = (uint8_t)(0xE0u | (cp >> 12));
                dst[o++] = (uint8_t)(0x80u | ((cp >> 6) & 0x3Fu));
                dst[o++] = (uint8_t)(0x80u | (cp & 0x3Fu));
            }
        } else {
            if (o + 4u <= V3_NFC_BUF_CAP) {
                dst[o++] = (uint8_t)(0xF0u | (cp >> 18));
                dst[o++] = (uint8_t)(0x80u | ((cp >> 12) & 0x3Fu));
                dst[o++] = (uint8_t)(0x80u | ((cp >> 6) & 0x3Fu));
                dst[o++] = (uint8_t)(0x80u | (cp & 0x3Fu));
            }
        }
    }
    return o;
}

static int uni_cat(const v3_model *m, uint32_t cp) {
    if (cp < 128) {
        if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) return 1;
        if (cp >= '0' && cp <= '9') return 2;
        return 0;
    }
    uint32_t lo = 0, hi = m->tok_n_ranges;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        const uint8_t *r = m->tok_ranges + 12ull * mid;
        if (cp < rd32(r)) hi = mid;
        else if (cp > rd32(r + 4)) lo = mid + 1;
        else return (int)rd32(r + 8);
    }
    return 0;
}

static int is_space(uint32_t cp) {
    return cp == ' ' || (cp >= 0x09 && cp <= 0x0D) || cp == 0x85 || cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
           cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}
static int is_nl(uint32_t cp) { return cp == '\r' || cp == '\n'; }

static size_t pretok_match(const v3_model *m, const uint8_t *s, size_t n, size_t i) {
    uint32_t l0, l1;
    uint32_t c0 = utf8_next(s, n, i, &l0);
    if (c0 == '\'' && i + 1 < n) {
        uint8_t a = s[i + 1] | 0x20, b = (i + 2 < n) ? (s[i + 2] | 0x20) : 0;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
    }
    {
        size_t j = i;
        int c0cat = uni_cat(m, c0);
        if (c0cat != 1 && !is_nl(c0) && c0cat != 2) j = i + l0;
        if (j < n && uni_cat(m, utf8_next(s, n, j, &l1)) == 1) {
            while (j < n) { uint32_t cp = utf8_next(s, n, j, &l1); if (uni_cat(m, cp) != 1) break; j += l1; }
            return j - i;
        }
        if (c0cat == 1) {
            j = i;
            while (j < n) { uint32_t cp = utf8_next(s, n, j, &l1); if (uni_cat(m, cp) != 1) break; j += l1; }
            return j - i;
        }
    }
    if (uni_cat(m, c0) == 2) return l0;
    {
        size_t j = i;
        if (c0 == ' ') j = i + 1;
        size_t k = j;
        while (k < n) {
            uint32_t cp = utf8_next(s, n, k, &l1);
            if (is_space(cp) || uni_cat(m, cp) != 0) break;
            k += l1;
        }
        if (k > j) {
            while (k < n && (s[k] == '\r' || s[k] == '\n')) k++;
            return k - i;
        }
    }
    if (is_space(c0)) {
        size_t j = i, last_nl_end = 0;
        while (j < n) {
            uint32_t cp = utf8_next(s, n, j, &l1);
            if (!is_space(cp)) break;
            j += l1;
            if (is_nl(cp)) last_nl_end = j;
        }
        if (last_nl_end) return last_nl_end - i;
        if (j >= n) return j - i;
        size_t back = j;
        {
            size_t p = j - 1;
            while (p > i && (s[p] & 0xC0) == 0x80) p--;
            back = p;
        }
        if (back > i) return back - i;
        return j - i;
    }
    return l0;
}

static void bpe_heap_push(uint64_t *hp, size_t *hsz, uint64_t val) {
    size_t i = (*hsz)++;
    hp[i] = val;
    while (i > 0) {
        size_t p = (i - 1) >> 1;
        if (hp[p] <= hp[i]) break;
        uint64_t tmp = hp[p]; hp[p] = hp[i]; hp[i] = tmp;
        i = p;
    }
}

static uint64_t bpe_heap_pop(uint64_t *hp, size_t *hsz) {
    uint64_t out = hp[0];
    size_t n = --(*hsz);
    if (n > 0) {
        hp[0] = hp[n];
        size_t i = 0;
        for (;;) {
            size_t l = (i << 1) + 1;
            if (l >= n) break;
            size_t m = (l + 1 < n && hp[l + 1] < hp[l]) ? (l + 1) : l;
            if (hp[i] <= hp[m]) break;
            uint64_t tmp = hp[i]; hp[i] = hp[m]; hp[m] = tmp;
            i = m;
        }
    }
    return out;
}

static size_t bpe_piece(v3_tokenizer *t, const uint8_t *p, size_t len, uint32_t *out, size_t max_out, int *truncated) {
    const v3_model *m = t->m;
    uint32_t *w = t->piece;
    int32_t *prv = t->prev, *nxt = t->next;
    uint64_t *hp = t->heap;
    size_t n = 0;
    for (size_t i = 0; i < len && n < V3_TOK_MAX_PIECE; i++, n++) {
        w[n] = rd32(m->tok_bytes + 4u * p[i]);
        prv[n] = (int32_t)n - 1;
        nxt[n] = (int32_t)(n + 1);
    }
    if (n == 0) return 0;
    nxt[n - 1] = -1;

    size_t hsz = 0;
    for (size_t i = 0; i + 1 < n; i++) {
        uint32_t r, res;
        if (merge_lookup(t, w[i], w[i + 1], &r, &res))
            bpe_heap_push(hp, &hsz, ((uint64_t)r << 32) | (uint32_t)i);
    }

    while (hsz > 0) {
        uint64_t top = bpe_heap_pop(hp, &hsz);
        uint32_t want_r = (uint32_t)(top >> 32);
        int32_t i = (int32_t)(uint32_t)top;
        if (w[i] == 0xFFFFFFFFu) continue;
        int32_t j = nxt[i];
        if (j < 0 || w[j] == 0xFFFFFFFFu) continue;
        uint32_t cur_r, res;
        if (!merge_lookup(t, w[i], w[j], &cur_r, &res) || cur_r != want_r) continue;

        w[i] = res;
        w[j] = 0xFFFFFFFFu;
        int32_t nj = nxt[j];
        nxt[i] = nj;
        if (nj >= 0) prv[nj] = i;

        int32_t pi = prv[i];
        if (pi >= 0) {
            uint32_t rl, resl;
            if (merge_lookup(t, w[pi], w[i], &rl, &resl))
                bpe_heap_push(hp, &hsz, ((uint64_t)rl << 32) | (uint32_t)pi);
        }
        if (nj >= 0) {
            uint32_t rr, resr;
            if (merge_lookup(t, w[i], w[nj], &rr, &resr))
                bpe_heap_push(hp, &hsz, ((uint64_t)rr << 32) | (uint32_t)i);
        }
    }

    size_t k = 0;
    for (int32_t cur = 0; cur >= 0; cur = nxt[cur]) {
        if (k < max_out) out[k++] = w[cur];
        else { *truncated = 1; break; }
    }
    return k;
}

/* Token-aligned 4 KB overlap windowing for single pieces exceeding V3_TOK_MAX_PIECE (32 KB) */
static int bpe_piece_long(v3_tokenizer *t, const uint8_t *src, size_t ml, uint32_t *out, size_t max_out, size_t *k_io) {
    size_t k = *k_io;
    size_t pos = 0;
    while (pos < ml) {
        size_t rem = ml - pos;
        if (rem <= V3_TOK_MAX_PIECE) {
            int trunc = 0;
            size_t got = bpe_piece(t, src + pos, rem, out + k, max_out - k, &trunc);
            k += got;
            *k_io = k;
            return trunc ? V3_E_SIZE : V3_OK;
        }
        int trunc = 0;
        size_t got = bpe_piece(t, src + pos, V3_TOK_MAX_PIECE, t->win_toks, V3_TOK_MAX_PIECE, &trunc);
        if (got == 0) break;
        size_t commit_n = got;
        size_t tail_bytes = 0;
        while (commit_n > 1u && tail_bytes < 4096u) {
            const uint8_t *tb = 0; uint32_t tbl = 0;
            v3_token_bytes(t->m, t->win_toks[commit_n - 1u], &tb, &tbl);
            tail_bytes += (size_t)tbl;
            commit_n--;
        }
        size_t commit_bytes = V3_TOK_MAX_PIECE - tail_bytes;
        if (commit_bytes == 0u) commit_bytes = V3_TOK_MAX_PIECE;
        if (commit_n > max_out - k) {
            for (size_t j = 0; j < max_out - k; j++) out[k + j] = t->win_toks[j];
            *k_io = max_out;
            return V3_E_SIZE;
        }
        for (size_t j = 0; j < commit_n; j++) out[k + j] = t->win_toks[j];
        k += commit_n;
        pos += commit_bytes;
    }
    *k_io = k;
    return V3_OK;
}

static int span_has_nfc_cand(const uint8_t *s, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (s[i] >= 0xCCu) return 1;
    return 0;
}

static int match_special(const v3_model *m, const uint8_t *s, size_t n, size_t i, uint32_t *id, size_t *len) {
    if (s[i] != '<') return 0;
    for (uint32_t k = 0; k < m->tok_n_special; k++) {
        uint32_t tid = rd32(m->tok_special + 4ull * k);
        const uint8_t *b; uint32_t bl;
        v3_token_bytes(m, tid, &b, &bl);
        if (bl == 0 || bl > n - i) continue;
        uint32_t j = 0;
        while (j < bl && s[i + j] == b[j]) j++;
        if (j == bl) { *id = tid; *len = bl; return 1; }
    }
    return 0;
}

int v3_encode_ex(v3_tokenizer *t, const uint8_t *text, size_t len, uint32_t *out, size_t max_out, size_t *n_out, int allow_special) {
    if (!t || !t->m || !n_out) return V3_E_SIZE;
    *n_out = 0;
    if (len == 0) return V3_OK;
    if (!text || (max_out > 0 && !out)) return V3_E_SIZE;

    const v3_model *m = t->m;
    size_t i = 0, k = 0;
    while (i < len) {
        if (allow_special) {
            uint32_t sid; size_t sl;
            if (match_special(m, text, len, i, &sid, &sl)) {
                if (k >= max_out) { *n_out = k; return V3_E_SIZE; }
                out[k++] = sid;
                i += sl;
                continue;
            }
        }
        size_t end = len;
        if (allow_special) {
            end = i;
            while (end < len) {
                uint32_t tmp_id; size_t tmp_l;
                if (text[end] == '<' && match_special(m, text, len, end, &tmp_id, &tmp_l)) break;
                end++;
            }
        }
        while (i < end) {
            if (!span_has_nfc_cand(text + i, end - i)) {
                while (i < end) {
                    size_t ml = pretok_match(m, text, end, i);
                    int rc = bpe_piece_long(t, text + i, ml, out, max_out, &k);
                    if (rc != V3_OK) { *n_out = k; return rc; }
                    i += ml;
                }
                break;
            }
            /* Fast-skip pure-<0xCC pretok pieces well ahead of the next >=0xCC byte */
            {
                size_t ml0 = pretok_match(m, text, end, i);
                size_t chk = (i + ml0 + 4u < end) ? (ml0 + 4u) : (end - i);
                if (!span_has_nfc_cand(text + i, chk)) {
                    int rc = bpe_piece_long(t, text + i, ml0, out, max_out, &k);
                    if (rc != V3_OK) { *n_out = k; return rc; }
                    i += ml0;
                    continue;
                }
            }
            /* Bounded UAX #15 NFC span ending at a clean pretok boundary */
            size_t sub_end = (end - i <= 1024u) ? end : (i + 1024u);
            if (sub_end < end) {
                size_t best = 0;
                for (size_t p = i + 1u; p <= sub_end; p++) {
                    if ((text[p] & 0xC0u) == 0x80u) continue;
                    size_t la = (end - p < 4u) ? (end - p) : 4u;
                    if (span_has_nfc_cand(text + p, la)) continue;
                    if (text[p - 1u] == '\n' && text[p] != '\r' && text[p] != '\n') { best = p; break; }
                    if (text[p - 1u] >= 0x80u && text[p] < 0x80u && text[p] != '\'' &&
                        uni_cat(m, text[p]) == 0) {
                        best = p;
                        break;
                    }
                }
                if (best > i) sub_end = best;
                else {
                    while (sub_end > i + 1u && (text[sub_end] & 0xC0u) == 0x80u) sub_end--;
                    while (sub_end > i + 1u) {
                        uint32_t cl = 1u;
                        uint32_t cp = utf8_next_strict(text, end, sub_end, &cl);
                        if (nfc_ccc(cp) == 0u) break;
                        sub_end--;
                        while (sub_end > i + 1u && (text[sub_end] & 0xC0u) == 0x80u) sub_end--;
                    }
                }
            } else {
                /* Even when end - i <= 1024, split off any pure-ASCII suffix right after a non-ASCII run */
                for (size_t p = i + 1u; p < sub_end; p++) {
                    if (text[p - 1u] >= 0x80u && text[p] < 0x80u && text[p] != '\'' &&
                        uni_cat(m, text[p]) == 0 && !span_has_nfc_cand(text + p, sub_end - p)) {
                        sub_end = p;
                        break;
                    }
                }
            }
            size_t nfc_len = nfc_normalize_span(t, text + i, sub_end - i);
            size_t ni = 0;
            while (ni < nfc_len) {
                size_t ml = pretok_match(m, t->nfc_buf, nfc_len, ni);
                int rc = bpe_piece_long(t, t->nfc_buf + ni, ml, out, max_out, &k);
                if (rc != V3_OK) { *n_out = k; return rc; }
                ni += ml;
            }
            i = sub_end;
        }
    }
    *n_out = k;
    return V3_OK;
}

int v3_encode(v3_tokenizer *t, const uint8_t *text, size_t len, uint32_t *out, size_t max_out, size_t *n_out) {
    return v3_encode_ex(t, text, len, out, max_out, n_out, 1);
}

/* ============================================================================
 * CAAI / OPI Constitutional Output Governor (pre-sampler logit shield)
 * ========================================================================== */
static uint8_t gov_fold_cp(uint32_t cp) {
    if (cp >= 0xFF01u && cp <= 0xFF5Eu) cp -= 0xFEE0u;
    if (cp < 0x80u) {
        if (cp <= 0x20u || cp == 0x7Fu || cp == '\'' || cp == '"' || cp == '`' || cp == '+' || cp == '\\' || cp == '_' || cp == '-')
            return 0u;
        if (cp >= 'A' && cp <= 'Z') return (uint8_t)(cp + 32u);
        return (uint8_t)cp;
    }
    /* Zero-width / invisible format / smart quotes / fullwidth joiners -> stripped */
    if (cp == 0x00ADu || cp == 0x034Fu || cp == 0x061Cu || cp == 0x180Eu ||
        (cp >= 0x200Bu && cp <= 0x200Fu) || (cp >= 0x2018u && cp <= 0x201Fu) ||
        (cp >= 0x202Au && cp <= 0x202Eu) || (cp >= 0x2060u && cp <= 0x206Fu) || cp == 0xFEFFu)
        return 0u;
    /* Cyrillic & Greek visual homoglyph folding */
    switch (cp) {
    case 0x0410u: case 0x0430u: case 0x0391u: case 0x03B1u: return 'a';
    case 0x0412u: case 0x0432u: case 0x0392u: case 0x03B2u: return 'b';
    case 0x0421u: case 0x0441u: return 'c';
    case 0x0415u: case 0x0435u: case 0x0395u: case 0x03B5u: return 'e';
    case 0x041Du: case 0x043Du: case 0x0397u: return 'h';
    case 0x0406u: case 0x0456u: case 0x0399u: case 0x03B9u: return 'i';
    case 0x0408u: case 0x0458u: return 'j';
    case 0x041Au: case 0x043Au: case 0x039Au: case 0x03BAu: return 'k';
    case 0x041Cu: case 0x043Cu: case 0x039Cu: return 'm';
    case 0x039Du: case 0x03BDu: return 'n';
    case 0x041Eu: case 0x043Eu: case 0x039Fu: case 0x03BFu: return 'o';
    case 0x0420u: case 0x0440u: case 0x03A1u: case 0x03C1u: return 'p';
    case 0x0405u: case 0x0455u: case 0x03A3u: case 0x03C3u: case 0x03C2u: return 's';
    case 0x0422u: case 0x0442u: case 0x03A4u: case 0x03C4u: return 't';
    case 0x0425u: case 0x0445u: case 0x03A7u: case 0x03C7u: return 'x';
    case 0x0423u: case 0x0443u: case 0x03A5u: case 0x03C5u: return 'y';
    case 0x0396u: return 'z';
    default: return '?';
    }
}

static void gov_push_canon(uint8_t *hist, uint32_t *hlen, uint32_t cap, uint8_t ch) {
    if (ch == 0u) return;
    if (*hlen < cap) {
        hist[(*hlen)++] = ch;
    } else {
        for (uint32_t i = 1u; i < cap; i++) hist[i - 1u] = hist[i];
        hist[cap - 1u] = ch;
    }
}

static void gov_feed_bytes(uint8_t *hist, uint32_t *hlen, uint32_t cap,
                           uint8_t *rem, uint32_t *rlen, const uint8_t *b, size_t bl) {
    for (size_t i = 0; i < bl; i++) {
        uint8_t c = b[i];
        if (*rlen > 0u) {
            if ((c & 0xC0u) == 0x80u) {
                rem[(*rlen)++] = c;
                uint8_t lead = rem[0];
                uint32_t need = ((lead & 0xE0u) == 0xC0u) ? 2u : (((lead & 0xF0u) == 0xE0u) ? 3u : 4u);
                if (*rlen == need) {
                    uint32_t cl = 1u;
                    uint32_t cp = utf8_next_strict(rem, need, 0u, &cl);
                    *rlen = 0u;
                    if (cp < 0x110000u) gov_push_canon(hist, hlen, cap, gov_fold_cp(cp));
                }
                continue;
            }
            *rlen = 0u;
        }
        if (c < 0x80u) {
            gov_push_canon(hist, hlen, cap, gov_fold_cp(c));
        } else if ((c & 0xE0u) == 0xC0u || (c & 0xF0u) == 0xE0u || (c & 0xF8u) == 0xF0u) {
            rem[0] = c;
            *rlen = 1u;
        }
    }
}

void v3_gov_init(v3_governor *g) {
    if (!g) return;
    uint8_t *p = (uint8_t *)g;
    for (size_t i = 0; i < sizeof(*g); i++) p[i] = 0u;
}

void v3_gov_reset_stream(v3_governor *g) {
    if (!g) return;
    g->hist_len = 0u;
    g->utf8_len = 0u;
    g->raw_len = 0u;
}

static void gov_push_raw(v3_governor *g, const uint8_t *b, uint32_t bl) {
    for (uint32_t i = 0u; i < bl; i++) {
        if (g->raw_len < V3_GOV_RAW_CAP) {
            g->raw_tail[g->raw_len++] = b[i];
        } else {
            for (uint32_t j = 1u; j < V3_GOV_RAW_CAP; j++) g->raw_tail[j - 1u] = g->raw_tail[j];
            g->raw_tail[V3_GOV_RAW_CAP - 1u] = b[i];
        }
    }
}

static int gov_b64_val(uint8_t c) {
    if (c >= 'A' && c <= 'Z') return (int)(c - 'A');
    if (c >= 'a' && c <= 'z') return (int)(c - 'a') + 26;
    if (c >= '0' && c <= '9') return (int)(c - '0') + 52;
    if (c == '+' || c == '-') return 62;
    if (c == '/' || c == '_') return 63;
    return -1;
}

static int gov_hex_val(uint8_t c) {
    if (c >= '0' && c <= '9') return (int)(c - '0');
    if (c >= 'a' && c <= 'f') return (int)(c - 'a') + 10;
    if (c >= 'A' && c <= 'F') return (int)(c - 'A') + 10;
    return -1;
}

static int gov_is_hex_sep(uint8_t c) {
    return c == ' ' || c == '\\' || c == 'x' || c == 'X' || c == ':' ||
           c == ',' || c == '-' || c == '_' || c == '\r' || c == '\n';
}

int v3_gov_add_rule(v3_governor *g, const uint8_t *pat, size_t len) {
    if (!g || !pat || len == 0u) return V3_E_SIZE;
    uint8_t canon[V3_GOV_RULE_CAP];
    uint8_t rem[4] = {0, 0, 0, 0};
    uint32_t clen = 0u, rlen = 0u;
    gov_feed_bytes(canon, &clen, V3_GOV_RULE_CAP, rem, &rlen, pat, len);
    if (clen < 4u) return V3_E_SIZE;
    for (uint32_t r = 0; r < g->n_rules; r++) {
        if (g->rule_len[r] == clen) {
            uint32_t k = 0;
            while (k < clen && g->rules[r][k] == canon[k]) k++;
            if (k == clen) { g->active = 1; return V3_OK; }
        }
    }
    if (g->n_rules >= V3_GOV_MAX_RULES) return V3_E_LIMIT;
    uint32_t idx = g->n_rules++;
    for (uint32_t k = 0; k < clen; k++) g->rules[idx][k] = canon[k];
    g->rule_len[idx] = clen;
    g->active = 1;
    return V3_OK;
}

static int span_contains_ci(const uint8_t *s, size_t n, const char *kw, size_t kw_len) {
    if (n < kw_len) return 0;
    for (size_t i = 0; i + kw_len <= n; i++) {
        size_t k = 0;
        while (k < kw_len) {
            uint8_t c = s[i + k];
            if (c >= 'A' && c <= 'Z') c = (uint8_t)(c + 32u);
            if (c != (uint8_t)kw[k]) break;
            k++;
        }
        if (k == kw_len) return 1;
    }
    return 0;
}

void v3_gov_arm_system(v3_governor *g, const uint8_t *sys_text, size_t len) {
    if (!g) return;
    v3_gov_init(g);
    if (!sys_text || len == 0u) return;
    int has_prohib = span_contains_ci(sys_text, len, "never", 5u) ||
                     span_contains_ci(sys_text, len, "forbid", 6u) ||
                     span_contains_ci(sys_text, len, "prohibit", 8u) ||
                     span_contains_ci(sys_text, len, "must not", 8u) ||
                     span_contains_ci(sys_text, len, "do not output", 13u) ||
                     span_contains_ci(sys_text, len, "do not emit", 11u);
    if (!has_prohib) return;

    if (span_contains_ci(sys_text, len, "syscall_exec", 12u)) {
        v3_gov_add_rule(g, (const uint8_t *)"syscall_exec", 12u);
        v3_gov_add_rule(g, (const uint8_t *)"syscallexec", 11u);
    }

    /* Also extract quoted literals ('...', "...", `...`) containing '_' or '(' after prohibition cues */
    for (size_t i = 0; i < len; i++) {
        uint8_t q = sys_text[i];
        if (q != '\'' && q != '"' && q != '`') continue;
        size_t j = i + 1u;
        int has_sig = 0;
        while (j < len && sys_text[j] != q && (j - i - 1u) < V3_GOV_RULE_CAP) {
            if (sys_text[j] == '_' || sys_text[j] == '(') has_sig = 1;
            j++;
        }
        size_t slen = j - (i + 1u);
        if (j < len && sys_text[j] == q && has_sig && slen >= 5u && slen < V3_GOV_RULE_CAP) {
            const uint8_t *lit = sys_text + i + 1u;
            v3_gov_add_rule(g, lit, slen);
            if (lit[slen - 1u] == '(' && slen >= 6u)
                v3_gov_add_rule(g, lit, slen - 1u);
            /* Add underscore-stripped variant to block underscore omission */
            uint8_t stripped[V3_GOV_RULE_CAP];
            size_t st_len = 0;
            size_t base_l = (lit[slen - 1u] == '(') ? (slen - 1u) : slen;
            for (size_t k = 0; k < base_l && st_len < V3_GOV_RULE_CAP; k++)
                if (lit[k] != '_') stripped[st_len++] = lit[k];
            if (st_len >= 5u) v3_gov_add_rule(g, stripped, st_len);
            i = j;
        }
    }
}

static int gov_hist_violates(const v3_governor *g, const uint8_t *hist, uint32_t hlen, uint32_t prev_hlen) {
    for (uint32_t r = 0; r < g->n_rules; r++) {
        uint32_t rl = g->rule_len[r];
        if (hlen < rl) continue;
        /* Check any alignment ending after the previous unmasked prefix (both forward and reversed) */
        uint32_t start_min = (prev_hlen >= rl) ? (prev_hlen - rl + 1u) : 0u;
        for (uint32_t pos = start_min; pos + rl <= hlen; pos++) {
            uint32_t kf = 0u, kr = 0u;
            while (kf < rl && hist[pos + kf] == g->rules[r][kf]) kf++;
            if (kf == rl) return 1;
            while (kr < rl && hist[pos + kr] == g->rules[r][rl - 1u - kr]) kr++;
            if (kr == rl) return 1;
        }
    }
    return 0;
}

static int gov_encoded_tail_violates(const v3_governor *g, const uint8_t *tb, uint32_t tbl) {
    uint32_t take_tb = (tbl <= 64u) ? tbl : 64u;
    int has_b64 = 0, has_hex = 0;
    for (uint32_t i = 0u; i < take_tb; i++) {
        if (gov_b64_val(tb[i]) >= 0) has_b64 = 1;
        if (gov_hex_val(tb[i]) >= 0) has_hex = 1;
    }
    if (!has_b64 && !has_hex) return 0;

    uint8_t comb[160];
    uint8_t buf[192];
    uint8_t dec[96];
    uint8_t dhist[V3_GOV_HIST_CAP];
    uint8_t drem[4];
    uint32_t clen = 0u;
    for (uint32_t i = 0u; i < g->raw_len; i++) comb[clen++] = g->raw_tail[i];
    for (uint32_t i = 0u; i < take_tb; i++) comb[clen++] = tb[i];
    uint32_t bound = take_tb + 3u;

    /* 1. Streaming 4-phase Base64 / Base64URL tail check (handles partial 2/3-char groups & unaligned tails) */
    if (has_b64) {
        uint32_t end = clen;
        while (end > 0u && (comb[end - 1u] == '\r' || comb[end - 1u] == '\n')) end--;
        uint32_t eq = 0u;
        while (end > 0u && comb[end - 1u] == '=' && eq < 2u) { end--; eq++; }
        uint32_t start = end, cnt_b64 = 0u;
        while (start > 0u && cnt_b64 < 128u) {
            uint8_t ch = comb[start - 1u];
            if (ch == '\r' || ch == '\n') { start--; continue; }
            if (gov_b64_val(ch) >= 0) { start--; cnt_b64++; continue; }
            break;
        }
        if (cnt_b64 >= 8u) {
            uint32_t nb64 = 0u;
            for (uint32_t i = start; i < end && nb64 < 128u; i++) {
                int v = gov_b64_val(comb[i]);
                if (v >= 0) buf[nb64++] = (uint8_t)v;
            }
            for (uint32_t off = 0u; off < 4u && off + 6u <= nb64; off++) {
                uint32_t rem_n = nb64 - off, p = off, dlen = 0u;
                while (rem_n >= 4u && dlen + 3u <= sizeof(dec)) {
                    uint32_t v = ((uint32_t)buf[p] << 18) | ((uint32_t)buf[p + 1u] << 12) |
                                 ((uint32_t)buf[p + 2u] << 6) | (uint32_t)buf[p + 3u];
                    dec[dlen++] = (uint8_t)((v >> 16) & 0xFFu);
                    dec[dlen++] = (uint8_t)((v >> 8) & 0xFFu);
                    dec[dlen++] = (uint8_t)(v & 0xFFu);
                    p += 4u; rem_n -= 4u;
                }
                if (rem_n == 3u && dlen + 2u <= sizeof(dec)) {
                    uint32_t v = ((uint32_t)buf[p] << 18) | ((uint32_t)buf[p + 1u] << 12) | ((uint32_t)buf[p + 2u] << 6);
                    dec[dlen++] = (uint8_t)((v >> 16) & 0xFFu);
                    dec[dlen++] = (uint8_t)((v >> 8) & 0xFFu);
                } else if (rem_n == 2u && dlen + 1u <= sizeof(dec)) {
                    uint32_t v = ((uint32_t)buf[p] << 18) | ((uint32_t)buf[p + 1u] << 12);
                    dec[dlen++] = (uint8_t)((v >> 16) & 0xFFu);
                }
                uint32_t dhlen = 0u, drlen = 0u;
                gov_feed_bytes(dhist, &dhlen, V3_GOV_HIST_CAP, drem, &drlen, dec, dlen);
                uint32_t prev_dh = (dhlen > bound) ? (dhlen - bound) : 0u;
                if (gov_hist_violates(g, dhist, dhlen, prev_dh)) return 1;
            }
        }
    }

    /* 2. Streaming 2-phase Hex / \xNN / 0xNN tail check */
    if (has_hex) {
        uint32_t start = clen, cnt_hex = 0u;
        while (start > 0u && cnt_hex < 192u) {
            uint8_t ch = comb[start - 1u];
            if (gov_hex_val(ch) >= 0) { start--; cnt_hex++; continue; }
            if (gov_is_hex_sep(ch)) { start--; continue; }
            break;
        }
        if (cnt_hex >= 8u) {
            uint32_t nnib = 0u;
            for (uint32_t i = start; i < clen && nnib < 192u; ) {
                if (comb[i] == '0' && i + 2u < clen && (comb[i + 1u] == 'x' || comb[i + 1u] == 'X') &&
                    gov_hex_val(comb[i + 2u]) >= 0) {
                    i += 2u;
                    continue;
                }
                int v = gov_hex_val(comb[i]);
                if (v >= 0) buf[nnib++] = (uint8_t)v;
                i++;
            }
            if (nnib >= 8u) {
                for (uint32_t off = 0u; off < 2u && off + 8u <= nnib; off++) {
                    uint32_t dlen = 0u;
                    for (uint32_t p = off; p + 1u < nnib && dlen < sizeof(dec); p += 2u)
                        dec[dlen++] = (uint8_t)((buf[p] << 4) | buf[p + 1u]);
                    uint32_t dhlen = 0u, drlen = 0u;
                    gov_feed_bytes(dhist, &dhlen, V3_GOV_HIST_CAP, drem, &drlen, dec, dlen);
                    uint32_t prev_dh = (dhlen > bound) ? (dhlen - bound) : 0u;
                    if (gov_hist_violates(g, dhist, dhlen, prev_dh)) return 1;
                }
            }
        }
    }
    return 0;
}

void v3_govern_logits(const v3_governor *g, const v3_model *m, float *logits, uint32_t top_k_guard) {
    if (!g || !g->active || g->n_rules == 0u || !m || !logits) return;
    uint32_t n = m->vocab_size;
    uint32_t k = (top_k_guard == 0u || top_k_guard > 64u) ? 64u : top_k_guard;
    if (k > n) k = n;

    for (uint32_t pass = 0; pass < 4u; pass++) {
        uint32_t idx[64];
        float val[64];
        uint32_t cnt = 0;
        for (uint32_t i = 0; i < n; i++) {
            float lv = logits[i];
            if (lv <= -3.0e38f) continue;
            if (cnt < k) {
                uint32_t j = cnt++;
                while (j > 0 && val[j - 1] < lv) { val[j] = val[j - 1]; idx[j] = idx[j - 1]; j--; }
                val[j] = lv; idx[j] = i;
            } else if (lv > val[k - 1]) {
                uint32_t j = k - 1;
                while (j > 0 && val[j - 1] < lv) { val[j] = val[j - 1]; idx[j] = idx[j - 1]; j--; }
                val[j] = lv; idx[j] = i;
            }
        }
        int any_masked = 0;
        for (uint32_t c = 0; c < cnt; c++) {
            uint32_t tid = idx[c];
            const uint8_t *tb = 0; uint32_t tbl = 0;
            if (v3_token_bytes(m, tid, &tb, &tbl) != V3_OK || tbl == 0u) continue;
            uint8_t thist[V3_GOV_HIST_CAP];
            uint8_t trem[4];
            uint32_t thlen = g->hist_len, trlen = g->utf8_len;
            for (uint32_t i = 0; i < thlen; i++) thist[i] = g->hist[i];
            for (uint32_t i = 0; i < trlen; i++) trem[i] = g->utf8_rem[i];
            /* Keep at most 31 bytes of suffix before feeding candidate so rolling window never shifts out prev_hlen */
            if (thlen > 31u) {
                uint32_t drop = thlen - 31u;
                for (uint32_t i = drop; i < thlen; i++) thist[i - drop] = thist[i];
                thlen = 31u;
            }
            uint32_t prev_hlen = thlen;
            gov_feed_bytes(thist, &thlen, V3_GOV_HIST_CAP, trem, &trlen, tb, tbl);
            if (gov_hist_violates(g, thist, thlen, prev_hlen) || gov_encoded_tail_violates(g, tb, tbl)) {
                logits[tid] = -3.4e38f;
                any_masked = 1;
            }
        }
        if (!any_masked) break;
    }
}

void v3_gov_commit(v3_governor *g, const v3_model *m, uint32_t id) {
    if (!g || !g->active || !m) return;
    if (id == m->eos_id || id == m->im_end_id || id == m->im_start_id) {
        v3_gov_reset_stream(g);
        return;
    }
    const uint8_t *tb = 0; uint32_t tbl = 0;
    if (v3_token_bytes(m, id, &tb, &tbl) != V3_OK || tbl == 0u) return;
    gov_feed_bytes(g->hist, &g->hist_len, V3_GOV_HIST_CAP, g->utf8_rem, &g->utf8_len, tb, tbl);
    gov_push_raw(g, tb, tbl);
}

/* ============================================================================
 * Sampling
 * ========================================================================== */
uint32_t v3_argmax(const float *logits, uint32_t n) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
    return best;
}

uint32_t v3_sample_topk(const float *logits, uint32_t n, uint32_t k, float temp, uint64_t *rng) {
    if (temp <= 0.0f || k <= 1) return v3_argmax(logits, n);
    if (k > 64) k = 64;
    uint32_t idx[64];
    float val[64];
    uint32_t cnt = 0;
    for (uint32_t i = 0; i < n; i++) {
        float lv = logits[i];
        if (cnt < k) {
            uint32_t j = cnt++;
            while (j > 0 && val[j - 1] < lv) { val[j] = val[j - 1]; idx[j] = idx[j - 1]; j--; }
            val[j] = lv; idx[j] = i;
        } else if (lv > val[k - 1]) {
            uint32_t j = k - 1;
            while (j > 0 && val[j - 1] < lv) { val[j] = val[j - 1]; idx[j] = idx[j - 1]; j--; }
            val[j] = lv; idx[j] = i;
        }
    }
    float p[64], sum = 0.0f;
    for (uint32_t j = 0; j < cnt; j++) { p[j] = v3_expf((val[j] - val[0]) / temp); sum += p[j]; }
    uint64_t x = *rng ? *rng : 0x9E3779B97F4A7C15ull;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *rng = x;
    float r = (float)((x >> 40) & 0xFFFFFF) / 16777216.0f * sum;
    for (uint32_t j = 0; j < cnt; j++) { r -= p[j]; if (r <= 0.0f) return idx[j]; }
    return idx[cnt - 1];
}
