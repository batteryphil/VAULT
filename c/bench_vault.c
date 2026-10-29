/*
 * bench_vault.c — 100,000-Token C99 Head-to-Head Paradigm Benchmark for VAULT
 * ===========================================================================
 * VAULT: Variance-Weighted Associative Unitary Lattice Tensor
 * Copyright (c) 2026 Philip Bartsch (batteryphil) — MIT License.
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "vault.h"
#include "../../PRIME-Moment-Attention/c/prime_moment.h"

#define D_KEY 32
#define D_VAL 64
#define VOCAB_UNIQUE 512
#define ZIPF_COMMON 32

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint64_t rng_state = 0xCAFEBABE2026ULL;
static uint64_t next_u64(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static float rand_normal(void) {
    float u1 = ((float)(next_u64() & 0xFFFFFFULL) + 1.0f) / 16777217.0f;
    float u2 = ((float)(next_u64() & 0xFFFFFFULL) + 1.0f) / 16777217.0f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307f * u2);
}

static float cosine_sim(const float *a, const float *b, int dim) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na < 1e-12f || nb < 1e-12f) return 0.0f;
    return dot / sqrtf(na * nb);
}

typedef struct {
    float S[D_VAL * D_KEY];
    float P[D_KEY * D_KEY];
} dense_rls_t;

static void dense_rls_init(dense_rls_t *rls) {
    memset(rls->S, 0, sizeof(rls->S));
    memset(rls->P, 0, sizeof(rls->P));
    for (int i = 0; i < D_KEY; i++) rls->P[i * D_KEY + i] = 100.0f;
}

static void dense_rls_write(dense_rls_t *rls, const float *k_in, const float *v_in) {
    float kn[D_KEY];
    float sq = 0.0f;
    for (int i = 0; i < D_KEY; i++) sq += k_in[i] * k_in[i];
    float inv = 1.0f / sqrtf(sq + 1e-12f);
    for (int i = 0; i < D_KEY; i++) kn[i] = k_in[i] * inv;

    float tr = 0.0f;
    for (int i = 0; i < D_KEY; i++) tr += rls->P[i * D_KEY + i];
    if (tr < 0.08f * D_KEY * 100.0f) {
        for (int i = 0; i < D_KEY * D_KEY; i++) rls->P[i] *= 0.85f;
        for (int i = 0; i < D_KEY; i++) rls->P[i * D_KEY + i] += 15.0f;
    }

    float err[D_VAL], u[D_KEY];
    for (int i = 0; i < D_VAL; i++) {
        float vh = 0.0f;
        for (int j = 0; j < D_KEY; j++) vh += rls->S[i * D_KEY + j] * kn[j];
        err[i] = v_in[i] - vh;
    }
    float k_dot_u = 0.0f;
    for (int i = 0; i < D_KEY; i++) {
        float sum = 0.0f;
        for (int j = 0; j < D_KEY; j++) sum += rls->P[i * D_KEY + j] * kn[j];
        u[i] = sum;
        k_dot_u += kn[i] * sum;
    }
    float gamma = 1.0f / (1.0f + (k_dot_u > 0.0f ? k_dot_u : 0.0f));
    for (int i = 0; i < D_VAL; i++) {
        for (int j = 0; j < D_KEY; j++) rls->S[i * D_KEY + j] += gamma * err[i] * u[j];
    }
    for (int i = 0; i < D_KEY; i++) {
        for (int j = 0; j < D_KEY; j++) rls->P[i * D_KEY + j] -= gamma * u[i] * u[j];
    }
}

static void dense_rls_read(const dense_rls_t *rls, const float *q_in, float *v_out) {
    float qn[D_KEY];
    float sq = 0.0f;
    for (int i = 0; i < D_KEY; i++) sq += q_in[i] * q_in[i];
    float inv = 1.0f / sqrtf(sq + 1e-12f);
    for (int i = 0; i < D_KEY; i++) qn[i] = q_in[i] * inv;
    for (int i = 0; i < D_VAL; i++) {
        float sum = 0.0f;
        for (int j = 0; j < D_KEY; j++) sum += rls->S[i * D_KEY + j] * qn[j];
        v_out[i] = sum;
    }
}

typedef struct {
    float M[D_VAL * D_KEY];
    float Z[D_KEY];
    int use_exp_kernel;
    float decay;
} flat_linear_t;

static void flat_linear_transform(const float *x, float *phi, int use_exp) {
    float sq = 0.0f;
    for (int i = 0; i < D_KEY; i++) sq += x[i] * x[i];
    float rms = 1.0f / sqrtf(sq / (float)D_KEY + 1e-6f);
    for (int i = 0; i < D_KEY; i++) {
        float v = x[i] * rms;
        if (v > 3.0f) v = 3.0f;
        if (v < -3.0f) v = -3.0f;
        if (use_exp) {
            phi[i] = expf(1.5f * v);
        } else {
            phi[i] = (v >= 0.0f) ? (v + 1.0f) : expf(v);
        }
    }
}

static void flat_linear_write(flat_linear_t *fl, const float *k_in, const float *v_in) {
    float phi[D_KEY];
    flat_linear_transform(k_in, phi, fl->use_exp_kernel);
    float d = fl->decay;
    for (int j = 0; j < D_KEY; j++) fl->Z[j] = d * fl->Z[j] + phi[j];
    for (int i = 0; i < D_VAL; i++) {
        for (int j = 0; j < D_KEY; j++) {
            fl->M[i * D_KEY + j] = d * fl->M[i * D_KEY + j] + v_in[i] * phi[j];
        }
    }
}

static void flat_linear_read(const flat_linear_t *fl, const float *q_in, float *v_out) {
    float phi[D_KEY];
    flat_linear_transform(q_in, phi, fl->use_exp_kernel);
    float den = 1e-4f;
    for (int j = 0; j < D_KEY; j++) den += phi[j] * fl->Z[j];
    float inv_den = 1.0f / den;
    for (int i = 0; i < D_VAL; i++) {
        float num = 0.0f;
        for (int j = 0; j < D_KEY; j++) num += fl->M[i * D_KEY + j] * phi[j];
        v_out[i] = num * inv_den;
    }
}

int main(void) {
    printf("=========================================================================================\n");
    printf("VAULT vs. EXISTING PARADIGMS: 100,000-Token Associative Horizon Benchmark (C99)\n");
    printf("=========================================================================================\n");

    float *keys = (float*)malloc(VOCAB_UNIQUE * D_KEY * sizeof(float));
    float *vals = (float*)malloc(VOCAB_UNIQUE * D_VAL * sizeof(float));
    for (int idx = 0; idx < VOCAB_UNIQUE; idx++) {
        float k_sq = 0.0f, v_sq = 0.0f;
        for (int j = 0; j < D_KEY; j++) {
            float r = rand_normal();
            keys[idx * D_KEY + j] = r;
            k_sq += r * r;
        }
        for (int j = 0; j < D_VAL; j++) {
            float r = rand_normal();
            vals[idx * D_VAL + j] = r;
            v_sq += r * r;
        }
        float inv_k = 1.0f / sqrtf(k_sq);
        float inv_v = 1.0f / sqrtf(v_sq);
        for (int j = 0; j < D_KEY; j++) keys[idx * D_KEY + j] *= inv_k;
        for (int j = 0; j < D_VAL; j++) vals[idx * D_VAL + j] *= inv_v;
    }

    int horizons[10] = {100, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000};
    int n_horizons = 10;
    int n_trials = 24;

    vault_state_t *vlt = vault_create(D_KEY, D_VAL);
    prime_config_t pm_cfg = prime_default_config(1, D_VAL);
    pm_cfg.decay = 0.9995f;
    pm_cfg.use_delta_rule = 0;
    prime_state_t *pm_state = prime_state_create(1, D_VAL);

    dense_rls_t rls;
    flat_linear_t lin_elu = { .use_exp_kernel = 0, .decay = 0.9995f };
    flat_linear_t gist_exp = { .use_exp_kernel = 1, .decay = 0.9995f };

    printf("State Memory Footprints:\n");
    printf("  - VAULT Total State        : %.2f KB (Active L1 per step: %.2f KB across 4 Co-Prime vaults)\n",
           (double)vlt->state_bytes / 1024.0,
           (double)(4 * (D_VAL * D_KEY + D_KEY * D_KEY) * sizeof(float)) / 1024.0);
    printf("  - PRIME-Moment (12-layer)  : 4,720.00 KB (393.2 KB/layer for H=12, D=64)\n");
    printf("-----------------------------------------------------------------------------------------\n");
    printf("%-10s | %-12s | %-12s | %-14s | %-14s | %-16s\n",
           "Horizon T", "Linear(ELU)", "Gist(exp)", "PRIME-Moment", "Dense-RLS", "VAULT (C99)");
    printf("-----------------------------------------------------------------------------------------\n");

    FILE *fp_json = fopen("benchmarks/c99_100k_results.json", "w");
    if (fp_json) fprintf(fp_json, "{\n  \"horizons\": [\n");

    for (int h_idx = 0; h_idx < n_horizons; h_idx++) {
        int T = horizons[h_idx];

        vault_reset(vlt);
        prime_state_reset(pm_state);
        dense_rls_init(&rls);
        memset(&lin_elu.M, 0, sizeof(lin_elu.M));
        memset(&lin_elu.Z, 0, sizeof(lin_elu.Z));
        memset(&gist_exp.M, 0, sizeof(gist_exp.M));
        memset(&gist_exp.Z, 0, sizeof(gist_exp.Z));

        for (int t = 0; t < n_trials; t++) {
            const float *k_t = keys + t * D_KEY;
            const float *v_t = vals + t * D_VAL;

            vault_write_step(vlt, k_t, v_t);
            dense_rls_write(&rls, k_t, v_t);
            flat_linear_write(&lin_elu, k_t, v_t);
            flat_linear_write(&gist_exp, k_t, v_t);

            float k_pad[D_VAL] = {0}, q_pad[D_VAL] = {0}, out_pad[D_VAL] = {0};
            memcpy(k_pad, k_t, D_KEY * sizeof(float));
            memcpy(q_pad, k_t, D_KEY * sizeof(float));
            prime_step(&pm_cfg, pm_state, q_pad, k_pad, v_t, out_pad);
        }

        for (int step = 0; step < T; step++) {
            int tok_id;
            if ((next_u64() % 100ULL) < 85ULL) {
                tok_id = n_trials + (int)(next_u64() % ZIPF_COMMON);
            } else {
                tok_id = n_trials + ZIPF_COMMON + (int)(next_u64() % (VOCAB_UNIQUE - n_trials - ZIPF_COMMON));
            }
            const float *k_t = keys + tok_id * D_KEY;
            const float *v_t = vals + tok_id * D_VAL;

            vault_write_step(vlt, k_t, v_t);
            dense_rls_write(&rls, k_t, v_t);
            flat_linear_write(&lin_elu, k_t, v_t);
            flat_linear_write(&gist_exp, k_t, v_t);

            float k_pad[D_VAL] = {0}, out_pad[D_VAL] = {0};
            memcpy(k_pad, k_t, D_KEY * sizeof(float));
            prime_step(&pm_cfg, pm_state, k_pad, k_pad, v_t, out_pad);
        }

        float sum_lin = 0.0f, sum_gist = 0.0f, sum_pm = 0.0f, sum_rls = 0.0f, sum_vlt = 0.0f;
        int top1_vlt = 0, top1_rls = 0, top1_pm = 0;

        for (int t = 0; t < n_trials; t++) {
            const float *q_t = keys + t * D_KEY;
            const float *v_true = vals + t * D_VAL;
            float v_pred[D_VAL];

            flat_linear_read(&lin_elu, q_t, v_pred);
            sum_lin += cosine_sim(v_pred, v_true, D_VAL);

            flat_linear_read(&gist_exp, q_t, v_pred);
            sum_gist += cosine_sim(v_pred, v_true, D_VAL);

            {
                float q_pad[D_VAL] = {0}, q_sq[D_VAL] = {0}, num[D_VAL];
                memcpy(q_pad, q_t, D_KEY * sizeof(float));
                float inv_sqrt = 1.0f / sqrtf((float)D_VAL);
                for (int d = 0; d < D_VAL; d++) {
                    q_pad[d] *= inv_sqrt;
                    q_sq[d] = q_pad[d] * q_pad[d];
                    num[d] = pm_state->s0[d];
                }
                for (int d = 0; d < D_VAL; d++) {
                    const float *s1_r = pm_state->s1 + d * D_VAL;
                    const float *s2_r = pm_state->s2 + d * D_VAL;
                    for (int e = 0; e < D_VAL; e++) {
                        num[e] += q_pad[d] * s1_r[e] + 0.5f * q_sq[d] * s2_r[e];
                    }
                }
                float c_pm = cosine_sim(num, v_true, D_VAL);
                sum_pm += c_pm;
                int best_id = 0; float best_c = -2.0f;
                for (int cand = 0; cand < VOCAB_UNIQUE; cand++) {
                    float c = cosine_sim(num, vals + cand * D_VAL, D_VAL);
                    if (c > best_c) { best_c = c; best_id = cand; }
                }
                if (best_id == t) top1_pm++;
            }

            dense_rls_read(&rls, q_t, v_pred);
            sum_rls += cosine_sim(v_pred, v_true, D_VAL);
            {
                int best_id = 0; float best_c = -2.0f;
                for (int cand = 0; cand < VOCAB_UNIQUE; cand++) {
                    float c = cosine_sim(v_pred, vals + cand * D_VAL, D_VAL);
                    if (c > best_c) { best_c = c; best_id = cand; }
                }
                if (best_id == t) top1_rls++;
            }

            vault_read_step(vlt, q_t, v_pred);
            sum_vlt += cosine_sim(v_pred, v_true, D_VAL);
            {
                int best_id = 0; float best_c = -2.0f;
                for (int cand = 0; cand < VOCAB_UNIQUE; cand++) {
                    float c = cosine_sim(v_pred, vals + cand * D_VAL, D_VAL);
                    if (c > best_c) { best_c = c; best_id = cand; }
                }
                if (best_id == t) top1_vlt++;
            }
        }

        float avg_lin  = sum_lin / n_trials;
        float avg_gist = sum_gist / n_trials;
        float avg_pm   = sum_pm / n_trials;
        float avg_rls  = sum_rls / n_trials;
        float avg_vlt  = sum_vlt / n_trials;
        float acc_vlt  = 100.0f * (float)top1_vlt / (float)n_trials;

        printf("%-10d | %12.4f | %12.4f | %14.4f | %14.4f | %8.4f (%5.1f%%)\n",
               T, avg_lin, avg_gist, avg_pm, avg_rls, avg_vlt, acc_vlt);

        if (fp_json) {
            fprintf(fp_json,
                "    {\"T\": %d, \"linear_elu\": %.5f, \"gist_exp\": %.5f, \"prime_moment\": %.5f, "
                "\"dense_rls\": %.5f, \"vault_cos\": %.5f, \"vault_top1_pct\": %.2f}%s\n",
                T, avg_lin, avg_gist, avg_pm, avg_rls, avg_vlt, acc_vlt,
                (h_idx + 1 < n_horizons) ? "," : "");
        }
    }

    int bench_steps = 200000;
    vault_reset(vlt);
    double t0 = now_sec();
    for (int s = 0; s < bench_steps; s++) {
        int id = s & (ZIPF_COMMON - 1);
        vault_write_step(vlt, keys + id * D_KEY, vals + id * D_VAL);
    }
    double elapsed = now_sec() - t0;
    double tok_per_sec = (double)bench_steps / elapsed;
    double skip_pct = 100.0 * (double)vlt->skipped_writes / (double)vlt->step_count;

    printf("-----------------------------------------------------------------------------------------\n");
    printf("C99 Throughput (200k tokens)   : %.0f tok/s (%.3f us/token)\n",
           tok_per_sec, (elapsed * 1e6) / bench_steps);
    printf("Innovation Gate Redundancy Skip: %.2f%% of redundant writes bypassed automatically!\n",
           skip_pct);
    printf("=========================================================================================\n");

    if (fp_json) {
        fprintf(fp_json, "  ],\n  \"c99_tok_per_sec\": %.1f,\n  \"c99_us_per_tok\": %.4f,\n  \"skip_pct\": %.2f,\n  \"state_kb\": %.2f\n}\n",
                tok_per_sec, (elapsed * 1e6) / bench_steps, skip_pct, (double)vlt->state_bytes / 1024.0);
        fclose(fp_json);
    }

    vault_free(vlt);
    prime_state_free(pm_state);
    free(keys);
    free(vals);
    return 0;
}
