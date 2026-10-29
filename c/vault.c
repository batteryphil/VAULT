/*
 * vault.c — Native Zero-Dependency C99 Implementation of the VAULT Paradigm
 * =========================================================================
 * VAULT: Variance-Weighted Associative Unitary Lattice Tensor
 * Copyright (c) 2026 Philip Bartsch (batteryphil) — MIT License.
 */

#include "vault.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_VAULT_DIM 128

static const int BANK_SIZES[VAULT_NUM_BANKS] = {
    VAULT_BANK_0, VAULT_BANK_1, VAULT_BANK_2, VAULT_BANK_3
};

static const int BANK_OFFSETS[VAULT_NUM_BANKS] = {
    0,
    VAULT_BANK_0,
    VAULT_BANK_0 + VAULT_BANK_1,
    VAULT_BANK_0 + VAULT_BANK_1 + VAULT_BANK_2
};

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

vault_state_t* vault_create(int d_key, int d_val) {
    if (d_key <= 0 || d_key > 64 || d_val <= 0 || d_val > MAX_VAULT_DIM) {
        return NULL;
    }

    vault_state_t *st = (vault_state_t*)calloc(1, sizeof(vault_state_t));
    if (!st) return NULL;

    st->d_key = d_key;
    st->d_val = d_val;
    st->ridge_mu = 0.01f;
    st->innovation_thresh = 0.01f;
    st->min_rank_ratio = 0.08f;
    st->rejuv_rate = 0.15f;

    size_t s_bytes = (size_t)VAULT_TOTAL_CELLS * d_val * d_key * sizeof(float);
    size_t p_bytes = (size_t)VAULT_TOTAL_CELLS * d_key * d_key * sizeof(float);
    size_t load_bytes = (size_t)VAULT_TOTAL_CELLS * sizeof(float);
    size_t gtrm_curr_bytes = (size_t)d_key * d_val * sizeof(float);
    size_t gtrm_lvl_bytes = (size_t)VAULT_DYADIC_LEVELS * d_key * d_val * sizeof(float);

    st->state_bytes = s_bytes + p_bytes + load_bytes + gtrm_curr_bytes + gtrm_lvl_bytes;

    st->s_cells = (float*)calloc(VAULT_TOTAL_CELLS * d_val * d_key, sizeof(float));
    st->p_cells = (float*)calloc(VAULT_TOTAL_CELLS * d_key * d_key, sizeof(float));
    st->cell_load = (float*)calloc(VAULT_TOTAL_CELLS, sizeof(float));
    st->gtrm_curr = (float*)calloc(d_key * d_val, sizeof(float));
    st->gtrm_levels = (float*)calloc(VAULT_DYADIC_LEVELS * d_key * d_val, sizeof(float));

    if (!st->s_cells || !st->p_cells || !st->cell_load || !st->gtrm_curr || !st->gtrm_levels) {
        vault_free(st);
        return NULL;
    }

    uint64_t rng = 0x2026BA77E29FULL;
    uint64_t valid_bits = (d_key == 64) ? ~0ULL : ((1ULL << d_key) - 1ULL);
    for (int c = 0; c < VAULT_TOTAL_CELLS; c++) {
        uint64_t r1 = splitmix64(&rng) & valid_bits;
        uint64_t r2 = splitmix64(&rng) & valid_bits;
        st->router_masks[c].pos_mask = r1 & ~r2;
        st->router_masks[c].neg_mask = r2 & ~r1;
    }

    vault_reset(st);
    return st;
}

void vault_reset(vault_state_t *st) {
    if (!st) return;
    int K = st->d_key;
    int V = st->d_val;
    float inv_mu = 1.0f / st->ridge_mu;

    memset(st->s_cells, 0, (size_t)VAULT_TOTAL_CELLS * V * K * sizeof(float));
    memset(st->p_cells, 0, (size_t)VAULT_TOTAL_CELLS * K * K * sizeof(float));
    memset(st->cell_load, 0, (size_t)VAULT_TOTAL_CELLS * sizeof(float));
    memset(st->gtrm_curr, 0, (size_t)K * V * sizeof(float));
    memset(st->gtrm_levels, 0, (size_t)VAULT_DYADIC_LEVELS * K * V * sizeof(float));

    for (int c = 0; c < VAULT_TOTAL_CELLS; c++) {
        float *P = st->p_cells + c * K * K;
        for (int i = 0; i < K; i++) {
            P[i * K + i] = inv_mu;
        }
    }

    st->apl_state = 0.0f;
    st->step_count = 0;
    st->novel_writes = 0;
    st->skipped_writes = 0;
}

void vault_free(vault_state_t *st) {
    if (!st) return;
    free(st->s_cells);
    free(st->p_cells);
    free(st->cell_load);
    free(st->gtrm_curr);
    free(st->gtrm_levels);
    free(st);
}

static void apply_prime_kernel_c(const float *k_in, float *phi_out, int K) {
    float sq_sum = 0.0f;
    for (int i = 0; i < K; i++) {
        sq_sum += k_in[i] * k_in[i];
    }
    float rms = 1.0f / sqrtf(sq_sum / (float)K + 1e-6f);

    float norm_sq = 0.0f;
    for (int i = 0; i < K; i++) {
        float x = k_in[i] * rms;
        if (x > 3.0f) x = 3.0f;
        if (x < -3.0f) x = -3.0f;
        float e = expf(1.5f * x);
        phi_out[i] = e;
        norm_sq += e * e;
    }
    float inv_norm = 1.0f / sqrtf(norm_sq + 1e-12f);
    for (int i = 0; i < K; i++) {
        phi_out[i] *= inv_norm;
    }
}

void vault_route(
    const vault_state_t *st,
    const float *k_in,
    int out_cells[VAULT_NUM_BANKS]
) {
    int K = st->d_key;
    float abs_sum = 0.0f;
    for (int i = 0; i < K; i++) {
        abs_sum += fabsf(k_in[i]);
    }
    float thresh = 0.25f * (abs_sum / (float)K);

    uint64_t xp = 0ULL;
    uint64_t xn = 0ULL;
    for (int i = 0; i < K; i++) {
        if (k_in[i] > thresh)       xp |= (1ULL << i);
        else if (k_in[i] < -thresh) xn |= (1ULL << i);
    }

    for (int h = 0; h < VAULT_NUM_BANKS; h++) {
        int P_h = BANK_SIZES[h];
        int off = BANK_OFFSETS[h];
        int best_c = 0;
        int best_score = -999999;

        for (int c = 0; c < P_h; c++) {
            uint64_t wp = st->router_masks[off + c].pos_mask;
            uint64_t wn = st->router_masks[off + c].neg_mask;
            int pos_hits = __builtin_popcountll((wp & xp) | (wn & xn));
            int neg_hits = __builtin_popcountll((wp & xn) | (wn & xp));
            int score = pos_hits - neg_hits;
            if (score > best_score) {
                best_score = score;
                best_c = c;
            }
        }
        out_cells[h] = off + best_c;
    }
}

int vault_write_step(
    vault_state_t *st,
    const float *k_in,
    const float *v_in
) {
    int K = st->d_key;
    int V = st->d_val;
    st->step_count++;

    int active_cells[VAULT_NUM_BANKS];
    vault_route(st, k_in, active_cells);

    float phi_k[MAX_VAULT_DIM];
    apply_prime_kernel_c(k_in, phi_k, K);

    float err_vec[MAX_VAULT_DIM];
    float u_vec[MAX_VAULT_DIM];
    float max_trace = (float)K / st->ridge_mu;
    float inv_mu = 1.0f / st->ridge_mu;
    int wrote_any = 0;

    for (int h = 0; h < VAULT_NUM_BANKS; h++) {
        int cell_idx = active_cells[h];
        float *S = st->s_cells + (size_t)cell_idx * V * K;
        float *P = st->p_cells + (size_t)cell_idx * K * K;

        float err_sq = 0.0f;
        for (int i = 0; i < V; i++) {
            const float *s_row = S + i * K;
            float v_hat = 0.0f;
            for (int j = 0; j < K; j++) {
                v_hat += s_row[j] * phi_k[j];
            }
            float d = v_in[i] - v_hat;
            err_vec[i] = d;
            err_sq += d * d;
        }

        if (err_sq < st->innovation_thresh * st->innovation_thresh) {
            continue;
        }
        wrote_any = 1;

        float trace = 0.0f;
        for (int i = 0; i < K; i++) {
            trace += P[i * K + i];
        }
        if (trace < st->min_rank_ratio * max_trace) {
            float keep = 1.0f - st->rejuv_rate;
            float add_diag = st->rejuv_rate * inv_mu;
            for (int i = 0; i < K * K; i++) P[i] *= keep;
            for (int i = 0; i < K; i++) P[i * K + i] += add_diag;
            st->cell_load[cell_idx] *= keep;
        }

        float k_dot_u = 0.0f;
        for (int i = 0; i < K; i++) {
            const float *p_row = P + i * K;
            float sum = 0.0f;
            for (int j = 0; j < K; j++) {
                sum += p_row[j] * phi_k[j];
            }
            u_vec[i] = sum;
            k_dot_u += phi_k[i] * sum;
        }
        if (k_dot_u < 1e-6f) k_dot_u = 1e-6f;
        /* Exact Nullspace Delta Overwrite:
         * u_vec = P * phi_k is already orthogonal to all other keys in this cell.
         * Scaling S update by 1 / k_dot_u sets S * phi_k = v_in exactly (100% overwrite
         * of stale values for an existing key with zero interference on other keys),
         * while P only needs to project out phi_k when phi_k is a novel key direction (k_dot_u > 2.0f). */
        float gamma_s = 1.0f / k_dot_u;

        for (int i = 0; i < V; i++) {
            float *s_row = S + i * K;
            float g_err = gamma_s * err_vec[i];
            for (int j = 0; j < K; j++) {
                s_row[j] += g_err * u_vec[j];
            }
        }

        if (k_dot_u > 2.0f) {
            float gamma_p = 1.0f / (1.0f + k_dot_u);
            for (int i = 0; i < K; i++) {
                float *p_row = P + i * K;
                float g_ui = gamma_p * u_vec[i];
                for (int j = 0; j < K; j++) {
                    p_row[j] -= g_ui * u_vec[j];
                }
            }
            st->cell_load[cell_idx] += 1.0f;
        }
    }

    if (wrote_any) st->novel_writes++;
    else           st->skipped_writes++;

    for (int i = 0; i < K; i++) {
        float pk = 0.004f * phi_k[i];
        float *g_row = st->gtrm_curr + i * V;
        for (int j = 0; j < V; j++) {
            g_row[j] = 0.996f * g_row[j] + pk * v_in[j];
        }
    }

    if ((st->step_count & 255ULL) == 0ULL) {
        uint64_t chunk_idx = st->step_count >> 8;
        int tz = __builtin_ctzll(chunk_idx);
        if (tz >= VAULT_DYADIC_LEVELS) tz = VAULT_DYADIC_LEVELS - 1;
        size_t kv_len = (size_t)K * V;
        for (int lvl = 0; lvl <= tz; lvl++) {
            float *dst = st->gtrm_levels + (size_t)lvl * kv_len;
            const float *src = (lvl == 0) ? st->gtrm_curr : (st->gtrm_levels + (size_t)(lvl - 1) * kv_len);
            for (size_t idx = 0; idx < kv_len; idx++) {
                dst[idx] = 0.5f * dst[idx] + 0.5f * src[idx];
            }
        }
    }

    return wrote_any;
}

void vault_read_step(
    const vault_state_t *st,
    const float *q_in,
    float *v_out
) {
    int K = st->d_key;
    int V = st->d_val;

    int active_cells[VAULT_NUM_BANKS];
    vault_route(st, q_in, active_cells);

    float phi_q[MAX_VAULT_DIM];
    apply_prime_kernel_c(q_in, phi_q, K);

    float preds[VAULT_NUM_BANKS][MAX_VAULT_DIM];
    float loads[VAULT_NUM_BANKS];
    float mean_load = 0.0f;

    for (int h = 0; h < VAULT_NUM_BANKS; h++) {
        int cell_idx = active_cells[h];
        const float *S = st->s_cells + (size_t)cell_idx * V * K;
        loads[h] = st->cell_load[cell_idx];
        mean_load += loads[h];

        for (int i = 0; i < V; i++) {
            const float *s_row = S + i * K;
            float sum = 0.0f;
            for (int j = 0; j < K; j++) {
                sum += s_row[j] * phi_q[j];
            }
            preds[h][i] = sum;
        }
    }
    mean_load = (mean_load * 0.25f) + 1.0f;

    float v_med[MAX_VAULT_DIM];
    for (int i = 0; i < V; i++) {
        float a = preds[0][i], b = preds[1][i], c = preds[2][i], d = preds[3][i];
        float t;
        if (a > b) { t = a; a = b; b = t; }
        if (c > d) { t = c; c = d; d = t; }
        if (a > c) { t = a; a = c; c = t; }
        if (b > d) { t = b; b = d; d = t; }
        if (b > c) { t = b; b = c; c = t; }
        v_med[i] = 0.5f * (b + c);
    }

    float weights[VAULT_NUM_BANKS];
    float w_sum = 0.0f;
    for (int h = 0; h < VAULT_NUM_BANKS; h++) {
        float dev2 = 0.0f;
        for (int i = 0; i < V; i++) {
            float diff = preds[h][i] - v_med[i];
            dev2 += diff * diff;
        }
        dev2 /= (float)V;
        float norm_load = loads[h] / mean_load;
        float w = 1.0f / (dev2 + 0.01f * norm_load + 1e-4f);
        weights[h] = w;
        w_sum += w;
    }

    float inv_w_sum = 1.0f / (w_sum + 1e-12f);
    for (int i = 0; i < V; i++) {
        float acc = 0.0f;
        for (int h = 0; h < VAULT_NUM_BANKS; h++) {
            acc += weights[h] * preds[h][i];
        }
        v_out[i] = acc * inv_w_sum;
    }
}
