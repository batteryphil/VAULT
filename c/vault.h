/*
 * vault.h — Native Zero-Dependency C99 Header for the VAULT Paradigm
 * ==================================================================
 * VAULT: Variance-Weighted Associative Unitary Lattice Tensor
 * Copyright (c) 2026 Philip Bartsch (batteryphil) — MIT License.
 *
 * Unifies:
 *   1. Bit-Sliced 1.58-Bit Ternary {pos_mask, neg_mask} POPCNT Router (buildbit-os / synapse-os)
 *   2. Sparse Connectome Routing + FusedAPLNorm Homeostatic Inhibition (BioNPC)
 *   3. Co-Prime Residue Vaults P = {17, 19, 23, 29} / {83, 89, 97, 101} (prime-revisited)
 *   4. PRIME-Net High-Gain Kernel exp(1.5 * RMSNorm(x)) (gist-memory / PRIME-Net)
 *   5. Sherman-Morrison Orthogonal Nullspace Vault Cells (PRIME-Moment-Attention)
 *   6. Dyadic 256-Token GTRM Cascade via __builtin_ctzll (prime_gtrm.h + thalamic-bloom)
 */

#ifndef VAULT_H
#define VAULT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VAULT_NUM_BANKS 4
#define VAULT_BANK_0 17
#define VAULT_BANK_1 19
#define VAULT_BANK_2 23
#define VAULT_BANK_3 29
#define VAULT_TOTAL_CELLS (17 + 19 + 23 + 29) /* 88 vault cells; CRT product = 215,441 chords */
#define VAULT_DYADIC_LEVELS 12                /* 256 * 2^12 = 1,048,576 token macro horizon */

typedef struct {
    uint64_t pos_mask;
    uint64_t neg_mask;
} vault_trit64_t;

typedef struct {
    int d_key;                  /* Vault cell key dimension (e.g. 32 or 64) */
    int d_val;                  /* Vault cell value dimension (e.g. 64) */
    float ridge_mu;             /* Initial inverse covariance scale (e.g. 0.01f) */
    float innovation_thresh;    /* Surprise gate threshold (e.g. 0.01f) */
    float min_rank_ratio;       /* Minimum free nullspace rank before local rejuvenation */
    float rejuv_rate;           /* Local vault cell rejuvenation blend factor */
    size_t state_bytes;         /* Total bytes allocated for recurrent state */
    uint64_t step_count;        /* Total tokens processed */
    uint64_t novel_writes;      /* Count of novel writes locked into vaults */
    uint64_t skipped_writes;    /* Count of redundant writes skipped by Innovation Gate */

    /* APL Homeostatic Inhibition State (from BioNPC FusedAPLNorm) */
    float apl_state;

    /* Bit-Sliced Ternary Router Masks: [VAULT_TOTAL_CELLS] */
    vault_trit64_t router_masks[VAULT_TOTAL_CELLS];

    /* Co-Prime Nullspace Vault Cells (flattened contiguous arrays across all 88 cells):
     *   s_cells   : [VAULT_TOTAL_CELLS, d_val, d_key]
     *   p_cells   : [VAULT_TOTAL_CELLS, d_key, d_key]
     *   cell_load : [VAULT_TOTAL_CELLS]
     */
    float *s_cells;
    float *p_cells;
    float *cell_load;

    /* Dyadic 256-Token GTRM Macro-Cascade (from prime_gtrm.h + thalamic-bloom):
     *   gtrm_curr   : [d_key, d_val]
     *   gtrm_levels : [VAULT_DYADIC_LEVELS, d_key, d_val]
     */
    float *gtrm_curr;
    float *gtrm_levels;
} vault_state_t;

/* Allocate and initialize VAULT state */
vault_state_t* vault_create(int d_key, int d_val);

/* Reset VAULT state to initial condition */
void vault_reset(vault_state_t *st);

/* Free VAULT state */
void vault_free(vault_state_t *st);

/* Route a key vector [d_key] to 4 co-prime vault cell indices via Ternary POPCNT */
void vault_route(
    const vault_state_t *st,
    const float *k_in,
    int out_cells[VAULT_NUM_BANKS]
);

/* O(1) Single-Step Write (with Innovation Gate + Sherman-Morrison Nullspace + Dyadic GTRM) */
int vault_write_step(
    vault_state_t *st,
    const float *k_in,
    const float *v_in
);

/* O(1) Single-Step Read (with PRIME-Net Discovered Variance-Weighted Harmonic Consensus) */
void vault_read_step(
    const vault_state_t *st,
    const float *q_in,
    float *v_out
);

#ifdef __cplusplus
}
#endif

#endif /* VAULT_H */
