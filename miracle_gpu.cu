/**
 * miracle_gpu.cu: TO-DO.
 * 
 * Copyright (c) Michele Collevati
 */


#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <limits.h>
#include <thrust/sort.h>


#include "cnf_formula_gpu.cuh"
#include "miracle_gpu.cuh"
#include "utils.cuh"
#include "launch_parameters_gpu.cuh"


/**
 * Global variables
 */


__device__ Lit *d_lits;                 // Device array of assigned literals.
__device__ int d_lits_len;              /**
                                         * Device length of d_lits, which is
                                         * the number of assigned literals.
                                         */

__device__ int *d_lit_occ;              // Device array of literal occurrences.
__device__ int d_lit_occ_len;           /**
                                         * Device length of d_lit_occ, which is
                                         * mrc->phi->num_vars * 2.
                                         */
static int lit_occ_len;                 /**
                                         * Length of d_lit_occ, which is
                                         * mrc->phi->num_vars * 2.
                                         */

__device__ int *d_cum_lit_occ;          /**
                                         * Device array of cumulative literal
                                         * occurrences.
                                         */
__device__ int d_cum_lit_occ_len;       /**
                                         * Device length of d_cum_lit_occ,
                                         * which is mrc->phi->num_vars * 2.
                                         */
static int cum_lit_occ_len;             /**
                                         * Length of d_cum_lit_occ, which is
                                         * mrc->phi->num_vars * 2.
                                         */

__device__ float *d_lit_weights;        // Device array of literal weights.
__device__ int d_lit_weights_len;       /**
                                         * Device length of d_lit_weights,
                                         * which is mrc->phi->num_vars * 2.
                                         */
static int lit_weights_len;             /**
                                         * Length of d_lit_weights, which is
                                         * mrc->phi->num_vars * 2.
                                         */

__device__ int *d_var_occ;              /**
                                         * Device array of variable
                                         * occurrences.
                                         */
__device__ int d_var_occ_len;           /**
                                         * Device length of d_var_occ, which is
                                         * mrc->phi->num_vars.
                                         */
static int var_occ_len;                 /**
                                         * Length of d_var_occ, which is
                                         * mrc->phi->num_vars.
                                         */

__device__ float *d_var_weights;        // Device array of variable weights.
__device__ int d_var_weights_len;       /**
                                         * Device length of d_var_weights,
                                         * which is mrc->phi->num_vars.
                                         */
static int var_weights_len;             /**
                                         * Length of d_var_weights, which is
                                         * mrc->phi->num_vars.
                                         */

__device__ int *d_var_availability;     /**
                                         * Device array of variable
                                         * availability.
                                         */
__device__ int d_var_availability_len;  /**
                                         * Device length of d_var_availability,
                                         * which is mrc->phi->num_vars.
                                         */
static int var_availability_len;        /**
                                         * Length of d_var_availability, which
                                         * is mrc->phi->num_vars.
                                         */

__device__ int *d_clause_sizes;         // Device array of clause sizes.
__device__ int *d_clause_sizes_copy;    // Device copy of array d_clause_sizes.
__device__ int d_clause_sizes_len;      /**
                                         * Device length of d_clause_sizes,
                                         * which is mrc->phi->num_clauses.
                                         */
static int clause_sizes_len;            /**
                                         * Length of d_clause_sizes, which is
                                         * mrc->phi->num_clauses.
                                         */

__device__ int *d_clause_idxs;          // Device array of clause indices.
__device__ int d_clause_idxs_len;       /**
                                         * Device length of d_clause_idxs,
                                         * which is mrc->phi->num_clauses.
                                         */
static int clause_idxs_len;             /**
                                         * Length of d_clause_idxs, which is
                                         * mrc->phi->num_clauses.
                                         */

__device__ int *d_clause_indices;       /**
                                         * Device indexing array of clauses
                                         * ordered by size.
                                         */
static int *clause_indices;             /**
                                         * Indexing array of clauses ordered by
                                         * size.
                                         */
__device__ int d_clause_indices_len;    /**
                                         * Device current length of
                                         * d_clause_indices and clause_indices,
                                         * which is the number of different
                                         * sizes + 1.
                                         */
static int clause_indices_len;          /**
                                         * Current length of d_clause_indices
                                         * and clause_indices, which is the
                                         * number of different sizes + 1.
                                         */


/**
 * Auxiliary function prototypes
 */


/**
 * @brief Initializes auxiliary data structures.
 * 
 * @param [in]mrc A miracle.
 * @retval None.
 */
static void init_aux_data_structs(Miracle *mrc);


/**
 * @brief Destroys auxiliary data structures.
 * 
 * @retval None.
 */
static void destroy_aux_data_structs();


/**
 * @brief If two_sided = true, computes the JW-TS heuristic on the device,
 * otherwise computes the JW-OS heuristic on the device.
 * 
 * @param [in]d_mrc A device miracle.
 * @param [in]two_sided A flag to choose JW-TS or JW-OS.
 * @retval The branching literal.
 */
static Lit JW_xS_heuristic(Miracle *d_mrc, bool two_sided);


/**
 * @brief If dlcs = true, computes the DLCS heuristic on the device,
 * otherwise computes the DLIS heuristic on the device.
 * 
 * @param [in]d_mrc A device miracle.
 * @param [in]dlcs A flag to choose DLCS or DLIS.
 * @retval The branching literal.
 */
static Lit DLxS_heuristic(Miracle *d_mrc, bool dlcs);


/**
 * @brief If rdlcs = true, computes the RDLCS heuristic on the device,
 * otherwise computes the RDLIS heuristic on the device.
 * 
 * @param [in]d_mrc A device miracle.
 * @param [in]rdlcs A flag to choose RDLCS or RDLIS.
 * @retval The branching literal.
 */
static Lit RDLxS_heuristic(Miracle *d_mrc, bool rdlcs);


/**
 * Kernels
 */


/**
 * @brief Updates device variable assignments with assigned literals (to be
 * called before update_clause_sat_krn).
 * 
 * @param [in/out]mrc A device miracle.
 * @retval None.
 */
__global__ void update_var_ass_krn(Miracle *mrc);


/**
 * @brief Updates device clause satisfiability (to be called after
 * update_var_ass_krn).
 * 
 * @param [in/out]mrc A device miracle.
 * @retval None.
 */
__global__ void update_clause_sat_krn(Miracle *mrc);


/**
 * @brief Increases the device decision level.
 * 
 * @param [in/out]mrc A device miracle.
 * @retval None.
 */
__global__ void increase_dec_lvl_krn(Miracle *mrc);


/**
 * @brief Restores device clause satisfiability by backjumping to a decision
 * level.
 * 
 * @param [in]bj_dec_lvl A backjump decision level. A bj_dec_lvl < 1 resets the
 * device miracle.
 * @param [in/out]mrc A device miracle.
 * @retval None.
 */
__global__ void restore_clause_sat_krn(int bj_dec_lvl, Miracle *mrc);


/**
 * @brief Restores device variable assignments by backjumping to a decision
 * level.
 * 
 * @param [in]bj_dec_lvl A backjump decision level. A bj_dec_lvl < 1 resets the
 * device miracle.
 * @param [in/out]mrc A device miracle.
 * @retval None.
 */
__global__ void restore_var_ass_krn(int bj_dec_lvl, Miracle *mrc);


/**
 * @brief Restores the device decision level to a backjump decision level.
 * 
 * @param [in]bj_dec_lvl A backjump decision level. A bj_dec_lvl < 1 resets the
 * device miracle.
 * @param [in/out]mrc A device miracle.
 * @retval None.
 */
__global__ void restore_dec_lvl_krn(int bj_dec_lvl, Miracle *mrc);


/**
 * @brief Weighs the literals in unresolved clauses according to the JW weight
 * function (to be called before JW_TS_weigh_vars_unres_clauses_krn).
 * 
 * @param [in]mrc A device miracle.
 * @retval None.
 */
__global__ void JW_weigh_lits_unres_clauses_krn(Miracle *mrc);


/**
 * @brief Weighs the variables in unresolved clauses according to the JW-TS
 * weight function (to be called after JW_weigh_lits_unres_clauses_krn).
 * 
 * @param [in]mrc A device miracle.
 * @retval None.
 */
__global__ void JW_TS_weigh_vars_unres_clauses_krn(Miracle *mrc);


/**
 * @brief Computes the clause sizes.
 * 
 * @param [in]mrc A device miracle.
 * @retval None.
 */
__global__ void compute_clause_sizes_krn(Miracle *mrc);


/**
 * @brief Counts the number of occurrences of literals in the smallest
 * unresolved clauses.
 * 
 * @param [in]mrc A device miracle.
 * @param [in]smallest_c_size The smallest clause size.
 * @retval None.
 */
__global__ void count_lits_smallest_unres_clauses_krn(Miracle *mrc,
                                                      int smallest_c_size);


/**
 * @brief Weighs the variables in the smallest unresolved clauses according to
 * the POSIT weight function.
 * 
 * @param [in]mrc A device miracle.
 * @param [in]n A constant of the POSIT weight function.
 * @retval None.
 */
__global__ void POSIT_weigh_vars_smallest_unres_clauses_krn(Miracle *mrc,
                                                            const int n);


/**
 * @brief Counts the number of occurrences of literals in unresolved clauses
 * (to be called before count_vars_unres_clauses_krn).
 * 
 * @param [in]mrc A device miracle.
 * @retval None.
 */
__global__ void count_lits_unres_clauses_krn(Miracle *mrc);


/**
 * @brief Counts the number of occurrences of variables in unresolved clauses
 * (to be called after count_lits_unres_clauses_krn).
 * 
 * @param [in]mrc A device miracle.
 * @retval None.
 */
__global__ void count_vars_unres_clauses_krn(Miracle *mrc);


/**
 * @brief Initializes the variable availability.
 *
 * @param [in]mrc A device miracle.
 * @retval None.
 */
__global__ void init_var_availability_krn(Miracle *mrc);


/**
 * @brief Initializes the clause indices.
 *
 * @param [in]mrc A device miracle.
 * @retval None.
 */
__global__ void init_clause_idxs_krn(Miracle *mrc);


/**
 * @brief Counts the number of occurrences of literals in unresolved clauses
 * having the same size.
 *
 * @param [in]mrc A device miracle.
 * @param [in]num_clauses_same_size The number of clauses having the same size.
 * @param [in]i An index of clause_indices for retrieving clauses having the
 * same size.
 * @retval None.
 */
__global__ void count_lits_unres_clauses_same_size_krn(
                                                    Miracle *mrc,
                                                    int num_clauses_same_size,
                                                    int i
                                                      );


/**
 * @brief Weighs the variables in unresolved clauses having the same size
 * according to the BOHM weight function.
 *
 * @param [in]mrc A device miracle.
 * @param [in]alpha A constant of the BOHM weight function.
 * @param [in]beta A constant of the BOHM weight function.
 * @retval None.
 */
__global__ void BOHM_weigh_vars_unres_clauses_same_size_krn(Miracle *mrc,
                                                            const int alpha,
                                                            const int beta);


/**
 * @brief Updates the variable availability according to the BOHM Variable
 * Selection Heuristic.
 *
 * @param [in]greatest_v_weight The greatest variable BOHM weight.
 * @retval None.
 */
__global__ void BOHM_update_var_availability_krn(float greatest_v_weight);


/**
 * API definition
 */


Miracle *mrc_gpu_transfer_miracle_host_to_dev(Miracle *mrc) {
    Miracle *d_mrc;
    gpuErrchk( cudaMalloc((void**)&d_mrc, sizeof *d_mrc) );
    gpuErrchk( cudaMemcpy(d_mrc, mrc, sizeof *d_mrc, cudaMemcpyHostToDevice) );

    CNF_Formula *d_phi = cnf_gpu_transfer_formula_host_to_dev(mrc->phi);
    gpuErrchk( cudaMemcpy(&(d_mrc->phi), &d_phi,
                          sizeof d_phi,
                          cudaMemcpyHostToDevice) );

    int *d_var_ass;
    gpuErrchk( cudaMalloc((void**)&d_var_ass,
                          sizeof *d_var_ass * mrc->var_ass_len) );
    gpuErrchk( cudaMemcpy(&(d_mrc->var_ass), &d_var_ass,
                          sizeof d_var_ass,
                          cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_var_ass, mrc->var_ass,
                          sizeof *d_var_ass * mrc->var_ass_len,
                          cudaMemcpyHostToDevice) );

    int *d_clause_sat;
    gpuErrchk( cudaMalloc((void**)&d_clause_sat,
                          sizeof *d_clause_sat * mrc->clause_sat_len) );
    gpuErrchk( cudaMemcpy(&(d_mrc->clause_sat), &d_clause_sat,
                          sizeof d_clause_sat,
                          cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_clause_sat, mrc->clause_sat,
                          sizeof *d_clause_sat * mrc->clause_sat_len,
                          cudaMemcpyHostToDevice) );

    init_aux_data_structs(mrc);

    return d_mrc;
}


Miracle *mrc_gpu_transfer_miracle_dev_to_host(Miracle *d_mrc) {
    Miracle *mrc = (Miracle *)malloc(sizeof *mrc);
    gpuErrchk( cudaMemcpy(mrc, d_mrc, sizeof *mrc, cudaMemcpyDeviceToHost) );

    CNF_Formula *d_phi;
    gpuErrchk( cudaMemcpy(&d_phi, &(d_mrc->phi),
                          sizeof d_phi,
                          cudaMemcpyDeviceToHost) );
    mrc->phi = cnf_gpu_transfer_formula_dev_to_host(d_phi);

    mrc->var_ass = (int *)malloc(sizeof *(mrc->var_ass) * mrc->var_ass_len);
    int *d_var_ass;
    gpuErrchk( cudaMemcpy(&d_var_ass, &(d_mrc->var_ass),
                          sizeof d_var_ass,
                          cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(mrc->var_ass, d_var_ass,
                          sizeof *d_var_ass * mrc->var_ass_len,
                          cudaMemcpyDeviceToHost) );

    mrc->clause_sat = (int *)malloc(sizeof *(mrc->clause_sat) *
                                    mrc->clause_sat_len);
    int *d_clause_sat;
    gpuErrchk( cudaMemcpy(&d_clause_sat, &(d_mrc->clause_sat),
                          sizeof d_clause_sat,
                          cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(mrc->clause_sat, d_clause_sat,
                          sizeof *d_clause_sat * mrc->clause_sat_len,
                          cudaMemcpyDeviceToHost) );

    return mrc;
}


void mrc_gpu_destroy_miracle(Miracle *d_mrc) {
    CNF_Formula *d_phi;
    gpuErrchk( cudaMemcpy(&d_phi, &(d_mrc->phi),
                          sizeof d_phi,
                          cudaMemcpyDeviceToHost) );
    cnf_gpu_destroy_formula(d_phi);

    int *d_var_ass;
    gpuErrchk( cudaMemcpy(&d_var_ass, &(d_mrc->var_ass),
                          sizeof d_var_ass,
                          cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaFree(d_var_ass) );

    int *d_clause_sat;
    gpuErrchk( cudaMemcpy(&d_clause_sat, &(d_mrc->clause_sat),
                          sizeof d_clause_sat,
                          cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaFree(d_clause_sat) );

    gpuErrchk( cudaFree(d_mrc) );

    destroy_aux_data_structs();
}


void mrc_gpu_assign_lits(Lit *lits, int lits_len, Miracle *d_mrc) {
    // Set d_lits_len and d_lits.
    gpuErrchk( cudaMemcpyToSymbol(d_lits_len, &lits_len,
                                  sizeof lits_len, 0UL,
                                  cudaMemcpyHostToDevice) );
    Lit *dev_lits;
    gpuErrchk( cudaMemcpyFromSymbol(&dev_lits, d_lits,
                                    sizeof dev_lits, 0UL,
                                    cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(dev_lits, lits,
                          sizeof *lits * lits_len,
                          cudaMemcpyHostToDevice) );

    // Update device variable assignments.
    int num_blks = gpu_num_blocks(lits_len);
    int num_thds_per_blk = gpu_num_threads_per_block();

    update_var_ass_krn<<<num_blks, num_thds_per_blk>>>(d_mrc);
    
    gpuErrchk( cudaPeekAtLastError() );

    // Update device clause satisfiability.
    int clause_sat_len;
    gpuErrchk( cudaMemcpy(&clause_sat_len, &(d_mrc->clause_sat_len),
                          sizeof clause_sat_len,
                          cudaMemcpyDeviceToHost) );

    num_blks = gpu_num_blocks(clause_sat_len);
    num_thds_per_blk = gpu_num_threads_per_block();

    update_clause_sat_krn<<<num_blks, num_thds_per_blk>>>(d_mrc);
    
    gpuErrchk( cudaPeekAtLastError() );
}


void mrc_gpu_increase_decision_level(Miracle *d_mrc) {
    increase_dec_lvl_krn<<<1, 1>>>(d_mrc);

    gpuErrchk( cudaPeekAtLastError() );
}


void mrc_gpu_backjump(int bj_dec_lvl, Miracle *d_mrc) {
    // Restore device clause satisfiability.
    int clause_sat_len;
    gpuErrchk( cudaMemcpy(&clause_sat_len, &(d_mrc->clause_sat_len),
                          sizeof clause_sat_len,
                          cudaMemcpyDeviceToHost) );
    
    int num_blks = gpu_num_blocks(clause_sat_len);
    int num_thds_per_blk = gpu_num_threads_per_block();

    restore_clause_sat_krn<<<num_blks, num_thds_per_blk>>>(bj_dec_lvl, d_mrc);

    gpuErrchk( cudaPeekAtLastError() );

    // Restore device variable assignments.
    int var_ass_len;
    gpuErrchk( cudaMemcpy(&var_ass_len, &(d_mrc->var_ass_len),
                          sizeof var_ass_len,
                          cudaMemcpyDeviceToHost) );
    num_blks = gpu_num_blocks(var_ass_len);
    num_thds_per_blk = gpu_num_threads_per_block();
    
    restore_var_ass_krn<<<num_blks, num_thds_per_blk>>>(bj_dec_lvl, d_mrc);

    gpuErrchk( cudaPeekAtLastError() );

    // Restore device decision level.
    restore_dec_lvl_krn<<<1, 1>>>(bj_dec_lvl, d_mrc);

    gpuErrchk( cudaPeekAtLastError() );
}


Lit mrc_gpu_JW_OS_heuristic(Miracle *d_mrc) {
    return JW_xS_heuristic(d_mrc, false);
}


Lit mrc_gpu_JW_TS_heuristic(Miracle *d_mrc) {
    return JW_xS_heuristic(d_mrc, true);
}


Lit mrc_gpu_BOHM_heuristic(Miracle *d_mrc, const int alpha, const int beta) {
    // Init d_var_availability.
    int num_blks = gpu_num_blocks(var_availability_len);
    int num_thds_per_blk = gpu_num_threads_per_block();

    init_var_availability_krn<<<num_blks, num_thds_per_blk>>>(d_mrc);

    gpuErrchk( cudaPeekAtLastError() );

    // Clear d_cum_lit_occ.
    int *dev_cum_lit_occ;
    gpuErrchk( cudaMemcpyFromSymbol(&dev_cum_lit_occ, d_cum_lit_occ,
                                    sizeof dev_cum_lit_occ, 0UL,
                                    cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemset(dev_cum_lit_occ, 0,
                          sizeof *dev_cum_lit_occ * cum_lit_occ_len) );

    // Clear d_clause_sizes.
    int *dev_clause_sizes;
    gpuErrchk( cudaMemcpyFromSymbol(&dev_clause_sizes, d_clause_sizes,
                                    sizeof dev_clause_sizes, 0UL,
                                    cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemset(dev_clause_sizes, 0,
                          sizeof *dev_clause_sizes * clause_sizes_len) );

    // Compute the clause sizes.
    num_blks = gpu_num_blocks(clause_sizes_len);
    num_thds_per_blk = gpu_num_threads_per_block();

    compute_clause_sizes_krn<<<num_blks, num_thds_per_blk>>>(d_mrc);

    gpuErrchk( cudaPeekAtLastError() );

    // Init d_clause_idxs.
    num_blks = gpu_num_blocks(clause_idxs_len);
    num_thds_per_blk = gpu_num_threads_per_block();

    init_clause_idxs_krn<<<num_blks, num_thds_per_blk>>>(d_mrc);

    gpuErrchk( cudaPeekAtLastError() );

    int *dev_clause_idxs;
    gpuErrchk( cudaMemcpyFromSymbol(&dev_clause_idxs, d_clause_idxs,
                                    sizeof dev_clause_idxs, 0UL,
                                    cudaMemcpyDeviceToHost) );

    // Sort the clauses by increasing size.
    thrust::sort_by_key(thrust::device,
                        dev_clause_sizes,
                        dev_clause_sizes + clause_sizes_len,
                        dev_clause_idxs);

    // Build the indexing array of clauses ordered by size.
    int *dev_clause_sizes_copy;
    gpuErrchk( cudaMemcpyFromSymbol(&dev_clause_sizes_copy,
                                    d_clause_sizes_copy,
                                    sizeof dev_clause_sizes_copy, 0UL,
                                    cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(dev_clause_sizes_copy, dev_clause_sizes,
                          sizeof *dev_clause_sizes * clause_sizes_len,
                          cudaMemcpyDeviceToDevice) );

    int *dev_clause_indices;
    gpuErrchk( cudaMemcpyFromSymbol(&dev_clause_indices, d_clause_indices,
                                    sizeof dev_clause_indices, 0UL,
                                    cudaMemcpyDeviceToHost) );

    thrust::sequence(thrust::device,
                     dev_clause_indices,
                     dev_clause_indices + clause_sizes_len + 1);

    clause_indices_len = (thrust::unique_by_key(
                                    thrust::device,
                                    dev_clause_sizes_copy,
                                    dev_clause_sizes_copy + clause_sizes_len,
                                    dev_clause_indices)
                         ).second - dev_clause_indices;

    gpuErrchk( cudaMemcpy(clause_indices, dev_clause_indices,
                          sizeof *clause_indices * clause_indices_len,
                          cudaMemcpyDeviceToHost) );

    clause_indices[clause_indices_len] = clause_sizes_len;

    gpuErrchk( cudaMemcpy(&(dev_clause_indices[clause_indices_len]),
                          &(clause_indices[clause_indices_len]),
                          sizeof *clause_indices,
                          cudaMemcpyHostToDevice) );

    clause_indices_len++;

    gpuErrchk( cudaMemcpyToSymbol(d_clause_indices_len, &clause_indices_len,
                                  sizeof clause_indices_len, 0UL,
                                  cudaMemcpyHostToDevice) );

    int min_c_size;     // Minimum clause size.
    gpuErrchk( cudaMemcpy(&min_c_size, &(dev_clause_sizes[0]),
                          sizeof min_c_size,
                          cudaMemcpyDeviceToHost) );

    int *dev_lit_occ;
    gpuErrchk( cudaMemcpyFromSymbol(&dev_lit_occ, d_lit_occ,
                                    sizeof dev_lit_occ, 0UL,
                                    cudaMemcpyDeviceToHost) );
    float *dev_var_weights;
    gpuErrchk( cudaMemcpyFromSymbol(&dev_var_weights, d_var_weights,
                                    sizeof dev_var_weights, 0UL,
                                    cudaMemcpyDeviceToHost) );

    int num_clauses_same_size;      // Number of clauses having the same size.
    float greatest_v_weight;        // Greatest variable weight.

    for (int i = min_c_size == 0 ? 1 : 0;
         i < clause_indices_len - 1;
         i++) {
        // Clear d_lit_occ.
        gpuErrchk( cudaMemset(dev_lit_occ, 0,
                              sizeof *dev_lit_occ * lit_occ_len) );

        // Clear d_var_weights.
        gpuErrchk( cuda_memset_float(dev_var_weights, -1.0, var_weights_len) );

        num_clauses_same_size = clause_indices[i+1] - clause_indices[i];
        num_blks = gpu_num_blocks(num_clauses_same_size);
        num_thds_per_blk = gpu_num_threads_per_block();

        count_lits_unres_clauses_same_size_krn<<<num_blks,
                                                 num_thds_per_blk>>>(
                                                        d_mrc,
                                                        num_clauses_same_size,
                                                        i);

        gpuErrchk( cudaPeekAtLastError() );

        num_blks = gpu_num_blocks(var_weights_len);
        num_thds_per_blk = gpu_num_threads_per_block();

        BOHM_weigh_vars_unres_clauses_same_size_krn<<<num_blks,
                                                      num_thds_per_blk>>>(
                                                                    d_mrc,
                                                                    alpha,
                                                                    beta
                                                                         );

        gpuErrchk( cudaPeekAtLastError() );

        greatest_v_weight = find_max_float(dev_var_weights, var_weights_len);

        num_blks = gpu_num_blocks(var_availability_len);
        num_thds_per_blk = gpu_num_threads_per_block();

        BOHM_update_var_availability_krn<<<num_blks, num_thds_per_blk>>>(
                                                            greatest_v_weight
                                                                        );

        gpuErrchk( cudaPeekAtLastError() );
    }

    int *dev_var_availability;
    gpuErrchk( cudaMemcpyFromSymbol(&dev_var_availability, d_var_availability,
                                    sizeof dev_var_availability, 0UL,
                                    cudaMemcpyDeviceToHost) );

    Var bvar = (Var)find_min_int(dev_var_availability, var_availability_len);
    Lidx pos_lidx = varpol_to_lidx(bvar, true);
    Lidx neg_lidx = varpol_to_lidx(bvar, false);
    int lc_i_pos_lidx;
    int lc_i_neg_lidx;
    gpuErrchk( cudaMemcpy(&lc_i_pos_lidx, &(dev_cum_lit_occ[pos_lidx]),
                          sizeof lc_i_pos_lidx,
                          cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(&lc_i_neg_lidx, &(dev_cum_lit_occ[neg_lidx]),
                          sizeof lc_i_neg_lidx,
                          cudaMemcpyDeviceToHost) );

    return lc_i_pos_lidx >= lc_i_neg_lidx ? lidx_to_lit(pos_lidx) :
                                            lidx_to_lit(neg_lidx);
}


Lit mrc_gpu_POSIT_heuristic(Miracle *d_mrc, const int n) {
    // Clear d_clause_sizes.
    int *dev_clause_sizes;
    gpuErrchk( cudaMemcpyFromSymbol(&dev_clause_sizes, d_clause_sizes,
                                    sizeof dev_clause_sizes, 0UL,
                                    cudaMemcpyDeviceToHost) );
    gpuErrchk( cuda_memset_int(dev_clause_sizes, INT_MAX, clause_sizes_len) );

    int num_blks = gpu_num_blocks(clause_sizes_len);
    int num_thds_per_blk = gpu_num_threads_per_block();

    compute_clause_sizes_krn<<<num_blks, num_thds_per_blk>>>(d_mrc);

    gpuErrchk( cudaPeekAtLastError() );

    // Smallest clause size.
    int smallest_c_size = find_min_int(dev_clause_sizes, clause_sizes_len);

    // Clear d_lit_occ.
    int *dev_lit_occ;
    gpuErrchk( cudaMemcpyFromSymbol(&dev_lit_occ, d_lit_occ,
                                    sizeof dev_lit_occ, 0UL,
                                    cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemset(dev_lit_occ, 0,
                          sizeof *dev_lit_occ * lit_occ_len) );

    int clause_sat_len;
    gpuErrchk( cudaMemcpy(&clause_sat_len, &(d_mrc->clause_sat_len),
                          sizeof clause_sat_len,
                          cudaMemcpyDeviceToHost) );

    num_blks = gpu_num_blocks(clause_sat_len);
    num_thds_per_blk = gpu_num_threads_per_block();

    count_lits_smallest_unres_clauses_krn<<<num_blks, num_thds_per_blk>>>(
                                                                d_mrc,
                                                                smallest_c_size
                                                                         );

    gpuErrchk( cudaPeekAtLastError() );

    // Clear d_var_weights.
    float *dev_var_weights;
    gpuErrchk( cudaMemcpyFromSymbol(&dev_var_weights, d_var_weights,
                                    sizeof dev_var_weights, 0UL,
                                    cudaMemcpyDeviceToHost) );
    gpuErrchk( cuda_memset_float(dev_var_weights, -1.0, var_weights_len) );

    num_blks = gpu_num_blocks(var_weights_len);
    num_thds_per_blk = gpu_num_threads_per_block();

    POSIT_weigh_vars_smallest_unres_clauses_krn<<<num_blks,
                                                  num_thds_per_blk>>>(d_mrc,
                                                                      n);

    gpuErrchk( cudaPeekAtLastError() );

    Var bvar = (Var)find_idx_max_float(dev_var_weights, var_weights_len);
    Lidx pos_lidx = varpol_to_lidx(bvar, true);
    Lidx neg_lidx = varpol_to_lidx(bvar, false);
    int lo_min_pos_lidx;
    int lo_min_neg_lidx;
    gpuErrchk( cudaMemcpy(&lo_min_pos_lidx, &(dev_lit_occ[pos_lidx]),
                          sizeof lo_min_pos_lidx,
                          cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(&lo_min_neg_lidx, &(dev_lit_occ[neg_lidx]),
                          sizeof lo_min_neg_lidx,
                          cudaMemcpyDeviceToHost) );

    return lo_min_pos_lidx >= lo_min_neg_lidx ? lidx_to_lit(neg_lidx) :
                                                lidx_to_lit(pos_lidx);
}


Lit mrc_gpu_DLIS_heuristic(Miracle *d_mrc) {
    return DLxS_heuristic(d_mrc, false);
}


Lit mrc_gpu_DLCS_heuristic(Miracle *d_mrc) {
    return DLxS_heuristic(d_mrc, true);
}


Lit mrc_gpu_RDLIS_heuristic(Miracle *d_mrc) {
    return RDLxS_heuristic(d_mrc, false);
}


Lit mrc_gpu_RDLCS_heuristic(Miracle *d_mrc) {
    return RDLxS_heuristic(d_mrc, true);
}


/**
 * Auxiliary function definitions
 */


static void init_aux_data_structs(Miracle *mrc) {
    int *dev_lits_len;
    gpuErrchk( cudaGetSymbolAddress((void**)&dev_lits_len, d_lits_len) );
    Lit *dev_lits;
    gpuErrchk( cudaMalloc((void**)&dev_lits,
                          sizeof *dev_lits * mrc->phi->num_vars) );
    gpuErrchk( cudaMemcpyToSymbol(d_lits, &dev_lits,
                                  sizeof dev_lits, 0UL,
                                  cudaMemcpyHostToDevice) );

    lit_occ_len = mrc->phi->num_vars * 2;
    gpuErrchk( cudaMemcpyToSymbol(d_lit_occ_len, &lit_occ_len,
                                  sizeof lit_occ_len, 0UL,
                                  cudaMemcpyHostToDevice) );
    int *dev_lit_occ;
    gpuErrchk( cudaMalloc((void**)&dev_lit_occ,
                          sizeof *dev_lit_occ * lit_occ_len) );
    gpuErrchk( cudaMemcpyToSymbol(d_lit_occ, &dev_lit_occ,
                                  sizeof dev_lit_occ, 0UL,
                                  cudaMemcpyHostToDevice) );

    cum_lit_occ_len = mrc->phi->num_vars * 2;
    gpuErrchk( cudaMemcpyToSymbol(d_cum_lit_occ_len, &cum_lit_occ_len,
                                  sizeof cum_lit_occ_len, 0UL,
                                  cudaMemcpyHostToDevice) );
    int *dev_cum_lit_occ;
    gpuErrchk( cudaMalloc((void**)&dev_cum_lit_occ,
                          sizeof *dev_cum_lit_occ * cum_lit_occ_len) );
    gpuErrchk( cudaMemcpyToSymbol(d_cum_lit_occ, &dev_cum_lit_occ,
                                  sizeof dev_cum_lit_occ, 0UL,
                                  cudaMemcpyHostToDevice) );

    lit_weights_len = mrc->phi->num_vars * 2;
    gpuErrchk( cudaMemcpyToSymbol(d_lit_weights_len, &lit_weights_len,
                                  sizeof lit_weights_len, 0UL,
                                  cudaMemcpyHostToDevice) );
    float *dev_lit_weights;
    gpuErrchk( cudaMalloc((void**)&dev_lit_weights,
                          sizeof *dev_lit_weights * lit_weights_len) );
    gpuErrchk( cudaMemcpyToSymbol(d_lit_weights, &dev_lit_weights,
                                  sizeof dev_lit_weights, 0UL,
                                  cudaMemcpyHostToDevice) );

    var_occ_len = mrc->phi->num_vars;
    gpuErrchk( cudaMemcpyToSymbol(d_var_occ_len, &var_occ_len,
                                  sizeof var_occ_len, 0UL,
                                  cudaMemcpyHostToDevice) );
    int *dev_var_occ;
    gpuErrchk( cudaMalloc((void**)&dev_var_occ,
                          sizeof *dev_var_occ * var_occ_len) );
    gpuErrchk( cudaMemcpyToSymbol(d_var_occ, &dev_var_occ,
                                  sizeof dev_var_occ, 0UL,
                                  cudaMemcpyHostToDevice) );

    var_weights_len = mrc->phi->num_vars;
    gpuErrchk( cudaMemcpyToSymbol(d_var_weights_len, &var_weights_len,
                                  sizeof var_weights_len, 0UL,
                                  cudaMemcpyHostToDevice) );
    float *dev_var_weights;
    gpuErrchk( cudaMalloc((void**)&dev_var_weights,
                          sizeof *dev_var_weights * var_weights_len) );
    gpuErrchk( cudaMemcpyToSymbol(d_var_weights, &dev_var_weights,
                                  sizeof dev_var_weights, 0UL,
                                  cudaMemcpyHostToDevice) );

    var_availability_len = mrc->phi->num_vars;
    gpuErrchk( cudaMemcpyToSymbol(d_var_availability_len,
                                  &var_availability_len,
                                  sizeof var_availability_len, 0UL,
                                  cudaMemcpyHostToDevice) );
    int *dev_var_availability;
    gpuErrchk( cudaMalloc((void**)&dev_var_availability,
                          sizeof *dev_var_availability *
                          var_availability_len) );
    gpuErrchk( cudaMemcpyToSymbol(d_var_availability, &dev_var_availability,
                                  sizeof dev_var_availability, 0UL,
                                  cudaMemcpyHostToDevice) );

    clause_sizes_len = mrc->phi->num_clauses;
    gpuErrchk( cudaMemcpyToSymbol(d_clause_sizes_len, &clause_sizes_len,
                                  sizeof clause_sizes_len, 0UL,
                                  cudaMemcpyHostToDevice) );
    int *dev_clause_sizes;
    gpuErrchk( cudaMalloc((void**)&dev_clause_sizes,
                          sizeof *dev_clause_sizes * clause_sizes_len) );
    gpuErrchk( cudaMemcpyToSymbol(d_clause_sizes, &dev_clause_sizes,
                                  sizeof dev_clause_sizes, 0UL,
                                  cudaMemcpyHostToDevice) );
    int *dev_clause_sizes_copy;
    gpuErrchk( cudaMalloc((void**)&dev_clause_sizes_copy,
                          sizeof *dev_clause_sizes_copy * clause_sizes_len) );
    gpuErrchk( cudaMemcpyToSymbol(d_clause_sizes_copy, &dev_clause_sizes_copy,
                                  sizeof dev_clause_sizes_copy, 0UL,
                                  cudaMemcpyHostToDevice) );

    clause_idxs_len = mrc->phi->num_clauses;
    gpuErrchk( cudaMemcpyToSymbol(d_clause_idxs_len, &clause_idxs_len,
                                  sizeof clause_idxs_len, 0UL,
                                  cudaMemcpyHostToDevice) );
    int *dev_clause_idxs;
    gpuErrchk( cudaMalloc((void**)&dev_clause_idxs,
                          sizeof *dev_clause_idxs * clause_idxs_len) );
    gpuErrchk( cudaMemcpyToSymbol(d_clause_idxs, &dev_clause_idxs,
                                  sizeof dev_clause_idxs, 0UL,
                                  cudaMemcpyHostToDevice) );

    clause_indices_len = 0;
    gpuErrchk( cudaMemcpyToSymbol(d_clause_indices_len, &clause_indices_len,
                                  sizeof clause_indices_len, 0UL,
                                  cudaMemcpyHostToDevice) );
    int *dev_clause_indices;
    gpuErrchk( cudaMalloc((void**)&dev_clause_indices,
                          sizeof *dev_clause_indices *
                          (clause_sizes_len + 1)) );
    gpuErrchk( cudaMemcpyToSymbol(d_clause_indices, &dev_clause_indices,
                                  sizeof dev_clause_indices, 0UL,
                                  cudaMemcpyHostToDevice) );
    clause_indices = (int *)malloc(sizeof *clause_indices *
                                   (clause_sizes_len + 1));
}


static void destroy_aux_data_structs() {
    Lit *dev_lits;
    gpuErrchk( cudaMemcpyFromSymbol(&dev_lits, d_lits,
                                    sizeof dev_lits, 0UL,
                                    cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaFree(dev_lits) );

    int *dev_lit_occ;
    gpuErrchk( cudaMemcpyFromSymbol(&dev_lit_occ, d_lit_occ,
                                    sizeof dev_lit_occ, 0UL,
                                    cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaFree(dev_lit_occ) );

    int *dev_cum_lit_occ;
    gpuErrchk( cudaMemcpyFromSymbol(&dev_cum_lit_occ, d_cum_lit_occ,
                                    sizeof dev_cum_lit_occ, 0UL,
                                    cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaFree(dev_cum_lit_occ) );

    float *dev_lit_weights;
    gpuErrchk( cudaMemcpyFromSymbol(&dev_lit_weights, d_lit_weights,
                                    sizeof dev_lit_weights, 0UL,
                                    cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaFree(dev_lit_weights) );

    int *dev_var_occ;
    gpuErrchk( cudaMemcpyFromSymbol(&dev_var_occ, d_var_occ,
                                    sizeof dev_var_occ, 0UL,
                                    cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaFree(dev_var_occ) );

    float *dev_var_weights;
    gpuErrchk( cudaMemcpyFromSymbol(&dev_var_weights, d_var_weights,
                                    sizeof dev_var_weights, 0UL,
                                    cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaFree(dev_var_weights) );

    int *dev_var_availability;
    gpuErrchk( cudaMemcpyFromSymbol(&dev_var_availability, d_var_availability,
                                    sizeof dev_var_availability, 0UL,
                                    cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaFree(dev_var_availability) );

    int *dev_clause_sizes;
    gpuErrchk( cudaMemcpyFromSymbol(&dev_clause_sizes, d_clause_sizes,
                                    sizeof dev_clause_sizes, 0UL,
                                    cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaFree(dev_clause_sizes) );
    int *dev_clause_sizes_copy;
    gpuErrchk( cudaMemcpyFromSymbol(&dev_clause_sizes_copy,
                                    d_clause_sizes_copy,
                                    sizeof dev_clause_sizes_copy, 0UL,
                                    cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaFree(dev_clause_sizes_copy) );

    int *dev_clause_idxs;
    gpuErrchk( cudaMemcpyFromSymbol(&dev_clause_idxs, d_clause_idxs,
                                    sizeof dev_clause_idxs, 0UL,
                                    cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaFree(dev_clause_idxs) );

    int *dev_clause_indices;
    gpuErrchk( cudaMemcpyFromSymbol(&dev_clause_indices, d_clause_indices,
                                    sizeof dev_clause_indices, 0UL,
                                    cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaFree(dev_clause_indices) );
    free(clause_indices);
}


static Lit JW_xS_heuristic(Miracle *d_mrc, bool two_sided) {
    // Clear d_lit_weights.
    float *dev_lit_weights;
    gpuErrchk( cudaMemcpyFromSymbol(&dev_lit_weights, d_lit_weights,
                                    sizeof dev_lit_weights, 0UL,
                                    cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemset(dev_lit_weights, 0,
                          sizeof *dev_lit_weights * lit_weights_len) );

    int clause_sat_len;
    gpuErrchk( cudaMemcpy(&clause_sat_len, &(d_mrc->clause_sat_len),
                          sizeof clause_sat_len,
                          cudaMemcpyDeviceToHost) );

    int num_blks = gpu_num_blocks(clause_sat_len);
    int num_thds_per_blk = gpu_num_threads_per_block();

    JW_weigh_lits_unres_clauses_krn<<<num_blks, num_thds_per_blk>>>(d_mrc);

    gpuErrchk( cudaPeekAtLastError() );

    Lidx blidx;     // JW_xS branching literal index.

    if (two_sided) {
        // Clear d_var_weights.
        float *dev_var_weights;
        gpuErrchk( cudaMemcpyFromSymbol(&dev_var_weights, d_var_weights,
                                        sizeof dev_var_weights, 0UL,
                                        cudaMemcpyDeviceToHost) );
        gpuErrchk( cuda_memset_float(dev_var_weights, -1.0, var_weights_len) );

        int num_blks = gpu_num_blocks(var_weights_len);
        int num_thds_per_blk = gpu_num_threads_per_block();

        JW_TS_weigh_vars_unres_clauses_krn<<<num_blks,
                                             num_thds_per_blk>>>(d_mrc);

        gpuErrchk( cudaPeekAtLastError() );

        Var bvar = (Var)find_idx_max_float(dev_var_weights, var_weights_len);
        Lidx pos_lidx = varpol_to_lidx(bvar, true);
        Lidx neg_lidx = varpol_to_lidx(bvar, false);
        float lw_pos_lidx;
        float lw_neg_lidx;
        gpuErrchk( cudaMemcpy(&lw_pos_lidx, &(dev_lit_weights[pos_lidx]),
                              sizeof lw_pos_lidx,
                              cudaMemcpyDeviceToHost) );
        gpuErrchk( cudaMemcpy(&lw_neg_lidx, &(dev_lit_weights[neg_lidx]),
                              sizeof lw_neg_lidx,
                              cudaMemcpyDeviceToHost) );

        blidx = lw_pos_lidx >= lw_neg_lidx ? pos_lidx : neg_lidx;
    } else {
        blidx = (Lidx)find_idx_max_float(dev_lit_weights, lit_weights_len);
    }

    return lidx_to_lit(blidx);
}


static Lit DLxS_heuristic(Miracle *d_mrc, bool dlcs) {
    // Clear d_lit_occ.
    int *dev_lit_occ;
    gpuErrchk( cudaMemcpyFromSymbol(&dev_lit_occ, d_lit_occ,
                                    sizeof dev_lit_occ, 0UL,
                                    cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemset(dev_lit_occ, 0,
                          sizeof *dev_lit_occ * lit_occ_len) );

    int clause_sat_len;
    gpuErrchk( cudaMemcpy(&clause_sat_len, &(d_mrc->clause_sat_len),
                          sizeof clause_sat_len,
                          cudaMemcpyDeviceToHost) );

    int num_blks = gpu_num_blocks(clause_sat_len);
    int num_thds_per_blk = gpu_num_threads_per_block();

    count_lits_unres_clauses_krn<<<num_blks, num_thds_per_blk>>>(d_mrc);

    gpuErrchk( cudaPeekAtLastError() );

    Lidx blidx;     // DLxS branching literal index.

    if (dlcs) {
        // Clear d_var_occ.
        int *dev_var_occ;
        gpuErrchk( cudaMemcpyFromSymbol(&dev_var_occ, d_var_occ,
                                        sizeof dev_var_occ, 0UL,
                                        cudaMemcpyDeviceToHost) );
        gpuErrchk( cuda_memset_int(dev_var_occ, -1, var_occ_len) );

        int num_blks = gpu_num_blocks(var_occ_len);
        int num_thds_per_blk = gpu_num_threads_per_block();

        count_vars_unres_clauses_krn<<<num_blks, num_thds_per_blk>>>(d_mrc);

        gpuErrchk( cudaPeekAtLastError() );

        Var bvar = (Var)find_idx_max_int(dev_var_occ, var_occ_len);
        Lidx pos_lidx = varpol_to_lidx(bvar, true);
        Lidx neg_lidx = varpol_to_lidx(bvar, false);
        int lo_pos_lidx;
        int lo_neg_lidx;
        gpuErrchk( cudaMemcpy(&lo_pos_lidx, &(dev_lit_occ[pos_lidx]),
                              sizeof lo_pos_lidx,
                              cudaMemcpyDeviceToHost) );
        gpuErrchk( cudaMemcpy(&lo_neg_lidx, &(dev_lit_occ[neg_lidx]),
                              sizeof lo_neg_lidx,
                              cudaMemcpyDeviceToHost) );

        blidx = lo_pos_lidx >= lo_neg_lidx ? pos_lidx : neg_lidx;
    } else {
        blidx = (Lidx)find_idx_max_int(dev_lit_occ, lit_occ_len);
    }

    return lidx_to_lit(blidx);
}


static Lit RDLxS_heuristic(Miracle *d_mrc, bool rdlcs) {
    init_PRNG();

    if (rdlcs && (rand() % 2)) {
        return neg_lit(mrc_gpu_DLCS_heuristic(d_mrc));
    } else if (rdlcs) {
        return mrc_gpu_DLCS_heuristic(d_mrc);
    } else if (rand() % 2) {
        return neg_lit(mrc_gpu_DLIS_heuristic(d_mrc));
    } else {
        return mrc_gpu_DLIS_heuristic(d_mrc);
    }
}


/**
 * Kernel definitions
 */


__global__ void update_var_ass_krn(Miracle *mrc) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int lits_lgth = d_lits_len;
    Lit lit;
    Var var;
    bool pol;
    int dec_lvl = mrc->dec_lvl;

    while (gid < lits_lgth) {
        lit = d_lits[gid];
        var = lit_to_var(lit);
        pol = lit_to_pol(lit);

        mrc->var_ass[var] = pol ? dec_lvl : -(dec_lvl);
        
        gid += stride;
    }
}


__global__ void update_clause_sat_krn(Miracle *mrc) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int clause_sat_lgth = mrc->clause_sat_len;
    Lidx lidx;
    Var var;
    bool pol;
    int dec_lvl = mrc->dec_lvl;

    while (gid < clause_sat_lgth) {
        if (!(mrc->clause_sat[gid])) {
            for (int l = mrc->phi->clause_indices[gid];
                 l < mrc->phi->clause_indices[gid+1];
                 l++) {
                lidx = mrc->phi->clauses[l];
                var = lidx_to_var(lidx);
                pol = lidx_to_pol(lidx);

                if ((pol && mrc->var_ass[var] > 0) ||
                    (!pol && mrc->var_ass[var] < 0)) {
                    mrc->clause_sat[gid] = dec_lvl;
                    break;
                }
            }
        }

        gid += stride;
    }
}


__global__ void increase_dec_lvl_krn(Miracle *mrc) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    if (gid == 0) {
        mrc->dec_lvl++;
    }
}


__global__ void restore_clause_sat_krn(int bj_dec_lvl, Miracle *mrc) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int clause_sat_lgth = mrc->clause_sat_len;

    while (gid < clause_sat_lgth) {
        if (mrc->clause_sat[gid] > bj_dec_lvl) {
            mrc->clause_sat[gid] = 0;
        }

        gid += stride;
    }
}


__global__ void restore_var_ass_krn(int bj_dec_lvl, Miracle *mrc) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int var_ass_lgth = mrc->var_ass_len;

    while (gid < var_ass_lgth) {
        if (abs(mrc->var_ass[gid]) > bj_dec_lvl) {
            mrc->var_ass[gid] = 0;
        }

        gid += stride;
    }
}


__global__ void restore_dec_lvl_krn(int bj_dec_lvl, Miracle *mrc) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    if (gid == 0) {
        mrc->dec_lvl = bj_dec_lvl < 1 ? 1 : bj_dec_lvl;
    }
}


__global__ void JW_weigh_lits_unres_clauses_krn(Miracle *mrc) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int clause_sat_lgth = mrc->clause_sat_len;
    int c_size;     // Clause size.
    Lidx lidx;
    Var var;
    float weight;

    while (gid < clause_sat_lgth) {
        if (!(mrc->clause_sat[gid])) {
            c_size = 0;

            for (int l = mrc->phi->clause_indices[gid];
                 l < mrc->phi->clause_indices[gid+1];
                 l++) {
                lidx = mrc->phi->clauses[l];
                var = lidx_to_var(lidx);

                if (!(mrc->var_ass[var])) {
                    c_size++;
                }
            }

            for (int l = mrc->phi->clause_indices[gid];
                 l < mrc->phi->clause_indices[gid+1];
                 l++) {
                lidx = mrc->phi->clauses[l];
                var = lidx_to_var(lidx);

                if (!(mrc->var_ass[var])) {
                    weight = powf(2.0, (float)-c_size);
                    atomicAdd(&(d_lit_weights[lidx]), weight);
                }
            }
        }

        gid += stride;
    }
}


__global__ void JW_TS_weigh_vars_unres_clauses_krn(Miracle *mrc) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int var_weights_lgth = d_var_weights_len;
    Lidx pos_lidx;
    Lidx neg_lidx;

    while (gid < var_weights_lgth) {
        if (!(mrc->var_ass[gid])) {
            pos_lidx = varpol_to_lidx(gid, true);
            neg_lidx = varpol_to_lidx(gid, false);

            d_var_weights[gid] = abs(d_lit_weights[pos_lidx] -
                                     d_lit_weights[neg_lidx]);
        }

        gid += stride;
    }
}


__global__ void compute_clause_sizes_krn(Miracle *mrc) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int clause_sizes_lgth = d_clause_sizes_len;
    int c_size;     // Clause size.
    Lidx lidx;
    Var var;

    while (gid < clause_sizes_lgth) {
        if (!(mrc->clause_sat[gid])) {
            c_size = 0;

            for (int l = mrc->phi->clause_indices[gid];
                 l < mrc->phi->clause_indices[gid+1];
                 l++) {
                lidx = mrc->phi->clauses[l];
                var = lidx_to_var(lidx);

                if (!(mrc->var_ass[var])) {
                    c_size++;
                }
            }

            d_clause_sizes[gid] = c_size;
        }

        gid += stride;
    }
}


__global__ void count_lits_smallest_unres_clauses_krn(Miracle *mrc,
                                                      int smallest_c_size) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int clause_sat_lgth = mrc->clause_sat_len;
    Lidx lidx;
    Var var;

    while (gid < clause_sat_lgth) {
        if (!(mrc->clause_sat[gid]) &&
            (d_clause_sizes[gid] == smallest_c_size)) {
            for (int l = mrc->phi->clause_indices[gid];
                 l < mrc->phi->clause_indices[gid+1];
                 l++) {
                lidx = mrc->phi->clauses[l];
                var = lidx_to_var(lidx);

                if (!(mrc->var_ass[var])) {
                    atomicAdd(&(d_lit_occ[lidx]), 1);
                }
            }
        }

        gid += stride;
    }
}


__global__ void POSIT_weigh_vars_smallest_unres_clauses_krn(Miracle *mrc,
                                                            const int n) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int var_weights_lgth = d_var_weights_len;
    Lidx pos_lidx;
    Lidx neg_lidx;

    while (gid < var_weights_lgth) {
        if (!(mrc->var_ass[gid])) {
            pos_lidx = varpol_to_lidx(gid, true);
            neg_lidx = varpol_to_lidx(gid, false);

            d_var_weights[gid] = (float)
                                 (d_lit_occ[pos_lidx] * d_lit_occ[neg_lidx] *
                                  (int)(pow(2, n) + 0.5) +
                                  d_lit_occ[pos_lidx] + d_lit_occ[neg_lidx]);
        }

        gid += stride;
    }
}


__global__ void count_lits_unres_clauses_krn(Miracle *mrc) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int clause_sat_lgth = mrc->clause_sat_len;
    Lidx lidx;
    Var var;

    while (gid < clause_sat_lgth) {
        if (!(mrc->clause_sat[gid])) {
            for (int l = mrc->phi->clause_indices[gid];
                 l < mrc->phi->clause_indices[gid+1];
                 l++) {
                lidx = mrc->phi->clauses[l];
                var = lidx_to_var(lidx);

                if (!(mrc->var_ass[var])) {
                    atomicAdd(&(d_lit_occ[lidx]), 1);
                }
            }
        }

        gid += stride;
    }
}


__global__ void count_vars_unres_clauses_krn(Miracle *mrc) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int var_occ_lgth = d_var_occ_len;
    Lidx pos_lidx;
    Lidx neg_lidx;

    while (gid < var_occ_lgth) {
        if (!(mrc->var_ass[gid])) {
            pos_lidx = varpol_to_lidx(gid, true);
            neg_lidx = varpol_to_lidx(gid, false);

            d_var_occ[gid] = d_lit_occ[pos_lidx] + d_lit_occ[neg_lidx];
        }

        gid += stride;
    }
}


__global__ void init_var_availability_krn(Miracle *mrc) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int var_availability_lgth = d_var_availability_len;

    while (gid < var_availability_lgth) {
        d_var_availability[gid] = mrc->var_ass[gid] ? INT_MAX : gid;

        gid += stride;
    }
}


__global__ void init_clause_idxs_krn(Miracle *mrc) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int clause_idxs_lgth = d_clause_idxs_len;

    while (gid < clause_idxs_lgth) {
        d_clause_idxs[gid] = gid;

        gid += stride;
    }
}


__global__ void count_lits_unres_clauses_same_size_krn(
                                                    Miracle *mrc,
                                                    int num_clauses_same_size,
                                                    int i
                                                      ) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int cidx = d_clause_indices[i] + gid;
    int c;
    Lidx lidx;
    Var var;

    while (gid <  num_clauses_same_size) {
        c = d_clause_idxs[cidx];

        for (int l = mrc->phi->clause_indices[c];
             l < mrc->phi->clause_indices[c+1];
             l++) {
            lidx = mrc->phi->clauses[l];
            var = lidx_to_var(lidx);

            if (d_var_availability[var] != INT_MAX) {
                // Update lc_i(l).
                atomicAdd(&(d_lit_occ[lidx]), 1);
                // Update the summation of lc_i(l).
                atomicAdd(&(d_cum_lit_occ[lidx]), 1);
            }
        }
        
        cidx += stride;
        gid += stride;
    }
}


__global__ void BOHM_weigh_vars_unres_clauses_same_size_krn(Miracle *mrc,
                                                            const int alpha,
                                                            const int beta) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int var_weights_lgth = d_var_weights_len;
    Lidx pos_lidx;
    Lidx neg_lidx;
    int lc_i_pos_lidx;
    int lc_i_neg_lidx;

    while (gid < var_weights_lgth) {
        if (d_var_availability[gid] != INT_MAX) {
            pos_lidx = varpol_to_lidx(gid, true);
            neg_lidx = varpol_to_lidx(gid, false);
            lc_i_pos_lidx = d_lit_occ[pos_lidx];
            lc_i_neg_lidx = d_lit_occ[neg_lidx];

            // Compute w_i(v).
            d_var_weights[gid] = (float)
                                 (alpha * max(lc_i_pos_lidx, lc_i_neg_lidx) +
                                  beta * min(lc_i_pos_lidx, lc_i_neg_lidx));
        }

        gid += stride;
    }
}


__global__ void BOHM_update_var_availability_krn(float greatest_v_weight) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int var_availability_lgth = d_var_availability_len;

    while (gid < var_availability_lgth) {
        if (d_var_availability[gid] != INT_MAX &&
            d_var_weights[gid] < greatest_v_weight) {
            d_var_availability[gid] = INT_MAX;
        }

        gid += stride;
    }
}
