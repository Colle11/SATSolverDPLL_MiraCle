/**
 * driver.cu: TO-DO.
 * 
 * Copyright (c) Michele Collevati
 */


#include <stdlib.h>
#include <stdio.h>


#include "miracle.cuh"
#include "miracle_dynamic.cuh"
#include "miracle_gpu.cuh"
#include "launch_parameters_gpu.cuh"


#define NUM_ARGS (1)    // Number of program arguments.


int main(int argc, char *argv[]) {
    char *prog_name = argv[0];      // Program name.

    if ((argc - 1) != NUM_ARGS) {
        fprintf(stderr, "usage: %s filename\n", prog_name);
        exit(EXIT_FAILURE);
    }

    char *filename = argv[1];

    Lit lits[] = {1};
    int lits_len = 1;
    const int POSIT_n = 3;

    // Testing MiraCle (serial version).
    Miracle *mrc = mrc_create_miracle(filename);
    mrc_assign_lits(lits, lits_len, mrc);
    Lit RAND_blit = mrc_RAND_heuristic(mrc);
    Lit JW_OS_blit = mrc_JW_OS_heuristic(mrc);
    Lit JW_TS_blit = mrc_JW_TS_heuristic(mrc);
    Lit POSIT_blit = mrc_POSIT_heuristic(mrc, POSIT_n);
    Lit DLIS_blit = mrc_DLIS_heuristic(mrc);
    Lit DLCS_blit = mrc_DLCS_heuristic(mrc);
    Lit RDLIS_blit = mrc_RDLIS_heuristic(mrc);
    Lit RDLCS_blit = mrc_RDLCS_heuristic(mrc);
    printf("RAND branching literal = %d\n", RAND_blit);
    printf("JW-OS branching literal = %d\n", JW_OS_blit);
    printf("JW-TS branching literal = %d\n", JW_TS_blit);
    printf("POSIT branching literal = %d\n", POSIT_blit);
    printf("DLIS branching literal = %d\n", DLIS_blit);
    printf("DLCS branching literal = %d\n", DLCS_blit);
    printf("RDLIS branching literal = %d\n", RDLIS_blit);
    printf("RDLCS branching literal = %d\n", RDLCS_blit);
    // mrc_destroy_miracle(mrc);
    // End testing MiraCle (serial version).

    // Testing Dynamic MiraCle (serial version).
    Miracle_Dyn *mrc_dyn = mrc_dyn_create_miracle(filename);
    mrc_dyn_assign_lits(lits, lits_len, mrc_dyn);
    Lit RAND_blit_dyn = mrc_dyn_RAND_heuristic(mrc_dyn);
    Lit JW_OS_blit_dyn = mrc_dyn_JW_OS_heuristic(mrc_dyn);
    Lit JW_TS_blit_dyn = mrc_dyn_JW_TS_heuristic(mrc_dyn);
    Lit POSIT_blit_dyn = mrc_dyn_POSIT_heuristic(mrc_dyn, POSIT_n);
    Lit DLIS_blit_dyn = mrc_dyn_DLIS_heuristic(mrc_dyn);
    Lit DLCS_blit_dyn = mrc_dyn_DLCS_heuristic(mrc_dyn);
    Lit RDLIS_blit_dyn = mrc_dyn_RDLIS_heuristic(mrc_dyn);
    Lit RDLCS_blit_dyn = mrc_dyn_RDLCS_heuristic(mrc_dyn);
    printf("RAND branching literal dynamic = %d\n", RAND_blit_dyn);
    printf("JW-OS branching literal dynamic = %d\n", JW_OS_blit_dyn);
    printf("JW-TS branching literal dynamic = %d\n", JW_TS_blit_dyn);
    printf("POSIT branching literal dynamic = %d\n", POSIT_blit_dyn);
    printf("DLIS branching literal dynamic = %d\n", DLIS_blit_dyn);
    printf("DLCS branching literal dynamic = %d\n", DLCS_blit_dyn);
    printf("RDLIS branching literal dynamic = %d\n", RDLIS_blit_dyn);
    printf("RDLCS branching literal dynamic = %d\n", RDLCS_blit_dyn);
    mrc_dyn_destroy_miracle(mrc_dyn);
    // End testing Dynamic MiraCle (serial version).

    // Testing MiraCle (parallel version).
    Miracle *d_mrc = mrc_gpu_transfer_miracle_host_to_dev(mrc);
    mrc_destroy_miracle(mrc);
    Lit JW_OS_blit_gpu = mrc_gpu_JW_OS_heuristic(d_mrc);
    Lit JW_TS_blit_gpu = mrc_gpu_JW_TS_heuristic(d_mrc);
    Lit POSIT_blit_gpu = mrc_gpu_POSIT_heuristic(d_mrc, POSIT_n);
    Lit DLIS_blit_gpu = mrc_gpu_DLIS_heuristic(d_mrc);
    Lit DLCS_blit_gpu = mrc_gpu_DLCS_heuristic(d_mrc);
    Lit RDLIS_blit_gpu = mrc_gpu_RDLIS_heuristic(d_mrc);
    Lit RDLCS_blit_gpu = mrc_gpu_RDLCS_heuristic(d_mrc);
    printf("JW-OS branching literal GPU = %d\n", JW_OS_blit_gpu);
    printf("JW-TS branching literal GPU = %d\n", JW_TS_blit_gpu);
    printf("POSIT branching literal GPU = %d\n", POSIT_blit_gpu);
    printf("DLIS branching literal GPU = %d\n", DLIS_blit_gpu);
    printf("DLCS branching literal GPU = %d\n", DLCS_blit_gpu);
    printf("RDLIS branching literal GPU = %d\n", RDLIS_blit_gpu);
    printf("RDLCS branching literal GPU = %d\n", RDLCS_blit_gpu);
    mrc_gpu_destroy_miracle(d_mrc);
    // End testing MiraCle (parallel version).

    exit(EXIT_SUCCESS);
}
