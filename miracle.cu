/**
 * miracle.cu: TO-DO.
 * 
 * Copyright (c) Michele Collevati
 */


#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <limits.h>
#include <string.h>


#include "miracle.cuh"
#include "utils.cuh"


/**
 * In branching heuristics, tie breaking by selecting the variable with the
 * smallest index.
 */
#define MIN_IDX_VAR


/**
 * Global variables
 */


static float *lit_weights;      // Array of literal weights.
static int lit_weights_len;     /**
                                 * Length of lit_weights, which is
                                 * mrc->phi->num_vars * 2.
                                 */

static int *clause_sizes;       // Array of clause sizes.
static int clause_sizes_len;    /**
                                 * Length of clause_sizes, which is
                                 * mrc->phi->num_clauses.
                                 */

static int *lit_cnts;           // Array of literal counters.
static int lit_cnts_len;        /**
                                 * Length of lit_cnts, which is
                                 * mrc->phi->num_vars * 2.
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
 * @brief If two_sided = true, computes the JW-TS heuristic,
 * otherwise computes the JW-OS heuristic.
 * 
 * @param [in]mrc A miracle.
 * @param [in]two_sided A flag to choose JW-TS or JW-OS.
 * @retval The branching literal.
 */
static Lit JW_xS_heuristic(Miracle *mrc, bool two_sided);


/**
 * @brief If dlcs = true, computes the DLCS heuristic,
 * otherwise computes the DLIS heuristic.
 * 
 * @param [in]mrc A miracle.
 * @param [in]dlcs A flag to choose DLCS or DLIS.
 * @retval The branching literal.
 */
static Lit DLxS_heuristic(Miracle *mrc, bool dlcs);


/**
 * @brief If rdlcs = true, computes the RDLCS heuristic,
 * otherwise computes the RDLIS heuristic.
 * 
 * @param [in]mrc A miracle.
 * @param [in]rdlcs A flag to choose RDLCS or RDLIS.
 * @retval The branching literal.
 */
static Lit RDLxS_heuristic(Miracle *mrc, bool rdlcs);


/**
 * API definition
 */


Miracle *mrc_create_miracle(char *filename) {
    Miracle *mrc = (Miracle *)malloc(sizeof *mrc);

    mrc->phi = cnf_parse_DIMACS(filename);

    mrc->dec_lvl = 1;

    mrc->var_ass_len = mrc->phi->num_vars;
    mrc->var_ass = (int *)calloc(mrc->var_ass_len,
                                 sizeof *(mrc->var_ass));

    mrc->clause_sat_len = mrc->phi->num_clauses;
    mrc->clause_sat = (int *)calloc(mrc->clause_sat_len,
                                    sizeof *(mrc->clause_sat));

    init_aux_data_structs(mrc);

    return mrc;
}


void mrc_destroy_miracle(Miracle *mrc) {
    cnf_destroy_formula(mrc->phi);
    free(mrc->var_ass);
    free(mrc->clause_sat);
    free(mrc);
    destroy_aux_data_structs();
}


void mrc_print_miracle(Miracle *mrc) {
    printf("*** MiraCle ***\n\n");

    cnf_print_formula(mrc->phi);
    
    printf("Decision level: %d\n", mrc->dec_lvl);

    printf("Variable assignments: ");
    for (int v = 0; v < mrc->var_ass_len; v++) {
        printf("%d ", mrc->var_ass[v]);
    }
    printf("\n");

    printf("Clause satisfiability: ");
    for (int c = 0; c < mrc->clause_sat_len; c++) {
        printf("%d ", mrc->clause_sat[c]);
    }
    printf("\n");

    printf("\n*** End MiraCle ***\n\n");
}


void mrc_assign_lits(Lit *lits, int lits_len, Miracle *mrc) {
    Lit lit;
    Var var;
    bool pol;
    
    // Update variable assignments.
    for (int l = 0; l < lits_len; l++) {
        lit = lits[l];
        var = lit_to_var(lit);
        pol = lit_to_pol(lit);

        mrc->var_ass[var] = pol ? mrc->dec_lvl : -(mrc->dec_lvl);
    }

    Lidx lidx;

    // Update clause satisfiability.
    for (int c = 0; c < mrc->clause_sat_len; c++) {
        if (!(mrc->clause_sat[c])) {
            for (int l = mrc->phi->clause_indices[c];
                 l < mrc->phi->clause_indices[c+1];
                 l++) {
                lidx = mrc->phi->clauses[l];
                var = lidx_to_var(lidx);
                pol = lidx_to_pol(lidx);
                
                if ((pol && mrc->var_ass[var] > 0) ||
                    (!pol && mrc->var_ass[var] < 0)) {
                    mrc->clause_sat[c] = mrc->dec_lvl;
                    break;
                }
            }
        }
    }
}


void mrc_backjump(int bj_dec_lvl, Miracle *mrc) {
    // Restore clause satisfiability.
    for (int c = 0; c < mrc->clause_sat_len; c++) {
        if (mrc->clause_sat[c] > bj_dec_lvl) {
            mrc->clause_sat[c] = 0;
        }
    }

    // Restore variable assignments.
    for (int v = 0; v < mrc->var_ass_len; v++) {
        if (abs(mrc->var_ass[v]) > bj_dec_lvl) {
            mrc->var_ass[v] = 0;
        }
    }

    // Restore decision level.
    mrc->dec_lvl = bj_dec_lvl < 1 ? 1 : bj_dec_lvl;
}


Lit mrc_RAND_heuristic(Miracle *mrc) {
    init_PRNG();

    int num_unass_vars = 0;
    Var bvar = UNDEF_VAR;

    for (Var v = 0; v < mrc->phi->num_vars; v++) {
        if (!(mrc->var_ass[v])) {
            // Variable Selection Heuristic.
            num_unass_vars++;

            if ((rand() % num_unass_vars) == 0) {
                bvar = v;
            }
        }
    }

    if (bvar == UNDEF_VAR) {
        fprintf(stderr, "Undefined variable \"bvar\" in function "
                "\"mrc_RAND_heuristic\".\n");
        exit(EXIT_FAILURE);
    }

    // Polarity Selection Heuristic.
    return rand() % 2 ? varpol_to_lit(bvar, false) : varpol_to_lit(bvar, true);
}


Lit mrc_JW_OS_heuristic(Miracle *mrc) {
    return JW_xS_heuristic(mrc, false);
}


Lit mrc_JW_TS_heuristic(Miracle *mrc) {
    return JW_xS_heuristic(mrc, true);
}


Lit mrc_POSIT_heuristic(Miracle *mrc, const int n) {
    // Clear clause_sizes.
    memset(clause_sizes, 0, sizeof *clause_sizes * clause_sizes_len);

    // Clear lit_cnts.
    memset(lit_cnts, 0, sizeof *lit_cnts * lit_cnts_len);

    int c_size;     // Clause size.
    Lidx lidx;
    Var var;
    int smallest_c_size = INT_MAX;      // Smallest clause size.
    Lidx pos_lidx;
    Lidx neg_lidx;
    int lc_min_pos_lidx;
    int lc_min_neg_lidx;
    int weight;
    int greatest_weight = -1;
    Var bvar = UNDEF_VAR;

    /**
     * Compute the clause sizes and the smallest clause size.
     */
    for (int c = 0; c < mrc->phi->num_clauses; c++) {
        if (!(mrc->clause_sat[c])) {
            c_size = 0;

            for (int l = mrc->phi->clause_indices[c];
                 l < mrc->phi->clause_indices[c+1];
                 l++) {
                lidx = mrc->phi->clauses[l];
                var = lidx_to_var(lidx);

                if (!(mrc->var_ass[var])) {
                    c_size++;
                }
            }

            if (c_size < smallest_c_size) {
                smallest_c_size = c_size;
            }

            clause_sizes[c] = c_size;
        }
    }

    for (int c = 0; c < mrc->phi->num_clauses; c++) {
        if (!(mrc->clause_sat[c]) && clause_sizes[c] == smallest_c_size) {
            for (int l = mrc->phi->clause_indices[c];
                 l < mrc->phi->clause_indices[c+1];
                 l++) {
                lidx = mrc->phi->clauses[l];
                var = lidx_to_var(lidx);

                if (!(mrc->var_ass[var])) {
                    // Variable Selection Heuristic.
                    lit_cnts[lidx]++;
                    
                    pos_lidx = varpol_to_lidx(var, true);
                    neg_lidx = varpol_to_lidx(var, false);
                    lc_min_pos_lidx = lit_cnts[pos_lidx];
                    lc_min_neg_lidx = lit_cnts[neg_lidx];
                    weight = lc_min_pos_lidx * lc_min_neg_lidx *
                             (int)(pow(2, n) + 0.5) +
                             lc_min_pos_lidx + lc_min_neg_lidx;
                    
                    if (weight > greatest_weight
#ifdef MIN_IDX_VAR
                        || ((weight == greatest_weight) && (var < bvar))
#endif
                       ) {
                        bvar = var;
                        greatest_weight = weight;
                    }
                }
            }
        }
    }

    if (bvar == UNDEF_VAR) {
        fprintf(stderr, "Undefined variable \"bvar\" in function "
                "\"mrc_POSIT_heuristic\".\n");
        exit(EXIT_FAILURE);
    }

    pos_lidx = varpol_to_lidx(bvar, true);
    neg_lidx = varpol_to_lidx(bvar, false);
    lc_min_pos_lidx = lit_cnts[pos_lidx];
    lc_min_neg_lidx = lit_cnts[neg_lidx];

    // Polarity Selection Heuristic.
    return lc_min_pos_lidx >= lc_min_neg_lidx ? lidx_to_lit(neg_lidx) :
                                                lidx_to_lit(pos_lidx);
}


Lit mrc_DLIS_heuristic(Miracle *mrc) {
    return DLxS_heuristic(mrc, false);
}


Lit mrc_DLCS_heuristic(Miracle *mrc) {
    return DLxS_heuristic(mrc, true);
}


Lit mrc_RDLIS_heuristic(Miracle *mrc) {
    return RDLxS_heuristic(mrc, false);
}


Lit mrc_RDLCS_heuristic(Miracle *mrc) {
    return RDLxS_heuristic(mrc, true);
}


/**
 * Auxiliary function definitions
 */


static void init_aux_data_structs(Miracle *mrc) {
    lit_weights_len = mrc->phi->num_vars * 2;
    lit_weights = (float *)calloc(lit_weights_len,
                                  sizeof *lit_weights);

    clause_sizes_len = mrc->phi->num_clauses;
    clause_sizes = (int *)calloc(clause_sizes_len,
                                 sizeof *clause_sizes);

    lit_cnts_len = mrc->phi->num_vars * 2;
    lit_cnts = (int *)calloc(lit_cnts_len,
                             sizeof *lit_cnts);
}


static void destroy_aux_data_structs() {
    free(lit_weights);
    free(clause_sizes);
    free(lit_cnts);
}


static Lit JW_xS_heuristic(Miracle *mrc, bool two_sided) {
    // Clear lit_weights.
    memset(lit_weights, 0, sizeof *lit_weights * lit_weights_len);

    int c_size;     // Clause size.
    Lidx lidx;
    Var var;
    Lidx pos_lidx;
    Lidx neg_lidx;
    float weight_pos_lidx;
    float weight_neg_lidx;
    float weight;
    float greatest_weight = -1.0;
    Var bvar = UNDEF_VAR;

    // Compute the JW weight of literals in unresolved clauses.
    for (int c = 0; c < mrc->phi->num_clauses; c++) {
        if (!(mrc->clause_sat[c])) {
            c_size = 0;

            for (int l = mrc->phi->clause_indices[c];
                 l < mrc->phi->clause_indices[c+1];
                 l++) {
                lidx = mrc->phi->clauses[l];
                var = lidx_to_var(lidx);

                if (!(mrc->var_ass[var])) {
                    c_size++;
                }
            }

            for (int l = mrc->phi->clause_indices[c];
                 l < mrc->phi->clause_indices[c+1];
                 l++) {
                lidx = mrc->phi->clauses[l];
                var = lidx_to_var(lidx);

                if (!(mrc->var_ass[var])) {
                    lit_weights[lidx] += powf(2.0, (float)-c_size);
                }
            }
        }
    }

    for (Var v = 0; v < mrc->phi->num_vars; v++) {
        if (!(mrc->var_ass[v])) {
            // Variable Selection Heuristic.
            pos_lidx = varpol_to_lidx(v, true);
            neg_lidx = varpol_to_lidx(v, false);
            weight_pos_lidx = lit_weights[pos_lidx];
            weight_neg_lidx = lit_weights[neg_lidx];
            weight = two_sided ? abs(weight_pos_lidx - weight_neg_lidx) :
                                 (weight_pos_lidx >= weight_neg_lidx ?
                                  weight_pos_lidx : weight_neg_lidx);

            if (weight > greatest_weight) {
                bvar = v;
                greatest_weight = weight;
            }
        }
    }

    if (bvar == UNDEF_VAR) {
        fprintf(stderr, "Undefined variable \"bvar\" in function "
                "\"JW_xS_heuristic\".\n");
        exit(EXIT_FAILURE);
    }

    pos_lidx = varpol_to_lidx(bvar, true);
    neg_lidx = varpol_to_lidx(bvar, false);

    // Polarity Selection Heuristic.
    return lit_weights[pos_lidx] >= lit_weights[neg_lidx] ?
           lidx_to_lit(pos_lidx) : lidx_to_lit(neg_lidx);
}


static Lit DLxS_heuristic(Miracle *mrc, bool dlcs) {
    // Clear lit_cnts.
    memset(lit_cnts, 0, sizeof *lit_cnts * lit_cnts_len);

    Lidx lidx;
    Var var;
    Lidx pos_lidx;
    Lidx neg_lidx;
    int sum;
    int largest_sum = -1;
    Var bvar = UNDEF_VAR;

    for (int c = 0; c < mrc->phi->num_clauses; c++) {
        if (!(mrc->clause_sat[c])) {
            for (int l = mrc->phi->clause_indices[c];
                 l < mrc->phi->clause_indices[c+1];
                 l++) {
                lidx = mrc->phi->clauses[l];
                var = lidx_to_var(lidx);

                if (!(mrc->var_ass[var])) {
                    // Variable Selection Heuristic.
                    lit_cnts[lidx]++;
                    
                    pos_lidx = varpol_to_lidx(var, true);
                    neg_lidx = varpol_to_lidx(var, false);
                    sum = dlcs ? lit_cnts[pos_lidx] + lit_cnts[neg_lidx] :
                                 lit_cnts[lidx];

                    if (sum > largest_sum
#ifdef MIN_IDX_VAR
                        || ((sum == largest_sum) && (var < bvar))
#endif
                       ) {
                        bvar = var;
                        largest_sum = sum;
                    }
                }
            }
        }
    }

    if (bvar == UNDEF_VAR) {
        fprintf(stderr, "Undefined variable \"bvar\" in function "
                "\"DLxS_heuristic\".\n");
        exit(EXIT_FAILURE);
    }

    pos_lidx = varpol_to_lidx(bvar, true);
    neg_lidx = varpol_to_lidx(bvar, false);
    int lc_pos_lidx = lit_cnts[pos_lidx];
    int lc_neg_lidx = lit_cnts[neg_lidx];

    // Polarity Selection Heuristic.
    return lc_pos_lidx >= lc_neg_lidx ? lidx_to_lit(pos_lidx) :
                                        lidx_to_lit(neg_lidx);
}


static Lit RDLxS_heuristic(Miracle *mrc, bool rdlcs) {
    init_PRNG();

    if (rdlcs && (rand() % 2)) {
        return neg_lit(mrc_DLCS_heuristic(mrc));
    } else if (rdlcs) {
        return mrc_DLCS_heuristic(mrc);
    } else if (rand() % 2) {
        return neg_lit(mrc_DLIS_heuristic(mrc));
    } else {
        return mrc_DLIS_heuristic(mrc);
    }
}
