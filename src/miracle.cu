/**
 * miracle.cu: definition of the Miracle API.
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
#include "sat_miracle.cuh"
#include "utils.cuh"


/**
 * In branching heuristics, tie breaking by selecting the variable with the
 * smallest index.
 */
#define MIN_IDX_VAR


/**
 * @brief Clause data type.
 */
typedef struct clause {
    int size;       // Clause size.
    int idx;        // Clause index.
} Clause;


/**
 * Global variables
 */


static float *lit_weights;          // Array of literal weights.
static int lit_weights_len;         /**
                                    * Length of lit_weights, which is
                                    * mrc->phi->num_vars * 2.
                                    */

static int *clause_sizes;           // Array of clause sizes.
static int clause_sizes_len;        /**
                                    * Length of clause_sizes, which is
                                    * mrc->phi->num_clauses.
                                    */

static Clause *clauses;             // Array of Clauses.
static int clauses_len;             /**
                                     * Length of clauses, which is
                                     * mrc->phi->num_clauses.
                                     */

static int *clause_indices;         // Array of Clause indices.
static int clause_indices_len;      /**
                                     * Current length of clause_indices,
                                     * which is the number of different sizes
                                     * + 1.
                                     */

static int *lit_occ;                // Array of literal occurrences.
static int lit_occ_len;             /**
                                     * Length of lit_occ, which is
                                     * mrc->phi->num_vars * 2.
                                     */

static int *cum_lit_occ;            // Array of cumulative literal occurrences.
static int cum_lit_occ_len;         /**
                                     * Length of cum_lit_occ, which is
                                     * mrc->phi->num_vars * 2.
                                     */

static bool *var_availability;      // Array of variable availability.
static int var_availability_len;    /**
                                     * Length of var_availability, which is
                                     * mrc->phi->num_vars.
                                     */

static float *var_weights;          // Array of variable weights.
static int var_weights_len;         /**
                                     * Length of var_weights, which is
                                     * mrc->phi->num_vars.
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
 * @param [in]sat_mrc A sat_miracle.
 * @param [in]two_sided A flag to choose JW-TS or JW-OS.
 * @retval The branching literal.
 */
static Lit JW_xS_heuristic(sat_miracle *sat_mrc, bool two_sided);


/**
 * @brief If dlcs = true, computes the DLCS heuristic,
 * otherwise computes the DLIS heuristic.
 * 
 * @param [in]sat_mrc A sat_miracle.
 * @param [in]dlcs A flag to choose DLCS or DLIS.
 * @retval The branching literal.
 */
static Lit DLxS_heuristic(sat_miracle *sat_mrc, bool dlcs);


/**
 * @brief If rdlcs = true, computes the RDLCS heuristic,
 * otherwise computes the RDLIS heuristic.
 * 
 * @param [in]sat_mrc A sat_miracle.
 * @param [in]rdlcs A flag to choose RDLCS or RDLIS.
 * @retval The branching literal.
 */
static Lit RDLxS_heuristic(sat_miracle *sat_mrc, bool rdlcs);


/**
 * @brief Compares two Clause elements.
 *
 * @param [in]a The first Clause element.
 * @param [in]b The second Clause element.
 * @retval A value that specifies the relationship between the two Clause
 * elements.
 */
static int compare_clauses(const void *a, const void *b);


/**
 * API definition
 */


Miracle *mrc_create_miracle(char *filename) {
    Miracle *mrc = (Miracle *)malloc(sizeof *mrc);

    mrc->phi = cnf_parse_DIMACS(filename);

    CNF_Formula *phi = mrc->phi;

    mrc->dec_lvl = 1;

    mrc->var_ass_len = phi->num_vars;
    mrc->var_ass = (int *)calloc(mrc->var_ass_len,
                                 sizeof *(mrc->var_ass));

    mrc->clause_sat_len = phi->num_clauses;
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
    int var_ass_len = mrc->var_ass_len;
    int *var_ass = mrc->var_ass;
    for (int v = 0; v < var_ass_len; v++) {
        printf("[%d]%d ", v, var_ass[v]);
    }
    printf("\n");

    printf("Clause satisfiability: ");
    int clause_sat_len = mrc->clause_sat_len;
    int *clause_sat = mrc->clause_sat;
    for (int c = 0; c < clause_sat_len; c++) {
        printf("[%d]%d ", c, clause_sat[c]);
    }
    printf("\n");

    printf("\n*** End MiraCle ***\n\n");
}


void mrc_assign_lits(Lit *lits, int lits_len, sat_miracle *sat_mrc) {
    Miracle *mrc = sat_mrc->mrc;

    Lit lit;
    Var var;
    bool pol;
    int *var_ass = mrc->var_ass;
    int dec_lvl = mrc->dec_lvl;
    
    // Update variable assignments.
    for (int l = 0; l < lits_len; l++) {
        lit = lits[l];
        var = lit_to_var(lit);
        pol = lit_to_pol(lit);

        var_ass[var] = pol ? dec_lvl : -dec_lvl;
    }

    Lidx lidx;
    int clause_sat_len = mrc->clause_sat_len;
    int *clause_sat = mrc->clause_sat;
    CNF_Formula *phi = mrc->phi;
    int *phi_clause_indices = phi->clause_indices;
    Lidx *phi_clauses = phi->clauses;
    int v_ass;

    // Update clause satisfiability.
    for (int c = 0; c < clause_sat_len; c++) {
        if (!(clause_sat[c])) {
            for (int l = phi_clause_indices[c];
                 l < phi_clause_indices[c+1];
                 l++) {
                lidx = phi_clauses[l];
                var = lidx_to_var(lidx);
                pol = lidx_to_pol(lidx);
                v_ass = var_ass[var];
                
                if ((pol && v_ass > 0) || (!pol && v_ass < 0)) {
                    clause_sat[c] = dec_lvl;
                    break;
                }
            }
        }
    }
}


void mrc_increase_decision_level(sat_miracle *sat_mrc) {
    sat_mrc->mrc->dec_lvl++;
}


void mrc_backjump(int bj_dec_lvl, sat_miracle *sat_mrc) {
    Miracle *mrc = sat_mrc->mrc;
    int clause_sat_len = mrc->clause_sat_len;
    int *clause_sat = mrc->clause_sat;

    // Restore clause satisfiability.
    for (int c = 0; c < clause_sat_len; c++) {
        if (clause_sat[c] > bj_dec_lvl) {
            clause_sat[c] = 0;
        }
    }

    int var_ass_len = mrc->var_ass_len;
    int *var_ass = mrc->var_ass;

    // Restore variable assignments.
    for (int v = 0; v < var_ass_len; v++) {
        if (abs(var_ass[v]) > bj_dec_lvl) {
            var_ass[v] = 0;
        }
    }

    // Restore decision level.
    mrc->dec_lvl = bj_dec_lvl < 1 ? 1 : bj_dec_lvl;
}


Lit mrc_RAND_heuristic(sat_miracle *sat_mrc) {
    Miracle *mrc = sat_mrc->mrc;

    init_PRNG();

    int num_unass_vars = 0;
    Var bvar = UNDEF_VAR;
    int num_vars = mrc->phi->num_vars;
    int *var_ass = mrc->var_ass;

    for (Var v = 0; v < num_vars; v++) {
        if (!(var_ass[v])) {
            // Variable Selection Heuristic.
            num_unass_vars++;

            if ((rand() % num_unass_vars) == 0) {
                bvar = v;
            }
        }
    }

    if (bvar == UNDEF_VAR) {
        // return UNDEF_LIT;

        fprintf(stderr, "Undefined variable \"bvar\" in function "
                "\"mrc_RAND_heuristic\".\n");
        exit(EXIT_FAILURE);
    }

    // Polarity Selection Heuristic.
    return rand() % 2 ? varpol_to_lit(bvar, false) : varpol_to_lit(bvar, true);
}


Lit mrc_JW_OS_heuristic(sat_miracle *sat_mrc) {
    return JW_xS_heuristic(sat_mrc, false);
}


Lit mrc_JW_TS_heuristic(sat_miracle *sat_mrc) {
    return JW_xS_heuristic(sat_mrc, true);
}


Lit mrc_BOHM_heuristic(sat_miracle *sat_mrc, const int alpha, const int beta) {
    Miracle *mrc = sat_mrc->mrc;
    int *var_ass = mrc->var_ass;

    // Init var_availability.
    for (Var v = 0; v < var_availability_len; v++) {
        var_availability[v] = !((bool)var_ass[v]);
    }

    // Clear cum_lit_occ.
    memset(cum_lit_occ, 0, sizeof *cum_lit_occ * cum_lit_occ_len);

    int c_size;     // Clause size.
    Lidx lidx;
    Var var;
    int *clause_sat = mrc->clause_sat;
    CNF_Formula *phi = mrc->phi;
    int *phi_clause_indices = phi->clause_indices;
    Lidx *phi_clauses = phi->clauses;

    // Compute the clause sizes.
    for (int c = 0; c < clauses_len; c++) {
        c_size = 0;

        if (!(clause_sat[c])) {
            for (int l = phi_clause_indices[c];
                 l < phi_clause_indices[c+1];
                 l++) {
                lidx = phi_clauses[l];
                var = lidx_to_var(lidx);

                if (!(var_ass[var])) {
                    c_size++;
                }
            }
        }

        clauses[c].size = c_size;
        clauses[c].idx = c;
    }

    // Sort the clauses by increasing size.
    qsort(clauses, clauses_len, sizeof *clauses, compare_clauses);

    // Build the array of Clause indices.
    clause_indices_len = 0;
    clause_indices[clause_indices_len] = 0;
    clause_indices_len++;
    
    for (int c = 1; c < clauses_len; c++) {
        if (clauses[c-1].size < clauses[c].size) {
            clause_indices[clause_indices_len] = c;
            clause_indices_len++;
        }
    }

    clause_indices[clause_indices_len] = clauses_len;
    clause_indices_len++;

    int c;
    Lidx pos_lidx;
    Lidx neg_lidx;
    int lc_i_pos_lidx;
    int lc_i_neg_lidx;
    float weight;
    float greatest_weight;
    Var bvar = UNDEF_VAR;

    for (int i = clauses[0].size == 0 ? 1 : 0;
         i < clause_indices_len - 1;
         i++) {
        // Clear lit_occ.
        memset(lit_occ, 0, sizeof *lit_occ * lit_occ_len);

        // Clear var_weights.
        memset(var_weights, 0, sizeof *var_weights * var_weights_len);

        // Reset greatest_weight.
        greatest_weight = -1.0;

        for (int cidx = clause_indices[i];
             cidx < clause_indices[i+1];
             cidx++) {
            c = clauses[cidx].idx;

            for (int l = phi_clause_indices[c];
                 l < phi_clause_indices[c+1];
                 l++) {
                lidx = phi_clauses[l];
                var = lidx_to_var(lidx);

                if (var_availability[var]) {
                    // Update lc_i(l).
                    lit_occ[lidx]++;
                    // Update the summation of lc_i(l).
                    cum_lit_occ[lidx]++;

                    // Compute w_i(v).
                    pos_lidx = varpol_to_lidx(var, true);
                    neg_lidx = varpol_to_lidx(var, false);
                    lc_i_pos_lidx = lit_occ[pos_lidx];
                    lc_i_neg_lidx = lit_occ[neg_lidx];
                    weight = (float)
                             (alpha * max(lc_i_pos_lidx, lc_i_neg_lidx) +
                              beta * min(lc_i_pos_lidx, lc_i_neg_lidx));
                    var_weights[var] = weight;

                    // Compute the greatest w_i(v).
                    if (weight > greatest_weight) {
                        greatest_weight = weight;
                    }
                }
            }
        }

        // Variable Selection Heuristic.
        bvar = UNDEF_VAR;

        for (Var v = 0; v < var_availability_len; v++) {
            if (var_availability[v]) {
                if (var_weights[v] < greatest_weight) {
                    var_availability[v] = false;
                } else if (bvar == UNDEF_VAR) {
                    bvar = v;
                }
            }
        }
    }

    if (bvar == UNDEF_VAR) {
        // return UNDEF_LIT;

        fprintf(stderr, "Undefined variable \"bvar\" in function "
                "\"mrc_BOHM_heuristic\".\n");
        exit(EXIT_FAILURE);
    }

    pos_lidx = varpol_to_lidx(bvar, true);
    neg_lidx = varpol_to_lidx(bvar, false);
    lc_i_pos_lidx = cum_lit_occ[pos_lidx];
    lc_i_neg_lidx = cum_lit_occ[neg_lidx];

    // Polarity Selection Heuristic.
    return lc_i_pos_lidx >= lc_i_neg_lidx ? lidx_to_lit(pos_lidx) :
                                            lidx_to_lit(neg_lidx);
}


Lit mrc_POSIT_heuristic(sat_miracle *sat_mrc, const int n) {
    Miracle *mrc = sat_mrc->mrc;

    // Clear clause_sizes.
    memset(clause_sizes, 0, sizeof *clause_sizes * clause_sizes_len);

    // Clear lit_occ.
    memset(lit_occ, 0, sizeof *lit_occ * lit_occ_len);

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
    CNF_Formula *phi = mrc->phi;
    int num_clauses = phi->num_clauses;
    int *clause_sat = mrc->clause_sat;
    int *phi_clause_indices = phi->clause_indices;
    Lidx *phi_clauses = phi->clauses;
    int *var_ass = mrc->var_ass;

    /**
     * Compute the clause sizes and the smallest clause size.
     */
    for (int c = 0; c < num_clauses; c++) {
        if (!(clause_sat[c])) {
            c_size = 0;

            for (int l = phi_clause_indices[c];
                 l < phi_clause_indices[c+1];
                 l++) {
                lidx = phi_clauses[l];
                var = lidx_to_var(lidx);

                if (!(var_ass[var])) {
                    c_size++;
                }
            }

            if (c_size < smallest_c_size) {
                smallest_c_size = c_size;
            }

            clause_sizes[c] = c_size;
        }
    }

    int exp2_n = (int)(exp2f((float)n) + 0.5);

    for (int c = 0; c < num_clauses; c++) {
        if (!(clause_sat[c]) && clause_sizes[c] == smallest_c_size) {
            for (int l = phi_clause_indices[c];
                 l < phi_clause_indices[c+1];
                 l++) {
                lidx = phi_clauses[l];
                var = lidx_to_var(lidx);

                if (!(var_ass[var])) {
                    // Variable Selection Heuristic.
                    lit_occ[lidx]++;
                    
                    pos_lidx = varpol_to_lidx(var, true);
                    neg_lidx = varpol_to_lidx(var, false);
                    lc_min_pos_lidx = lit_occ[pos_lidx];
                    lc_min_neg_lidx = lit_occ[neg_lidx];
                    weight = lc_min_pos_lidx * lc_min_neg_lidx * exp2_n +
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
        // return UNDEF_LIT;

        fprintf(stderr, "Undefined variable \"bvar\" in function "
                "\"mrc_POSIT_heuristic\".\n");
        exit(EXIT_FAILURE);
    }

    pos_lidx = varpol_to_lidx(bvar, true);
    neg_lidx = varpol_to_lidx(bvar, false);
    lc_min_pos_lidx = lit_occ[pos_lidx];
    lc_min_neg_lidx = lit_occ[neg_lidx];

    // Polarity Selection Heuristic.
    return lc_min_pos_lidx >= lc_min_neg_lidx ? lidx_to_lit(neg_lidx) :
                                                lidx_to_lit(pos_lidx);
}


Lit mrc_DLIS_heuristic(sat_miracle *sat_mrc) {
    return DLxS_heuristic(sat_mrc, false);
}


Lit mrc_DLCS_heuristic(sat_miracle *sat_mrc) {
    return DLxS_heuristic(sat_mrc, true);
}


Lit mrc_RDLIS_heuristic(sat_miracle *sat_mrc) {
    return RDLxS_heuristic(sat_mrc, false);
}


Lit mrc_RDLCS_heuristic(sat_miracle *sat_mrc) {
    return RDLxS_heuristic(sat_mrc, true);
}


/**
 * Auxiliary function definitions
 */


static void init_aux_data_structs(Miracle *mrc) {
    CNF_Formula *phi = mrc->phi;

    lit_weights_len = phi->num_vars * 2;
    lit_weights = (float *)calloc(lit_weights_len,
                                  sizeof *lit_weights);

    clause_sizes_len = phi->num_clauses;
    clause_sizes = (int *)calloc(clause_sizes_len,
                                 sizeof *clause_sizes);

    clauses_len = phi->num_clauses;
    clauses = (Clause *)malloc(sizeof *clauses * clauses_len);

    clause_indices_len = 0;
    clause_indices = (int *)malloc(sizeof *clause_indices *
                                   (clauses_len + 1));

    lit_occ_len = phi->num_vars * 2;
    lit_occ = (int *)calloc(lit_occ_len,
                            sizeof *lit_occ);

    cum_lit_occ_len = phi->num_vars * 2;
    cum_lit_occ = (int *)calloc(cum_lit_occ_len,
                                sizeof *cum_lit_occ);

    var_availability_len = phi->num_vars;
    var_availability = (bool *)malloc(sizeof *var_availability *
                                      var_availability_len);

    var_weights_len = phi->num_vars;
    var_weights = (float *)calloc(var_weights_len,
                                  sizeof *var_weights);
}


static void destroy_aux_data_structs() {
    free(lit_weights);
    free(clause_sizes);
    free(clauses);
    free(clause_indices);
    free(lit_occ);
    free(cum_lit_occ);
    free(var_availability);
    free(var_weights);
}


static Lit JW_xS_heuristic(sat_miracle *sat_mrc, bool two_sided) {
    Miracle *mrc = sat_mrc->mrc;

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
    CNF_Formula *phi = mrc->phi;
    int num_clauses = phi->num_clauses;
    int *clause_sat = mrc->clause_sat;
    int *phi_clause_indices = phi->clause_indices;
    Lidx *phi_clauses = phi->clauses;
    int *var_ass = mrc->var_ass;

    // Compute the JW weight of literals in unresolved clauses.
    for (int c = 0; c < num_clauses; c++) {
        if (!(clause_sat[c])) {
            c_size = 0;

            for (int l = phi_clause_indices[c];
                 l < phi_clause_indices[c+1];
                 l++) {
                lidx = phi_clauses[l];
                var = lidx_to_var(lidx);

                if (!(var_ass[var])) {
                    c_size++;
                }
            }

            weight = exp2f((float)-c_size);

            for (int l = phi_clause_indices[c];
                 l < phi_clause_indices[c+1];
                 l++) {
                lidx = phi_clauses[l];
                var = lidx_to_var(lidx);

                if (!(var_ass[var])) {
                    lit_weights[lidx] += weight;
                }
            }
        }
    }

    int num_vars = phi->num_vars;

    for (Var v = 0; v < num_vars; v++) {
        if (!(var_ass[v])) {
            // Variable Selection Heuristic.
            pos_lidx = varpol_to_lidx(v, true);
            neg_lidx = varpol_to_lidx(v, false);
            weight_pos_lidx = lit_weights[pos_lidx];
            weight_neg_lidx = lit_weights[neg_lidx];

            if (weight_pos_lidx > 0 || weight_neg_lidx > 0) {
                weight = two_sided ? abs(weight_pos_lidx - weight_neg_lidx) :
                                     (weight_pos_lidx >= weight_neg_lidx ?
                                      weight_pos_lidx : weight_neg_lidx);

                if (weight > greatest_weight) {
                    bvar = v;
                    greatest_weight = weight;
                }
            }
        }
    }

    if (bvar == UNDEF_VAR) {
        // return UNDEF_LIT;

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


static Lit DLxS_heuristic(sat_miracle *sat_mrc, bool dlcs) {
    Miracle *mrc = sat_mrc->mrc;

    // Clear lit_occ.
    memset(lit_occ, 0, sizeof *lit_occ * lit_occ_len);

    Lidx lidx;
    Var var;
    Lidx pos_lidx;
    Lidx neg_lidx;
    int sum;
    int largest_sum = -1;
    Var bvar = UNDEF_VAR;
    CNF_Formula *phi = mrc->phi;
    int num_clauses = phi->num_clauses;
    int *clause_sat = mrc->clause_sat;
    int *phi_clause_indices = phi->clause_indices;
    Lidx *phi_clauses = phi->clauses;
    int *var_ass = mrc->var_ass;

    for (int c = 0; c < num_clauses; c++) {
        if (!(clause_sat[c])) {
            for (int l = phi_clause_indices[c];
                 l < phi_clause_indices[c+1];
                 l++) {
                lidx = phi_clauses[l];
                var = lidx_to_var(lidx);

                if (!(var_ass[var])) {
                    // Variable Selection Heuristic.
                    lit_occ[lidx]++;
                    
                    pos_lidx = varpol_to_lidx(var, true);
                    neg_lidx = varpol_to_lidx(var, false);
                    sum = dlcs ? lit_occ[pos_lidx] + lit_occ[neg_lidx] :
                                 lit_occ[lidx];

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
        // return UNDEF_LIT;

        fprintf(stderr, "Undefined variable \"bvar\" in function "
                "\"DLxS_heuristic\".\n");
        exit(EXIT_FAILURE);
    }

    pos_lidx = varpol_to_lidx(bvar, true);
    neg_lidx = varpol_to_lidx(bvar, false);
    int lc_pos_lidx = lit_occ[pos_lidx];
    int lc_neg_lidx = lit_occ[neg_lidx];

    // Polarity Selection Heuristic.
    return lc_pos_lidx >= lc_neg_lidx ? lidx_to_lit(pos_lidx) :
                                        lidx_to_lit(neg_lidx);
}


static Lit RDLxS_heuristic(sat_miracle *sat_mrc, bool rdlcs) {
    init_PRNG();

    if (rdlcs && (rand() % 2)) {
        return neg_lit(mrc_DLCS_heuristic(sat_mrc));
    } else if (rdlcs) {
        return mrc_DLCS_heuristic(sat_mrc);
    } else if (rand() % 2) {
        return neg_lit(mrc_DLIS_heuristic(sat_mrc));
    } else {
        return mrc_DLIS_heuristic(sat_mrc);
    }
}


static int compare_clauses(const void *a, const void *b) {
    return (((Clause *)a)->size - ((Clause *)b)->size);
}
