/*
 * Program to implement a SAT solver using the DPLL algorithm with unit
 * propagation Sukrut Rao CS15BTECH11036
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>
#include <float.h>
#include <math.h>
#include <signal.h>
#include <sys/wait.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>

#include "utils.cuh"
#include "sig_handling.h"
#ifdef MRC
#include "miracle.cuh"
#endif
#ifdef MRC_DYN
#include "miracle_dynamic.cuh"
#endif
#ifdef MRC_GPU
#include "miracle.cuh"
#include "miracle_gpu.cuh"
#include "launch_parameters_gpu.cuh"
#endif

#define NUM_ARGS (1)    // Number of solver arguments.
#define POSIT_n (3)     // Constant of the POSIT weight function.
#define BOHM_alpha (1)  // Constant of the BOHM weight function.
#define BOHM_beta (2)   // Constant of the BOHM weight function.

// #define NO_MRC
// #define MRC
// #define MRC_DYN
// #define MRC_GPU

// #define JW_OS
// #define JW_TS
// #define BOHM
// #define POSIT
// #define DLIS
// #define DLCS
// #define RDLIS
// #define RDLCS

// #define STATS

using namespace std;

#ifdef MRC
static Lit *lits;       // Array of assigned literals.
static int lits_len;    // Length of lits, which is the number of assigned literals.
#endif
#ifdef MRC_DYN
static Lit *lits;       // Array of assigned literals.
static int lits_len;    // Length of lits, which is the number of assigned literals.
#endif
#ifdef MRC_GPU
static Lit *lits;       // Array of assigned literals.
static int lits_len;    // Length of lits, which is the number of assigned literals.
#endif

#ifdef STATS
clock_t solve_tic;                     // Solving time in clock.
clock_t solve_toc;                     // Solving time out clock.
clock_t heur_tic;                      // Heuristic time in clock.
clock_t heur_toc;                      // Heuristic time out clock.
double solving_time;                   // Solving time.
static double heur_time;               // Heuristic time.
double max_heuristic_time;             // Maximum heuristic computation time.
double min_heuristic_time;             // Minimum heuristic computation time.
double avg_heuristic_time;             // Average heuristic computation time.
double tot_heuristic_time;             // Total heuristic computation time.
int num_heur;                          // Number of heuristic calls.
int timeout_expired;                   // Flag for timeout expiration.
int escape;                            // Flag for SIGINT.
static int timeout;                    // In s.
#endif


/*
 * enum for different types of return flags defined
 */
enum Cat {
  satisfied,   // when a satisfying assignment has been found
  unsatisfied, // when no satisfying assignment has been found after
               // exhaustively searching
  normal,   // when no satisfying assignment has been found till now, and DPLL()
            // has exited normally
  completed // when the DPLL algorithm has completed execution
};

/*
 * class to represent a boolean formula
 */
class Formula {
public:
  // a vector that stores the value assigned to each variable, where
  // -1 - unassigned
  // 0 - true
  // 1 - false
  vector<int> literals;
#ifdef NO_MRC
  vector<int> literal_frequency; // vector to store the number of occurrences of
                                 // each literal

  // vector to store the difference in number of occurrences with
  // positive and negative polarity of each literal
  vector<int> literal_polarity;
#endif

  // vector to store the clauses
  // for each clauses, if the variable n is of positive polarity, then 2n is
  // stored if the variable n is of negative polarity, then 2n+1 is stored here,
  // n is assumed to be zero indexed
  vector<vector<int>> clauses;
  Formula() {}

  // copy constructor for copying a formula - each member is copied over
  Formula(const Formula &f) {
    literals = f.literals;
    clauses = f.clauses;
#ifdef NO_MRC
    literal_frequency = f.literal_frequency;
    literal_polarity = f.literal_polarity;
#endif
  }
};

/*
 * class to represent the structure and functions of the SAT Solver
 */
class SATSolverDPLL {
private:
  Formula formula;               // the initial formula given as input
  int literal_count;             // the number of variables in the formula
  int clause_count;              // the number of clauses in the formula
#ifdef NO_MRC
  int unit_propagate(Formula &); // performs unit propagation
#endif
#ifdef MRC
  int unit_propagate(Formula &, Miracle *mrc, Lit blit);
#endif
#ifdef MRC_DYN
  int unit_propagate(Formula &, Miracle_Dyn *mrc_dyn, Lit blit);
#endif
#ifdef MRC_GPU
  int unit_propagate(Formula &, Miracle *d_mrc, Lit blit);
#endif
#ifdef NO_MRC
  int DPLL(Formula);             // performs DPLL recursively
#endif
#ifdef MRC
  int DPLL(Formula, Miracle *mrc, Lit blit);
#endif
#ifdef MRC_DYN
  int DPLL(Formula, Miracle_Dyn *mrc_dyn, Lit blit);
#endif
#ifdef MRC_GPU
  int DPLL(Formula, Miracle *d_mrc, Lit blit);
#endif
  int apply_transform(Formula &,
                      int); // applies the value of the literal in every clause
  void show_result(Formula &, int); // displays the result
public:
  SATSolverDPLL() {}
  void initialize(char *filename); // initializes the values
#ifdef NO_MRC
  void solve();      // calls the solver
#endif
#ifdef MRC
  void solve(Miracle *mrc);
#endif
#ifdef MRC_DYN
  void solve(Miracle_Dyn *mrc_dyn);
#endif
#ifdef MRC_GPU
  void solve(Miracle *d_mrc);
#endif
#ifdef MRC_DYN
  void print_debug_info(Formula &f, Miracle_Dyn *mrc_dyn);
#endif
#ifdef STATS
  void print_stats(double solving_time,
                   double max_heuristic_time,
                   double min_heuristic_time,
                   double avg_heuristic_time,
                   double tot_heuristic_time,
                   int num_heur);
#endif
};

/*
 * function that accepts the inputs from the user and initializes the attributes
 * in the solver
 */
void SATSolverDPLL::initialize(char *filename) {
  char c;   // store first character
  string s; // dummy string
  ifstream dimacs_cnf_file(filename);

  while (true) {
    dimacs_cnf_file >> c;
    if (c == 'c') // if comment
    {
      getline(dimacs_cnf_file, s); // ignore
    } else             // else, if would be a p
    {
      dimacs_cnf_file >> s; // this would be cnf
      break;
    }
  }
  dimacs_cnf_file >> literal_count;
  dimacs_cnf_file >> clause_count;

  // set the vectors to their appropriate sizes and initial values
  formula.literals.clear();
  formula.literals.resize(literal_count, -1);
  formula.clauses.clear();
  formula.clauses.resize(clause_count);
#ifdef NO_MRC
  formula.literal_frequency.clear();
  formula.literal_frequency.resize(literal_count, 0);
  formula.literal_polarity.clear();
  formula.literal_polarity.resize(literal_count, 0);
#endif

  int literal; // store the incoming literal value
  // iterate over the clauses
  for (int i = 0; i < clause_count; i++) {
    while (true) // while the ith clause gets more literals
    {
      dimacs_cnf_file >> literal;
      if (literal > 0) // if the variable has positive polarity
      {
        formula.clauses[i].push_back(2 *
                                     (literal - 1)); // store it in the form 2n
        // increment frequency and polarity of the literal
#ifdef NO_MRC
        formula.literal_frequency[literal - 1]++;
        formula.literal_polarity[literal - 1]++;
#endif
      } else if (literal < 0) // if the variable has negative polarity
      {
        formula.clauses[i].push_back(2 * ((-1) * literal - 1) +
                                     1); // store it in the form 2n+1
        // increment frequency and decrement polarity of the literal
#ifdef NO_MRC
        formula.literal_frequency[-1 - literal]++;
        formula.literal_polarity[-1 - literal]--;
#endif
      } else {
        break; // read 0, so move to next clause
      }
    }
  }

#ifdef MRC
  lits = (Lit *)malloc(sizeof *lits * literal_count);
  lits_len = 0;
#endif
#ifdef MRC_DYN
  lits = (Lit *)malloc(sizeof *lits * literal_count);
  lits_len = 0;
#endif
#ifdef MRC_GPU
  lits = (Lit *)malloc(sizeof *lits * literal_count);
  lits_len = 0;
#endif
#ifdef STATS
  max_heuristic_time = -DBL_MAX;
  min_heuristic_time = DBL_MAX;
  tot_heuristic_time = 0;
  num_heur = 0;
  timeout_expired = 0;
  escape = 0;
  timeout = 900;   // In s.

  // Set SIGINT handler.
  install_handler();

  // Set SIGALRM handler.
  install_alarmhandler();
#endif
}

/*
 * function to perform unit resolution in a given formula
 * arguments: f - the formula to perform unit resolution on
 * return value: int - the status of the solver after unit resolution, a member
 * of the Cat enum Cat::satisfied - the formula has been satisfied
 *               Cat::unsatisfied - the formula can no longer be satisfied
 *               Cat::normal - normal exit
 */
#ifdef NO_MRC
int SATSolverDPLL::unit_propagate(Formula &f) {
#endif
#ifdef MRC
int SATSolverDPLL::unit_propagate(Formula &f, Miracle *mrc, Lit blit) {
  Lidx lidx;
  lits_len = 0;

  if (blit != 0)
  {
    lits[lits_len] = blit;
    lits_len++;
    mrc_increase_decision_level(mrc);
  }
#endif
#ifdef MRC_DYN
int SATSolverDPLL::unit_propagate(Formula &f, Miracle_Dyn *mrc_dyn, Lit blit) {
  Lidx lidx;
  lits_len = 0;

  if (blit != 0)
  {
    lits[lits_len] = blit;
    lits_len++;
    mrc_dyn_increase_decision_level(mrc_dyn);
  }
#endif
#ifdef MRC_GPU
int SATSolverDPLL::unit_propagate(Formula &f, Miracle *d_mrc, Lit blit) {
  Lidx lidx;
  lits_len = 0;

  if (blit != 0)
  {
    lits[lits_len] = blit;
    lits_len++;
    mrc_gpu_increase_decision_level(d_mrc);
  }
#endif
  bool unit_clause_found =
      false; // stores whether the current iteration found a unit clause
  if (f.clauses.size() == 0) // if the formula contains no clauses
  {
    return Cat::satisfied; // it is vacuously satisfied
  }
  do {
    unit_clause_found = false;
    // iterate over the clauses in f
    for (int i = 0; i < f.clauses.size(); i++) {
      if (f.clauses[i].size() ==
          1) // if the size of a clause is 1, it is a unit clause
      {
        unit_clause_found = true;
        f.literals[f.clauses[i][0] / 2] =
            f.clauses[i][0] % 2; // 0 - if true, 1 - if false, set the literal
#ifdef NO_MRC
        f.literal_frequency[f.clauses[i][0] / 2] =
            -1; // once assigned, reset the frequency to mark it closed
#endif
#ifdef MRC
        lidx = (Lidx)f.clauses[i][0];
        lits[lits_len] = lidx_to_lit(lidx);
        lits_len++;
#endif
#ifdef MRC_DYN
        lidx = (Lidx)f.clauses[i][0];
        lits[lits_len] = lidx_to_lit(lidx);
        lits_len++;
#endif
#ifdef MRC_GPU
        lidx = (Lidx)f.clauses[i][0];
        lits[lits_len] = lidx_to_lit(lidx);
        lits_len++;
#endif
        int result = apply_transform(f, f.clauses[i][0] /
                                            2); // apply this change through f
        // if this caused the formula to be either satisfied or unsatisfied,
        // return the result flag
        if (result == Cat::satisfied || result == Cat::unsatisfied) {
          return result;
        }
        break; // exit the loop to check for another unit clause from the start
      } else if (f.clauses[i].size() == 0) // if a given clause is empty
      {
        return Cat::unsatisfied; // the formula is unsatisfiable in this branch
      }
    }
  } while (unit_clause_found);

#ifdef MRC
  if (lits_len > 0) {
    mrc_assign_lits(lits, lits_len, mrc);
  }
#endif
#ifdef MRC_DYN
  if (lits_len > 0) {
    mrc_dyn_assign_lits(lits, lits_len, mrc_dyn);
  }
#endif
#ifdef MRC_GPU
  if (lits_len > 0) {
    mrc_gpu_assign_lits(lits, lits_len, d_mrc);
  }
#endif

  return Cat::normal; // if reached here, the unit resolution ended normally
}

/*
 * applies a value of a literal to all clauses in a given formula
 * arguments: f - the formula to apply on
 *            literal_to_apply - the literal which has just been set
 * return value: int - the return status flag, a member of the Cat enum
 *               Cat::satisfied - the formula has been satisfied
 *               Cat::unsatisfied - the formula can no longer be satisfied
 *               Cat::normal - normal exit
 */
int SATSolverDPLL::apply_transform(Formula &f, int literal_to_apply) {
  int value_to_apply = f.literals[literal_to_apply]; // the value to apply, 0 -
                                                     // if true, 1 - if false
  // iterate over the clauses in f
  for (int i = 0; i < f.clauses.size(); i++) {
    // iterate over the variables in the clause
    for (int j = 0; j < f.clauses[i].size(); j++) {
      // if this is true, then the literal appears with the same polarity as it
      // is being applied that is, if assigned true, it appears positive if
      // assigned false, it appears negative, in this clause hence, the clause
      // has now become true
      if ((2 * literal_to_apply + value_to_apply) == f.clauses[i][j]) {
        f.clauses.erase(f.clauses.begin() +
                        i); // remove the clause from the list
        i--;                // reset iterator
        if (f.clauses.size() ==
            0) // if all clauses have been removed, the formula is satisfied
        {
          return Cat::satisfied;
        }
        break; // move to the next clause
      } else if (f.clauses[i][j] / 2 ==
                 literal_to_apply) // the literal appears with opposite polarity
      {
        f.clauses[i].erase(
            f.clauses[i].begin() +
            j); // remove the literal from the clause, as it is false in it
        j--;    // reset the iterator
        if (f.clauses[i].size() ==
            0) // if the clause is empty, the formula is unsatisfiable currently
        {
          return Cat::unsatisfied;
        }
        break; // move to the next clause
      }
    }
  }
  // if reached here, the function is exiting normally
  return Cat::normal;
}

/*
 * function to perform the recursive DPLL on a given formula
 * argument: f - the formula to perform DPLL on
 * return value: int - the return status flag, a member of the Cat enum
 *               Cat::normal - exited normally
 *               Cat::completed - result has been found, exit recursion all the
 * way
 */
#ifdef NO_MRC
int SATSolverDPLL::DPLL(Formula f) {
  int result = unit_propagate(f); // perform unit propagation on the formula
#endif
#ifdef MRC
int SATSolverDPLL::DPLL(Formula f, Miracle *mrc, Lit blit) {
  int dec_lvl = mrc->dec_lvl;
  Lit bl;
  Var bv;
  bool pol;
  int i;

  int result = unit_propagate(f, mrc, blit);
#endif
#ifdef MRC_DYN
int SATSolverDPLL::DPLL(Formula f, Miracle_Dyn *mrc_dyn, Lit blit) {
  int dec_lvl = mrc_dyn->dec_lvl;
  Lit bl;
  Var bv;
  bool pol;
  int i;

  int result = unit_propagate(f, mrc_dyn, blit);
#endif
#ifdef MRC_GPU
int SATSolverDPLL::DPLL(Formula f, Miracle *d_mrc, Lit blit) {
  int dec_lvl;
  gpuErrchk( cudaMemcpy(&dec_lvl, &(d_mrc->dec_lvl),
                        sizeof dec_lvl,
                        cudaMemcpyDeviceToHost) );
  Lit bl;
  Var bv;
  bool pol;
  int i;

  int result = unit_propagate(f, d_mrc, blit);
#endif
  if (result == Cat::satisfied) // if formula satisfied, show result and return
  {
    show_result(f, result);
    return Cat::completed;
  } else if (result == Cat::unsatisfied) // if formula not satisfied in this
                                         // branch, return normally
  {
#ifdef MRC
    mrc_backjump(dec_lvl, mrc);
#endif
#ifdef MRC_DYN
    mrc_dyn_backjump(dec_lvl, mrc_dyn);
#endif
#ifdef MRC_GPU
    mrc_gpu_backjump(dec_lvl, d_mrc);
#endif
    return Cat::normal;
  }
  // find the variable with maximum frequency in f, which will be the next to be
  // assigned a value already assigned variables have this field reset to -1 in
  // order to ignore them
#ifdef NO_MRC
#ifdef STATS
  heur_tic = clock();
#endif
  int i = distance(
      f.literal_frequency.begin(),
      max_element(f.literal_frequency.begin(), f.literal_frequency.end()));
#ifdef STATS
  heur_toc = clock();
#endif
#endif
#ifdef MRC
#ifdef STATS
  heur_tic = clock();
#endif
  #ifdef JW_OS
  bl = mrc_JW_OS_heuristic(mrc);
  #endif
  #ifdef JW_TS
  bl = mrc_JW_TS_heuristic(mrc);
  #endif
  #ifdef BOHM
  bl = mrc_BOHM_heuristic(mrc, BOHM_alpha, BOHM_beta);
  #endif
  #ifdef POSIT
  bl = mrc_POSIT_heuristic(mrc, POSIT_n);
  #endif
  #ifdef DLIS
  bl = mrc_DLIS_heuristic(mrc);
  #endif
  #ifdef DLCS
  bl = mrc_DLCS_heuristic(mrc);
  #endif
  #ifdef RDLIS
  bl = mrc_RDLIS_heuristic(mrc);
  #endif
  #ifdef RDLCS
  bl = mrc_RDLCS_heuristic(mrc);
  #endif

  bv = lit_to_var(bl);
  pol = lit_to_pol(bl);
  i = (int)bv;
#ifdef STATS
  heur_toc = clock();
#endif
#endif
#ifdef MRC_DYN
#ifdef STATS
  heur_tic = clock();
#endif
  #ifdef JW_OS
  bl = mrc_dyn_JW_OS_heuristic(mrc_dyn);
  #endif
  #ifdef JW_TS
  bl = mrc_dyn_JW_TS_heuristic(mrc_dyn);
  #endif
  #ifdef BOHM
  bl = mrc_dyn_BOHM_heuristic(mrc_dyn, BOHM_alpha, BOHM_beta);
  #endif
  #ifdef POSIT
  bl = mrc_dyn_POSIT_heuristic(mrc_dyn, POSIT_n);
  #endif
  #ifdef DLIS
  bl = mrc_dyn_DLIS_heuristic(mrc_dyn);
  #endif
  #ifdef DLCS
  bl = mrc_dyn_DLCS_heuristic(mrc_dyn);
  #endif
  #ifdef RDLIS
  bl = mrc_dyn_RDLIS_heuristic(mrc_dyn);
  #endif
  #ifdef RDLCS
  bl = mrc_dyn_RDLCS_heuristic(mrc_dyn);
  #endif

  bv = lit_to_var(bl);
  pol = lit_to_pol(bl);
  i = (int)bv;
#ifdef STATS
  heur_toc = clock();
#endif
#endif
#ifdef MRC_GPU
#ifdef STATS
  heur_tic = clock();
#endif
  #ifdef JW_OS
  bl = mrc_gpu_JW_OS_heuristic(d_mrc);
  #endif
  #ifdef JW_TS
  bl = mrc_gpu_JW_TS_heuristic(d_mrc);
  #endif
  #ifdef BOHM
  bl = mrc_gpu_BOHM_heuristic(d_mrc, BOHM_alpha, BOHM_beta);
  #endif
  #ifdef POSIT
  bl = mrc_gpu_POSIT_heuristic(d_mrc, POSIT_n);
  #endif
  #ifdef DLIS
  bl = mrc_gpu_DLIS_heuristic(d_mrc);
  #endif
  #ifdef DLCS
  bl = mrc_gpu_DLCS_heuristic(d_mrc);
  #endif
  #ifdef RDLIS
  bl = mrc_gpu_RDLIS_heuristic(d_mrc);
  #endif
  #ifdef RDLCS
  bl = mrc_gpu_RDLCS_heuristic(d_mrc);
  #endif

  bv = lit_to_var(bl);
  pol = lit_to_pol(bl);
  i = (int)bv;
#ifdef STATS
  heur_toc = clock();
#endif
#endif
#ifdef STATS
  num_heur++;
  heur_time = ((double)(heur_toc - heur_tic)) / CLOCKS_PER_SEC;    // In s.
  heur_time *= 1000;   // In ms.

  tot_heuristic_time += heur_time;

  if (heur_time > max_heuristic_time) {
    max_heuristic_time = heur_time;
  }

  if (heur_time < min_heuristic_time) {
    min_heuristic_time = heur_time;
  }
#endif
  // need to apply twice, once true, the other false
  for (int j = 0; j < 2; j++) {
    Formula new_f = f; // copy the formula before recursing
#ifdef NO_MRC
    if (new_f.literal_polarity[i] >
        0) // if the number of literals with positive polarity are greater
#endif
#ifdef MRC
    if (pol)
#endif
#ifdef MRC_DYN
    if (pol)
#endif
#ifdef MRC_GPU
    if (pol)
#endif
    {
      new_f.literals[i] = j; // assign positive first
    } else                   // if not
    {
      new_f.literals[i] = (j + 1) % 2; // assign negative first
    }
#ifdef NO_MRC
    new_f.literal_frequency[i] =
        -1; // reset the frequency to -1 to ignore in the future
#endif
#ifdef MRC
    if (j == 1)
    {
      bl = neg_lit(bl);
    }
#endif
#ifdef MRC_DYN
    if (j == 1)
    {
      bl = neg_lit(bl);
    }
#endif
#ifdef MRC_GPU
    if (j == 1)
    {
      bl = neg_lit(bl);
    }
#endif
    int transform_result =
        apply_transform(new_f, i); // apply the change to all the clauses
    if (transform_result ==
        Cat::satisfied) // if formula satisfied, show result and return
    {
      show_result(new_f, transform_result);
      return Cat::completed;
    } else if (transform_result == Cat::unsatisfied) // if formula not satisfied
                                                     // in this branch, return
                                                     // normally
    {
      continue;
    }
#ifdef NO_MRC
    int dpll_result = DPLL(new_f); // recursively call DPLL on the new formula
#endif
#ifdef MRC
    int dpll_result = DPLL(new_f, mrc, bl);
#endif
#ifdef MRC_DYN
    int dpll_result = DPLL(new_f, mrc_dyn, bl);
#endif
#ifdef MRC_GPU
    int dpll_result = DPLL(new_f, d_mrc, bl);
#endif
    if (dpll_result == Cat::completed) // propagate the result, if completed
    {
      return dpll_result;
    }
  }
#ifdef MRC
  mrc_backjump(dec_lvl, mrc);
#endif
#ifdef MRC_DYN
  mrc_dyn_backjump(dec_lvl, mrc_dyn);
#endif
#ifdef MRC_GPU
  mrc_gpu_backjump(dec_lvl, d_mrc);
#endif
  // if the control reaches here, the function has returned normally
  return Cat::normal;
}

/*
 * function to display the result of the solver
 * arguments: f - the formula when it was satisfied or shown to be unsatisfiable
 *            result - the result flag, a member of the Cat enum
 */
void SATSolverDPLL::show_result(Formula &f, int result) {
  if (result == Cat::satisfied) // if the formula is satisfiable
  {
    cout << "SAT" << endl;
    for (int i = 0; i < f.literals.size(); i++) {
      if (i != 0) {
        cout << " ";
      }
      if (f.literals[i] != -1) {
        cout << pow(-1, f.literals[i]) * (i + 1);
      } else // for literals which can take either value, arbitrarily assign
             // them to be true
      {
        cout << (i + 1);
      }
    }
    cout << " 0" << endl;
  } else // if the formula is unsatisfiable
  {
    cout << "UNSAT" << endl;
  }
}

/*
 * function to call the solver
 */
#ifdef NO_MRC
void SATSolverDPLL::solve() {
  int result = DPLL(formula); // final result of DPLL on the original formula
  // if normal return till the end, then the formula could not be satisfied in
  // any branch, so it is unsatisfiable
#endif
#ifdef MRC
void SATSolverDPLL::solve(Miracle *mrc) {
  int result = DPLL(formula, mrc, 0);
#endif
#ifdef MRC_DYN
void SATSolverDPLL::solve(Miracle_Dyn *mrc_dyn) {
  int result = DPLL(formula, mrc_dyn, 0);
#endif
#ifdef MRC_GPU
void SATSolverDPLL::solve(Miracle *d_mrc) {
  int result = DPLL(formula, d_mrc, 0);
#endif
  if (result == Cat::normal) {
    show_result(formula, Cat::unsatisfied); // the argument formula is a dummy
                                            // here, the result is UNSAT
  }
}

#ifdef MRC_DYN
/*
 * function to print debugging information about the formula and the miracle
 */
void SATSolverDPLL::print_debug_info(Formula &f, Miracle_Dyn *mrc_dyn) {
  printf("******************************************************************");
  printf("\n");
  printf("**********************    DEBUG INFO    **************************");
  printf("\n");
  printf("******************************************************************");
  printf("\n\n");

  mrc_dyn_print_miracle(mrc_dyn);

  printf("******************************************************************");
  printf("\n\n");

  printf("*** SATSolverDPLL ***\n\n");

  printf("Literals: ");
  for (int l = 0; l < f.literals.size(); l++) {
    printf("[%d]%d ", l, f.literals[l]);
  }
  printf("\n");

  printf("Number of unresolved clauses: %d\n", f.clauses.size());

  printf("Clauses: ");
  for (int c = 0; c < f.clauses.size(); c++) {
    printf("(");
    for (int l = 0; l < f.clauses[c].size(); l++) {
      printf("%d ", f.clauses[c][l]);
    }
    printf(") ");
  }
  printf("\n");

  printf("Clause sizes: ");
  for (int c = 0; c < f.clauses.size(); c++) {
    printf("%d ", f.clauses[c].size());
  }
  printf("\n");

  printf("\n*** End SATSolverDPLL ***\n\n");

  printf("******************************************************************");
  printf("\n\n");

  printf("*** Correctness test ***\n\n");

  printf("*** Variable assignments test ***\n\n");

  int v_ass;
  int v_ass_conv;
  for (int v = 0; v < f.literals.size(); v++) {
    v_ass = mrc_dyn->var_ass[v];
    v_ass_conv = v_ass == 0 ? -1 : (v_ass > 0 ? 0 : 1);

    if (v_ass_conv != f.literals[v]) {
      printf("mrc_dyn->var_ass_conv[%d] = %d    !=    f.literals[%d] = %d\n", v, v_ass_conv, v, f.literals[v]);
      exit(EXIT_FAILURE);
    }
  }

  printf("OK!\n");
  
  printf("\n*** End variable assignments test ***\n\n");

  printf("*** Clause sizes test ***\n\n");

  int i = 0;
  for (int c = 0; c < mrc_dyn->clause_sat_len; c++) {
    if (!(mrc_dyn->clause_sat[c])) {
      if (mrc_dyn->unres_clause_size[c] != f.clauses[i].size()) {
        printf("mrc_dyn->unres_clause_size[%d] = %d    !=    f.clauses[%d].size() = %d\n", c, mrc_dyn->unres_clause_size[c], i, f.clauses[i].size());
        exit(EXIT_FAILURE);
      }
      
      i++;
    }
  }

  printf("OK!\n");

  printf("\n*** End clause sizes test ***\n\n");

  printf("\n*** End correctness test ***\n\n");

  printf("******************************************************************");
  printf("\n");
  printf("*********************    END DEBUG INFO    ***********************");
  printf("\n");
  printf("******************************************************************");
  printf("\n\n");
}
#endif

/*
 * function to print solving statistics
 */
#ifdef STATS
void SATSolverDPLL::print_stats(double solving_time,
                                double max_heuristic_time,
                                double min_heuristic_time,
                                double avg_heuristic_time,
                                double tot_heuristic_time,
                                int num_heur) {
  printf("******************************************************************");
  printf("\n");
  printf("*************************    STATS    ****************************");
  printf("\n");
  printf("******************************************************************");
  printf("\n\n");

  printf("Solving time: %f ms\n", solving_time);
  printf("Maximum heuristic computation time: %f ms\n", max_heuristic_time);
  printf("Minimum heuristic computation time: %f ms\n", min_heuristic_time);
  avg_heuristic_time = tot_heuristic_time / num_heur;
  printf("Average heuristic computation time: %f ms\n", avg_heuristic_time);
  printf("Total heuristic computation time: %f ms\n", tot_heuristic_time);
  printf("%% of the solving time used in the heuristic computation: %f %%\n",
         (tot_heuristic_time * 100) / solving_time);
  printf("Number of heuristic calls: %d\n", num_heur);
  
  printf("\n");
  printf("******************************************************************");
  printf("\n");
  printf("***********************    END STATS    **************************");
  printf("\n");
  printf("******************************************************************");
  printf("\n\n");
}
#endif

int main(int argc, char *argv[]) {
  char *prog = argv[0];   // Program name.

  if ((argc - 1) != NUM_ARGS) {
    fprintf(stderr, "usage: %s filename\n", prog);
    exit(EXIT_FAILURE);
  }

  char *filename = argv[1];

  SATSolverDPLL solver; // create the solver
  solver.initialize(filename);  // initialize
#ifdef NO_MRC
#ifdef STATS
  alarm(timeout);
  solve_tic = clock();
#endif
  solver.solve();       // solve
#ifdef STATS
  solve_toc = clock();
#endif
#endif
#ifdef MRC
  Miracle *mrc = mrc_create_miracle(filename);
#ifdef STATS
  alarm(timeout);
  solve_tic = clock();
#endif
  solver.solve(mrc);
#ifdef STATS
  solve_toc = clock();
#endif
  mrc_destroy_miracle(mrc);
#endif
#ifdef MRC_DYN
  Miracle_Dyn *mrc_dyn = mrc_dyn_create_miracle(filename);
#ifdef STATS
  alarm(timeout);
  solve_tic = clock();
#endif
  solver.solve(mrc_dyn);
#ifdef STATS
  solve_toc = clock();
#endif
  mrc_dyn_destroy_miracle(mrc_dyn);
#endif
#ifdef MRC_GPU
  gpu_set_device(0);
  gpu_set_num_threads_per_block(512);

  Miracle *mrc = mrc_create_miracle(filename);
  Miracle *d_mrc = mrc_gpu_transfer_miracle_host_to_dev(mrc);
#ifdef STATS
  alarm(timeout);
  solve_tic = clock();
#endif
  solver.solve(d_mrc);
#ifdef STATS
  solve_toc = clock();
#endif
  mrc_destroy_miracle(mrc);
  mrc_gpu_destroy_miracle(d_mrc);
#endif
#ifdef STATS
  solving_time = ((double)(solve_toc - solve_tic)) / CLOCKS_PER_SEC;    // In s.
  solving_time *= 1000;   // In ms.

  solver.print_stats(solving_time,
                     max_heuristic_time,
                     min_heuristic_time,
                     avg_heuristic_time,
                     tot_heuristic_time,
                     num_heur);
#endif
  return 0;
}
