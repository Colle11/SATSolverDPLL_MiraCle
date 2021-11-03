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

#include "launch_parameters_gpu.cuh"
#include "utils.cuh"
#include "miracle.cuh"
#include "miracle_dynamic.cuh"
#include "miracle_gpu.cuh"

#define NUM_ARGS (1)    // Number of solver arguments.
#define POSIT_n (3)     // Constant of the POSIT weight function.

#define NO_MRC
// #define MRC
// #define MRC_DYN
// #define MRC_GPU

#define JW_OS
// #define JW_TS
// #define POSIT
// #define DLIS
// #define DLCS
// #define RDLIS
// #define RDLCS

using namespace std;

/**
 * Global variables
 */
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
  int unit_propagate(Formula &, Miracle *mrc);
#endif
#ifdef MRC_DYN
  int unit_propagate(Formula &, Miracle_Dyn *mrc_dyn);
#endif
#ifdef MRC_GPU
  int unit_propagate(Formula &, Miracle *d_mrc);
#endif
#ifdef NO_MRC
  int DPLL(Formula);                          // performs DPLL recursively
#endif
#ifdef MRC
  int DPLL(Formula, Miracle *mrc);
#endif
#ifdef MRC_DYN
  int DPLL(Formula, Miracle_Dyn *mrc_dyn);
#endif
#ifdef MRC_GPU
  int DPLL(Formula, Miracle *d_mrc);
#endif
  int apply_transform(Formula &,
                      int); // applies the value of the literal in every clause
  void show_result(Formula &, int); // displays the result
public:
  SATSolverDPLL() {}
  void initialize(char *filename); // intiializes the values
#ifdef NO_MRC
  void solve();                       // calls the solver
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
int SATSolverDPLL::unit_propagate(Formula &f, Miracle *mrc) {
  Lidx lidx;
  lits_len = 0;
#endif
#ifdef MRC_DYN
int SATSolverDPLL::unit_propagate(Formula &f, Miracle_Dyn *mrc_dyn) {
  Lidx lidx;
  lits_len = 0;
#endif
#ifdef MRC_GPU
int SATSolverDPLL::unit_propagate(Formula &f, Miracle *d_mrc) {
  Lidx lidx;
  lits_len = 0;
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
int SATSolverDPLL::DPLL(Formula f, Miracle *mrc) {
  int dec_lvl = mrc->dec_lvl;

  int result = unit_propagate(f, mrc); // perform unit propagation on the formula
#endif
#ifdef MRC_DYN
int SATSolverDPLL::DPLL(Formula f, Miracle_Dyn *mrc_dyn) {
  int dec_lvl = mrc_dyn->dec_lvl;

  int result = unit_propagate(f, mrc_dyn); // perform unit propagation on the formula
#endif
#ifdef MRC_GPU
int SATSolverDPLL::DPLL(Formula f, Miracle *d_mrc) {
  int dec_lvl;
  gpuErrchk( cudaMemcpy(&dec_lvl, &(d_mrc->dec_lvl),
                        sizeof dec_lvl,
                        cudaMemcpyDeviceToHost) );

  int result = unit_propagate(f, d_mrc); // perform unit propagation on the formula
#endif
  if (result == Cat::satisfied) // if formula satisfied, show result and return
  {
    show_result(f, result);
#ifdef MRC
  mrc_backjump(dec_lvl - 1, mrc);
#endif
#ifdef MRC_DYN
  mrc_dyn_backjump(dec_lvl - 1, mrc_dyn);
#endif
#ifdef MRC_GPU
  mrc_gpu_backjump(dec_lvl - 1, d_mrc);
#endif
    return Cat::completed;
  } else if (result == Cat::unsatisfied) // if formula not satisfied in this
                                         // branch, return normally
  {
#ifdef MRC
    mrc_backjump(dec_lvl - 1, mrc);
#endif
#ifdef MRC_DYN
    mrc_dyn_backjump(dec_lvl - 1, mrc_dyn);
#endif
#ifdef MRC_GPU
    mrc_gpu_backjump(dec_lvl - 1, d_mrc);
#endif
    return Cat::normal;
  }
  // find the variable with maximum frequency in f, which will be the next to be
  // assigned a value already assigned variables have this field reset to -1 in
  // order to ignore them
#ifdef NO_MRC
  int i = distance(
      f.literal_frequency.begin(),
      max_element(f.literal_frequency.begin(), f.literal_frequency.end()));
#endif
#ifdef MRC
  #ifdef JW_OS
  Lit blit = mrc_JW_OS_heuristic(mrc);
  #endif
  #ifdef JW_TS
  Lit blit = mrc_JW_TS_heuristic(mrc);
  #endif
  #ifdef POSIT
  Lit blit = mrc_POSIT_heuristic(mrc, POSIT_n);
  #endif
  #ifdef DLIS
  Lit blit = mrc_DLIS_heuristic(mrc);
  #endif
  #ifdef DLCS
  Lit blit = mrc_DLCS_heuristic(mrc);
  #endif
  #ifdef RDLIS
  Lit blit = mrc_RDLIS_heuristic(mrc);
  #endif
  #ifdef RDLCS
  Lit blit = mrc_RDLCS_heuristic(mrc);
  #endif

  Var bvar = lit_to_var(blit);
  bool pol = lit_to_pol(blit);
  int i = (int)bvar;
#endif
#ifdef MRC_DYN
  #ifdef JW_OS
  Lit blit = mrc_dyn_JW_OS_heuristic(mrc_dyn);
  #endif
  #ifdef JW_TS
  Lit blit = mrc_dyn_JW_TS_heuristic(mrc_dyn);
  #endif
  #ifdef POSIT
  Lit blit = mrc_dyn_POSIT_heuristic(mrc_dyn, POSIT_n);
  #endif
  #ifdef DLIS
  Lit blit = mrc_dyn_DLIS_heuristic(mrc_dyn);
  #endif
  #ifdef DLCS
  Lit blit = mrc_dyn_DLCS_heuristic(mrc_dyn);
  #endif
  #ifdef RDLIS
  Lit blit = mrc_dyn_RDLIS_heuristic(mrc_dyn);
  #endif
  #ifdef RDLCS
  Lit blit = mrc_dyn_RDLCS_heuristic(mrc_dyn);
  #endif

  Var bvar = lit_to_var(blit);
  bool pol = lit_to_pol(blit);
  int i = (int)bvar;
#endif
#ifdef MRC_GPU
  #ifdef JW_OS
  Lit blit = mrc_gpu_JW_OS_heuristic(d_mrc);
  #endif
  #ifdef JW_TS
  Lit blit = mrc_gpu_JW_TS_heuristic(d_mrc);
  #endif
  #ifdef POSIT
  Lit blit = mrc_gpu_POSIT_heuristic(d_mrc, POSIT_n);
  #endif
  #ifdef DLIS
  Lit blit = mrc_gpu_DLIS_heuristic(d_mrc);
  #endif
  #ifdef DLCS
  Lit blit = mrc_gpu_DLCS_heuristic(d_mrc);
  #endif
  #ifdef RDLIS
  Lit blit = mrc_gpu_RDLIS_heuristic(d_mrc);
  #endif
  #ifdef RDLCS
  Lit blit = mrc_gpu_RDLCS_heuristic(d_mrc);
  #endif

  Var bvar = lit_to_var(blit);
  bool pol = lit_to_pol(blit);
  int i = (int)bvar;
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
    mrc_increase_decision_level(mrc);

    if (j == 1)
    {
      blit = neg_lit(blit);
    }

    mrc_assign_lits(&blit, 1, mrc);
#endif
#ifdef MRC_DYN
    mrc_dyn_increase_decision_level(mrc_dyn);

    if (j == 1)
    {
      blit = neg_lit(blit);
    }

    mrc_dyn_assign_lits(&blit, 1, mrc_dyn);
#endif
#ifdef MRC_GPU
    mrc_gpu_increase_decision_level(d_mrc);

    if (j == 1)
    {
      blit = neg_lit(blit);
    }

    mrc_gpu_assign_lits(&blit, 1, d_mrc);
#endif
    int transform_result =
        apply_transform(new_f, i); // apply the change to all the clauses
    if (transform_result ==
        Cat::satisfied) // if formula satisfied, show result and return
    {
      show_result(new_f, transform_result);
#ifdef MRC
      mrc_backjump(dec_lvl - 1, mrc);
#endif
#ifdef MRC_DYN
      mrc_dyn_backjump(dec_lvl - 1, mrc_dyn);
#endif
#ifdef MRC_GPU
      mrc_gpu_backjump(dec_lvl - 1, d_mrc);
#endif
      return Cat::completed;
    } else if (transform_result == Cat::unsatisfied) // if formula not satisfied
                                                     // in this branch, return
                                                     // normally
    {
#ifdef MRC
      mrc_backjump(dec_lvl - 1, mrc);
#endif
#ifdef MRC_DYN
      mrc_dyn_backjump(dec_lvl - 1, mrc_dyn);
#endif
#ifdef MRC_GPU
      mrc_gpu_backjump(dec_lvl - 1, d_mrc);
#endif
      continue;
    }
#ifdef NO_MRC
    int dpll_result = DPLL(new_f); // recursively call DPLL on the new formula
#endif
#ifdef MRC
    int dpll_result = DPLL(new_f, mrc);
#endif
#ifdef MRC_DYN
    int dpll_result = DPLL(new_f, mrc_dyn);
#endif
#ifdef MRC_GPU
    int dpll_result = DPLL(new_f, d_mrc);
#endif
    if (dpll_result == Cat::completed) // propagate the result, if completed
    {
#ifdef MRC
      mrc_backjump(dec_lvl - 1, mrc);
#endif
#ifdef MRC_DYN
      mrc_dyn_backjump(dec_lvl - 1, mrc_dyn);
#endif
#ifdef MRC_GPU
      mrc_gpu_backjump(dec_lvl - 1, d_mrc);
#endif
      return dpll_result;
    }
  }
#ifdef MRC
  mrc_backjump(dec_lvl - 1, mrc);
#endif
#ifdef MRC_DYN
  mrc_dyn_backjump(dec_lvl - 1, mrc_dyn);
#endif
#ifdef MRC_GPU
  mrc_gpu_backjump(dec_lvl - 1, d_mrc);
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
  int result = DPLL(formula, mrc);
#endif
#ifdef MRC_DYN
void SATSolverDPLL::solve(Miracle_Dyn *mrc_dyn) {
  int result = DPLL(formula, mrc_dyn);
#endif
#ifdef MRC_GPU
void SATSolverDPLL::solve(Miracle *d_mrc) {
  int result = DPLL(formula, d_mrc);
#endif
  if (result == Cat::normal) {
    show_result(formula, Cat::unsatisfied); // the argument formula is a dummy
                                            // here, the result is UNSAT
  }
}

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
  clock_t begin = clock();

  solver.solve();       // solve

  clock_t end = clock();
#endif
#ifdef MRC
  Miracle *mrc = mrc_create_miracle(filename);

  clock_t begin = clock();
  
  solver.solve(mrc);       // solve

  clock_t end = clock();

  mrc_destroy_miracle(mrc);
#endif
#ifdef MRC_DYN
  Miracle_Dyn *mrc_dyn = mrc_dyn_create_miracle(filename);

  clock_t begin = clock();
  
  solver.solve(mrc_dyn);       // solve

  clock_t end = clock();

  mrc_dyn_destroy_miracle(mrc_dyn);
#endif
#ifdef MRC_GPU
  gpu_set_device(0);
  gpu_set_num_threads_per_block(512);

  Miracle *mrc = mrc_create_miracle(filename);
  Miracle *d_mrc = mrc_gpu_transfer_miracle_host_to_dev(mrc);

  clock_t begin = clock();

  solver.solve(d_mrc);       // solve

  clock_t end = clock();

  mrc_destroy_miracle(mrc);
  mrc_gpu_destroy_miracle(d_mrc);
#endif
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  time_spent *= 1000;
  printf("Time to solve: %3.1f ms\n", time_spent);

  return 0;
}
