
#ifdef STATS
#ifndef FLAGLOADSIGHANDLER_H
#include <signal.h>
#include <sys/wait.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <time.h>

#ifdef MRC_GPU
#include "launch_parameters_gpu.cuh"
#endif

#ifdef POSIT
extern int POSIT_n;
#endif

#ifdef BOHM
extern int BOHM_alpha;
extern int BOHM_beta;
#endif

extern clock_t solve_tic;
extern clock_t solve_toc;
extern double solving_time;

#if defined MRC || defined MRC_DYN || defined MRC_GPU
extern double miracle_time;

extern double max_inc_dec_lvl_time;
extern double min_inc_dec_lvl_time;
extern double avg_inc_dec_lvl_time;
extern double tot_inc_dec_lvl_time;
extern int num_inc_dec_lvl;

extern double max_assign_time;
extern double min_assign_time;
extern double avg_assign_time;
extern double tot_assign_time;
extern int num_assign;

extern double max_bj_time;
extern double min_bj_time;
extern double avg_bj_time;
extern double tot_bj_time;
extern int num_bj;
#endif

extern double max_heur_time;
extern double min_heur_time;
extern double avg_heur_time;
extern double tot_heur_time;
extern int num_heur;

extern int timeout_expired;
extern int escape;
extern int timeout;

void install_alarmhandler();
void install_handler();

void my_catchint(int signo);
void my_catchalarm(int signo);

void print_stats();




void install_handler() {

  static struct sigaction act;
  act.sa_handler = my_catchint; /* registrazione dell'handler */

  sigfillset(&(act.sa_mask)); /* tutti i segnali saranno ignorati
                                 DURANTE l'esecuzione dell'handler */

  /* imposto l'handler per il segnale SIGINT */
  sigaction(SIGINT, &act, NULL); 

}

void install_alarmhandler() {

  static struct sigaction act;
  act.sa_handler = my_catchalarm;
  sigfillset(&(act.sa_mask));
  sigaction(SIGALRM, &act, NULL); 
}

 /* Questo e' l'handler. Semplice. */
void my_catchint(int signo) {
	if ((signo==SIGINT)) {
    solve_toc = clock();
    solving_time = ((double)(solve_toc - solve_tic)) / CLOCKS_PER_SEC;  // In s.
    solving_time *= 1000;   // In ms.

		escape = 1;
		fprintf(stderr,"\nCATCHING SIG_INT: forced exit.\n");fflush(stderr);

    print_stats();

    exit(2);
	}
}


void my_catchalarm(int signo) {
	if ((signo==SIGALRM)) {
    solve_toc = clock();
    solving_time = ((double)(solve_toc - solve_tic)) / CLOCKS_PER_SEC;  // In s.
    solving_time *= 1000;   // In ms.

		timeout_expired = 1;
		fprintf(stderr,"\nTIMEOUT EXPIRED: forced exit.\n");fflush(stderr);

    print_stats();

		exit(2);
	}
}


void print_stats() {
  printf("******************************************************************");
  printf("\n");
  printf("*************************    STATS    ****************************");
  printf("\n");
  printf("******************************************************************");
  printf("\n\n");

  if (timeout_expired) {
    printf("Timeout expired: YES\n");
  } else {
    printf("Timeout expired: NO\n");
  }

  if (escape) {
    printf("SIGINT captured: YES\n");
  } else {
    printf("SIGINT captured: NO\n");
  }

  printf("Timeout: %d s\n", timeout);
#ifdef MRC_GPU
  printf("Number of threads per block: %d\n", gpu_num_threads_per_block());
#endif
#ifdef POSIT
  printf("POSIT n: %d\n", POSIT_n);
#endif
#ifdef BOHM
  printf("BOHM alpha: %d\n", BOHM_alpha);
  printf("BOHM beta: %d\n", BOHM_beta);
#endif
  printf("\n");

  printf("Solving time: %f ms\n", solving_time);
  printf("\n");

#if defined MRC || defined MRC_DYN || defined MRC_GPU
  printf("MiraCle time: %f ms\n", miracle_time);
  printf("%% of solving time used in MiraCle calls: %f %%\n",
         (miracle_time * 100) / solving_time);
  printf("\n");

  printf("Maximum decision level increase time: %f ms\n", max_inc_dec_lvl_time);
  printf("Minimum decision level increase time: %f ms\n", min_inc_dec_lvl_time);
  avg_inc_dec_lvl_time = tot_inc_dec_lvl_time / num_inc_dec_lvl;
  printf("Average decision level increase time: %f ms\n", avg_inc_dec_lvl_time);
  printf("Total decision level increase time: %f ms\n", tot_inc_dec_lvl_time);
  printf("%% of MiraCle time used in increase decision level calls: %f %%\n",
         (tot_inc_dec_lvl_time * 100) / miracle_time);
  printf("Number of increase decision level calls: %d\n", num_inc_dec_lvl);
  printf("\n");

  printf("Maximum assignment time: %f ms\n", max_assign_time);
  printf("Minimum assignment time: %f ms\n", min_assign_time);
  avg_assign_time = tot_assign_time / num_assign;
  printf("Average assignment time: %f ms\n", avg_assign_time);
  printf("Total assignment time: %f ms\n", tot_assign_time);
  printf("%% of MiraCle time used in assignment calls: %f %%\n",
         (tot_assign_time * 100) / miracle_time);
  printf("Number of assignment calls: %d\n", num_assign);
  printf("\n");

  printf("Maximum backjumping time: %f ms\n", max_bj_time);
  printf("Minimum backjumping time: %f ms\n", min_bj_time);
  avg_bj_time = tot_bj_time / num_bj;
  printf("Average backjumping time: %f ms\n", avg_bj_time);
  printf("Total backjumping time: %f ms\n", tot_bj_time);
  printf("%% of MiraCle time used in backjumping calls: %f %%\n",
         (tot_bj_time * 100) / miracle_time);
  printf("Number of backjumping calls: %d\n", num_bj);
  printf("\n");
#endif

  printf("Maximum heuristic time: %f ms\n", max_heur_time);
  printf("Minimum heuristic time: %f ms\n", min_heur_time);
  avg_heur_time = tot_heur_time / num_heur;
  printf("Average heuristic time: %f ms\n", avg_heur_time);
  printf("Total heuristic time: %f ms\n", tot_heur_time);
#if defined MRC || defined MRC_DYN || defined MRC_GPU
  printf("%% of MiraCle time used in heuristic calls: %f %%\n",
         (tot_heur_time * 100) / miracle_time);
#endif
#ifdef NO_MRC
  printf("%% of solving time used in heuristic calls: %f %%\n",
         (tot_heur_time * 100) / solving_time);
#endif
  printf("Number of heuristic calls: %d\n", num_heur);
  printf("\n");

  printf("******************************************************************");
  printf("\n");
  printf("***********************    END STATS    **************************");
  printf("\n");
  printf("******************************************************************");
  printf("\n\n");
}



#define FLAGLOADSIGHANDLER_H 1
#endif
#endif
