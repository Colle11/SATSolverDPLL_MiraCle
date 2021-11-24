
#ifndef FLAGLOADSIGHANDLER_H
#include <signal.h>
#include <sys/wait.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <time.h>

extern int timeout_expired;
extern int escape;
extern clock_t solve_tic;
extern clock_t solve_toc;
extern double solving_time;
extern double max_heuristic_time;
extern double min_heuristic_time;
extern double avg_heuristic_time;
extern double tot_heuristic_time;
extern int num_heur;

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



#define FLAGLOADSIGHANDLER_H 1
#endif
