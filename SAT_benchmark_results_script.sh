#!/bin/bash

if test $# -gt 0; then
	echo "Usage: $0"
	exit 1
fi

make clean
make

mkdir -p ./SAT_benchmark_results/heuristic_quality_evaluation/sat
mkdir -p ./SAT_benchmark_results/heuristic_quality_evaluation/unsat
mkdir -p ./SAT_benchmark_results/heuristic_computation_time_evaluation

for f in ./SAT_benchmarks/heuristic_quality_evaluation/sat/*.cnf
do
    s=$(echo "$f" | cut -d'/' -f 5 | cut -d'.' -f 1)
	./build/bin/SATSolverDPLL_NO_MRC_STATS $f > "./SAT_benchmark_results/heuristic_quality_evaluation/sat/${s}_NO_MRC_STATS.txt"
    echo "./build/bin/SATSolverDPLL_NO_MRC_STATS $f"
    ./build/bin/SATSolverDPLL_MRC_GPU_JW_OS_STATS $f > "./SAT_benchmark_results/heuristic_quality_evaluation/sat/${s}_MRC_GPU_JW_OS_STATS.txt"
    echo "./build/bin/SATSolverDPLL_MRC_GPU_JW_OS_STATS $f"
    ./build/bin/SATSolverDPLL_MRC_GPU_JW_TS_STATS $f > "./SAT_benchmark_results/heuristic_quality_evaluation/sat/${s}_MRC_GPU_JW_TS_STATS.txt"
    echo "./build/bin/SATSolverDPLL_MRC_GPU_JW_TS_STATS $f"
    ./build/bin/SATSolverDPLL_MRC_GPU_BOHM_STATS $f > "./SAT_benchmark_results/heuristic_quality_evaluation/sat/${s}_MRC_GPU_BOHM_STATS.txt"
    echo "./build/bin/SATSolverDPLL_MRC_GPU_BOHM_STATS $f"
    ./build/bin/SATSolverDPLL_MRC_GPU_POSIT_STATS $f > "./SAT_benchmark_results/heuristic_quality_evaluation/sat/${s}_MRC_GPU_POSIT_STATS.txt"
    echo "./build/bin/SATSolverDPLL_MRC_GPU_POSIT_STATS $f"
    ./build/bin/SATSolverDPLL_MRC_GPU_DLIS_STATS $f > "./SAT_benchmark_results/heuristic_quality_evaluation/sat/${s}_MRC_GPU_DLIS_STATS.txt"
    echo "./build/bin/SATSolverDPLL_MRC_GPU_DLIS_STATS $f"
    ./build/bin/SATSolverDPLL_MRC_GPU_DLCS_STATS $f > "./SAT_benchmark_results/heuristic_quality_evaluation/sat/${s}_MRC_GPU_DLCS_STATS.txt"
    echo "./build/bin/SATSolverDPLL_MRC_GPU_DLCS_STATS $f"
done

for f in ./SAT_benchmarks/heuristic_quality_evaluation/unsat/*.cnf
do
    s=$(echo "$f" | cut -d'/' -f 5 | cut -d'.' -f 1)
	./build/bin/SATSolverDPLL_NO_MRC_STATS $f > "./SAT_benchmark_results/heuristic_quality_evaluation/unsat/${s}_NO_MRC_STATS.txt"
    echo "./build/bin/SATSolverDPLL_NO_MRC_STATS $f"
    ./build/bin/SATSolverDPLL_MRC_GPU_JW_OS_STATS $f > "./SAT_benchmark_results/heuristic_quality_evaluation/unsat/${s}_MRC_GPU_JW_OS_STATS.txt"
    echo "./build/bin/SATSolverDPLL_MRC_GPU_JW_OS_STATS $f"
    ./build/bin/SATSolverDPLL_MRC_GPU_JW_TS_STATS $f > "./SAT_benchmark_results/heuristic_quality_evaluation/unsat/${s}_MRC_GPU_JW_TS_STATS.txt"
    echo "./build/bin/SATSolverDPLL_MRC_GPU_JW_TS_STATS $f"
    ./build/bin/SATSolverDPLL_MRC_GPU_BOHM_STATS $f > "./SAT_benchmark_results/heuristic_quality_evaluation/unsat/${s}_MRC_GPU_BOHM_STATS.txt"
    echo "./build/bin/SATSolverDPLL_MRC_GPU_BOHM_STATS $f"
    ./build/bin/SATSolverDPLL_MRC_GPU_POSIT_STATS $f > "./SAT_benchmark_results/heuristic_quality_evaluation/unsat/${s}_MRC_GPU_POSIT_STATS.txt"
    echo "./build/bin/SATSolverDPLL_MRC_GPU_POSIT_STATS $f"
    ./build/bin/SATSolverDPLL_MRC_GPU_DLIS_STATS $f > "./SAT_benchmark_results/heuristic_quality_evaluation/unsat/${s}_MRC_GPU_DLIS_STATS.txt"
    echo "./build/bin/SATSolverDPLL_MRC_GPU_DLIS_STATS $f"
    ./build/bin/SATSolverDPLL_MRC_GPU_DLCS_STATS $f > "./SAT_benchmark_results/heuristic_quality_evaluation/unsat/${s}_MRC_GPU_DLCS_STATS.txt"
    echo "./build/bin/SATSolverDPLL_MRC_GPU_DLCS_STATS $f"
done

for f in ./SAT_benchmarks/heuristic_computation_time_evaluation/*.cnf
do
    s=$(echo "$f" | cut -d'/' -f 4 | cut -d'.' -f 1)
	./build/bin/SATSolverDPLL_NO_MRC_STATS $f > "./SAT_benchmark_results/heuristic_computation_time_evaluation/${s}_NO_MRC_STATS.txt"
    echo "./build/bin/SATSolverDPLL_NO_MRC_STATS $f"
    ./build/bin/SATSolverDPLL_MRC_POSIT_STATS $f > "./SAT_benchmark_results/heuristic_computation_time_evaluation/${s}_MRC_POSIT_STATS.txt"
    echo "./build/bin/SATSolverDPLL_MRC_POSIT_STATS $f"
    ./build/bin/SATSolverDPLL_MRC_DYN_POSIT_STATS $f > "./SAT_benchmark_results/heuristic_computation_time_evaluation/${s}_MRC_DYN_POSIT_STATS.txt"
    echo "./build/bin/SATSolverDPLL_MRC_DYN_POSIT_STATS $f"
    ./build/bin/SATSolverDPLL_MRC_GPU_POSIT_STATS $f > "./SAT_benchmark_results/heuristic_computation_time_evaluation/${s}_MRC_GPU_POSIT_STATS.txt"
    echo "./build/bin/SATSolverDPLL_MRC_GPU_POSIT_STATS $f"
done
