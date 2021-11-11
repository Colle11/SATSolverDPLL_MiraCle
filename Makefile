miracle: solver.cu cnf_formula.cu miracle.cu miracle_dynamic.cu cnf_formula_gpu.cu miracle_gpu.cu utils.cu
		nvcc -ccbin g++ -m64 -gencode arch=compute_50,code=sm_50 -o SATSolverDPLL_MiraCle solver.cu cnf_formula.cu miracle.cu miracle_dynamic.cu cnf_formula_gpu.cu miracle_gpu.cu utils.cu -DMRC_GPU -DPOSIT

clean:
		rm SATSolverDPLL_MiraCle
