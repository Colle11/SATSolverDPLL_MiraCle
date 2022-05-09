BUILD_DIR := ./build
BIN_DIR := $(BUILD_DIR)/bin
SRC_DIR := ./src

NVCC := nvcc
NVCCFLAGS := -gencode arch=compute_50,code=sm_50

.PHONY: all clean

all: $(BIN_DIR) \
	 $(BIN_DIR)/SATSolverDPLL_NO_MRC_STATS \
	 $(BIN_DIR)/SATSolverDPLL_MRC_JW_OS_STATS $(BIN_DIR)/SATSolverDPLL_MRC_JW_TS_STATS $(BIN_DIR)/SATSolverDPLL_MRC_BOHM_STATS $(BIN_DIR)/SATSolverDPLL_MRC_POSIT_STATS $(BIN_DIR)/SATSolverDPLL_MRC_DLIS_STATS $(BIN_DIR)/SATSolverDPLL_MRC_DLCS_STATS \
	 $(BIN_DIR)/SATSolverDPLL_MRC_DYN_JW_OS_STATS $(BIN_DIR)/SATSolverDPLL_MRC_DYN_JW_TS_STATS $(BIN_DIR)/SATSolverDPLL_MRC_DYN_BOHM_STATS $(BIN_DIR)/SATSolverDPLL_MRC_DYN_POSIT_STATS $(BIN_DIR)/SATSolverDPLL_MRC_DYN_DLIS_STATS $(BIN_DIR)/SATSolverDPLL_MRC_DYN_DLCS_STATS \
	 $(BIN_DIR)/SATSolverDPLL_MRC_GPU_JW_OS_STATS $(BIN_DIR)/SATSolverDPLL_MRC_GPU_JW_TS_STATS $(BIN_DIR)/SATSolverDPLL_MRC_GPU_BOHM_STATS $(BIN_DIR)/SATSolverDPLL_MRC_GPU_POSIT_STATS $(BIN_DIR)/SATSolverDPLL_MRC_GPU_DLIS_STATS $(BIN_DIR)/SATSolverDPLL_MRC_GPU_DLCS_STATS

$(BIN_DIR):
	mkdir -p $(BIN_DIR)


### NO_MRC ###

$(BIN_DIR)/SATSolverDPLL_NO_MRC_STATS: $(BIN_DIR) $(SRC_DIR)/solver.cu $(SRC_DIR)/utils.cuh $(BUILD_DIR)/utils.o $(SRC_DIR)/sig_handling.h
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/solver.cu $(BUILD_DIR)/utils.o -DNO_MRC -DSTATS -o $@

### END NO_MRC ###


### MRC ###

$(BIN_DIR)/SATSolverDPLL_MRC_JW_OS_STATS: $(BIN_DIR) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/cnf_formula_gpu.o $(SRC_DIR)/sat_miracle.cuh $(BUILD_DIR)/sat_miracle.o $(BUILD_DIR)/miracle.o $(BUILD_DIR)/miracle_gpu.o $(SRC_DIR)/utils.cuh $(BUILD_DIR)/utils.o $(SRC_DIR)/sig_handling.h
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/cnf_formula_gpu.o $(BUILD_DIR)/sat_miracle.o $(BUILD_DIR)/miracle.o $(BUILD_DIR)/miracle_gpu.o $(BUILD_DIR)/utils.o -DMRC -DJW_OS -DSTATS -o $@

$(BIN_DIR)/SATSolverDPLL_MRC_JW_TS_STATS: $(BIN_DIR) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/cnf_formula_gpu.o $(SRC_DIR)/sat_miracle.cuh $(BUILD_DIR)/sat_miracle.o $(BUILD_DIR)/miracle.o $(BUILD_DIR)/miracle_gpu.o $(SRC_DIR)/utils.cuh $(BUILD_DIR)/utils.o $(SRC_DIR)/sig_handling.h
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/cnf_formula_gpu.o $(BUILD_DIR)/sat_miracle.o $(BUILD_DIR)/miracle.o $(BUILD_DIR)/miracle_gpu.o $(BUILD_DIR)/utils.o -DMRC -DJW_TS -DSTATS -o $@

$(BIN_DIR)/SATSolverDPLL_MRC_BOHM_STATS: $(BIN_DIR) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/cnf_formula_gpu.o $(SRC_DIR)/sat_miracle.cuh $(BUILD_DIR)/sat_miracle.o $(BUILD_DIR)/miracle.o $(BUILD_DIR)/miracle_gpu.o $(SRC_DIR)/utils.cuh $(BUILD_DIR)/utils.o $(SRC_DIR)/sig_handling.h
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/cnf_formula_gpu.o $(BUILD_DIR)/sat_miracle.o $(BUILD_DIR)/miracle.o $(BUILD_DIR)/miracle_gpu.o $(BUILD_DIR)/utils.o -DMRC -DBOHM -DSTATS -o $@

$(BIN_DIR)/SATSolverDPLL_MRC_POSIT_STATS: $(BIN_DIR) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/cnf_formula_gpu.o $(SRC_DIR)/sat_miracle.cuh $(BUILD_DIR)/sat_miracle.o $(BUILD_DIR)/miracle.o $(BUILD_DIR)/miracle_gpu.o $(SRC_DIR)/utils.cuh $(BUILD_DIR)/utils.o $(SRC_DIR)/sig_handling.h
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/cnf_formula_gpu.o $(BUILD_DIR)/sat_miracle.o $(BUILD_DIR)/miracle.o $(BUILD_DIR)/miracle_gpu.o $(BUILD_DIR)/utils.o -DMRC -DPOSIT -DSTATS -o $@

$(BIN_DIR)/SATSolverDPLL_MRC_DLIS_STATS: $(BIN_DIR) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/cnf_formula_gpu.o $(SRC_DIR)/sat_miracle.cuh $(BUILD_DIR)/sat_miracle.o $(BUILD_DIR)/miracle.o $(BUILD_DIR)/miracle_gpu.o $(SRC_DIR)/utils.cuh $(BUILD_DIR)/utils.o $(SRC_DIR)/sig_handling.h
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/cnf_formula_gpu.o $(BUILD_DIR)/sat_miracle.o $(BUILD_DIR)/miracle.o $(BUILD_DIR)/miracle_gpu.o $(BUILD_DIR)/utils.o -DMRC -DDLIS -DSTATS -o $@

$(BIN_DIR)/SATSolverDPLL_MRC_DLCS_STATS: $(BIN_DIR) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/cnf_formula_gpu.o $(SRC_DIR)/sat_miracle.cuh $(BUILD_DIR)/sat_miracle.o $(BUILD_DIR)/miracle.o $(BUILD_DIR)/miracle_gpu.o $(SRC_DIR)/utils.cuh $(BUILD_DIR)/utils.o $(SRC_DIR)/sig_handling.h
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/cnf_formula_gpu.o $(BUILD_DIR)/sat_miracle.o $(BUILD_DIR)/miracle.o $(BUILD_DIR)/miracle_gpu.o $(BUILD_DIR)/utils.o -DMRC -DDLCS -DSTATS -o $@

### END MRC ###


### MRC_DYN ###

$(BIN_DIR)/SATSolverDPLL_MRC_DYN_JW_OS_STATS: $(BIN_DIR) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(SRC_DIR)/miracle_dynamic.cuh $(BUILD_DIR)/miracle_dynamic.o $(SRC_DIR)/utils.cuh $(BUILD_DIR)/utils.o $(SRC_DIR)/sig_handling.h
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/miracle_dynamic.o $(BUILD_DIR)/utils.o -DMRC_DYN -DJW_OS -DSTATS -o $@

$(BIN_DIR)/SATSolverDPLL_MRC_DYN_JW_TS_STATS: $(BIN_DIR) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(SRC_DIR)/miracle_dynamic.cuh $(BUILD_DIR)/miracle_dynamic.o $(SRC_DIR)/utils.cuh $(BUILD_DIR)/utils.o $(SRC_DIR)/sig_handling.h
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/miracle_dynamic.o $(BUILD_DIR)/utils.o -DMRC_DYN -DJW_TS -DSTATS -o $@

$(BIN_DIR)/SATSolverDPLL_MRC_DYN_BOHM_STATS: $(BIN_DIR) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(SRC_DIR)/miracle_dynamic.cuh $(BUILD_DIR)/miracle_dynamic.o $(SRC_DIR)/utils.cuh $(BUILD_DIR)/utils.o $(SRC_DIR)/sig_handling.h
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/miracle_dynamic.o $(BUILD_DIR)/utils.o -DMRC_DYN -DBOHM -DSTATS -o $@

$(BIN_DIR)/SATSolverDPLL_MRC_DYN_POSIT_STATS: $(BIN_DIR) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(SRC_DIR)/miracle_dynamic.cuh $(BUILD_DIR)/miracle_dynamic.o $(SRC_DIR)/utils.cuh $(BUILD_DIR)/utils.o $(SRC_DIR)/sig_handling.h
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/miracle_dynamic.o $(BUILD_DIR)/utils.o -DMRC_DYN -DPOSIT -DSTATS -o $@

$(BIN_DIR)/SATSolverDPLL_MRC_DYN_DLIS_STATS: $(BIN_DIR) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(SRC_DIR)/miracle_dynamic.cuh $(BUILD_DIR)/miracle_dynamic.o $(SRC_DIR)/utils.cuh $(BUILD_DIR)/utils.o $(SRC_DIR)/sig_handling.h
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/miracle_dynamic.o $(BUILD_DIR)/utils.o -DMRC_DYN -DDLIS -DSTATS -o $@

$(BIN_DIR)/SATSolverDPLL_MRC_DYN_DLCS_STATS: $(BIN_DIR) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(SRC_DIR)/miracle_dynamic.cuh $(BUILD_DIR)/miracle_dynamic.o $(SRC_DIR)/utils.cuh $(BUILD_DIR)/utils.o $(SRC_DIR)/sig_handling.h
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/miracle_dynamic.o $(BUILD_DIR)/utils.o -DMRC_DYN -DDLCS -DSTATS -o $@

### END MRC_DYN ###


### MRC_GPU ###

$(BIN_DIR)/SATSolverDPLL_MRC_GPU_JW_OS_STATS: $(BIN_DIR) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/cnf_formula_gpu.o $(SRC_DIR)/sat_miracle.cuh $(BUILD_DIR)/sat_miracle.o $(BUILD_DIR)/miracle.o $(BUILD_DIR)/miracle_gpu.o $(SRC_DIR)/launch_parameters_gpu.cuh $(SRC_DIR)/utils.cuh $(BUILD_DIR)/utils.o $(SRC_DIR)/sig_handling.h
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/cnf_formula_gpu.o $(BUILD_DIR)/sat_miracle.o $(BUILD_DIR)/miracle.o $(BUILD_DIR)/miracle_gpu.o $(BUILD_DIR)/utils.o -DMRC_GPU -DJW_OS -DSTATS -o $@

$(BIN_DIR)/SATSolverDPLL_MRC_GPU_JW_TS_STATS: $(BIN_DIR) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/cnf_formula_gpu.o $(SRC_DIR)/sat_miracle.cuh $(BUILD_DIR)/sat_miracle.o $(BUILD_DIR)/miracle.o $(BUILD_DIR)/miracle_gpu.o $(SRC_DIR)/launch_parameters_gpu.cuh $(SRC_DIR)/utils.cuh $(BUILD_DIR)/utils.o $(SRC_DIR)/sig_handling.h
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/cnf_formula_gpu.o $(BUILD_DIR)/sat_miracle.o $(BUILD_DIR)/miracle.o $(BUILD_DIR)/miracle_gpu.o $(BUILD_DIR)/utils.o -DMRC_GPU -DJW_TS -DSTATS -o $@

$(BIN_DIR)/SATSolverDPLL_MRC_GPU_BOHM_STATS: $(BIN_DIR) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/cnf_formula_gpu.o $(SRC_DIR)/sat_miracle.cuh $(BUILD_DIR)/sat_miracle.o $(BUILD_DIR)/miracle.o $(BUILD_DIR)/miracle_gpu.o $(SRC_DIR)/launch_parameters_gpu.cuh $(SRC_DIR)/utils.cuh $(BUILD_DIR)/utils.o $(SRC_DIR)/sig_handling.h
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/cnf_formula_gpu.o $(BUILD_DIR)/sat_miracle.o $(BUILD_DIR)/miracle.o $(BUILD_DIR)/miracle_gpu.o $(BUILD_DIR)/utils.o -DMRC_GPU -DBOHM -DSTATS -o $@

$(BIN_DIR)/SATSolverDPLL_MRC_GPU_POSIT_STATS: $(BIN_DIR) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/cnf_formula_gpu.o $(SRC_DIR)/sat_miracle.cuh $(BUILD_DIR)/sat_miracle.o $(BUILD_DIR)/miracle.o $(BUILD_DIR)/miracle_gpu.o $(SRC_DIR)/launch_parameters_gpu.cuh $(SRC_DIR)/utils.cuh $(BUILD_DIR)/utils.o $(SRC_DIR)/sig_handling.h
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/cnf_formula_gpu.o $(BUILD_DIR)/sat_miracle.o $(BUILD_DIR)/miracle.o $(BUILD_DIR)/miracle_gpu.o $(BUILD_DIR)/utils.o -DMRC_GPU -DPOSIT -DSTATS -o $@

$(BIN_DIR)/SATSolverDPLL_MRC_GPU_DLIS_STATS: $(BIN_DIR) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/cnf_formula_gpu.o $(SRC_DIR)/sat_miracle.cuh $(BUILD_DIR)/sat_miracle.o $(BUILD_DIR)/miracle.o $(BUILD_DIR)/miracle_gpu.o $(SRC_DIR)/launch_parameters_gpu.cuh $(SRC_DIR)/utils.cuh $(BUILD_DIR)/utils.o $(SRC_DIR)/sig_handling.h
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/cnf_formula_gpu.o $(BUILD_DIR)/sat_miracle.o $(BUILD_DIR)/miracle.o $(BUILD_DIR)/miracle_gpu.o $(BUILD_DIR)/utils.o -DMRC_GPU -DDLIS -DSTATS -o $@

$(BIN_DIR)/SATSolverDPLL_MRC_GPU_DLCS_STATS: $(BIN_DIR) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/cnf_formula_gpu.o $(SRC_DIR)/sat_miracle.cuh $(BUILD_DIR)/sat_miracle.o $(BUILD_DIR)/miracle.o $(BUILD_DIR)/miracle_gpu.o $(SRC_DIR)/launch_parameters_gpu.cuh $(SRC_DIR)/utils.cuh $(BUILD_DIR)/utils.o $(SRC_DIR)/sig_handling.h
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/solver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/cnf_formula_gpu.o $(BUILD_DIR)/sat_miracle.o $(BUILD_DIR)/miracle.o $(BUILD_DIR)/miracle_gpu.o $(BUILD_DIR)/utils.o -DMRC_GPU -DDLCS -DSTATS -o $@

### END MRC_GPU ###


$(BUILD_DIR)/cnf_formula.o: $(BIN_DIR) $(SRC_DIR)/cnf_formula.cuh $(SRC_DIR)/cnf_formula.cu $(SRC_DIR)/cnf_formula_types.cuh $(SRC_DIR)/utils.cuh
	$(NVCC) $(NVCCFLAGS) -c $(SRC_DIR)/cnf_formula.cu -o $@

$(BUILD_DIR)/cnf_formula_gpu.o: $(BIN_DIR) $(SRC_DIR)/cnf_formula_gpu.cuh $(SRC_DIR)/cnf_formula_gpu.cu $(SRC_DIR)/cnf_formula.cuh $(SRC_DIR)/utils.cuh
	$(NVCC) $(NVCCFLAGS) -c $(SRC_DIR)/cnf_formula_gpu.cu -o $@

$(BUILD_DIR)/miracle.o: $(BIN_DIR) $(SRC_DIR)/miracle.cuh $(SRC_DIR)/miracle.cu $(SRC_DIR)/cnf_formula.cuh $(SRC_DIR)/sat_miracle.cuh $(SRC_DIR)/utils.cuh
	$(NVCC) $(NVCCFLAGS) -c $(SRC_DIR)/miracle.cu -o $@

$(BUILD_DIR)/miracle_dynamic.o: $(BIN_DIR) $(SRC_DIR)/miracle_dynamic.cuh $(SRC_DIR)/miracle_dynamic.cu $(SRC_DIR)/cnf_formula.cuh $(SRC_DIR)/utils.cuh
	$(NVCC) $(NVCCFLAGS) -c $(SRC_DIR)/miracle_dynamic.cu -o $@

$(BUILD_DIR)/miracle_gpu.o: $(BIN_DIR) $(SRC_DIR)/miracle_gpu.cuh $(SRC_DIR)/miracle_gpu.cu $(SRC_DIR)/cnf_formula_gpu.cuh $(SRC_DIR)/miracle.cuh $(SRC_DIR)/sat_miracle.cuh $(SRC_DIR)/utils.cuh $(SRC_DIR)/launch_parameters_gpu.cuh
	$(NVCC) $(NVCCFLAGS) -c $(SRC_DIR)/miracle_gpu.cu -o $@

$(BUILD_DIR)/sat_miracle.o: $(BIN_DIR) $(SRC_DIR)/sat_miracle.cuh $(SRC_DIR)/sat_miracle.cu $(SRC_DIR)/miracle.cuh $(SRC_DIR)/miracle_gpu.cuh
	$(NVCC) $(NVCCFLAGS) -c $(SRC_DIR)/sat_miracle.cu -o $@

$(BUILD_DIR)/utils.o: $(BIN_DIR) $(SRC_DIR)/utils.cuh $(SRC_DIR)/utils.cu $(SRC_DIR)/launch_parameters_gpu.cuh
	$(NVCC) $(NVCCFLAGS) -c $(SRC_DIR)/utils.cu -o $@


clean:
	rm -rf $(BUILD_DIR)
