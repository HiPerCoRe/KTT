#pragma once

#ifdef KTT_PYTHON

#include <string>

namespace ktt
{

inline const std::string ProfileBasedSearcherName = "ProfileBasedSearcher";
inline const std::string ProfileBasedSearcherFile = "ProfileBasedSearcher.py";
inline const std::string ProfileBasedSearcherModule =
  std::string() +
	"#!/usr/bin/env python  -W ignore::DeprecationWarning\n" +
	"\n" +
	"'''\n" +
	"Searcher which explores configurations according to observed bottlenecks\n" +
	"and ML model created on historical data (on the same tuning space, but\n" +
	"possibly different HW and input size). For more information, see\n" +
	"J. Filipovic et al. Using hardware performance counters to speed up\n" +
	"autotuning convergence on GPUs. JPDC, vol. 160, 2021.\n" +
	"'''\n" +
	"\n" +
	"import random\n" +
	"import csv\n" +
	"import pickle\n" +
	"import numpy as np\n" +
	"import signal\n" +
	"import pdb\n" +
	"\n" +
	"import pyktt as ktt\n" +
	"\n" +
	"np.printoptions(precision=5, suppress=True)\n" +
	"\n" +
	"# verbosity level (0, 1, 2, 3)\n" +
	"VERBOSE = 3\n" +
	"if VERBOSE > 2:\n" +
	"    signal.signal(signal.SIGINT, lambda sig, frame: pdb.Pdb().set_trace(frame))\n" +
	"    #pdb.Pdb().set_trace()\n" +
	"\n" +
	"# all constant used by the searcher\n" +
	"# CORR_SHIFT: value added to correlation (positive forces to search parameters with weak correlation but strong variation)\n" +
	"# EXP: exponent used to prioritize configurations with high score (probab = score ^ EXP, where score is in <0, 1> )\n" +
	"# REACT_TO_INST_BOTTLENECKS: minimal instructions bottlenecks, which affects scoring of tuning configurations\n" +
	"# CUTOFF: maximal score of configurations, which are discarded from tuning space\n" +
	"# BATCH: number of configuration from which the fastest one is profiled\n" +
	"CORR_SHIFT = 0.0\n" +
	"EXP = 8\n" +
	"REACT_TO_INST_BOTTLENECKS = 0.7\n" +
	"CUTOFF = -0.25\n" +
	"BATCH = 1\n" +
	"\n" +
	"########################### auxiliary functions ################################\n" +
	"\n" +
	"def loadStatisticsCounters(stat):\n" +
	"    stat.seek(0, 0)\n" +
	"    counters = []\n" +
	"    words = stat.readline().split(',')\n" +
	"    for i in range(1,len(words)) :\n" +
	"        counters.append(words[i].rstrip())\n" +
	"\n" +
	"    return counters\n" +
	"\n" +
	"def loadStatistics (stat, tuningParams, profCounters):\n" +
	"    stat.seek(0, 0)\n" +
	"    # check headers\n" +
	"    words = stat.readline().split(',')\n" +
	"    for i in range(1,len(profCounters)+1):\n" +
	"        if profCounters[i-1] != words[i].rstrip():\n" +
	"            print(\"Error, mismatch tuning parameters: expected \" + profCounters[i-1] + \", but have \" + words[i])\n" +
	"            exit()\n" +
	"\n" +
	"    # load data\n" +
	"    statistics = {}\n" +
	"    for line in stat.readlines():\n" +
	"        words = line.split(',')\n" +
	"        if len(words) <= 1: break\n" +
	"        if not words[0] in tuningParams:\n" +
	"            print(\"Error, unknown tuning parameter \" + words[0])\n" +
	"            exit()\n" +
	"        row = []\n" +
	"        for j in range(1, len(words)):\n" +
	"            row.append(float(words[j]))\n" +
	"        statistics[words[0]] = row\n" +
	"\n" +
	"    return statistics\n" +
	"\n" +
	"def loadCompleteMappingCounters(tuningSpace, rangeC) :\n" +
	"    words = tuningSpace.readline().split(',')\n" +
	"    counters = []\n" +
	"    #for i in range(tuningInt[1]+1, len(words)) :\n" +
	"    for i in rangeC :\n" +
	"        if i < len(words) :\n" +
	"            counters.append(words[i].rstrip())\n" +
	"    for j in range(i+1, len(words)) :\n" +
	"        counters.append(words[j].rstrip())\n" +
	"\n" +
	"    return counters\n" +
	"\n" +
	"def loadModels(modelFiles) :\n" +
	"    tuningparamsNames = []\n" +
	"    TPassignmentsUnsorted = {}\n" +
	"    TPassignments = {}\n" +
	"    conditionsAllModels = []\n" +
	"    PCassignments = {}\n" +
	"    PCassignmentsAllModels = []\n" +
	"    #read all the models\n" +
	"    for m in modelFiles :\n" +
	"        counters = []\n" +
	"        with open(m) as modelFile:\n" +
	"            modelReader = csv.reader(modelFile, delimiter = ',')\n" +
	"            lc = 0\n" +
	"            afterCondition = False\n" +
	"            for row in modelReader:\n" +
	"                #skip the first line\n" +
	"                if lc == 0:\n" +
	"                    lc = 1\n" +
	"                    continue\n" +
	"                elif row[0] != \"Condition\" and not afterCondition:\n" +
	"                    tuningparamsNames.append(row[0])\n" +
	"                    TPassignmentsUnsorted[row[0]] = row[1]\n" +
	"                elif row[0] == \"Condition\":\n" +
	"                    condition = row[1]\n" +
	"                    afterCondition = True\n" +
	"                else :\n" +
	"                    PCassignments[row[0]] = row[1]\n" +
	"                    counters.append(row[0])\n" +
	"                lc = lc+1\n" +
	"        conditionsAllModels.append(condition)\n" +
	"        PCassignmentsAllModels.append(PCassignments)\n" +
	"    #\"sort\" TPassignments to correspond with the order of tuningparamsNames\n" +
	"    for j in range(0, len(tuningparamsNames)) :\n" +
	"        TPassignments[tuningparamsNames[j]] = TPassignmentsUnsorted[tuningparamsNames[j]]\n" +
	"    return [tuningparamsNames, TPassignments, conditionsAllModels, PCassignmentsAllModels, counters]\n" +
	"\n" +
	"#def prepareForModelsEvaluation(TPassignments, conditionsAllModels, tuningSpace) :\n" +
	"\n" +
	" #       applicableModelsOrder = prepareForModelsEvaluation(tuningparamsAssignments, conditions, configurationsData)\n" +
	"\n" +
	"def loadCompleteMapping(tuningSpace, rangeT, rangeC) :\n" +
	"    tuningSpace.seek(0)\n" +
	"    wordsHead = tuningSpace.readline().split(',')\n" +
	"    #pcInt = [tuningInt[1]+1, len(wordsHead)-1]\n" +
	"    myRange = []\n" +
	"    for i in rangeC :\n" +
	"        if i < len(wordsHead) :\n" +
	"            myRange.append(i)\n" +
	"    restPCs = list(range(rangeC[-1]+1, len(wordsHead)))\n" +
	"    pcInt = myRange + restPCs\n" +
	"\n" +
	"    space = []\n" +
	"    for line in tuningSpace.readlines() :\n" +
	"        words = line.split(',')\n" +
	"        if len(words) <= 1: break\n" +
	"\n" +
	"        tunRow = []\n" +
	"        #for i in range(tuningInt[0], tuningInt[1]+1) :\n" +
	"        for i in rangeT :\n" +
	"            tunRow.append(float(words[i]))\n" +
	"        #pcRow = {}\n" +
	"        #for i in range(pcInt[0], pcInt[1]+1) :\n" +
	"        #    pcRow[wordsHead[i].rstrip()] = float(words[i])\n" +
	"        pcRow = []\n" +
	"        #for i in range(pcInt[0], pcInt[1]+1) :\n" +
	"        for i in pcInt :\n" +
	"            pcRow.append(float(words[i]))\n" +
	"\n" +
	"        spaceRow = [tunRow, pcRow]\n" +
	"        space.append(spaceRow)\n" +
	"\n" +
	"    return space\n" +
	"\n" +
	"def setComputeBound():\n" +
	"    global REACT_TO_INST_BOTTLENECKS\n" +
	"    REACT_TO_INST_BOTTLENECKS = 0.5\n" +
	"\n" +
	"def setMemoryBound():\n" +
	"    global REACT_TO_INST_BOTTLENECKS\n" +
	"    REACT_TO_INST_BOTTLENECKS = 0.7\n" +
	"\n" +
	"\n" +
	"####################### GPU arch. dependent functions ##########################\n" +
	"\n" +
	"# analyzeBottlenecks\n" +
	"# analysis of bottlenecks, observes profiling counters and scores bottlenecks\n" +
	"# in interval <0, 1>\n" +
	"# GPU dependent, implemented for CUDA compute capabilities 3.0 - 7.5\n" +
	"\n" +
	"def analyzeBottlenecks (countersNames, countersData, cc, multiprocessors, cores):\n" +
	"    bottlenecks = {}\n" +
	"    # analyze global memory\n" +
	"    if cc < 7.0 :\n" +
	"        DRAMutil = countersData[countersNames.index(\"dram_utilization\")]\n" +
	"        DRAMldTrans = countersData[countersNames.index(\"dram_read_transactions\")]\n" +
	"        DRAMstTrans = countersData[countersNames.index(\"dram_write_transactions\")]\n" +
	"    else :\n" +
	"        DRAMutil = countersData[countersNames.index(\"dram__throughput.avg.pct_of_peak_sustained_elapsed\")]/10.0\n" +
	"        DRAMldTrans = countersData[countersNames.index(\"dram__sectors_read.sum\")]\n" +
	"        DRAMstTrans = countersData[countersNames.index(\"dram__sectors_write.sum\")]\n" +
	"    if DRAMldTrans + DRAMstTrans > 0 and DRAMutil > 0 :\n" +
	"        bnDRAMRead = (DRAMldTrans / (DRAMldTrans + DRAMstTrans)) * (DRAMutil / 10.0)\n" +
	"        bnDRAMWrite = (DRAMstTrans / (DRAMldTrans + DRAMstTrans)) * (DRAMutil / 10.0)\n" +
	"    else :\n" +
	"        bnDRAMRead = 0\n" +
	"        bnDRAMWrite = 0\n" +
	"    bottlenecks['bnDRAMRead'] = bnDRAMRead\n" +
	"    bottlenecks['bnDRAMWrite'] = bnDRAMWrite\n" +
	"\n" +
	"    # analyze cache system\n" +
	"    if cc < 7.0 :\n" +
	"        L2util = countersData[countersNames.index(\"l2_utilization\")]\n" +
	"        L2ldTrans = countersData[countersNames.index(\"l2_read_transactions\")]\n" +
	"        L2stTrans = countersData[countersNames.index(\"l2_write_transactions\")]\n" +
	"        texUtil = countersData[countersNames.index(\"tex_utilization\")]\n" +
	"        #texFuUtil = countersData[countersNames.index(\"tex_fu_utilization\")]\n" +
	"        texTrans = countersData[countersNames.index(\"tex_cache_transactions\")]\n" +
	"    else :\n" +
	"        L2util = countersData[countersNames.index(\"lts__t_sectors.avg.pct_of_peak_sustained_elapsed\")]/10.0\n" +
	"        L2ldTrans = countersData[countersNames.index(\"lts__t_sectors_op_read.sum\")]\n" +
	"        L2stTrans = countersData[countersNames.index(\"lts__t_sectors_op_write.sum\")]\n" +
	"        texUtil = countersData[countersNames.index(\"l1tex__t_requests_pipe_lsu_mem_global_op_ld.avg.pct_of_peak_sustained_active\")]/10.0\n" +
	"        #texFuUtil = countersData[countersNames.index(\"tex_fu_utilization\")]\n" +
	"        texTrans = countersData[countersNames.index(\"l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum\")]\n" +
	"    bnL2Read = (L2ldTrans / (L2ldTrans + L2stTrans)) * (L2util / 10.0)\n" +
	"    bnL2Write = (L2stTrans / (L2ldTrans + L2stTrans)) * (L2util / 10.0)\n" +
	"    #bnTex = max(texUtil / 10.0, texFuUtil / 10.0)\n" +
	"    bnTex = texUtil / 10.0\n" +
	"    bottlenecks['bnL2Read'] = bnL2Read\n" +
	"    bottlenecks['bnL2Write'] = bnL2Write\n" +
	"    bottlenecks['bnTex'] = bnTex\n" +
	"\n" +
	"    # analyze local (non-registers private in OpenCL) memory\n" +
	"    if cc < 7.0 :\n" +
	"        locOverhead = countersData[countersNames.index(\"local_memory_overhead\")]\n" +
	"    else :\n" +
	"        #XXX this is highly experimental computation\n" +
	"        locOverhead = 100.0 * countersNames.index(\"l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum\") / L2stTrans\n" +
	"    bottlenecks['bnLocal'] = (locOverhead/100.0) * max(DRAMutil/10.0, L2util/10.0, texUtil/10.0)#, texFuUtil/10.0)\n" +
	"\n" +
	"    # analyze shared memory\n" +
	"    if cc < 7.0 :\n" +
	"        if cc < 4.0 :\n" +
	"            SMutil = countersData[countersNames.index(\"shared_efficiency\")]\n" +
	"        else :\n" +
	"            SMutil = countersData[countersNames.index(\"shared_utilization\")]\n" +
	"        SMldTrans = countersData[countersNames.index(\"shared_load_transactions\")]\n" +
	"        SMstTrans = countersData[countersNames.index(\"shared_store_transactions\")]\n" +
	"    else :\n" +
	"        SMutil = countersData[countersNames.index(\"l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed\")]/10.0\n" +
	"        SMldTrans = countersData[countersNames.index(\"l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum\")]\n" +
	"        SMstTrans = countersData[countersNames.index(\"l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum\")]\n" +
	"\n" +
	"    if (SMldTrans + SMstTrans > 0):\n" +
	"        bnSMRead = (SMldTrans / (SMldTrans + SMstTrans)) * (SMutil / 10.0)\n" +
	"        bnSMWrite = (SMstTrans / (SMldTrans + SMstTrans)) * (SMutil / 10.0)\n" +
	"    else:\n" +
	"        bnSMRead = 0\n" +
	"        bnSMWrite = 0\n" +
	"    bottlenecks['bnSMRead'] = bnSMRead\n" +
	"    bottlenecks['bnSMWrite'] = bnSMWrite\n" +
	"\n" +
	"    # analyze multiprocessor parallelism\n" +
	"    if cc < 7.0 :\n" +
	"        occupancy = countersData[countersNames.index(\"achieved_occupancy\")]\n" +
	"    else :\n" +
	"        occupancy = countersData[countersNames.index(\"sm__warps_active.avg.pct_of_peak_sustained_active\")]/100.0\n" +
	"    bnMPparal = 1.0 - occupancy\n" +
	"    bottlenecks['bnMPparal'] = bnMPparal\n" +
	"\n" +
	"    # analyze global parallelism\n" +
	"    if cc < 7.0 :\n" +
	"        smEfficiency = countersData[countersNames.index(\"sm_efficiency\")]\n" +
	"    else :\n" +
	"        smEfficiency = countersData[countersNames.index(\"smsp__cycles_active.avg.pct_of_peak_sustained_elapsed\")]\n" +
	"    bnGparal = (100.0 - smEfficiency) / 100.0\n" +
	"    bottlenecks['bnGparal'] = bnGparal\n" +
	"\n" +
	"    threadBlocks = countersData[countersNames.index(\"Global size\")] / countersData[countersNames.index(\"Local size\")]\n" +
	"    bnTailEffect = 1 - (threadBlocks / (((threadBlocks + multiprocessors-1) / multiprocessors) * multiprocessors))\n" +
	"    bottlenecks['bnTailEffect'] = bnTailEffect\n" +
	"    #print(bnTailEffect, threadBlocks, countersData[countersNames.index(\"Global size\")], countersData[countersNames.index(\"Local size\")])\n" +
	"\n" +
	"    bnThreads = max(0, (cores * 5 - countersData[countersNames.index(\"Global size\")]) / (cores * 5))\n" +
	"    bottlenecks['bnThreads'] = bnThreads\n" +
	"\n" +
	"    # analyze instructions\n" +
	"    # insctruction counts\n" +
	"    if cc < 7.0 :\n" +
	"        spInstr = countersData[countersNames.index(\"inst_fp_32\")]\n" +
	"        dpInstr = countersData[countersNames.index(\"inst_fp_64\")]\n" +
	"        intInstr = countersData[countersNames.index(\"inst_integer\")]\n" +
	"        #commInstr = countersData[countersNames.index(\"inst_inter_thread_communication\")]\n" +
	"        miscInstr = countersData[countersNames.index(\"inst_misc\")]\n" +
	"        ldstInstr = countersData[countersNames.index(\"inst_compute_ld_st\")]\n" +
	"        ctrlInst = countersData[countersNames.index(\"inst_control\")]\n" +
	"        bconvInstr = countersData[countersNames.index(\"inst_bit_convert\")]\n" +
	"        execInstr = countersData[countersNames.index(\"inst_executed\")]\n" +
	"    else :\n" +
	"        spInstr = countersData[countersNames.index(\"smsp__sass_thread_inst_executed_op_fp32_pred_on.sum\")]\n" +
	"        dpInstr = countersData[countersNames.index(\"smsp__sass_thread_inst_executed_op_fp64_pred_on.sum\")]\n" +
	"        intInstr = countersData[countersNames.index(\"smsp__sass_thread_inst_executed_op_integer_pred_on.sum\")]\n" +
	"        #commInstr = countersData[countersNames.index(\"inst_inter_thread_communication\")]\n" +
	"        miscInstr = countersData[countersNames.index(\"smsp__sass_thread_inst_executed_op_misc_pred_on.sum\")]\n" +
	"        ldstInstr = countersData[countersNames.index(\"smsp__sass_thread_inst_executed_op_memory_pred_on.sum\")]\n" +
	"        ctrlInst = countersData[countersNames.index(\"smsp__sass_thread_inst_executed_op_control_pred_on.sum\")]\n" +
	"        bconvInstr = countersData[countersNames.index(\"smsp__sass_thread_inst_executed_op_conversion_pred_on.sum\")]\n" +
	"        execInstr = countersData[countersNames.index(\"smsp__inst_executed.sum\")]\n" +
	"\n" +
	"    #instruction utilization\n" +
	"    if cc < 7.0  :\n" +
	"        if cc < 4.0 :\n" +
	"            spUtil = countersData[countersNames.index(\"flop_sp_efficiency\")]\n" +
	"            dpUtil = countersData[countersNames.index(\"flop_dp_efficiency\")]\n" +
	"            sfuUtil = 0 #XXX we don't have this counter\n" +
	"        else :\n" +
	"            spUtil = countersData[countersNames.index(\"single_precision_fu_utilization\")]\n" +
	"            dpUtil = countersData[countersNames.index(\"double_precision_fu_utilization\")]\n" +
	"            sfuUtil = countersData[countersNames.index(\"special_fu_utilization\")]\n" +
	"        cfUtil = countersData[countersNames.index(\"cf_fu_utilization\")]\n" +
	"        ldstUtil = countersData[countersNames.index(\"ldst_fu_utilization\")]\n" +
	"        texFuUtil = countersData[countersNames.index(\"tex_fu_utilization\")]\n" +
	"        instrSlotUtil = countersData[countersNames.index(\"issue_slot_utilization\")]\n" +
	"        if cc >= 4.0 :\n" +
	"            instrEffExec = countersData[countersNames.index(\"warp_execution_efficiency\")]\n" +
	"            instrEffPred = countersData[countersNames.index(\"warp_nonpred_execution_efficiency\")]\n" +
	"        else :\n" +
	"            instrEffExec = 100\n" +
	"            instrEffPred = 100\n" +
	"    else :\n" +
	"        spUtil = countersData[countersNames.index(\"smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active\")]/10.0\n" +
	"        dpUtil = countersData[countersNames.index(\"smsp__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active\")]/10.0\n" +
	"        sfuUtil = countersData[countersNames.index(\"smsp__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active\")]/10.0\n" +
	"        cfUtil = 0.0 #XXX we don't have this counter\n" +
	"        ldstUtil = countersData[countersNames.index(\"smsp__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active\")]/10.0\n" +
	"        texFuUtil = countersData[countersNames.index(\"smsp__inst_executed_pipe_tex.avg.pct_of_peak_sustained_active\")]/10.0\n" +
	"        instrSlotUtil = countersData[countersNames.index(\"smsp__issue_active.avg.pct_of_peak_sustained_active\")]\n" +
	"        instrEffExec = countersData[countersNames.index(\"smsp__thread_inst_executed_per_inst_executed.ratio\")]*100.0/32.0\n" +
	"        instrEffPred = countersData[countersNames.index(\"smsp__thread_inst_executed_per_inst_executed.pct\")]\n" +
	"\n" +
	"    instrExecFitted = execInstr*32.0 * (100.0/instrEffExec) * (100.0/instrEffPred) #XXX this should be equal to spInstr+dpInstr+intInstr+miscInstr+ldstInstr+ctrlInst+bconvInstr\n" +
	"    if cc < 7.0 :\n" +
	"        instrUtilFitted = instrSlotUtil/100.0\n" +
	"    else :\n" +
	"        instrUtilFitted = min(1.0, instrSlotUtil/50.0) # dual-issue causes max 50% utilization of instruction of single type\n" +
	"\n" +
	"    spUtilApprox = (spInstr/instrExecFitted) * instrUtilFitted\n" +
	"    dpUtilApprox = (dpInstr/instrExecFitted) * instrUtilFitted\n" +
	"    ldstUtilApprox = (ldstInstr/instrExecFitted) * instrUtilFitted\n" +
	"    cfUtilApprox = (ctrlInst/instrExecFitted) * instrUtilFitted\n" +
	"    intUtilApprox = (intInstr/instrExecFitted) * instrUtilFitted\n" +
	"    miscUtilApprox = (miscInstr/instrExecFitted) * instrUtilFitted\n" +
	"    bconvUtilApprox = (bconvInstr/instrExecFitted) * instrUtilFitted\n" +
	"\n" +
	"    #print(\"single_precision_fu_utilization reported/computed: \", spUtil, (spInstr/instrExecFitted) * instrUtilFitted)\n" +
	"#    #workaround is to bottleneck instructions only if utilization is significant\n" +
	"#    maxUtil = max(spUtil, dpUtil, cfUtil)\n" +
	"#    maxUtilInst = max(spInstr, dpInstr, ctrlInst) #XXX should select the same category as the line above\n" +
	"#    if maxUtil > 6:\n" +
	"#        intUtilApprox = intInstr/maxUtilInst * maxUtil\n" +
	"#        miscUtilApprox = miscInstr/maxUtilInst * maxUtil\n" +
	"#        bconvUtilApprox = bconvInstr/maxUtilInst * maxUtil\n" +
	"#    else:\n" +
	"#        intUtilApprox = 0.0\n" +
	"#        miscUtilApprox = 0.0\n" +
	"#        bconvUtilApprox = 0.0\n" +
	"\n" +
	"#    bnSP = spUtil/10.0\n" +
	"    bnSP = spUtilApprox\n" +
	"#    bnDP = dpUtil/10.0\n" +
	"    bnDP = dpUtilApprox\n" +
	"    bnSFU = sfuUtil/10.0\n" +
	"#    bnCF = cfUtil/10.0\n" +
	"    bnCF = cfUtilApprox\n" +
	"#    bnLDST = ldstUtil/10.0\n" +
	"    bnLDST = ldstUtilApprox\n" +
	"    bnTexFu = texFuUtil/10.0\n" +
	"    bnInt = intUtilApprox\n" +
	"    bnMisc = miscUtilApprox\n" +
	"    bnBconv = bconvUtilApprox\n" +
	"\n" +
	"    bottlenecks['bnSP'] = bnSP\n" +
	"    bottlenecks['bnDP'] = bnDP\n" +
	"    bottlenecks['bnSFU'] = bnSFU\n" +
	"    bottlenecks['bnCF'] = bnCF\n" +
	"    bottlenecks['bnLDST'] = bnLDST\n" +
	"    bottlenecks['bnTexFu'] = bnTexFu\n" +
	"    bottlenecks['bnInt'] = bnInt\n" +
	"    bottlenecks['bnMisc'] = bnMisc\n" +
	"    bottlenecks['bnBconv'] = bnBconv\n" +
	"\n" +
	"    issueWeight = 0.0\n" +
	"    maxInstrUtil = max(spUtilApprox/instrUtilFitted, dpUtilApprox/instrUtilFitted, sfuUtil/10.0, cfUtilApprox/instrUtilFitted, ldstUtilApprox/instrUtilFitted, intUtilApprox/instrUtilFitted, miscUtilApprox/instrUtilFitted, bconvUtilApprox/instrUtilFitted)\n" +
	"    if maxInstrUtil > REACT_TO_INST_BOTTLENECKS :\n" +
	"        issueWeight = (maxInstrUtil -  REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)\n" +
	"    bnInstIssue = (100.0 - instrSlotUtil) / 100 * issueWeight\n" +
	"    #bnInstIssue = (100.0 - instrSlotUtil) / 100 * max(spUtilApprox/instrUtilFitted, dpUtilApprox/instrUtilFitted, sfuUtil/instrUtilFitted/10.0, cfUtilApprox/instrUtilFitted, ldstUtilApprox/instrUtilFitted, texFuUtil/instrUtilFitted/10.0, intUtilApprox/instrUtilFitted, miscUtilApprox/instrUtilFitted, bconvUtilApprox/instrUtilFitted)\n" +
	"    bottlenecks['bnInstIssue'] = bnInstIssue\n" +
	"\n" +
	"    if VERBOSE > 1 :\n" +
	"        print(\"bottlenecks\", bottlenecks)\n" +
	"\n" +
	"    return bottlenecks\n" +
	"\n" +
	"# computeChanges\n" +
	"# computes how to change profiling counters according to bottlenecks\n" +
	"# absolute value of computed changes means its importance, the sign means\n" +
	"# required direction (increase/decrease the counter)\n" +
	"# GPU dependent, implemented for CUDA compute capabilities 3.0 - 7.5\n" +
	"# Note: this function is separated from analyzeBottlenecks in order to manage\n" +
	"# portability across arch. easily (computed bottlenecks are arch. independent)\n" +
	"\n" +
	"def computeChanges(bottlenecks, countersNames, cc):\n" +
	"    # set how important is to change particular profiling counters\n" +
	"    changeImportance = [0.0]*len(countersNames)\n" +
	"\n" +
	"    # memory-subsystem related counters\n" +
	"    if cc < 7.0 :\n" +
	"        changeImportance[countersNames.index('dram_read_transactions')] = - bottlenecks['bnDRAMRead']\n" +
	"        changeImportance[countersNames.index('dram_write_transactions')] = - bottlenecks['bnDRAMWrite']\n" +
	"        changeImportance[countersNames.index('l2_read_transactions')] = - bottlenecks['bnL2Read']\n" +
	"        changeImportance[countersNames.index('l2_write_transactions')] = - bottlenecks['bnL2Write']\n" +
	"        changeImportance[countersNames.index('tex_cache_transactions')] = - bottlenecks['bnTex']\n" +
	"        changeImportance[countersNames.index('local_memory_overhead')] = - bottlenecks['bnLocal']\n" +
	"        changeImportance[countersNames.index('shared_load_transactions')] = - bottlenecks['bnSMRead']\n" +
	"        changeImportance[countersNames.index('shared_store_transactions')] = - bottlenecks['bnSMWrite']\n" +
	"    else:\n" +
	"        changeImportance[countersNames.index('dram__sectors_read.sum')] = - bottlenecks['bnDRAMRead']\n" +
	"        changeImportance[countersNames.index('dram__sectors_write.sum')] = - bottlenecks['bnDRAMWrite']\n" +
	"        changeImportance[countersNames.index('lts__t_sectors_op_read.sum')] = - bottlenecks['bnL2Read']\n" +
	"        changeImportance[countersNames.index('lts__t_sectors_op_write.sum')] = - bottlenecks['bnL2Write']\n" +
	"        #TODO solve additive counters more elegantly?\n" +
	"        changeImportance[countersNames.index('l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum')] = - bottlenecks['bnTex']\n" +
	"        changeImportance[countersNames.index('l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum')] = - bottlenecks['bnLocal']\n" +
	"        changeImportance[countersNames.index('l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum')] = - bottlenecks['bnLocal']\n" +
	"        changeImportance[countersNames.index('l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum')] = - bottlenecks['bnSMRead']\n" +
	"        changeImportance[countersNames.index('l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum')] = - bottlenecks['bnSMWrite']\n" +
	"\n" +
	"    # instructions related counters\n" +
	"    if cc < 7.0 :\n" +
	"        if bottlenecks['bnSP'] > REACT_TO_INST_BOTTLENECKS :\n" +
	"            changeImportance[countersNames.index('inst_fp_32')] = - (bottlenecks['bnSP'] - REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)\n" +
	"            changeImportance[countersNames.index('flop_sp_efficiency')] = (bottlenecks['bnSP'] - REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)\n" +
	"        if bottlenecks['bnDP'] > REACT_TO_INST_BOTTLENECKS :\n" +
	"            changeImportance[countersNames.index('inst_fp_64')] = - (bottlenecks['bnDP']- REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)\n" +
	"        #changeImportance[countersNames.index('special_fu_utilization')] = + bottlenecks['bnSFU'] #TODO how to count SFU instructions?\n" +
	"        if bottlenecks['bnCF'] > REACT_TO_INST_BOTTLENECKS :\n" +
	"            changeImportance[countersNames.index('inst_control')] = - (bottlenecks['bnCF'] - REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)\n" +
	"        if bottlenecks['bnLDST'] > REACT_TO_INST_BOTTLENECKS :\n" +
	"            changeImportance[countersNames.index('inst_compute_ld_st')] = - (bottlenecks['bnLDST']  - REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)\n" +
	"        #changeImportance[countersNames.index('tex_fu_utilization')] = + bottlenecks['bnTexFu']\n" +
	"        if bottlenecks['bnInt'] > REACT_TO_INST_BOTTLENECKS :\n" +
	"            changeImportance[countersNames.index('inst_integer')] = - (bottlenecks['bnInt']  - REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)\n" +
	"        if bottlenecks['bnMisc'] > REACT_TO_INST_BOTTLENECKS :\n" +
	"            changeImportance[countersNames.index('inst_misc')] = - (bottlenecks['bnMisc'] - REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)\n" +
	"        if bottlenecks['bnBconv'] > REACT_TO_INST_BOTTLENECKS :\n" +
	"            changeImportance[countersNames.index('inst_bit_convert')] = - (bottlenecks['bnBconv'] - REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)\n" +
	"        #if bottlenecks['bnInstIssue'] > REACT_TO_INST_BOTTLENECKS :\n" +
	"        changeImportance[countersNames.index('issue_slot_utilization')] = bottlenecks['bnInstIssue'] #(bottlenecks['bnInstIssue'] - REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)\n" +
	"    else :\n" +
	"        if bottlenecks['bnSP'] > REACT_TO_INST_BOTTLENECKS :\n" +
	"            changeImportance[countersNames.index('smsp__sass_thread_inst_executed_op_fp32_pred_on.sum')] = - bottlenecks['bnSP']\n" +
	"        if bottlenecks['bnDP'] > REACT_TO_INST_BOTTLENECKS :\n" +
	"            changeImportance[countersNames.index('smsp__sass_thread_inst_executed_op_fp64_pred_on.sum')] = - bottlenecks['bnDP']\n" +
	"        #changeImportance[countersNames.index('special_fu_utilization')] = + bottlenecks['bnSFU'] #TODO how to count SFU instructions?\n" +
	"        if bottlenecks['bnCF'] > REACT_TO_INST_BOTTLENECKS :\n" +
	"            changeImportance[countersNames.index('smsp__sass_thread_inst_executed_op_control_pred_on.sum')] = - (bottlenecks['bnCF'] - REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)\n" +
	"        if bottlenecks['bnLDST'] > REACT_TO_INST_BOTTLENECKS :\n" +
	"            changeImportance[countersNames.index('smsp__sass_thread_inst_executed_op_memory_pred_on.sum')] = - (bottlenecks['bnLDST']  - REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)\n" +
	"        #changeImportance[countersNames.index('tex_fu_utilization')] = + bottlenecks['bnTexFu']\n" +
	"        if bottlenecks['bnInt'] > REACT_TO_INST_BOTTLENECKS :\n" +
	"            changeImportance[countersNames.index('smsp__sass_thread_inst_executed_op_integer_pred_on.sum')] = - (bottlenecks['bnInt']  - REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)\n" +
	"        if bottlenecks['bnMisc'] > REACT_TO_INST_BOTTLENECKS :\n" +
	"            changeImportance[countersNames.index('smsp__sass_thread_inst_executed_op_misc_pred_on.sum')] = - (bottlenecks['bnMisc'] - REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)\n" +
	"        if bottlenecks['bnBconv'] > REACT_TO_INST_BOTTLENECKS :\n" +
	"            changeImportance[countersNames.index('smsp__sass_thread_inst_executed_op_conversion_pred_on.sum')] = - (bottlenecks['bnBconv'] - REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)\n" +
	"        #if bottlenecks['bnInstIssue'] > REACT_TO_INST_BOTTLENECKS :\n" +
	"        changeImportance[countersNames.index('smsp__issue_active.avg.pct_of_peak_sustained_active')] = bottlenecks['bnInstIssue'] #(bottlenecks['bnInstIssue'] - REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)\n" +
	"\n" +
	"    #parallelism related counters\n" +
	"    if cc < 7.0 :\n" +
	"        changeImportance[countersNames.index('sm_efficiency')] = bottlenecks['bnGparal']\n" +
	"    else :\n" +
	"        changeImportance[countersNames.index('smsp__cycles_active.avg.pct_of_peak_sustained_elapsed')] = bottlenecks['bnGparal']\n" +
	"\n" +
	"    changeImportance[countersNames.index('Global size')] = bottlenecks['bnThreads']#bottlenecks['bnTailEffect'] + bottlenecks['bnThreads']\n" +
	"    #changeImportance[countersNames.index('Local size')] = - bottlenecks['bnTailEffect'] / 2\n" +
	"\n" +
	"    if VERBOSE > 1 :\n" +
	"        print(\"changeImportance\", changeImportance)\n" +
	"\n" +
	"    return changeImportance\n" +
	"\n" +
	"###################### GPU arch. independent functions #########################\n" +
	"\n" +
	"# scoreTuningConfigurationsStats\n" +
	"# scores all tuning configurations according to required changes of profiling\n" +
	"# counters and expected effect of the tuning parameters to profiling counters\n" +
	"# GPU independent\n" +
	"# This version uses offline computed statistics (average corr and var)\n" +
	"\n" +
	"def scoreTuningConfigurations(changeImportance, tuningparamsNames, actualConf, tuningSpace, correlations, variations, scoreDistrib):\n" +
	"    newScoreDistrib = [0.0] * len(tuningSpace)\n" +
	"\n" +
	"    # get maximal variations (could be moved)\n" +
	"    maxVariations = [0.0] * len(variations[tuningparamsNames[0]])\n" +
	"    for i in range(0, len(variations[tuningparamsNames[0]])) :\n" +
	"        for j in range(0, len(tuningparamsNames)) :\n" +
	"            maxVariations[i] = max(maxVariations[i], variations[tuningparamsNames[j]][i])\n" +
	"\n" +
	"    # compute changes size and direction of tuning parameters\n" +
	"    changes = []\n" +
	"    for i in range(0, len(tuningparamsNames)) :\n" +
	"        var = 0.0\n" +
	"        corr = 0.0\n" +
	"        for j in range(0, len(changeImportance)) :\n" +
	"            if maxVariations[j] > 0.0 :\n" +
	"                tmp = abs(changeImportance[j]) * variations[tuningparamsNames[i]][j] / maxVariations[j]\n" +
	"            else :\n" +
	"                tmp = 0.0\n" +
	"            if tmp != 0.0:\n" +
	"                var = var + tmp\n" +
	"                sign = 1.0\n" +
	"                if (changeImportance[j] < 0) :\n" +
	"                    sign = -1.0\n" +
	"                corr = corr + (correlations[tuningparamsNames[i]][j] + CORR_SHIFT)/(1.0 + CORR_SHIFT) * sign * tmp\n" +
	"        if (var > 0.0) :\n" +
	"            corr = corr / abs(var)\n" +
	"        else :\n" +
	"            corr = 0.0\n" +
	"        change = []\n" +
	"        change.append(var)\n" +
	"        change.append(corr)\n" +
	"        changes.append(change)\n" +
	"    if VERBOSE > 1 :\n" +
	"        print(\"changes\", changes)\n" +
	"\n" +
	"    # compute scores according to proposed changes\n" +
	"    # for each point in tuning space\n" +
	"    for i in range(0, len(tuningSpace)) :\n" +
	"        # for each tuning parameter\n" +
	"        for j in range(0, len(tuningSpace[0])) :\n" +
	"            # direction from actual configuration\n" +
	"            direction = tuningSpace[i][j] - actualConf[j]\n" +
	"            #print(tuningData[i][1], actualConf)\n" +
	"            if direction > 0 :\n" +
	"                direction = 1\n" +
	"            if direction < 0 :\n" +
	"                direction = -1\n" +
	"            s = direction * changes[j][0] * changes[j][1]\n" +
	"            #print(i, direction, changes[j], s)\n" +
	"            newScoreDistrib[i] = newScoreDistrib[i] + s\n" +
	"\n" +
	"    minScore = min(newScoreDistrib)\n" +
	"    maxScore = max(newScoreDistrib)\n" +
	"    if VERBOSE > 0 :\n" +
	"        print(\"scoreDistrib interval: \", minScore, maxScore)\n" +
	"    for i in range(0, len(tuningSpace)) :\n" +
	"        if newScoreDistrib[i] < CUTOFF :\n" +
	"            newScoreDistrib[i] = 0.0001\n" +
	"        else :\n" +
	"            if newScoreDistrib[i] < 0.0 :\n" +
	"                newScoreDistrib[i] = 1.0 - (newScoreDistrib[i] / minScore)\n" +
	"            else :\n" +
	"                if newScoreDistrib[i] > 0.0 :\n" +
	"                    newScoreDistrib[i] = 1.0 + (newScoreDistrib[i] / maxScore)\n" +
	"            newScoreDistrib[i] = newScoreDistrib[i]**EXP\n" +
	"        if newScoreDistrib[i] < 0.0001 :\n" +
	"            newScoreDistrib[i] = 0.0001\n" +
	"\n" +
	"        # if was 0, set to 0 (explored)\n" +
	"        if scoreDistrib[i] == 0.0 :\n" +
	"            newScoreDistrib[i] = 0.0\n" +
	"\n" +
	"    if VERBOSE > 2 :\n" +
	"        print(\"newScoreDistrib\", newScoreDistrib)\n" +
	"\n" +
	"    return newScoreDistrib\n" +
	"\n" +
	"# scoreTuningConfigurationsExact\n" +
	"# scores all tuning configurations according to required changes of profiling\n" +
	"# counters and expected effect of the tuning parameters to profiling counters\n" +
	"# GPU independent\n" +
	"# This version uses completely computed offline space\n" +
	"\n" +
	"def scoreTuningConfigurationsExact(changeImportance, tuningparamsNames, actualConf, tuningSpace, completeMapping, scoreDistrib):\n" +
	"    newScoreDistrib = [0.0] * len(tuningSpace)\n" +
	"    #search index of actualConf in completeMapping (some conf. can be missing, therefore, we need to check tuning parameters)\n" +
	"    actualPC = []\n" +
	"    for conf in completeMapping :\n" +
	"        if actualConf == conf[0] :\n" +
	"            actualPC = conf[1]\n" +
	"    if len(actualPC) == 0 :\n" +
	"        # the configuration is not known in the completeMapping, return uniform distrib\n" +
	"        for i in range(0, len(tuningSpace)) :\n" +
	"            uniformScoreDistrib = [1.0] * len(tuningSpace)\n" +
	"            if scoreDistrib[i] == 0.0 :\n" +
	"                uniformScoreDistrib[i] = 0.0\n" +
	"        return uniformScoreDistrib\n" +
	"\n" +
	"    cmIdx = 0\n" +
	"    # for each tuning configuration\n" +
	"    for i in range(0, len(tuningSpace)) :\n" +
	"        #seek for equivalent tuning configuration in the completeMapping\n" +
	"        #TODO this implementation assumes the same order of tuning configurations, create mapping between indexes instead\n" +
	"        myPC = []\n" +
	"        for j in range(cmIdx, len(completeMapping)) :\n" +
	"            if (tuningSpace[i] == completeMapping[j][0]) :\n" +
	"                myPC = completeMapping[j][1]\n" +
	"                cmIdx = j+1\n" +
	"                break\n" +
	"        if (len(myPC) == 0) :\n" +
	"            newScoreDistrib[i] = 0.0\n" +
	"        else :\n" +
	"            #score configuration\n" +
	"            for j in range(0, len(changeImportance)) :\n" +
	"                try:\n" +
	"                    newScoreDistrib[i] = newScoreDistrib[i] + changeImportance[j] * (myPC[j] - actualPC[j]) / (myPC[j]+actualPC[j])\n" +
	"                except ZeroDivisionError :\n" +
	"                    newScoreDistrib[i] = newScoreDistrib[i] + 0.0\n" +
	"\n" +
	"    minScore = min(newScoreDistrib)\n" +
	"    maxScore = max(newScoreDistrib)\n" +
	"    if VERBOSE > 0 :\n" +
	"        print(\"scoreDistrib interval: \", minScore, maxScore)\n" +
	"    for i in range(0, len(tuningSpace)) :\n" +
	"        if newScoreDistrib[i] < CUTOFF :\n" +
	"            newScoreDistrib[i] = 0.0\n" +
	"        else :\n" +
	"            if newScoreDistrib[i] < 0.0 :\n" +
	"                newScoreDistrib[i] = 1.0 - (newScoreDistrib[i] / minScore)\n" +
	"            else :\n" +
	"                if newScoreDistrib[i] > 0.0 :\n" +
	"                    newScoreDistrib[i] = 1.0 + (newScoreDistrib[i] / maxScore)\n" +
	"            newScoreDistrib[i] = newScoreDistrib[i]**EXP\n" +
	"        if newScoreDistrib[i] < 0.0001 :\n" +
	"            newScoreDistrib[i] = 0.0001\n" +
	"\n" +
	"        # if was 0, set to 0 (explored)\n" +
	"        if scoreDistrib[i] == 0.0 :\n" +
	"            newScoreDistrib[i] = 0.0\n" +
	"\n" +
	"    if VERBOSE > 2 :\n" +
	"        print(\"newScoreDistrib\", newScoreDistrib)\n" +
	"\n" +
	"    return newScoreDistrib\n" +
	"\n" +
	"\n" +
	"# makePredictions\n" +
	"# calculates predicted PC values for all tuning configurations and returns them as completeMapping\n" +
	"# first, it encodes the values of tuning parameters\n" +
	"# second, it finds the suitable models of profiling counters by evaluating the conditions. as there are multiple non-linear models, one for each combination of values of binary parameters, we need to find which one is applicable to this tuning configuration and its combination fo values of binary parameters. tis is true if the conditionsAllModels is satisfied\n" +
	"# third, it calculates the predicted values of profiling counters\n" +
	"\n" +
	"def makePredictions(tuningSpace, TPassignments, conditionsAllModels, PCassignmentsAllModels):\n" +
	"\n" +
	"    completeMapping = []\n" +
	"    # for each tuning configuration\n" +
	"    for i in range(0, len(tuningSpace)) :\n" +
	"        #evaluate the predictions for all tuning configurations\n" +
	"        #encode tuning parameters, the order of TP is the same, we sorted earlier\n" +
	"        k = list(TPassignments)\n" +
	"        for j in range(0, len(k)) :\n" +
	"            exec(TPassignments[k[j]].replace(k[j], str(tuningSpace[i][j])))\n" +
	"        applicableModel = -1\n" +
	"        for j in range(0, len(conditionsAllModels)) :\n" +
	"            #find the first model where the condition is satisfied\n" +
	"            if (eval(conditionsAllModels[j])) :\n" +
	"                applicableModel = j\n" +
	"                break\n" +
	"\n" +
	"        if (applicableModel == -1):\n" +
	"            #we have not found an applicable model\n" +
	"            continue\n" +
	"\n" +
	"        #evaluate the prediction formulas\n" +
	"        #TODO this implementation assumes the same order of PC\n" +
	"        PCassignments = PCassignmentsAllModels[applicableModel]\n" +
	"        k = list(PCassignments)\n" +
	"        myPC = []\n" +
	"        for j in range(0, len(k)) :\n" +
	"            myPC.append(eval(PCassignments[k[j]]))\n" +
	"        completeMapping.append([tuningSpace[i], myPC])\n" +
	"\n" +
	"    return completeMapping\n" +
	"\n" +
	"# randomSearchStep\n" +
	"# perform one step of random search (without memory)\n" +
	"\n" +
	"def randomSearchStep(tuningSpaceSize) :\n" +
	"    return int(random.random() * tuningSpaceSize)\n" +
	"\n" +
	"# weightedRandomSearchStep\n" +
	"# perform one step of random search using weighted probability based on\n" +
	"# profiling counters\n" +
	"\n" +
	"def weightedRandomSearchStep(scoreDistrib, tuningSpaceSize) :\n" +
	"    if (sum(scoreDistrib) == 0.0) :\n" +
	"        print(\"Weighted search error: no more tuning configurations.\")\n" +
	"        return randomSearchStep(tuningSpaceSize)\n" +
	"\n" +
	"    rnd = random.random() * sum(scoreDistrib)\n" +
	"    idx = 0\n" +
	"    tmp = 0.0\n" +
	"    for j in range (0, tuningSpaceSize):\n" +
	"        tmp = tmp + scoreDistrib[j]\n" +
	"        if rnd < tmp : break\n" +
	"        idx = idx + 1\n" +
	"    return idx\n" +
	"\n" +
	"def loadMLModel(trainedKnowledgeBase):\n" +
	"    return pickle.load(open(trainedKnowledgeBase, 'rb'))\n" +
	"\n" +
	"########################### Temp function\n" +
	"def scoreTuningConfigurationsPredictor(changeImportance, tuningparamsNames, actualConf, tuningSpace, scoreDistrib, loaded_model):\n" +
	"    def mulfunc(a, b, c):\n" +
	"        if (a * (b - c)) > 0.0:\n" +
	"            return 1.0\n" +
	"        if (a * (b - c)) < 0.0:\n" +
	"            return -1.0\n" +
	"        else:\n" +
	"            return 0.0\n" +
	"\n" +
	"    newScoreDistrib = [0.0] * len(tuningSpace)\n" +
	"    actualPC = []\n" +
	"\n" +
	"    # Using ML predictor\n" +
	"    predictedPC = loaded_model.predict([actualConf])\n" +
	"    actualPC = list(predictedPC.flatten())\n" +
	"\n" +
	"    if len(actualPC) == 0 :\n" +
	"        for i in range(0, len(tuningSpace)) :\n" +
	"            uniformScoreDistrib = [1.0] * len(tuningSpace)\n" +
	"            if scoreDistrib[i] == 0.0 :\n" +
	"                uniformScoreDistrib[i] = 0.0\n" +
	"        return uniformScoreDistrib\n" +
	"\n" +
	"\n" +
	"    #################################################### Using ML predictor\n" +
	"    predictedMyPC = loaded_model.predict(tuningSpace)\n" +
	"    predictedMyPC1 = np.array(predictedMyPC)\n" +
	"    actualPC1 = np.array(actualPC)\n" +
	"    n = len(changeImportance) - len(actualPC1)\n" +
	"    changeImportance = changeImportance[:len(changeImportance)-n]\n" +
	"    changeImportance1 = np.array(changeImportance)\n" +
	"\n" +
	"    vfunc = np.vectorize(mulfunc)\n" +
	"    mul = vfunc(changeImportance1, predictedMyPC1, actualPC1)\n" +
	"    res = np.array(mul * abs(changeImportance1 * 2.0 * (predictedMyPC1 - actualPC1) / (predictedMyPC1+actualPC1)))\n" +
	"    res = np.nan_to_num(res)\n" +
	"    newScoreDistrib = res.sum(axis=1)\n" +
	"\n" +
	"\n" +
	"    minScore = min(newScoreDistrib)\n" +
	"    maxScore = max(newScoreDistrib)\n" +
	"    if VERBOSE > 0 :\n" +
	"        print(\"scoreDistrib interval: \", minScore, maxScore)\n" +
	"    for i in range(0, len(tuningSpace)) :\n" +
	"        if newScoreDistrib[i] < CUTOFF :\n" +
	"            newScoreDistrib[i] = 0.0001\n" +
	"        else :\n" +
	"            if newScoreDistrib[i] < 0.0 :\n" +
	"                newScoreDistrib[i] = 1.0 - (newScoreDistrib[i] / minScore)\n" +
	"            else :\n" +
	"                if newScoreDistrib[i] > 0.0 :\n" +
	"                    newScoreDistrib[i] = 1.0 + (newScoreDistrib[i] / maxScore)\n" +
	"            newScoreDistrib[i] = newScoreDistrib[i]**EXP\n" +
	"        if newScoreDistrib[i] < 0.0001 :\n" +
	"            newScoreDistrib[i] = 0.0001\n" +
	"\n" +
	"        # if was 0, set to 0 (explored)\n" +
	"        if scoreDistrib[i] == 0.0 :\n" +
	"            newScoreDistrib[i] = 0.0\n" +
	"\n" +
	"    if VERBOSE > 2 :\n" +
	"        print(\"Predictro newScoreDistrib\", newScoreDistrib)\n" +
	"\n" +
	"    return newScoreDistrib\n" +
	"\n" +
	"# auxiliary functions\n" +
	"def readPCList (filename) :\n" +
	"    ret = []\n" +
	"    pcListFile = open(filename, 'r')\n" +
	"    if pcListFile.readline().rstrip() != 'Profiling counter' :\n" +
	"        print('Malformed PC list file!')\n" +
	"        return ret\n" +
	"    for line in pcListFile.readlines():\n" +
	"        ret.append(line.rstrip())\n" +
	"    pcListFile.close()\n" +
	"\n" +
	"    return ret\n" +
	"\n" +
	"class PyProfilingSearcher(ktt.Searcher):\n" +
	"    ccMajor = 0\n" +
	"    ccMinor = 0\n" +
	"    cc = 0\n" +
	"    multiprocessors = 0\n" +
	"    profilingCountersModel = 0\n" +
	"    bestDuration = -1\n" +
	"    bestConf = None\n" +
	"    preselectedBatch = []\n" +
	"    tuningParamsNames = []\n" +
	"    currentConfiguration = ktt.KernelConfiguration()\n" +
	"    tuner = None\n" +
	"    model = None\n" +
	"\n" +
	"    def __init__(self):\n" +
	"        ktt.Searcher.__init__(self)\n" +
	"\n" +
	"    def OnInitialize(self):\n" +
	"        for i in range(0, BATCH) :\n" +
	"            self.preselectedBatch.append(self.GetRandomConfiguration())\n" +
	"        self.currentConfiguration = self.preselectedBatch[0]\n" +
	"        if BATCH > 1:\n" +
	"            #if BATCH==1, we need to keep the only configuration in batch, so that the profiling can be run on it. otherwise, we end up with an empty batch with nothing to profile.\n" +
	"            self.preselectedBatch.pop(0)\n" +
	"\n" +
	"        tp = self.currentConfiguration.GetPairs()\n" +
	"        for p in tp :\n" +
	"            self.tuningParamsNames.append(p.GetName())\n" +
	"\n" +
	"    def Configure(self, tuner, modelFile):\n" +
	"        self.tuner = tuner\n" +
	"        self.ccMajor = tuner.GetCurrentDeviceInfo().GetCudaComputeCapabilityMajor()\n" +
	"        self.ccMinor = tuner.GetCurrentDeviceInfo().GetCudaComputeCapabilityMinor()\n" +
	"        self.cc = self.ccMajor + round(0.1 * self.ccMinor, 1)\n" +
	"        self.multiprocessors = tuner.GetCurrentDeviceInfo().GetMaxComputeUnits()\n" +
	"\n" +
	"        self.profilingCountersModel = readPCList(modelFile + \".pc\")\n" +
	"        self.model = loadMLModel(modelFile)\n" +
	"\n" +
	"    def GetUniqueConfigurations(self, configurations):\n" +
	"        uniqueConfigurations = []\n" +
	"        indicesConfigurations = []\n" +
	"        uniqueIndicesConfigurations = []\n" +
	"        for c in configurations:\n" +
	"            indicesConfigurations.append(self.GetIndex(c))\n" +
	"        uniqueIndicesConfigurations = list(set(indicesConfigurations))\n" +
	"\n" +
	"        for i in uniqueIndicesConfigurations:\n" +
	"            uniqueConfigurations.append(self.GetConfiguration(i))\n" +
	"        return uniqueConfigurations\n" +
	"\n" +
	"    def CalculateNextConfiguration(self, previousResult):\n" +
	"        # select the new configuration\n" +
	"        if len(self.preselectedBatch) > 0 and len(previousResult.GetResults()) > 0:\n" +
	"            # we are testing current batch\n" +
	"            if (self.bestConf == None) or (previousResult.GetKernelDuration() < self.bestDuration) :\n" +
	"                self.bestDuration = previousResult.GetKernelDuration()\n" +
	"                self.bestConf = self.currentConfiguration\n" +
	"                if VERBOSE > 2:\n" +
	"                    print(\"Found new best configuration \", self.bestDuration, self.bestConf, flush = True)\n" +
	"            if VERBOSE > 2:\n" +
	"                print(\"PreselectedBatch has \", len(self.preselectedBatch), \" remaining items:\")\n" +
	"                ind = []\n" +
	"                for c in self.preselectedBatch :\n" +
	"                    ind.append(self.GetIndex(c))\n" +
	"                print(ind, flush = True)\n" +
	"            self.currentConfiguration = self.preselectedBatch.pop(0)\n" +
	"        else :\n" +
	"            if self.bestDuration != -1:\n" +
	"                # we run the fastest one once again, but with profiling\n" +
	"                self.currentConfiguration = self.bestConf\n" +
	"                self.bestDuration = -1\n" +
	"                self.tuner.SetProfiling(True)\n" +
	"                if VERBOSE > 2:\n" +
	"                    print(\"Preselected batch empty, running profiling on configuration \" + str(self.GetIndex(self.currentConfiguration)), flush = True)\n" +
	"            else :\n" +
	"                # get PCs from the last tuning run\n" +
	"                if len(previousResult.GetResults()) > 1:\n" +
	"                    print(\"Warning: this version of profile-based searcher does not support searching kernels collections. Using counters from kernels 0 only.\")\n" +
	"                globalSize = previousResult.GetResults()[0].GetGlobalSize()\n" +
	"                localSize = previousResult.GetResults()[0].GetLocalSize()\n" +
	"                profilingCountersRun = previousResult.GetResults()[0].GetProfilingData().GetCounters() #FIXME this supposes there is no composition profiled\n" +
	"                pcNames = [\"Global size\", \"Local size\"]\n" +
	"                pcVals = [globalSize.GetTotalSize(), localSize.GetTotalSize()]\n" +
	"                for pd in profilingCountersRun :\n" +
	"                    pcNames.append(pd.GetName())\n" +
	"                    if (pd.GetType() == ktt.ProfilingCounterType.Int) :\n" +
	"                        pcVals.append(pd.GetValueInt())\n" +
	"                    elif (pd.GetType() == ktt.ProfilingCounterType.UnsignedInt) or (pd.GetType() == ktt.ProfilingCounterType.Throughput) or (pd.GetType() == ktt.ProfilingCounterType.UtilizationLevel):\n" +
	"                        pcVals.append(pd.GetValueUint())\n" +
	"                    elif (pd.GetType() == ktt.ProfilingCounterType.Double) or (pd.GetType() == ktt.ProfilingCounterType.Percent) :\n" +
	"                        pcVals.append(pd.GetValueDouble())\n" +
	"                    else :\n" +
	"                        print(\"Fatal error, unsupported PC value passed to profile-based searcher!\")\n" +
	"                        exit(1)\n" +
	"\n" +
	"                # select candidate configurations according to position of the best one plus some random sample\n" +
	"                #candidates = self.GetNeighbourConfigurations(self.bestConf, 2, 100)\n" +
	"                candidates = []\n" +
	"                count = 0\n" +
	"                noAddRandomConfigurations = self.GetUnexploredConfigurationsCount()\n" +
	"                while count < noAddRandomConfigurations:\n" +
	"                    for i in range (count, noAddRandomConfigurations) :\n" +
	"                        candidates.append(self.GetRandomConfiguration())\n" +
	"                    candidates = self.GetUniqueConfigurations(candidates)\n" +
	"                    count = len(candidates)\n" +
	"\n" +
	"\n" +
	"                print(\"Profile-based searcher: evaluating model for \" + str(len(candidates)) + \" candidates...\", flush = True)\n" +
	"\n" +
	"                # get tuning space from candidates\n" +
	"                candidatesTuningSpace = []\n" +
	"                for c in candidates :\n" +
	"                    tp = c.GetPairs()\n" +
	"                    candidateParams = []\n" +
	"                    for p in tp :\n" +
	"                        candidateParams.append(p.GetValue())\n" +
	"                    candidatesTuningSpace.append(candidateParams)\n" +
	"                myTuningSpace = []\n" +
	"                tp = self.bestConf.GetPairs()\n" +
	"                for p in tp :\n" +
	"                    myTuningSpace.append(p.GetValue())\n" +
	"\n" +
	"                if VERBOSE > 2:\n" +
	"                    print(\"Candidates space done\", flush = True)\n" +
	"\n" +
	"                # score the configurations\n" +
	"                scoreDistrib = [1.0]*len(candidates)\n" +
	"                bottlenecks = analyzeBottlenecks(pcNames, pcVals, 6.1, self.multiprocessors, self.convertSM2Cores() * self.multiprocessors)\n" +
	"                changes = computeChanges(bottlenecks, self.profilingCountersModel, self.cc)\n" +
	"                if VERBOSE > 2:\n" +
	"                    print(self.tuningParamsNames)\n" +
	"                scoreDistrib = scoreTuningConfigurationsPredictor(changes, self.tuningParamsNames, myTuningSpace, candidatesTuningSpace, scoreDistrib, self.model)\n" +
	"\n" +
	"                if VERBOSE > 2:\n" +
	"                    print(\"Scoring of the candidates done.\", flush = True)\n" +
	"\n" +
	"                # select next batch\n" +
	"                selectedIndices = []\n" +
	"                if len(candidates) > BATCH :\n" +
	"                    numInBatch = 0\n" +
	"                    while numInBatch < BATCH :\n" +
	"                        idx = weightedRandomSearchStep(scoreDistrib, len(candidates))\n" +
	"                        if selectedIndices == [] or idx not in selectedIndices:\n" +
	"                        #check if we have not chosen the same configuration in previous iterations\n" +
	"                            self.preselectedBatch.append(candidates[idx])\n" +
	"                            selectedIndices.append(idx)\n" +
	"                            numInBatch = numInBatch + 1\n" +
	"                            #scoreDistrib[idx] = 0.0\n" +
	"                else:\n" +
	"                    for i in range(0, len(candidates)):\n" +
	"                        self.preselectedBatch.append(candidates[i])\n" +
	"\n" +
	"                if VERBOSE == 3:\n" +
	"                    print(\"Turning off profiling, new batch selected with length \", len(self.preselectedBatch), \" with configurations \")\n" +
	"                    ind = []\n" +
	"                    for c in self.preselectedBatch :\n" +
	"                        ind.append(self.GetIndex(c))\n" +
	"                    print(ind, flush = True)\n" +
	"                self.currentConfiguration = self.preselectedBatch[0]\n" +
	"                if BATCH > 1:\n" +
	"                    self.preselectedBatch.pop(0)\n" +
	"                self.bestConf = None\n" +
	"                self.tuner.SetProfiling(False)\n" +
	"\n" +
	"        return True\n" +
	"\n" +
	"    def GetCurrentConfiguration(self):\n" +
	"        return self.currentConfiguration\n" +
	"\n" +
	"    def convertSM2Cores(self):\n" +
	"        smToCoresDict = {\n" +
	"            0x30: 192,\n" +
	"            0x32: 192,\n" +
	"            0x35: 192,\n" +
	"            0x37: 192,\n" +
	"            0x50: 128,\n" +
	"            0x52: 128,\n" +
	"            0x53: 128,\n" +
	"            0x60: 64,\n" +
	"            0x61: 128,\n" +
	"            0x62: 128,\n" +
	"            0x70: 64,\n" +
	"            0x72: 64,\n" +
	"            0x75: 64,\n" +
	"            0x80: 64,\n" +
	"            0x86: 64\n" +
	"        }\n" +
	"        defaultSM = 64\n" +
	"\n" +
	"        compact = (self.ccMajor << 4) + self.ccMinor\n" +
	"        if compact in smToCoresDict:\n" +
	"            return smToCoresDict[compact]\n" +
	"        else:\n" +
	"            print(\"Warning: unknown number of cores for SM \" + str(self.ccMajor) + \".\" + str(self.ccMinor) + \", using default value of \" + str(defaultSM))\n" +
	"            return defaultSM\n" +
	"\n" +
	"def executeSearcher(tuner, kernel, model):\n" +
	"    searcher = PyProfilingSearcher()\n" +
	"    tuner.SetSearcher(kernel, searcher)\n" +
	"    searcher.Configure(tuner, model)\n" +
	"";
} // namespace ktt

#endif // KTT_PYTHON
