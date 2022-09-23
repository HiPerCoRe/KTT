#!/usr/bin/env python  -W ignore::DeprecationWarning

'''
Searcher which explores configurations according to observed bottlenecks
and ML model created on historical data (on the same tuning space, but
possibly different HW and input size). For more information, see
J. Filipovic et al. Using hardware performance counters to speed up
autotuning convergence on GPUs. JPDC, vol. 160, 2021.
'''

import random
import math
from operator import add
import csv
import pickle
import numpy as np
import pandas as pd
import sys

import pyktt as ktt

np.printoptions(precision=5, suppress=True)

# verbosity level (0, 1, 2, 3)
VERBOSE = 0

# all constant used by the searcher
# CORR_SHIFT: value added to correlation (positive forces to search parameters with weak correlation but strong variation)
# EXP: exponent used to prioritize configurations with high score (probab = score ^ EXP, where score is in <0, 1> )
# REACT_TO_INST_BOTTLENECKS: minimal instructions bottlenecks, which affects scoring of tuning configurations
# CUTOFF: maximal score of configurations, which are discarded from tuning space
# BATCH: number of configuration from which the fastest one is profiled
CORR_SHIFT = 0.0
EXP = 8
REACT_TO_INST_BOTTLENECKS = 0.7
CUTOFF = -0.25
BATCH = 5

########################### auxiliary functions ################################

def loadStatisticsCounters(stat):
    stat.seek(0, 0)
    counters = []
    words = stat.readline().split(',')
    for i in range(1,len(words)) :
        counters.append(words[i].rstrip())

    return counters

def loadStatistics (stat, tuningParams, profCounters):
    stat.seek(0, 0)
    # check headers
    words = stat.readline().split(',')
    for i in range(1,len(profCounters)+1):
        if profCounters[i-1] != words[i].rstrip():
            print("Error, mismatch tuning parameters: expected " + profCounters[i-1] + ", but have " + words[i])
            exit()

    # load data
    statistics = {}
    for line in stat.readlines():
        words = line.split(',')
        if len(words) <= 1: break
        if not words[0] in tuningParams:
            print("Error, unknown tuning parameter " + words[0])
            exit()
        row = []
        for j in range(1, len(words)):
            row.append(float(words[j]))
        statistics[words[0]] = row

    return statistics

def loadCompleteMappingCounters(tuningSpace, rangeC) :
    words = tuningSpace.readline().split(',')
    counters = []
    #for i in range(tuningInt[1]+1, len(words)) :
    for i in rangeC :
        if i < len(words) :
            counters.append(words[i].rstrip())
    for j in range(i+1, len(words)) :
        counters.append(words[j].rstrip())

    return counters

def loadModels(modelFiles) :
    tuningparamsNames = []
    TPassignmentsUnsorted = {}
    TPassignments = {}
    conditionsAllModels = []
    PCassignments = {}
    PCassignmentsAllModels = []
    #read all the models
    for m in modelFiles :
        counters = []
        with open(m) as modelFile:
            modelReader = csv.reader(modelFile, delimiter = ',')
            lc = 0
            afterCondition = False
            for row in modelReader:
                #skip the first line
                if lc == 0:
                    lc = 1
                    continue
                elif row[0] != "Condition" and not afterCondition:
                    tuningparamsNames.append(row[0])
                    TPassignmentsUnsorted[row[0]] = row[1]
                elif row[0] == "Condition":
                    condition = row[1]
                    afterCondition = True
                else :
                    PCassignments[row[0]] = row[1]
                    counters.append(row[0])
                lc = lc+1
        conditionsAllModels.append(condition)
        PCassignmentsAllModels.append(PCassignments)
    #"sort" TPassignments to correspond with the order of tuningparamsNames
    for j in range(0, len(tuningparamsNames)) :
        TPassignments[tuningparamsNames[j]] = TPassignmentsUnsorted[tuningparamsNames[j]]
    return [tuningparamsNames, TPassignments, conditionsAllModels, PCassignmentsAllModels, counters]

def prepareForModelsEvaluation(TPassignments, conditionsAllModels, tuningSpace) :

        applicableModelsOrder = prepareForModelsEvaluation(tuningparamsAssignments, conditions, configurationsData)

def loadCompleteMapping(tuningSpace, rangeT, rangeC) :
    tuningSpace.seek(0)
    wordsHead = tuningSpace.readline().split(',')
    #pcInt = [tuningInt[1]+1, len(wordsHead)-1]
    myRange = []
    for i in rangeC :
        if i < len(wordsHead) :
            myRange.append(i)
    restPCs = list(range(rangeC[-1]+1, len(wordsHead)))
    pcInt = myRange + restPCs

    space = []
    for line in tuningSpace.readlines() :
        words = line.split(',')
        if len(words) <= 1: break

        tunRow = []
        #for i in range(tuningInt[0], tuningInt[1]+1) :
        for i in rangeT :
            tunRow.append(float(words[i]))
        #pcRow = {}
        #for i in range(pcInt[0], pcInt[1]+1) :
        #    pcRow[wordsHead[i].rstrip()] = float(words[i])
        pcRow = []
        #for i in range(pcInt[0], pcInt[1]+1) :
        for i in pcInt :
            pcRow.append(float(words[i]))

        spaceRow = [tunRow, pcRow]
        space.append(spaceRow)

    return space

def setComputeBound():
    global REACT_TO_INST_BOTTLENECKS
    REACT_TO_INST_BOTTLENECKS = 0.5

def setMemoryBound():
    global REACT_TO_INST_BOTTLENECKS
    REACT_TO_INST_BOTTLENECKS = 0.7


####################### GPU arch. dependent functions ##########################

# analyzeBottlenecks
# analysis of bottlenecks, observes profiling counters and scores bottlenecks
# in interval <0, 1>
# GPU dependent, implemented for CUDA compute capabilities 3.0 - 7.5

def analyzeBottlenecks (countersNames, countersData, cc, multiprocessors, cores):
    bottlenecks = {}
    # analyze global memory
    if cc < 7.0 :
        DRAMutil = countersData[countersNames.index("dram_utilization")]
        DRAMldTrans = countersData[countersNames.index("dram_read_transactions")]
        DRAMstTrans = countersData[countersNames.index("dram_write_transactions")]
    else :
        DRAMutil = countersData[countersNames.index("dram__throughput.avg.pct_of_peak_sustained_elapsed")]/10.0
        DRAMldTrans = countersData[countersNames.index("dram__sectors_read.sum")]
        DRAMstTrans = countersData[countersNames.index("dram__sectors_write.sum")]
    if DRAMldTrans + DRAMstTrans > 0 and DRAMutil > 0 :
        bnDRAMRead = (DRAMldTrans / (DRAMldTrans + DRAMstTrans)) * (DRAMutil / 10.0)
        bnDRAMWrite = (DRAMstTrans / (DRAMldTrans + DRAMstTrans)) * (DRAMutil / 10.0)
    else :
        bnDRAMRead = 0
        bnDRAMWrite = 0
    bottlenecks['bnDRAMRead'] = bnDRAMRead
    bottlenecks['bnDRAMWrite'] = bnDRAMWrite

    # analyze cache system
    if cc < 7.0 :
        L2util = countersData[countersNames.index("l2_utilization")]
        L2ldTrans = countersData[countersNames.index("l2_read_transactions")]
        L2stTrans = countersData[countersNames.index("l2_write_transactions")]
        texUtil = countersData[countersNames.index("tex_utilization")]
        #texFuUtil = countersData[countersNames.index("tex_fu_utilization")]
        texTrans = countersData[countersNames.index("tex_cache_transactions")]
    else :
        L2util = countersData[countersNames.index("lts__t_sectors.avg.pct_of_peak_sustained_elapsed")]/10.0
        L2ldTrans = countersData[countersNames.index("lts__t_sectors_op_read.sum")]
        L2stTrans = countersData[countersNames.index("lts__t_sectors_op_write.sum")]
        texUtil = countersData[countersNames.index("l1tex__t_requests_pipe_lsu_mem_global_op_ld.avg.pct_of_peak_sustained_active")]/10.0
        #texFuUtil = countersData[countersNames.index("tex_fu_utilization")]
        texTrans = countersData[countersNames.index("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum")]
    bnL2Read = (L2ldTrans / (L2ldTrans + L2stTrans)) * (L2util / 10.0)
    bnL2Write = (L2stTrans / (L2ldTrans + L2stTrans)) * (L2util / 10.0)
    #bnTex = max(texUtil / 10.0, texFuUtil / 10.0)
    bnTex = texUtil / 10.0
    bottlenecks['bnL2Read'] = bnL2Read
    bottlenecks['bnL2Write'] = bnL2Write
    bottlenecks['bnTex'] = bnTex

    # analyze local (non-registers private in OpenCL) memory
    if cc < 7.0 :
        locOverhead = countersData[countersNames.index("local_memory_overhead")]
    else :
        #XXX this is highly experimental computation
        locOverhead = 100.0 * countersNames.index("l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum") / L2stTrans
    bottlenecks['bnLocal'] = (locOverhead/100.0) * max(DRAMutil/10.0, L2util/10.0, texUtil/10.0)#, texFuUtil/10.0)

    # analyze shared memory
    if cc < 7.0 :
        if cc < 4.0 :
            SMutil = countersData[countersNames.index("shared_efficiency")]
        else :
            SMutil = countersData[countersNames.index("shared_utilization")]
        SMldTrans = countersData[countersNames.index("shared_load_transactions")]
        SMstTrans = countersData[countersNames.index("shared_store_transactions")]
    else :
        SMutil = countersData[countersNames.index("l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed")]/10.0
        SMldTrans = countersData[countersNames.index("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum")]
        SMstTrans = countersData[countersNames.index("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum")]

    if (SMldTrans + SMstTrans > 0):
        bnSMRead = (SMldTrans / (SMldTrans + SMstTrans)) * (SMutil / 10.0)
        bnSMWrite = (SMstTrans / (SMldTrans + SMstTrans)) * (SMutil / 10.0)
    else:
        bnSMRead = 0;
        bnSMWrite = 0;
    bottlenecks['bnSMRead'] = bnSMRead
    bottlenecks['bnSMWrite'] = bnSMWrite

    # analyze multiprocessor parallelism
    if cc < 7.0 :
        occupancy = countersData[countersNames.index("achieved_occupancy")]
    else :
        occupancy = countersData[countersNames.index("sm__warps_active.avg.pct_of_peak_sustained_active")]/100.0
    bnMPparal = 1.0 - occupancy
    bottlenecks['bnMPparal'] = bnMPparal

    # analyze global parallelism
    if cc < 7.0 :
        smEfficiency = countersData[countersNames.index("sm_efficiency")]
    else :
        smEfficiency = countersData[countersNames.index("smsp__cycles_active.avg.pct_of_peak_sustained_elapsed")]
    bnGparal = (100.0 - smEfficiency) / 100.0
    bottlenecks['bnGparal'] = bnGparal

    threadBlocks = countersData[countersNames.index("Global size")] / countersData[countersNames.index("Local size")]
    bnTailEffect = 1 - (threadBlocks / (((threadBlocks + multiprocessors-1) / multiprocessors) * multiprocessors))
    bottlenecks['bnTailEffect'] = bnTailEffect
    #print(bnTailEffect, threadBlocks, countersData[countersNames.index("Global size")], countersData[countersNames.index("Local size")])

    bnThreads = max(0, (cores * 5 - countersData[countersNames.index("Global size")]) / (cores * 5))
    bottlenecks['bnThreads'] = bnThreads

    # analyze instructions
    # insctruction counts
    if cc < 7.0 :
        spInstr = countersData[countersNames.index("inst_fp_32")]
        dpInstr = countersData[countersNames.index("inst_fp_64")]
        intInstr = countersData[countersNames.index("inst_integer")]
        #commInstr = countersData[countersNames.index("inst_inter_thread_communication")]
        miscInstr = countersData[countersNames.index("inst_misc")]
        ldstInstr = countersData[countersNames.index("inst_compute_ld_st")]
        ctrlInst = countersData[countersNames.index("inst_control")]
        bconvInstr = countersData[countersNames.index("inst_bit_convert")]
        execInstr = countersData[countersNames.index("inst_executed")]
    else :
        spInstr = countersData[countersNames.index("smsp__sass_thread_inst_executed_op_fp32_pred_on.sum")]
        dpInstr = countersData[countersNames.index("smsp__sass_thread_inst_executed_op_fp64_pred_on.sum")]
        intInstr = countersData[countersNames.index("smsp__sass_thread_inst_executed_op_integer_pred_on.sum")]
        #commInstr = countersData[countersNames.index("inst_inter_thread_communication")]
        miscInstr = countersData[countersNames.index("smsp__sass_thread_inst_executed_op_misc_pred_on.sum")]
        ldstInstr = countersData[countersNames.index("smsp__sass_thread_inst_executed_op_memory_pred_on.sum")]
        ctrlInst = countersData[countersNames.index("smsp__sass_thread_inst_executed_op_control_pred_on.sum")]
        bconvInstr = countersData[countersNames.index("smsp__sass_thread_inst_executed_op_conversion_pred_on.sum")]
        execInstr = countersData[countersNames.index("smsp__inst_executed.sum")]

    #instruction utilization
    if cc < 7.0  :
        if cc < 4.0 :
            spUtil = countersData[countersNames.index("flop_sp_efficiency")]
            dpUtil = countersData[countersNames.index("flop_dp_efficiency")]
            sfuUtil = 0 #XXX we don't have this counter
        else :
            spUtil = countersData[countersNames.index("single_precision_fu_utilization")]
            dpUtil = countersData[countersNames.index("double_precision_fu_utilization")]
            sfuUtil = countersData[countersNames.index("special_fu_utilization")]
        cfUtil = countersData[countersNames.index("cf_fu_utilization")]
        ldstUtil = countersData[countersNames.index("ldst_fu_utilization")]
        texFuUtil = countersData[countersNames.index("tex_fu_utilization")]
        instrSlotUtil = countersData[countersNames.index("issue_slot_utilization")]
        if cc >= 4.0 :
            instrEffExec = countersData[countersNames.index("warp_execution_efficiency")]
            instrEffPred = countersData[countersNames.index("warp_nonpred_execution_efficiency")]
        else :
            instrEffExec = 100
            instrEffPred = 100
    else :
        spUtil = countersData[countersNames.index("smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active")]/10.0
        dpUtil = countersData[countersNames.index("smsp__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active")]/10.0
        sfuUtil = countersData[countersNames.index("smsp__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active")]/10.0
        cfUtil = 0.0 #XXX we don't have this counter
        ldstUtil = countersData[countersNames.index("smsp__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active")]/10.0
        texFuUtil = countersData[countersNames.index("smsp__inst_executed_pipe_tex.avg.pct_of_peak_sustained_active")]/10.0
        instrSlotUtil = countersData[countersNames.index("smsp__issue_active.avg.pct_of_peak_sustained_active")]
        instrEffExec = countersData[countersNames.index("smsp__thread_inst_executed_per_inst_executed.ratio")]*100.0/32.0
        instrEffPred = countersData[countersNames.index("smsp__thread_inst_executed_per_inst_executed.pct")]

    instrExecFitted = execInstr*32.0 * (100.0/instrEffExec) * (100.0/instrEffPred) #XXX this should be equal to spInstr+dpInstr+intInstr+miscInstr+ldstInstr+ctrlInst+bconvInstr
    if cc < 7.0 :
        instrUtilFitted = instrSlotUtil/100.0
    else :
        instrUtilFitted = min(1.0, instrSlotUtil/50.0) # dual-issue causes max 50% utilization of instruction of single type

    spUtilApprox = (spInstr/instrExecFitted) * instrUtilFitted
    dpUtilApprox = (dpInstr/instrExecFitted) * instrUtilFitted
    ldstUtilApprox = (ldstInstr/instrExecFitted) * instrUtilFitted
    cfUtilApprox = (ctrlInst/instrExecFitted) * instrUtilFitted
    intUtilApprox = (intInstr/instrExecFitted) * instrUtilFitted
    miscUtilApprox = (miscInstr/instrExecFitted) * instrUtilFitted
    bconvUtilApprox = (bconvInstr/instrExecFitted) * instrUtilFitted

    #print("single_precision_fu_utilization reported/computed: ", spUtil, (spInstr/instrExecFitted) * instrUtilFitted)
#    #workaround is to bottleneck instructions only if utilization is significant
#    maxUtil = max(spUtil, dpUtil, cfUtil)
#    maxUtilInst = max(spInstr, dpInstr, ctrlInst) #XXX should select the same category as the line above
#    if maxUtil > 6:
#        intUtilApprox = intInstr/maxUtilInst * maxUtil
#        miscUtilApprox = miscInstr/maxUtilInst * maxUtil
#        bconvUtilApprox = bconvInstr/maxUtilInst * maxUtil
#    else:
#        intUtilApprox = 0.0
#        miscUtilApprox = 0.0
#        bconvUtilApprox = 0.0

#    bnSP = spUtil/10.0
    bnSP = spUtilApprox
#    bnDP = dpUtil/10.0
    bnDP = dpUtilApprox
    bnSFU = sfuUtil/10.0
#    bnCF = cfUtil/10.0
    bnCF = cfUtilApprox
#    bnLDST = ldstUtil/10.0
    bnLDST = ldstUtilApprox
    bnTexFu = texFuUtil/10.0
    bnInt = intUtilApprox
    bnMisc = miscUtilApprox
    bnBconv = bconvUtilApprox

    bottlenecks['bnSP'] = bnSP
    bottlenecks['bnDP'] = bnDP
    bottlenecks['bnSFU'] = bnSFU
    bottlenecks['bnCF'] = bnCF
    bottlenecks['bnLDST'] = bnLDST
    bottlenecks['bnTexFu'] = bnTexFu
    bottlenecks['bnInt'] = bnInt
    bottlenecks['bnMisc'] = bnMisc
    bottlenecks['bnBconv'] = bnBconv

    issueWeight = 0.0
    maxInstrUtil = max(spUtilApprox/instrUtilFitted, dpUtilApprox/instrUtilFitted, sfuUtil/10.0, cfUtilApprox/instrUtilFitted, ldstUtilApprox/instrUtilFitted, intUtilApprox/instrUtilFitted, miscUtilApprox/instrUtilFitted, bconvUtilApprox/instrUtilFitted)
    if maxInstrUtil > REACT_TO_INST_BOTTLENECKS :
        issueWeight = (maxInstrUtil -  REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)
    bnInstIssue = (100.0 - instrSlotUtil) / 100 * issueWeight
    #bnInstIssue = (100.0 - instrSlotUtil) / 100 * max(spUtilApprox/instrUtilFitted, dpUtilApprox/instrUtilFitted, sfuUtil/instrUtilFitted/10.0, cfUtilApprox/instrUtilFitted, ldstUtilApprox/instrUtilFitted, texFuUtil/instrUtilFitted/10.0, intUtilApprox/instrUtilFitted, miscUtilApprox/instrUtilFitted, bconvUtilApprox/instrUtilFitted)
    bottlenecks['bnInstIssue'] = bnInstIssue

    if VERBOSE > 1 :
        print("bottlenecks", bottlenecks)

    return bottlenecks

# computeChanges
# computes how to change profiling counters according to bottlenecks
# absolute value of computed changes means its importance, the sign means
# required direction (increase/decrease the counter)
# GPU dependent, implemented for CUDA compute capabilities 3.0 - 7.5
# Note: this function is separated from analyzeBottlenecks in order to manage
# portability across arch. easily (computed bottlenecks are arch. independent)

def computeChanges(bottlenecks, countersNames, cc):
    # set how important is to change particular profiling counters
    changeImportance = [0.0]*len(countersNames)

    # memory-subsystem related counters
    if cc < 7.0 :
        changeImportance[countersNames.index('dram_read_transactions')] = - bottlenecks['bnDRAMRead']
        changeImportance[countersNames.index('dram_write_transactions')] = - bottlenecks['bnDRAMWrite']
        changeImportance[countersNames.index('l2_read_transactions')] = - bottlenecks['bnL2Read']
        changeImportance[countersNames.index('l2_write_transactions')] = - bottlenecks['bnL2Write']
        changeImportance[countersNames.index('tex_cache_transactions')] = - bottlenecks['bnTex']
        changeImportance[countersNames.index('local_memory_overhead')] = - bottlenecks['bnLocal']
        changeImportance[countersNames.index('shared_load_transactions')] = - bottlenecks['bnSMRead']
        changeImportance[countersNames.index('shared_store_transactions')] = - bottlenecks['bnSMWrite']
    else:
        changeImportance[countersNames.index('dram__sectors_read.sum')] = - bottlenecks['bnDRAMRead']
        changeImportance[countersNames.index('dram__sectors_write.sum')] = - bottlenecks['bnDRAMWrite']
        changeImportance[countersNames.index('lts__t_sectors_op_read.sum')] = - bottlenecks['bnL2Read']
        changeImportance[countersNames.index('lts__t_sectors_op_write.sum')] = - bottlenecks['bnL2Write']
        #TODO solve additive counters more elegantly?
        changeImportance[countersNames.index('l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum')] = - bottlenecks['bnTex']
        changeImportance[countersNames.index('l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum')] = - bottlenecks['bnLocal']
        changeImportance[countersNames.index('l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum')] = - bottlenecks['bnLocal']
        changeImportance[countersNames.index('l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum')] = - bottlenecks['bnSMRead']
        changeImportance[countersNames.index('l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum')] = - bottlenecks['bnSMWrite']

    # instructions related counters
    if cc < 7.0 :
        if bottlenecks['bnSP'] > REACT_TO_INST_BOTTLENECKS :
            changeImportance[countersNames.index('inst_fp_32')] = - (bottlenecks['bnSP'] - REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)
            changeImportance[countersNames.index('flop_sp_efficiency')] = (bottlenecks['bnSP'] - REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)
        if bottlenecks['bnDP'] > REACT_TO_INST_BOTTLENECKS :
            changeImportance[countersNames.index('inst_fp_64')] = - (bottlenecks['bnDP']- REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)
        #changeImportance[countersNames.index('special_fu_utilization')] = + bottlenecks['bnSFU'] #TODO how to count SFU instructions?
        if bottlenecks['bnCF'] > REACT_TO_INST_BOTTLENECKS :
            changeImportance[countersNames.index('inst_control')] = - (bottlenecks['bnCF'] - REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)
        if bottlenecks['bnLDST'] > REACT_TO_INST_BOTTLENECKS :
            changeImportance[countersNames.index('inst_compute_ld_st')] = - (bottlenecks['bnLDST']  - REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)
        #changeImportance[countersNames.index('tex_fu_utilization')] = + bottlenecks['bnTexFu']
        if bottlenecks['bnInt'] > REACT_TO_INST_BOTTLENECKS :
            changeImportance[countersNames.index('inst_integer')] = - (bottlenecks['bnInt']  - REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)
        if bottlenecks['bnMisc'] > REACT_TO_INST_BOTTLENECKS :
            changeImportance[countersNames.index('inst_misc')] = - (bottlenecks['bnMisc'] - REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)
        if bottlenecks['bnBconv'] > REACT_TO_INST_BOTTLENECKS :
            changeImportance[countersNames.index('inst_bit_convert')] = - (bottlenecks['bnBconv'] - REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)
        #if bottlenecks['bnInstIssue'] > REACT_TO_INST_BOTTLENECKS :
        changeImportance[countersNames.index('issue_slot_utilization')] = bottlenecks['bnInstIssue'] #(bottlenecks['bnInstIssue'] - REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)
    else :
        if bottlenecks['bnSP'] > REACT_TO_INST_BOTTLENECKS :
            changeImportance[countersNames.index('smsp__sass_thread_inst_executed_op_fp32_pred_on.sum')] = - bottlenecks['bnSP']
        if bottlenecks['bnDP'] > REACT_TO_INST_BOTTLENECKS :
            changeImportance[countersNames.index('smsp__sass_thread_inst_executed_op_fp64_pred_on.sum')] = - bottlenecks['bnDP']
        #changeImportance[countersNames.index('special_fu_utilization')] = + bottlenecks['bnSFU'] #TODO how to count SFU instructions?
        if bottlenecks['bnCF'] > REACT_TO_INST_BOTTLENECKS :
            changeImportance[countersNames.index('smsp__sass_thread_inst_executed_op_control_pred_on.sum')] = - (bottlenecks['bnCF'] - REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)
        if bottlenecks['bnLDST'] > REACT_TO_INST_BOTTLENECKS :
            changeImportance[countersNames.index('smsp__sass_thread_inst_executed_op_memory_pred_on.sum')] = - (bottlenecks['bnLDST']  - REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)
        #changeImportance[countersNames.index('tex_fu_utilization')] = + bottlenecks['bnTexFu']
        if bottlenecks['bnInt'] > REACT_TO_INST_BOTTLENECKS :
            changeImportance[countersNames.index('smsp__sass_thread_inst_executed_op_integer_pred_on.sum')] = - (bottlenecks['bnInt']  - REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)
        if bottlenecks['bnMisc'] > REACT_TO_INST_BOTTLENECKS :
            changeImportance[countersNames.index('smsp__sass_thread_inst_executed_op_misc_pred_on.sum')] = - (bottlenecks['bnMisc'] - REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)
        if bottlenecks['bnBconv'] > REACT_TO_INST_BOTTLENECKS :
            changeImportance[countersNames.index('smsp__sass_thread_inst_executed_op_conversion_pred_on.sum')] = - (bottlenecks['bnBconv'] - REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)
        #if bottlenecks['bnInstIssue'] > REACT_TO_INST_BOTTLENECKS :
        changeImportance[countersNames.index('smsp__issue_active.avg.pct_of_peak_sustained_active')] = bottlenecks['bnInstIssue'] #(bottlenecks['bnInstIssue'] - REACT_TO_INST_BOTTLENECKS) / (1.0 - REACT_TO_INST_BOTTLENECKS)

    #parallelism related counters
    if cc < 7.0 :
        changeImportance[countersNames.index('sm_efficiency')] = bottlenecks['bnGparal']
    else :
        changeImportance[countersNames.index('smsp__cycles_active.avg.pct_of_peak_sustained_elapsed')] = bottlenecks['bnGparal']

    changeImportance[countersNames.index('Global size')] = bottlenecks['bnThreads']#bottlenecks['bnTailEffect'] + bottlenecks['bnThreads']
    #changeImportance[countersNames.index('Local size')] = - bottlenecks['bnTailEffect'] / 2

    if VERBOSE > 1 :
        print("changeImportance", changeImportance)

    return changeImportance

###################### GPU arch. independent functions #########################

# scoreTuningConfigurationsStats
# scores all tuning configurations according to required changes of profiling
# counters and expected effect of the tuning parameters to profiling counters
# GPU independent
# This version uses offline computed statistics (average corr and var)

def scoreTuningConfigurations(changeImportance, tuningparamsNames, actualConf, tuningSpace, correlations, variations, scoreDistrib):
    newScoreDistrib = [0.0] * len(tuningSpace)

    # get maximal variations (could be moved)
    maxVariations = [0.0] * len(variations[tuningparamsNames[0]])
    for i in range(0, len(variations[tuningparamsNames[0]])) :
        for j in range(0, len(tuningparamsNames)) :
            maxVariations[i] = max(maxVariations[i], variations[tuningparamsNames[j]][i])

    # compute changes size and direction of tuning parameters
    changes = []
    for i in range(0, len(tuningparamsNames)) :
        var = 0.0
        corr = 0.0
        for j in range(0, len(changeImportance)) :
            if maxVariations[j] > 0.0 :
                tmp = abs(changeImportance[j]) * variations[tuningparamsNames[i]][j] / maxVariations[j]
            else :
                tmp = 0.0
            if tmp != 0.0:
                var = var + tmp
                sign = 1.0
                if (changeImportance[j] < 0) :
                    sign = -1.0
                corr = corr + (correlations[tuningparamsNames[i]][j] + CORR_SHIFT)/(1.0 + CORR_SHIFT) * sign * tmp
        if (var > 0.0) :
            corr = corr / abs(var)
        else :
            corr = 0.0
        change = []
        change.append(var)
        change.append(corr)
        changes.append(change)
    if VERBOSE > 1 :
        print("changes", changes)

    # compute scores according to proposed changes
    # for each point in tuning space
    for i in range(0, len(tuningSpace)) :
        # for each tuning parameter
        for j in range(0, len(tuningSpace[0])) :
            # direction from actual configuration
            direction = tuningSpace[i][j] - actualConf[j]
            #print(tuningData[i][1], actualConf)
            if direction > 0 :
                direction = 1
            if direction < 0 :
                direction = -1
            s = direction * changes[j][0] * changes[j][1]
            #print(i, direction, changes[j], s)
            newScoreDistrib[i] = newScoreDistrib[i] + s

    minScore = min(newScoreDistrib)
    maxScore = max(newScoreDistrib)
    if VERBOSE > 0 :
        print("scoreDistrib interval: ", minScore, maxScore)
    for i in range(0, len(tuningSpace)) :
        if newScoreDistrib[i] < CUTOFF :
            newScoreDistrib[i] = 0.0
        else :
            if newScoreDistrib[i] < 0.0 :
                newScoreDistrib[i] = 1.0 - (newScoreDistrib[i] / minScore)
            else :
                if newScoreDistrib[i] > 0.0 :
                    newScoreDistrib[i] = 1.0 + (newScoreDistrib[i] / maxScore)
            newScoreDistrib[i] = newScoreDistrib[i]**EXP
            if newScoreDistrib[i] == 0.0 :
                newScoreDistrib[i] = 0.0001

        # if was 0, set to 0 (explored)
        if scoreDistrib[i] == 0.0 :
            newScoreDistrib[i] = 0.0

    if VERBOSE > 2 :
        print("newScoreDistrib", newScoreDistrib)

    return newScoreDistrib

# scoreTuningConfigurationsExact
# scores all tuning configurations according to required changes of profiling
# counters and expected effect of the tuning parameters to profiling counters
# GPU independent
# This version uses completely computed offline space

def scoreTuningConfigurationsExact(changeImportance, tuningparamsNames, actualConf, tuningSpace, completeMapping, scoreDistrib):
    newScoreDistrib = [0.0] * len(tuningSpace)
    #search index of actualConf in completeMapping (some conf. can be missing, therefore, we need to check tuning parameters)
    actualPC = []
    for conf in completeMapping :
        if actualConf == conf[0] :
            actualPC = conf[1]
    if len(actualPC) == 0 :
        # the configuration is not known in the completeMapping, return uniform distrib
        for i in range(0, len(tuningSpace)) :
            uniformScoreDistrib = [1.0] * len(tuningSpace)
            if scoreDistrib[i] == 0.0 :
                uniformScoreDistrib[i] = 0.0
        return uniformScoreDistrib

    cmIdx = 0
    # for each tuning configuration
    for i in range(0, len(tuningSpace)) :
        #seek for equivalent tuning configuration in the completeMapping
        #TODO this implementation assumes the same order of tuning configurations, create mapping between indexes instead
        myPC = []
        for j in range(cmIdx, len(completeMapping)) :
            if (tuningSpace[i] == completeMapping[j][0]) :
                myPC = completeMapping[j][1]
                cmIdx = j+1
                break
        if (len(myPC) == 0) :
            newScoreDistrib[i] = 0.0
        else :
            #score configuration
            for j in range(0, len(changeImportance)) :
                try:
                    newScoreDistrib[i] = newScoreDistrib[i] + changeImportance[j] * (myPC[j] - actualPC[j]) / (myPC[j]+actualPC[j])
                except ZeroDivisionError :
                        newScoreDistrib[i] = newScoreDistrib[i] + 0.0

    minScore = min(newScoreDistrib)
    maxScore = max(newScoreDistrib)
    if VERBOSE > 0 :
        print("scoreDistrib interval: ", minScore, maxScore)
    for i in range(0, len(tuningSpace)) :
        if newScoreDistrib[i] < CUTOFF :
            newScoreDistrib[i] = 0.0
        else :
            if newScoreDistrib[i] < 0.0 :
                newScoreDistrib[i] = 1.0 - (newScoreDistrib[i] / minScore)
            else :
                if newScoreDistrib[i] > 0.0 :
                    newScoreDistrib[i] = 1.0 + (newScoreDistrib[i] / maxScore)
            newScoreDistrib[i] = newScoreDistrib[i]**EXP
            if newScoreDistrib[i] == 0.0 :
                newScoreDistrib[i] = 0.0001

        # if was 0, set to 0 (explored)
        if scoreDistrib[i] == 0.0 :
            newScoreDistrib[i] = 0.0

    if VERBOSE > 2 :
        print("newScoreDistrib", newScoreDistrib)

    return newScoreDistrib


# makePredictions
# calculates predicted PC values for all tuning configurations and returns them as completeMapping
# first, it encodes the values of tuning parameters
# second, it finds the suitable models of profiling counters by evaluating the conditions. as there are multiple non-linear models, one for each combination of values of binary parameters, we need to find which one is applicable to this tuning configuration and its combination fo values of binary parameters. tis is true if the conditionsAllModels is satisfied
# third, it calculates the predicted values of profiling counters

def makePredictions(tuningSpace, TPassignments, conditionsAllModels, PCassignmentsAllModels):

    completeMapping = []
    # for each tuning configuration
    for i in range(0, len(tuningSpace)) :
        #evaluate the predictions for all tuning configurations
        #encode tuning parameters, the order of TP is the same, we sorted earlier
        k = list(TPassignments)
        for j in range(0, len(k)) :
            exec(TPassignments[k[j]].replace(k[j], str(tuningSpace[i][j])))
        applicableModel = -1
        for j in range(0, len(conditionsAllModels)) :
            #find the first model where the condition is satisfied
            if (eval(conditionsAllModels[j])) :
                applicableModel = j
                break

        if (applicableModel == -1):
            #we have not found an applicable model
            continue

        #evaluate the prediction formulas
        #TODO this implementation assumes the same order of PC
        PCassignments = PCassignmentsAllModels[applicableModel]
        k = list(PCassignments)
        myPC = []
        for j in range(0, len(k)) :
            myPC.append(eval(PCassignments[k[j]]))
        completeMapping.append([tuningSpace[i], myPC])

    return completeMapping

# randomSearchStep
# perform one step of random search (without memory)

def randomSearchStep(tuningSpaceSize) :
    return int(random.random() * tuningSpaceSize)

# weightedRandomSearchStep
# perform one step of random search using weighted probability based on
# profiling counters

def weightedRandomSearchStep(scoreDistrib, tuningSpaceSize) :
    if (sum(scoreDistrib) == 0.0) :
        print("Weighted search error: no more tuning configurations.")
        return randomSearchStep(tuningSpaceSize)

    rnd = random.random() * sum(scoreDistrib)
    idx = 0
    tmp = 0.0
    for j in range (0, tuningSpaceSize):
        tmp = tmp + scoreDistrib[j]
        if rnd < tmp : break
        idx = idx + 1
    return idx

def loadMLModel(trainedKnowledgeBase):
    return pickle.load(open(trainedKnowledgeBase, 'rb'))

########################### Temp function
def scoreTuningConfigurationsPredictor(changeImportance, tuningparamsNames, actualConf, tuningSpace, scoreDistrib, loaded_model):
    def mulfunc(a, b, c):
        if (a * (b - c)) > 0.0:
            return 1.0
        if (a * (b - c)) < 0.0:
            return -1.0
        else:
            return 0.0

    newScoreDistrib = [0.0] * len(tuningSpace)
    actualPC = []

    # Using ML predictor
    predictedPC = loaded_model.predict([actualConf])
    actualPC = list(predictedPC.flatten())

    if len(actualPC) == 0 :
        for i in range(0, len(tuningSpace)) :
            uniformScoreDistrib = [1.0] * len(tuningSpace)
            if scoreDistrib[i] == 0.0 :
                uniformScoreDistrib[i] = 0.0
        return uniformScoreDistrib

    cmIdx = 0

    #################################################### Using ML predictor
    predictedMyPC = loaded_model.predict(tuningSpace)
    predictedMyPC1 = np.array(predictedMyPC)
    actualPC1 = np.array(actualPC)
    n = len(changeImportance) - len(actualPC1)
    changeImportance = changeImportance[:len(changeImportance)-n]
    changeImportance1 = np.array(changeImportance)

    vfunc = np.vectorize(mulfunc)
    mul = vfunc(changeImportance1, predictedMyPC1, actualPC1)
    res = np.array(mul * abs(changeImportance1 * 2.0 * (predictedMyPC1 - actualPC1) / (predictedMyPC1+actualPC1)))
    res = np.nan_to_num(res)
    newScoreDistrib = res.sum(axis=1)


    minScore = min(newScoreDistrib)
    maxScore = max(newScoreDistrib)
    if VERBOSE > 0 :
        print("scoreDistrib interval: ", minScore, maxScore)
    for i in range(0, len(tuningSpace)) :
        if newScoreDistrib[i] < CUTOFF :
            newScoreDistrib[i] = 0.0
        else :
            if newScoreDistrib[i] < 0.0 :
                newScoreDistrib[i] = 1.0 - (newScoreDistrib[i] / minScore)
            else :
                if newScoreDistrib[i] > 0.0 :
                    newScoreDistrib[i] = 1.0 + (newScoreDistrib[i] / maxScore)
            newScoreDistrib[i] = newScoreDistrib[i]**EXP
            if newScoreDistrib[i] == 0.0 :
                newScoreDistrib[i] = 0.0001

        # if was 0, set to 0 (explored)
        if scoreDistrib[i] == 0.0 :
            newScoreDistrib[i] = 0.0

    if VERBOSE > 2 :
        print("Predictro newScoreDistrib", newScoreDistrib)

    return newScoreDistrib

# auxiliary functions
def readPCList (filename) :
    ret = []
    pcListFile = open(filename, 'r')
    if pcListFile.readline().rstrip() != 'Profiling counter' :
        print('Malformed PC list file!')
        return ret
    for line in pcListFile.readlines():
        ret.append(line.rstrip())
    pcListFile.close()

    return ret

class PyProfilingSearcher(ktt.Searcher):
    ccMajor = 0
    ccMinor = 0
    cc = 0
    multiprocessors = 0
    profilingCountersModel = 0
    bestDuration = -1
    bestConf = None
    preselectedBatch = []
    tuningParamsNames = []
    currentConfiguration = ktt.KernelConfiguration()
    tuner = None
    model = None

    def __init__(self):
        ktt.Searcher.__init__(self)

    def OnInitialize(self):
        for i in range(0, BATCH) :
            self.preselectedBatch.append(self.GetRandomConfiguration())
        self.currentConfiguration = self.preselectedBatch[0]
        self.preselectedBatch.pop(0)

        tp = self.currentConfiguration.GetPairs()
        for p in tp :
            self.tuningParamsNames.append(p.GetName())

    def Configure(self, tuner, modelFile):
        self.tuner = tuner
        self.ccMajor = tuner.GetCurrentDeviceInfo().GetCudaComputeCapabilityMajor()
        self.ccMinor = tuner.GetCurrentDeviceInfo().GetCudaComputeCapabilityMinor()
        self.cc = self.ccMajor + round(0.1 * self.ccMinor, 1)
        self.multiprocessors = tuner.GetCurrentDeviceInfo().GetMaxComputeUnits()

        self.profilingCountersModel = readPCList(modelFile + ".pc")
        self.model = loadMLModel(modelFile)

    def GetUniqueConfigurations(self, configurations):
        uniqueConfigurations = []
        indicesConfigurations = []
        uniqueIndicesConfigurations = []
        for c in configurations:
            indicesConfigurations.append(self.GetIndex(c))
        uniqueIndicesConfigurations = list(set(indicesConfigurations))

        for i in uniqueIndicesConfigurations:
            uniqueConfigurations.append(self.GetConfiguration(i))
        return uniqueConfigurations

    def CalculateNextConfiguration(self, previousResult):
        # select the new configuration
        if len(self.preselectedBatch) > 0:
            # we are testing current batch
            if (self.bestConf == None) or (previousResult.GetKernelDuration() < self.bestDuration) :
                self.bestDuration = previousResult.GetKernelDuration()
                self.bestConf = self.currentConfiguration
                if VERBOSE > 2:
                    print("Found new best configuration ", self.bestDuration, self.bestConf, flush = True)
            if VERBOSE > 2:
                print("PreselectedBatch has ", len(self.preselectedBatch), " remaining items:")
                ind = []
                for c in self.preselectedBatch :
                    ind.append(self.GetIndex(c))
                print(ind, flush = True)
            self.currentConfiguration = self.preselectedBatch.pop(0)
        else :
            if VERBOSE > 2:
                print("Preselected batch empty, running profiling.", flush = True)
            if self.bestDuration != -1 :
                # we run the fastest one once again, but with profiling
                self.currentConfiguration = self.bestConf
                self.bestDuration = -1
                self.tuner.SetProfiling(True)
            else :
                # get PCs from the last tuning run
                if len(previousResult.GetResults()) > 1:
                    print("Warning: this version of profile-based searcher does not support searching kernels collections. Using counters from kernels 0 only.")
                globalSize = previousResult.GetResults()[0].GetGlobalSize()
                localSize = previousResult.GetResults()[0].GetLocalSize()
                profilingCountersRun = previousResult.GetResults()[0].GetProfilingData().GetCounters() #FIXME this supposes there is no composition profiled
                pcNames = ["Global size", "Local size"]
                pcVals = [globalSize.GetTotalSize(), localSize.GetTotalSize()]
                for pd in profilingCountersRun :
                    pcNames.append(pd.GetName())
                    if (pd.GetType() == ktt.ProfilingCounterType.Int) :
                        pcVals.append(pd.GetValueInt())
                    elif (pd.GetType() == ktt.ProfilingCounterType.UnsignedInt) or (pd.GetType() == ktt.ProfilingCounterType.Throughput) or (pd.GetType() == ktt.ProfilingCounterType.UtilizationLevel):
                        pcVals.append(pd.GetValueUint())
                    elif (pd.GetType() == ktt.ProfilingCounterType.Double) or (pd.GetType() == ktt.ProfilingCounterType.Percent) :
                        pcVals.append(pd.GetValueDouble())
                    else :
                        print("Fatal error, unsupported PC value passed to profile-based searcher!")
                        exit(1)

                # select candidate configurations according to position of the best one plus some random sample
                candidates = self.GetNeighbourConfigurations(self.bestConf, 2, 100)
                for i in range (0, 10) :
                    candidates.append(self.GetRandomConfiguration())
                candidates = self.GetUniqueConfigurations(candidates)
                print("Profile-based searcher: evaluating model for " + str(len(candidates)) + " candidates...", flush = True)

                # get tuning space from candidates
                candidatesTuningSpace = []
                for c in candidates :
                    tp = c.GetPairs()
                    candidateParams = []
                    for p in tp :
                        candidateParams.append(p.GetValue())
                    candidatesTuningSpace.append(candidateParams)
                myTuningSpace = []
                tp = self.bestConf.GetPairs()
                for p in tp :
                    myTuningSpace.append(p.GetValue())

                # score the configurations
                scoreDistrib = [1.0]*len(candidates)
                bottlenecks = analyzeBottlenecks(pcNames, pcVals, 6.1, self.multiprocessors, self.convertSM2Cores() * self.multiprocessors)
                changes = computeChanges(bottlenecks, self.profilingCountersModel, self.cc)
                scoreDistrib = scoreTuningConfigurationsPredictor(changes, self.tuningParamsNames, myTuningSpace, candidatesTuningSpace, scoreDistrib, self.model)

                # select next batch
                selectedIndices = []
                if len(candidates) > BATCH :
                    numInBatch = 0
                    while numInBatch < BATCH:
                        idx = weightedRandomSearchStep(scoreDistrib, len(candidates))
                        if selectedIndices == [] or idx not in selectedIndices:
                        #check if we have not chosen the same configuration in previous iterations
                            self.preselectedBatch.append(candidates[idx])
                            selectedIndices.append(idx)
                            numInBatch = numInBatch + 1
                else:
                    for i in range(0, len(candidates)):
                        self.preselectedBatch.append(candidates[i])

                if VERBOSE == 3:
                    print("Turning off profiling, new batch selected with length ", len(self.preselectedBatch), " with configurations ")
                ind = []
                for c in self.preselectedBatch :
                    ind.append(self.GetIndex(c))
                print(ind, flush = True)
                self.currentConfiguration = self.preselectedBatch[0]
                self.preselectedBatch.pop(0)
                self.bestConf = None
                self.tuner.SetProfiling(False)

        return True

    def GetCurrentConfiguration(self):
        return self.currentConfiguration

    def convertSM2Cores(self):
        smToCoresDict = {
            0x30: 192,
            0x32: 192,
            0x35: 192,
            0x37: 192,
            0x50: 128,
            0x52: 128,
            0x53: 128,
            0x60: 64,
            0x61: 128,
            0x62: 128,
            0x70: 64,
            0x72: 64,
            0x75: 64,
            0x80: 64,
            0x86: 64
        }
        defaultSM = 64

        compact = (self.ccMajor << 4) + self.ccMinor
        if compact in smToCoresDict:
            return smToCoresDict[compact]
        else:
            print("Warning: unknown number of cores for SM " + str(self.ccMajor) + "." + str(self.ccMinor) + ", using default value of " + str(defaultSM))
            return defaultSM

def executeSearcher(tuner, kernel, model):
    searcher = PyProfilingSearcher()
    tuner.SetSearcher(kernel, searcher)
    searcher.Configure(tuner, model)
