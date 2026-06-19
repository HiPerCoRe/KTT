import re
from collections.abc import Callable
from enum import Enum


class CounterType(Enum):
    Stress = 0
    Operations = 1


class ProfilingCounter(Enum):
    UNKNOWN = 'UNKNOWN'

    # Hardware counters
    DRAM_RT = 'DRAM_RT'
    DRAM_WT = 'DRAM_WT'
    L2_RT = 'L2_RT'
    L2_WT = 'L2_WT'
    TEX_RWT = 'TEX_RWT'
    LOC_O = 'LOC_O'
    SHR_LT = 'SHR_LT'
    SHR_WT = 'SHR_WT'
    INST_F32 = 'INST_F32'
    INST_F64 = 'INST_F64'
    INST_INT = 'INST_INT'
    INST_MISC = 'INST_MISC'
    INST_LDST = 'INST_LDST'
    INST_CONT = 'INST_CONT'
    INST_BCONV = 'INST_BCONV'
    INST_EXE = 'INST_EXE'
    INST_ISSUE_U = 'INST_ISSUE_U'
    DRAM_U = 'DRAM_U'
    L2_U = 'L2_U'
    TEX_U = 'TEX_U'
    SHR_U = 'SHR_U'
    SM_E = 'SM_E'
    WARP_E = 'WARP_E'
    WARP_NP_E = 'WARP_NP_E'

    # Additional counters
    SM_WARP_U = 'SM_WARP_U'
    FMA_U = 'FMA_U'
    FP64_E = 'FP64_E'
    XU_E = 'XU_E'
    LSU_E = 'LSU_E'
    TEX_E = 'TEX_E'
    FU_CF_U = 'FU_CF_U'
    FU_RW_U = 'FU_RW_U'
    FU_TEX_U = 'FU_TEX_U'
    SRM_E = 'SRM_E'
    SP_E = 'SP_E'
    DP_E = 'DP_E'

    # Compiler counters
    GLOBAL_SIZE = 'GLOBAL_SIZE'
    LOCAL_SIZE = 'LOCAL_SIZE'
    LOCAL_MEMORY_SIZE = 'LOCAL_MEMORY_SIZE'
    MAX_WORK_GROUP_SIZE = 'MAX_WORK_GROUP_SIZE'
    PRIVATE_MEMORY_SIZE = 'PRIVATE_MEMORY_SIZE'
    CONST_MEMORY_SIZE = 'CONST_MEMORY_SIZE'
    REGISTERS_COUNT = 'REGISTERS_COUNT'

    # Artificial counters
    PARALLELISM = 'PARALLELISM'
    DRAM_WT_U = 'DRAM_WT_U'
    DRAM_RT_U = 'DRAM_RT_U'

    L2_WT_U = 'L2_WT_U'
    L2_RT_U = 'L2_RT_U'

    SHR_WT_U = 'SHR_WT_U'
    SHR_RT_U = 'SHR_RT_U'

    def __lt__(self, other: 'ProfilingCounter'):  # for sorting
        return self.value < other.value


# Vim macros are GOOD
_counterNameMappings = {
    'Global size': ProfilingCounter.GLOBAL_SIZE,
    'Local size': ProfilingCounter.LOCAL_SIZE,
    'Local memory size': ProfilingCounter.LOCAL_MEMORY_SIZE,
    'Maximum work-group size': ProfilingCounter.MAX_WORK_GROUP_SIZE,
    'Private memory size': ProfilingCounter.PRIVATE_MEMORY_SIZE,
    'Constant memory size': ProfilingCounter.CONST_MEMORY_SIZE,
    'Registers count': ProfilingCounter.REGISTERS_COUNT,
    'dram_read_transactions': ProfilingCounter.DRAM_RT,
    'dram__sectors_read.sum': ProfilingCounter.DRAM_RT,
    'dram_write_transactions': ProfilingCounter.DRAM_WT,
    'dram__sectors_write.sum': ProfilingCounter.DRAM_WT,
    'l2_read_transactions': ProfilingCounter.L2_RT,
    'lts__t_sectors_op_read.sum': ProfilingCounter.L2_RT,
    'l2_write_transactions': ProfilingCounter.L2_WT,
    'lts__t_sectors_op_write.sum': ProfilingCounter.L2_WT,
    'tex_cache_transactions': ProfilingCounter.TEX_RWT,
    'l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum': ProfilingCounter.TEX_RWT,
    'local_memory_overhead': ProfilingCounter.LOC_O,
    'l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum': ProfilingCounter.LOC_O,
    'shared_load_transactions': ProfilingCounter.SHR_LT,
    'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum': ProfilingCounter.SHR_LT,
    'shared_store_transactions': ProfilingCounter.SHR_WT,
    'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum': ProfilingCounter.SHR_WT,
    'inst_fp_32': ProfilingCounter.INST_F32,
    'smsp__sass_thread_inst_executed_op_fp32_pred_on.sum': ProfilingCounter.INST_F32,
    'inst_fp_64': ProfilingCounter.INST_F64,
    'smsp__sass_thread_inst_executed_op_fp64_pred_on.sum': ProfilingCounter.INST_F64,
    'inst_integer': ProfilingCounter.INST_INT,
    'smsp__sass_thread_inst_executed_op_integer_pred_on.sum': ProfilingCounter.INST_INT,
    'inst_misc': ProfilingCounter.INST_MISC,
    'smsp__sass_thread_inst_executed_op_misc_pred_on.sum': ProfilingCounter.INST_MISC,
    'inst_compute_ld_st': ProfilingCounter.INST_LDST,
    'smsp__sass_thread_inst_executed_op_memory_pred_on.sum': ProfilingCounter.INST_LDST,
    'inst_control': ProfilingCounter.INST_CONT,
    'smsp__sass_thread_inst_executed_op_control_pred_on.sum': ProfilingCounter.INST_CONT,
    'inst_bit_convert': ProfilingCounter.INST_BCONV,
    'smsp__sass_thread_inst_executed_op_conversion_pred_on.sum': ProfilingCounter.INST_BCONV,
    'inst_executed': ProfilingCounter.INST_EXE,
    'smsp__inst_executed.sum': ProfilingCounter.INST_EXE,
    'issue_slot_utilization': ProfilingCounter.INST_ISSUE_U,
    'smsp__issue_active.avg.pct_of_peak_sustained_active': ProfilingCounter.INST_ISSUE_U,
    'dram_utilization': ProfilingCounter.DRAM_U,
    'dram__throughput.avg.pct_of_peak_sustained_elapsed': ProfilingCounter.DRAM_U,
    'l2_utilization': ProfilingCounter.L2_U,
    'lts__t_sectors.avg.pct_of_peak_sustained_elapsed': ProfilingCounter.L2_U,
    'tex_utilization': ProfilingCounter.TEX_U,
    'l1tex__t_requests_pipe_lsu_mem_global_op_ld'
    + '.avg.pct_of_peak_sustained_active': ProfilingCounter.TEX_U,
    'shared_utilization': ProfilingCounter.SHR_U,
    'l1tex__data_pipe_lsu_wavefronts_mem_shared'
    + '.avg.pct_of_peak_sustained_elapsed': ProfilingCounter.SHR_U,
    'sm_efficiency': ProfilingCounter.SM_E,
    'smsp__cycles_active.avg.pct_of_peak_sustained_elapsed': ProfilingCounter.SM_E,
    'warp_execution_efficiency': ProfilingCounter.WARP_E,
    'smsp__thread_inst_executed_per_inst_executed.ratio': ProfilingCounter.WARP_E,
    'warp_nonpred_execution_efficiency': ProfilingCounter.WARP_NP_E,
    'smsp__thread_inst_executed_per_inst_executed.pct': ProfilingCounter.WARP_NP_E,
    # Additional counters (not from the paper)
    'sm__warps_active.avg.pct_of_peak_sustained_active': ProfilingCounter.SM_WARP_U,
    'smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active': ProfilingCounter.FMA_U,
    'smsp__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active': ProfilingCounter.FP64_E,
    'smsp__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active': ProfilingCounter.XU_E,
    'smsp__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active': ProfilingCounter.LSU_E,
    'smsp__inst_executed_pipe_tex.avg.pct_of_peak_sustained_active': ProfilingCounter.TEX_E,
    'cf_fu_utilization': ProfilingCounter.FU_CF_U,
    'ldst_fu_utilization': ProfilingCounter.FU_RW_U,
    'tex_fu_utilization': ProfilingCounter.FU_TEX_U,
    'shared_efficiency': ProfilingCounter.SRM_E,
    'flop_sp_efficiency': ProfilingCounter.SP_E,
    'flop_dp_efficiency': ProfilingCounter.DP_E,
    # Artificial
    'parallelism': ProfilingCounter.PARALLELISM,
}


_counterTypeMappings: dict[ProfilingCounter, CounterType] = {
    # Doesn't matter, really
    ProfilingCounter.UNKNOWN: CounterType.Operations,
    # Hardware counters
    ProfilingCounter.DRAM_RT: CounterType.Operations,
    ProfilingCounter.DRAM_WT: CounterType.Operations,
    ProfilingCounter.L2_RT: CounterType.Operations,
    ProfilingCounter.L2_WT: CounterType.Operations,
    ProfilingCounter.TEX_RWT: CounterType.Operations,
    ProfilingCounter.LOC_O: CounterType.Operations,
    ProfilingCounter.SHR_LT: CounterType.Operations,
    ProfilingCounter.SHR_WT: CounterType.Operations,
    ProfilingCounter.INST_F32: CounterType.Operations,
    ProfilingCounter.INST_F64: CounterType.Operations,
    ProfilingCounter.INST_INT: CounterType.Operations,
    ProfilingCounter.INST_MISC: CounterType.Operations,
    ProfilingCounter.INST_LDST: CounterType.Operations,
    ProfilingCounter.INST_CONT: CounterType.Operations,
    ProfilingCounter.INST_BCONV: CounterType.Operations,
    ProfilingCounter.INST_EXE: CounterType.Operations,
    ProfilingCounter.INST_ISSUE_U: CounterType.Stress,
    ProfilingCounter.DRAM_U: CounterType.Stress,
    ProfilingCounter.L2_U: CounterType.Stress,
    ProfilingCounter.TEX_U: CounterType.Stress,
    ProfilingCounter.SHR_U: CounterType.Stress,
    ProfilingCounter.SM_E: CounterType.Stress,
    ProfilingCounter.WARP_E: CounterType.Stress,
    ProfilingCounter.WARP_NP_E: CounterType.Stress,
    # Compiler counters
    ProfilingCounter.GLOBAL_SIZE: CounterType.Operations,
    ProfilingCounter.LOCAL_SIZE: CounterType.Operations,
    ProfilingCounter.LOCAL_MEMORY_SIZE: CounterType.Operations,
    ProfilingCounter.MAX_WORK_GROUP_SIZE: CounterType.Operations,
    ProfilingCounter.PRIVATE_MEMORY_SIZE: CounterType.Operations,
    ProfilingCounter.CONST_MEMORY_SIZE: CounterType.Operations,
    ProfilingCounter.REGISTERS_COUNT: CounterType.Operations,
    # Additional counters (not from the paper)
    ProfilingCounter.SM_WARP_U: CounterType.Stress,
    ProfilingCounter.FMA_U: CounterType.Stress,
    ProfilingCounter.FP64_E: CounterType.Stress,
    ProfilingCounter.XU_E: CounterType.Stress,
    ProfilingCounter.LSU_E: CounterType.Stress,
    ProfilingCounter.TEX_E: CounterType.Stress,
    ProfilingCounter.FU_CF_U: CounterType.Stress,
    ProfilingCounter.FU_RW_U: CounterType.Stress,
    ProfilingCounter.FU_TEX_U: CounterType.Stress,
    ProfilingCounter.SRM_E: CounterType.Stress,
    ProfilingCounter.SP_E: CounterType.Stress,
    ProfilingCounter.DP_E: CounterType.Stress,
    # Artificial counters
    ProfilingCounter.PARALLELISM: CounterType.Stress,
}


# Regular expressions and their converters
_counterValueMappings: dict[str, Callable[[float], float]] = {
    # Python dictionaries are ordered, so order is important
    r'^issue_slot_utilization$': lambda x: x / 100,
    r'^.*\.pct_of_.*$': lambda x: x / 100,
    r'^.*\.pct$': lambda x: x / 100,
    r'^.*_utilization$': lambda x: x / 10,
    r'^.*_efficiency$': lambda x: x / 100,
    r'smsp__thread_inst_executed_per_inst_executed\.ratio': lambda x: x / 32,
    r'^unknown$': lambda _: 0,
}


def _GetCounterName(counter: str) -> ProfilingCounter:
    return _counterNameMappings.get(counter, ProfilingCounter.UNKNOWN)


def GetCounterType(counter: ProfilingCounter) -> CounterType:
    return _counterTypeMappings[counter]


def _FindMapping(counter: str) -> Callable[[float], float]:
    for pattern, mapping in _counterValueMappings.items():
        if re.fullmatch(pattern, counter):
            return mapping

    return lambda x: x

def _streamingMultiprocessorsToCores(ccMajor:int, ccMinor: int):
    smCoresMapping = {
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
        0x86: 64,
    }
    defaultSM = 64

    compact = (ccMajor << 4) + ccMinor
    if compact in smCoresMapping:
        return smCoresMapping[compact]

    print(
        'Warning: unknown number of cores for SM',
        f'{ccMajor}.{ccMinor},',
        f'using default value of {defaultSM}',
    )
    return defaultSM

class CounterHeader:
    def __init__(self, counter: str):
        self.name = _GetCounterName(counter)
        self.type = GetCounterType(self.name)

        self._converter = _FindMapping(
            counter
            if self.name != ProfilingCounter.UNKNOWN
            else 'unknown'
        )

    def ConvertValue(self, value: float) -> float:
        return self._converter(value)


class ParallelismHeader(CounterHeader):
    def __init__(self, ccMajor: int, ccMinor: int):
        super().__init__('parallelism')

        cores = _streamingMultiprocessorsToCores(ccMajor, ccMinor)

        def _converter(threads: float) -> float:
            return min(1.0, threads / (5 * cores))

        self._converter = _converter
