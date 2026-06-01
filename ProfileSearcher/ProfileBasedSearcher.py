import numpy
from numpy.typing import NDArray
from pandas import DataFrame, Series

from modules.counter_mappings import (
    CounterType,
    GetCounterType,
    CounterHeader,
    ParallelismHeader,
    ProfilingCounter,
)
from lib.pyktt import KernelConfiguration, Searcher, KernelResult

from modules.info import BatchInfo, ModelInfo
from modules.tuning_space import TuningSpace
from modules.model import loadModel

# Profiler configuration (might move into the class?)
SCORE_EXPONENT = 7
SCORE_THRESHOLD = 0.425

NEIGHBOUR_DISTANCE = 2

Batch = list[KernelConfiguration]


class ProfileBasedSearcher(Searcher):
    batch: Batch = []
    currentConfiguration: KernelConfiguration = KernelConfiguration()

    # Best configuration information for the current batch
    bestConfiguration: KernelConfiguration | None = None
    bestDuration = -1

    def __init__(self) -> None:
        super().__init__()

    def Configure(self, tuner, modelInfo: ModelInfo, batchInfo: BatchInfo):
        self.tuner = tuner
        self.batchInfo = batchInfo

        self.spaceModel = loadModel(modelInfo.spacePath)
        self.counterModel = loadModel(modelInfo.counterPath)
        self.deltaModel = loadModel(modelInfo.deltaPath)

    def OnInitialize(self):
        candidateBatch = []
        self._FillBatchRandom(candidateBatch, self.batchInfo.randomSize)

        tuningSpace = self._BatchToSpacePredicted(candidateBatch)
        scores = self._ScoreTuningSpace(tuningSpace, tuningSpace.counters[0])
        self.batch = self._SelectBatchWeighted(candidateBatch, scores)

        self.currentConfiguration = self.batch.pop()

    def CalculateNextConfiguration(self, previousResult: KernelResult) -> bool:
        self._UpdateBestConfiguration(previousResult)

        if len(self.batch) != 0:
            self.currentConfiguration = self.batch.pop()
            return True

        # unlikely, the batch is emptied but no configuration ran properly
        if self.bestConfiguration is None:
            print(
                '[Warning] Searcher: Batch was emptied but no configuration ran properly'
            )

            # TODO: maybe do the same as OnInitialize() here?
            self._FillBatchRandom(self.batch, self.batchInfo.batchSize)
            self.currentConfiguration = self.batch.pop()

            return True

        if self.bestDuration != -1:  # rerun best config with profiling on
            self.currentConfiguration = self.bestConfiguration
            self.bestDuration = -1

            self.tuner.SetProfiling(True)
            return True

        if not previousResult.IsValid():
            print('[Error] Searcher: Profiling run failed, bailing out')
            self.tuner.SetProfiling(False)
            return False

        # candidate configurations are made from neighbouring and random configurations
        candidateBatch = self._GetCandidateBatch()
        if len(candidateBatch) <= self.batchInfo.batchSize:
            print(
                '[Info] Searcher: Too little configurations in the candidate batch'
            )
            print('[Info] Searcher: Skipping profiling the batch')

            self.batch = candidateBatch
            self.currentConfiguration = self.batch.pop()
            self.bestConfiguration = None
            self.tuner.SetProfiling(False)

            return True

        # we only do the profiling in case the batch is bigger than the batch size
        candidateSpace = self._BatchToSpacePredicted(candidateBatch)

        counters = self._GetCounters(previousResult)
        predictedCounters = self.counterModel.predict(counters)

        candidateScores = self._ScoreTuningSpace(
            candidateSpace, predictedCounters
        )

        self.batch = self._SelectBatchWeighted(candidateBatch, candidateScores)
        self.currentConfiguration = self.batch.pop()

        self.bestConfiguration = None
        self.tuner.SetProfiling(False)

        return True

    def GetCurrentConfiguration(self) -> KernelConfiguration:
        return self.currentConfiguration

    # Private functions

    def _SelectBatchWeighted(self, batch: Batch, scores: NDArray) -> Batch:
        selected = []

        for _ in range(self.batchInfo.batchSize):
            index = WeightedRandomStep(scores)
            selected.append(batch[index])

            scores[index] = 0.0

        selected.reverse()  # for pops to work efficiently, taking the best configs
        return selected

    def _ScoreTuningSpace(
        self, space: TuningSpace, currentCounters: NDArray
    ) -> NDArray:
        counterDeltas = currentCounters - space.counters
        timeDeltas = self.deltaModel.predict(counterDeltas).reshape(-1, 1)

        minDelta, maxDelta = timeDeltas.min(), timeDeltas.max()
        scores = (timeDeltas - minDelta) / (maxDelta - minDelta)

        scores[numpy.isnan(scores)] = 1.0  # edge case if the model is bad
        scores = numpy.power(scores, SCORE_EXPONENT)
        scores[scores < SCORE_THRESHOLD] = 0.00001

        return scores

    def _GetCounters(self, results: KernelResult) -> NDArray:
        kernelResults = results.GetResults()
        result = kernelResults[0]

        if len(kernelResults) != 1:
            print('whuh we dont support this, usin kernel 0 counters')

        counters = self._ExtractCounters(result)

        artificialCounters = self._GenerateArtificialCounters(counters)
        stressCounters = counters.loc[
            :,
            [GetCounterType(c) == CounterType.Stress for c in counters.columns],
        ]

        relevantCounters = stressCounters.join(artificialCounters)
        relevantCounters = relevantCounters.reindex(
            sorted(relevantCounters.columns), axis=1
        )

        return relevantCounters.to_numpy()

    def _GenerateArtificialCounters(self, counters: DataFrame) -> DataFrame:
        dramUtilization = _GetRWUtilization(
            counters,
            ProfilingCounter.DRAM_RT,
            ProfilingCounter.DRAM_WT,
            ProfilingCounter.DRAM_U,
        )

        l2Utilization = _GetRWUtilization(
            counters,
            ProfilingCounter.L2_RT,
            ProfilingCounter.L2_WT,
            ProfilingCounter.L2_U,
        )

        smUtilCounter = (
            ProfilingCounter.SHR_U
            if ProfilingCounter.SHR_U in counters.columns
            else ProfilingCounter.SM_E
        )
        smUtilization = _GetRWUtilization(
            counters,
            ProfilingCounter.SHR_LT,
            ProfilingCounter.SHR_WT,
            smUtilCounter,
        )

        ccMajor = (
            self.tuner.GetCurrentDeviceInfo().GetCudaComputeCapabilityMajor()
        )
        ccMinor = (
            self.tuner.GetCurrentDeviceInfo().GetCudaComputeCapabilityMinor()
        )

        converter = ParallelismHeader(ccMajor, ccMinor).ConvertValue
        parallelism = counters[ProfilingCounter.GLOBAL_SIZE].apply(converter)

        # TODO: make these optional, since some counters might not exist
        return DataFrame(
            {
                ProfilingCounter.DRAM_RT_U: dramUtilization['read'],
                ProfilingCounter.DRAM_WT_U: dramUtilization['write'],
                ProfilingCounter.L2_RT_U: l2Utilization['read'],
                ProfilingCounter.L2_WT_U: l2Utilization['write'],
                ProfilingCounter.SHR_RT_U: smUtilization['read'],
                ProfilingCounter.SHR_WT_U: smUtilization['write'],
                ProfilingCounter.PARALLELISM: parallelism,
            }
        )

    def _ExtractCounters(self, result) -> DataFrame:
        counters = {}

        for counter in result.GetProfilingData().GetCounters():
            counterType = counter.GetType().name
            if counterType != 'Double':
                continue

            header = CounterHeader(counter.GetName())
            if header.name == ProfilingCounter.UNKNOWN:
                continue

            counters[header.name] = [
                header.ConvertValue(counter.GetValueDouble())
            ]

        globalSize = result.GetGlobalSize().GetTotalSize()
        localSize = result.GetLocalSize().GetTotalSize()

        # globalSize * localSize, huh ? okay
        counters[ProfilingCounter.GLOBAL_SIZE] = globalSize * localSize
        counters[ProfilingCounter.LOCAL_SIZE] = localSize
        return DataFrame(counters)

    def _BatchToSpacePredicted(self, batch: Batch) -> TuningSpace:
        parameterNames: list[str] = [p.GetName() for p in batch[0].GetPairs()]
        parameterData = numpy.empty((len(batch), len(parameterNames)))

        for row, configuration in enumerate(batch):
            for col, pair in enumerate(configuration.GetPairs()):
                parameterData[row, col] = pair.GetValue()

        # Sort the parameter names for consistency with the models
        parameterSpace = DataFrame(data=parameterData, columns=parameterNames)
        parameterSpace = parameterSpace.reindex(sorted(parameterNames), axis=1)

        counterSpace = self.spaceModel.predict(parameterSpace.to_numpy())
        return TuningSpace(parameterSpace, counterSpace)

    def _GetCandidateBatch(self) -> Batch:
        batch = self.GetNeighbourConfigurations(
            self.bestConfiguration,
            NEIGHBOUR_DISTANCE,
            self.batchInfo.neighborSize,
        )
        batch = self._GetUniqueConfigurations(batch)
        size = self.batchInfo.neighborSize + self.batchInfo.randomSize

        self._FillBatchRandom(batch, size)
        return batch

    def _UpdateBestConfiguration(self, previousResult: KernelResult):
        if not previousResult.IsValid():
            return

        duration = previousResult.GetKernelDuration()

        if self.bestConfiguration is None or duration < self.bestDuration:
            self.bestConfiguration = self.currentConfiguration
            self.bestDuration = duration

    def _FillBatchRandom(self, batch: Batch, expectedSize: int) -> None:
        size = min(expectedSize, self.GetUnexploredConfigurationsCount())

        while len(batch) < size:
            for _ in range(len(batch), size):
                batch.append(self.GetRandomConfiguration())

            batch = self._GetUniqueConfigurations(batch)

    def _GetUniqueConfigurations(self, configurations: Batch) -> Batch:
        uniqueIndices = set([self.GetIndex(c) for c in configurations])
        return [self.GetConfiguration(i) for i in uniqueIndices]


# Other helper functions


def WeightedRandomStep(scores: NDArray) -> int:
    """
    Returns an index of where in the tuning space the next step is.
    The value of the index is random, influenced by the weights in
    `score_distribution`
    """

    # .sum() and .cumsum()[-1] are different, so
    # cumulative sum is used for the random value
    scoreCumsum = numpy.cumsum(scores)
    randomValue = numpy.random.random() * scoreCumsum[-1]
    indices = numpy.argwhere(randomValue < scoreCumsum)
    if indices.shape[0] == 0:
        print("This error keeps on happening, although it shouldn't")
        print('Random value:', randomValue)
        print('Cumulative sum latest 10:', scoreCumsum[-10:])
        exit(-1)

    return indices[0, 0]


def _GetRWUtilization(
    counters: DataFrame,
    readCounter: ProfilingCounter,
    writeCounter: ProfilingCounter,
    utilizationCounter: ProfilingCounter,
) -> dict[str, Series]:
    reads = counters[readCounter]
    writes = counters[writeCounter]
    total = reads + writes
    total[total == 0] = 1  # prevent NaNs

    utilization = counters[utilizationCounter]

    readUtilization = (reads / total) * utilization
    writeUtilization = (writes / total) * utilization

    return {'read': readUtilization, 'write': writeUtilization}
