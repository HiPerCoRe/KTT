import numpy
from numpy.typing import NDArray
from pandas import DataFrame

from lib.pyktt import KernelConfiguration, Searcher, KernelResult

from modules.info import BatchInfo, ModelInfo
from modules.tuning_space import TuningSpace
from modules.model import loadModel

# Profiler configuration (might move into the class?)
SCORE_EXPONENT = 7
SCORE_THRESHOLD = 0.425

NEIGHBOUR_DISTANCE = 2


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


Batch = list[KernelConfiguration]


class KatsBasedSearcher(Searcher):
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
        self._FillBatchRandom(self.batch, self.batchInfo.batchSize)
        self.currentConfiguration = self.batch.pop()

    def CalculateNextConfiguration(self, previousResult: KernelResult) -> bool:
        self._UpdateBestConfiguration(previousResult)

        if len(self.batch) != 0:
            self.currentConfiguration = self.batch.pop()
            return True

        # unlikely, the batch is emptied but no configuration ran properly
        if self.bestConfiguration is None:
            self._FillBatchRandom(self.batch, self.batchInfo.batchSize)
            self.currentConfiguration = self.batch.pop()

            return True

        if self.bestDuration != -1:  # rerun best config with profiling on
            self.currentConfiguration = self.bestConfiguration
            self.bestDuration = -1

            self.tuner.SetProfiling(True)
            return True

        # candidate configurations are made from neighbouring and random configurations
        candidateBatch = self._GetCandidateBatch()
        candidateSpace = self._BatchToSpacePredicted(candidateBatch)
        candidateScores = self._ScoreTuningSpace(candidateSpace, previousResult)

        self.batch = self._SelectBatchWeighted(candidateBatch, candidateScores)
        self.currentConfiguration = self.batch.pop()

        self.bestConfiguration = None
        self.tuner.SetProfiling(False)

        return True

    def GetCurrentConfiguration(self) -> KernelConfiguration:
        return self.currentConfiguration

    # Private functions

    def _SelectBatchWeighted(self, batch: Batch, scores: NDArray) -> Batch:
        return []  # TODO

    def _ScoreTuningSpace(
        self, space: TuningSpace, results: KernelResult
    ) -> NDArray:
        counters = self._GetCounters(results)
        predictedCounters = self.counter_model.predict(counters)

        counterDeltas = predictedCounters - space.counters
        timeDeltas = self.deltaModel.predict(counterDeltas).reshape(-1, 1)

        minDelta, maxDelta = timeDeltas.min(), timeDeltas.max()
        scores = (timeDeltas - minDelta) / (maxDelta - minDelta)

        scores[numpy.isnan(scores)] = 1.0  # edge case if the model is bad
        scores = numpy.power(scores, SCORE_EXPONENT)
        scores[scores < SCORE_THRESHOLD] = 0.00001

        return scores

    # NOTE: not an instance method, move out of the class?
    def _GetCounters(self, results: KernelResult) -> NDArray:
        kernelResults = results.GetResults()
        result = kernelResults[0]

        if len(kernelResults) != 1:
            print('whuh we dont support this, usin kernel 0 counters')

        countersData = result.GetProfilingData().GetCounters()
        for counterData in countersData:
            print('wa')  # TODO: this
            pass

        return numpy.ones(0)

    def _BatchToSpacePredicted(self, batch: Batch) -> TuningSpace:
        parameterNames = [p.GetName() for p in batch[0].GetPairs()]
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
