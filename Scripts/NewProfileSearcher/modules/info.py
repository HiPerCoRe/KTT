class ModelInfo:
    def __init__(self, deltaPath: str, spacePath: str, counterPath: str):
        self.deltaPath = deltaPath
        self.spacePath = spacePath
        self.counterPath = counterPath


class BatchInfo:
    def __init__(self, batchSize: int, neighborSize: int, randomSize: int):
        self.batchSize = batchSize
        self.neighborSize = neighborSize
        self.randomSize = randomSize
