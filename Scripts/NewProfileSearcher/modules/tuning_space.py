from numpy.typing import NDArray
import pandas


class TuningSpace:
    def __init__(self, parameters: pandas.DataFrame, counters: NDArray):
        self.parameters = parameters
        self.counters = counters
