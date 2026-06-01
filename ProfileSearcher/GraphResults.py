#!/bin/env python3

import glob
import sys
from typing import cast
import xml.etree.ElementTree as ElementTree

import numpy
import pandas
from pandas import DataFrame
from seaborn import lineplot

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# Change these to where you store the profiling results
PROFILED_DIRECTORY = './logs/profiled/2080-1070/b10-n40-r100/'
RANDOM_DIRECTORY = './logs/random/2080/'


def parseResults(directory: str) -> DataFrame:
    outputFiles = glob.glob(directory + 'output-*.xml')

    durations = []
    for file in outputFiles:
        xmlTree = ElementTree.parse(file)
        results = xmlTree.findall('./Results/KernelResult')
        durations.append([float(d.attrib['TotalDuration']) for d in results])

    iterations = numpy.concatenate(
        [numpy.arange(1, len(d) + 1) for d in durations]
    )

    bestDurations = []
    for duration in durations:
        bestTime = numpy.empty(len(duration))
        bestTime[0] = duration[0]

        for i in range(1, len(duration)):
            time = duration[i]
            bestTime[i] = time if time < bestTime[i - 1] else bestTime[i - 1]

        bestDurations.append(bestTime)

    return DataFrame(
        {
            'iteration': iterations,
            'time': numpy.concatenate(bestDurations),
        }
    )


def graphResults(outputPath: str | None):
    profiledResults = parseResults(PROFILED_DIRECTORY)
    profiledResults['name'] = 'Profiling'

    randomResults = parseResults(RANDOM_DIRECTORY)
    randomResults['name'] = 'Random'

    resultsPlot = lineplot(
        data=pandas.concat([profiledResults, randomResults]),
        x='iteration',
        y='time',
        hue='name',
    )

    if outputPath is None:
        plt.show(block=True)
        return

    figure = cast(Figure, resultsPlot.get_figure())
    figure.set_dpi(150)
    figure.set_size_inches(19.20, 10.80)

    figure.savefig(outputPath)
    plt.clf()


if __name__ == '__main__':
    outputPath = sys.argv[1] if len(sys.argv) == 2 else None
    graphResults(outputPath)
