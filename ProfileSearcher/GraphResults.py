#!/bin/env python3

import glob
import xml.etree.ElementTree as ElementTree

import numpy

# TODO: parameter or just do the entire ./logs/xml folder
LOG_DIRECTORY = './logs/xml/2080-1070/b25-n30-r20/'


if __name__ == '__main__':
    outputFiles = glob.glob(LOG_DIRECTORY + 'output-*.xml')

    durations = []
    for file in outputFiles:
        xmlTree = ElementTree.parse(file)
        results = xmlTree.findall('./Results/KernelResult')
        durations.append([float(d.attrib['TotalDuration']) for d in results])

    iterations = numpy.concatenate(
        [numpy.arange(1, len(d) + 1) for d in durations]
    )

    # TODO: graphing of the results, some comparison between the other things maybe
