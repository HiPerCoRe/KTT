#!/usr/bin/env python

"""Generate models (decision tree) to predict performance counters from training data (does not need KTT).

Usage:
  ./generate_models.py --problem <problem> --input-dir <input_directory> --output-dir <output_directory>


Options:
  -h Show this screen.
  --problem  computational problem used for generating the model: coulomb, mtran, gemm_reduced, nbody, conv or gemm (for a whole GEMM benchmark).
  --input-dir the input directory with raw autotuning data that will be used for training the model, should follow the structure <problem>/<gpu>-<problem>_output.csv
  --output-dir the output directory for generated models
"""

from docopt import docopt
import os
import sys

def generateModel(method, problFrom, gpuFrom, input_dir, output_directory) :
    command = "mkdir -p " + output_directory
    os.system(command)
    command = "mkdir -p " + output_directory + "/" + problFrom[0]
    os.system(command)
    if problFrom[0].startswith("gemm") :
        input_directory = input_dir + "/gemm-reduced"
    else:
        input_directory = input_dir + "/" +  problFrom[0]
    if method == "decision-tree":
        command = "python3 " + script_directory + "/" + "generate_decision_tree_model.py -i " + input_directory + "/" + gpuFrom[0] + "-" + problFrom[0] + "_output.csv "
        command = command + "-t 4:" + str(4+problFrom[1]) + " -c 2,3," + str(4+problFrom[1]) + ":" + str(4+problFrom[1]+gpuFrom[4])
        command = command + " --cc " + str(gpuFrom[1])
        print ("Executing " + command)
        os.system(command)
        command = "mv " + input_directory + "/" + gpuFrom[0] + "-" + problFrom[0] + "_output_DT.sav* " + output_directory +"/" + problFrom[0]
        print ("Executing " + command)
        os.system(command)

arguments = docopt(__doc__)
method = "decision-tree"
problem = arguments['<problem>']
input_dir = arguments['--input-dir']
output_directory = arguments['--output-dir']
script_directory = os.path.dirname(os.path.realpath(sys.argv[0]))

print("Creating " + method + " models for " + problem + " for different GPUs, taking training data from " + input_dir + ", saving them in " + output_directory)

# (processor_name, computing_capability, no of multiprocessors, no of cuda cores, no of profiling_counters)
processors = [["680", 3.0, 8, 1536, 35], ["750", 5.0, 4, 512, 41], ["1070", 6.1, 15, 1920, 43], ["2080", 7.5, 46, 2944, 38]]
# (problem_name, tuning_parameters, boundary)
problemsGeneral = [["coulomb", 8, '--compute_bound'], ["mtran", 9, '--memory_bound'], ["gemm-reduced", 15, '--compute_bound'], ["nbody", 8, '--compute_bound'], ["conv", 10, "--compute_bound"], ["vectorAdd", 1, "--memory_bound"]]
problemsGemm = [["gemm-reduced", 15, '--compute_bound'], ["gemm-128-128-128", 15, '--compute_bound'], ["gemm-16-4096-4096", 15, '--memory_bound'], ["gemm-4096-16-4096", 15, '--memory_bound'], ["gemm-4096-4096-16", 15, '--compute_bound']]
problemsFrom = [["gemm-reduced", 15]]

if problem == "all" :
    for probl in problemsGeneral :
        for gpuFrom in processors :
            generateModel(method, probl, gpuFrom, input_dir, output_directory)
    for problFrom in problemsGemm :
        for gpu in [processors[2]] :
            generateModel(method, problFrom, gpu, input_dir, output_directory)
elif problem == "gemm":
    for problFrom in problemsGemm :
        for gpu in [processors[2]] :
            generateModel(method, problFrom, gpu, input_dir, output_directory)
else:
    for probl in problemsGeneral:
        for gpuFrom in processors:
            if probl[0] == problem:
                generateModel(method, probl, gpuFrom, input_dir, output_directory)
