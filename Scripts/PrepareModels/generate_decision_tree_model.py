#This code is for predicting the value for profiling counter values

"""Generating train file for predicting the profiling counter values based on tuning parameters

Usage:
  generate_decision_tree_model.py -i <KTT_output> -t <tuningpar_interval> -c <counters_interval> --cc <compute_capability_GPU>

Options:
  -h Show this screen.
  -i Output CSV file from KTT (must include profiling counters)-This file will use as an input for the model.
  -t Interval <i:j,[k,l,m]>  or i:j of tuning parameters indices (for example- 1:5,[7,9]  means cols 1 to 4 and 7 and 9).
  -c Interval <i:j,[k,l,m]>  or i:j of profiler counter indices (for example- 1:5,[7,9]  means cols 1 to 4 and 7 and 9).
  --cc compute capability of the GPU processor

"""

import os
from sklearn.inspection import partial_dependence
# To use this experimental feature, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoLars
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error
from sklearn.metrics import r2_score
from sklearn import model_selection
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import pandas as pd
import datetime
from docopt import docopt
import json



np.set_printoptions(suppress=True)
TEST_SIZE = 50
SEED = 7

# JSON Format Detection Constants
PERFORMANCE_METRICS_T4 = {"time", "power_usage", "energy_consumption", "runtimes"}
PERFORMANCE_METRICS_LEGACY = {"Duration", "PowerUsage", "EnergyConsumption"}
IGNORED_MEASUREMENTS_T4 = {"temperature", "sm_frequency", "memory_frequency", "fan_speed", "duration_stdev"}
IGNORED_MEASUREMENTS_LEGACY = {"Temperature", "SMFrequency", "MemoryFrequency", "FanSpeed", "DurationStdev"}

# Device Compute Capability Lookup (sorted by word count desc, then length desc)
DEVICE_COMPUTE_CAPABILITY = {
    "RTX 5000 Ada": 8.9,
    "RTX 500 Ada": 8.9,
    "RTX 2080": 7.5,
    "GTX 1070": 6.1,
    "GTX 750": 5.0,
    "GTX 680": 3.0,
}

def detect_format(json_data):
    """
    Detect JSON schema format.
    
    Returns: "T4" | "Legacy"
    Raises: ValueError if format cannot be determined
    """
    # T4 indicators (lowercase keys)
    if "metadata" in json_data and "configuration" in str(json_data):
        return "T4"
    
    # Legacy indicators (PascalCase keys)
    if "Metadata" in json_data and "KttVersion" in str(json_data.get("Metadata", {})):
        return "Legacy"
    
    raise ValueError("Could not detect JSON format. Expected KTT Legacy or T4 schema.")

def matches_device(lookup_key, device_name):
    """
    Check if all words in lookup_key appear as complete words in device_name.
    Uses word-boundary regex matching to avoid false positives.
    
    Examples:
        matches_device("RTX 5000 Ada", "NVIDIA RTX 5000 Ada Generation") → True
        matches_device("RTX 5000 Ada", "NVIDIA RTX 500 Ada Generation") → False
        matches_device("GTX 1070", "GeForce GTX 1070 Ti") → True
    """
    import re
    device_lower = device_name.lower()
    key_lower = lookup_key.lower()
    
    key_words = key_lower.split()
    for word in key_words:
        pattern = r'\b' + re.escape(word) + r'\b'
        if not re.search(pattern, device_lower):
            return False
    return True

def get_compute_capability(device_name, cli_override=None):
    """
    Extract compute capability from device name or CLI override.
    
    Returns: float (compute capability)
    Raises: ValueError if device unknown and no CLI override
    """
    if cli_override:
        if device_name:
            print(f"Device detected: {device_name}")
        print(f"Compute capability: {cli_override} (from --cc override)")
        return cli_override
    
    if not device_name:
        raise ValueError(
            "No device name found in metadata and no --cc override provided. "
            "Provide --cc argument."
        )
    
    # Sort keys by word count (desc) then length (desc) for longest-match-first
    sorted_keys = sorted(
        DEVICE_COMPUTE_CAPABILITY.keys(),
        key=lambda k: (len(k.split()), len(k)),
        reverse=True
    )
    
    for key in sorted_keys:
        if matches_device(key, device_name):
            cc = DEVICE_COMPUTE_CAPABILITY[key]
            print(f"Device detected: {device_name}")
            print(f"Compute capability: {cc} (matched: '{key}')")
            return cc
    
    raise ValueError(
        f"Unknown device '{device_name}'. "
        f"Provide --cc argument or update device lookup table."
    )


class T4Parser:
    """Parser for T4 JSON format."""
    
    def extract_tuning_params(self, result):
        """Extract tuning parameter names and values from configuration."""
        config = result.get("configuration", {})
        # Filter out compiler flags (keys starting with --)
        params = {k: v for k, v in config.items() if not k.startswith("--")}
        return params
    
    def extract_profiling_counters(self, result):
        """Extract profiling counter names and values from measurements."""
        measurements = result.get("measurements", [])
        counters = {}
        
        for m in measurements:
            name = m.get("name", "")
            # Skip performance metrics and ignored measurements
            if name in PERFORMANCE_METRICS_T4 or name in IGNORED_MEASUREMENTS_T4:
                continue
            counters[name] = m.get("value", 0.0)
        
        return counters
    
    def extract_sizes(self, result):
        """Extract global and local sizes from compilation_data."""
        comp_data = result.get("compilation_data", {})
        global_size = comp_data.get("global_size", {"x": 1, "y": 1, "z": 1})
        local_size = comp_data.get("local_size", {"x": 1, "y": 1, "z": 1})
        
        # Compute products to match old CSV format
        global_product = global_size.get("x", 1) * global_size.get("y", 1) * global_size.get("z", 1)
        local_product = local_size.get("x", 1) * local_size.get("y", 1) * local_size.get("z", 1)
        
        return {
            "Global size": global_product,
            "Local size": local_product,
        }

    def extract_compilation_data(self, result):
        """Extract compilation data fields (work group size, memory sizes, registers)."""
        comp_data = result.get("compilation_data", {})

        return {
            "Maximum work-group size": comp_data.get("max_work_group_size", 0),
            "Local memory size": comp_data.get("local_memory_size", 0),
            "Private memory size": comp_data.get("private_memory_size", 0),
            "Constant memory size": comp_data.get("constant_memory_size", 0),
            "Registers count": comp_data.get("registers", 0),
        }

    def has_profiling_data(self, result):
        """Check if result has profiling counter data."""
        if "compilation_data" not in result:
            return False
        
        measurements = result.get("measurements", [])
        if not measurements:
            return False
        
        # Check if there's at least one profiling counter (non-performance metric)
        for m in measurements:
            name = m.get("name", "")
            if name not in PERFORMANCE_METRICS_T4 and name not in IGNORED_MEASUREMENTS_T4:
                return True
        
        return False
    
    def get_device_name(self, metadata):
        """Extract device name from metadata."""
        return metadata.get("device", None)


class LegacyParser:
    """Parser for Legacy JSON format."""
    
    def extract_tuning_params(self, result):
        """Extract tuning parameter names and values from Configuration.Pairs."""
        config_pairs = result.get("Configuration", [])
        params = {}
        
        for pair in config_pairs:
            name = pair.get("Name", "")
            value = pair.get("Value", 0)
            params[name] = value
        
        return params
    
    def extract_profiling_counters(self, result):
        """Extract profiling counter names and values from ProfilingData.Counters."""
        comp_results = result.get("ComputationResults", [])
        if not comp_results:
            return {}
        
        # Take first computation result
        first_result = comp_results[0]
        profiling_data = first_result.get("ProfilingData", {})
        counters_list = profiling_data.get("Counters", [])
        
        counters = {}
        for counter in counters_list:
            name = counter.get("Name", "")
            value = counter.get("Value", 0.0)
            counters[name] = value
        
        return counters
    
    def extract_sizes(self, result):
        """Extract global and local sizes from CompilationData."""
        comp_results = result.get("ComputationResults", [])
        if not comp_results:
            return {
                "Global size": 1,
                "Local size": 1,
            }
        
        first_result = comp_results[0]
        global_size = first_result.get("GlobalSize", {"X": 1, "Y": 1, "Z": 1})
        local_size = first_result.get("LocalSize", {"X": 1, "Y": 1, "Z": 1})
        
        # Compute products to match old CSV format
        global_product = global_size.get("X", 1) * global_size.get("Y", 1) * global_size.get("Z", 1)
        local_product = local_size.get("X", 1) * local_size.get("Y", 1) * local_size.get("Z", 1)
        
        return {
            "Global size": global_product,
            "Local size": local_product,
        }
    
    def extract_compilation_data(self, result):
        """Extract compilation data fields (work group size, memory sizes, registers)."""
        comp_results = result.get("ComputationResults", [])
        if not comp_results:
            return {
                "Maximum work-group size": 0,
                "Local memory size": 0,
                "Private memory size": 0,
                "Constant memory size": 0,
                "Registers count": 0,
            }

        first_result = comp_results[0]
        comp_data = first_result.get("CompilationData", {})

        return {
            "Maximum work-group size": comp_data.get("MaxWorkGroupSize", 0),
            "Local memory size": comp_data.get("LocalMemorySize", 0),
            "Private memory size": comp_data.get("PrivateMemorySize", 0),
            "Constant memory size": comp_data.get("ConstantMemorySize", 0),
            "Registers count": comp_data.get("RegistersCount", 0),
        }

    def has_profiling_data(self, result):
        """Check if result has profiling counter data."""
        comp_results = result.get("ComputationResults", [])
        if not comp_results:
            return False
        
        first_result = comp_results[0]
        
        # Check compilation data exists
        if "CompilationData" not in first_result:
            return False
        
        # Check profiling data exists and has counters
        profiling_data = first_result.get("ProfilingData", {})
        counters = profiling_data.get("Counters", [])
        
        return len(counters) > 0
    
    def get_kernel_name(self, results):
        """Get kernel name from results and validate consistency."""
        if not results:
            raise ValueError("No configurations found in input file")
        
        # Extract kernel names from all results
        kernel_names = set()
        for result in results:
            comp_results = result.get("ComputationResults", [])
            if comp_results:
                kernel_func = comp_results[0].get("KernelFunction", "")
                if kernel_func:
                    kernel_names.add(kernel_func)
        
        if len(kernel_names) == 0:
            raise ValueError("No kernel function names found in results")
        
        if len(kernel_names) > 1:
            raise ValueError(
                f"Input file contains multiple kernels: {sorted(kernel_names)}. "
                f"Only single-kernel files supported."
            )
        
        return kernel_names.pop()
    
    def get_device_name(self, metadata):
        """Extract device name from metadata."""
        return metadata.get("Device", None)


if __name__ == '__main__':
    # parse command line
    arguments = docopt(__doc__)
    tuningOutput = open(arguments['-i'])
    #tuningInt = list(map(int, arguments['-t'].split(',')))
    tuningRange = arguments['-t']
    #countersInt = list(map(int, arguments['-c'].split(',')))
    countersRange = arguments['-c']
    #if (len(tuningInt) != 2) or (len(countersInt) != 2):
    #    print("Intervals must be in format from,to!")
    #    exit()

    bench = os.path.splitext(arguments['-i'])[0]
    data = pd.read_csv(tuningOutput)
    array = data.values

    try:
        rangeT = eval("np.r_[" + tuningRange + "]")
        rangeC = eval("np.r_[" + countersRange + "]")
    except:
        print("Intervals must be in format <i:j,[k,l,m]> or i:j (for example- 1:5,[7,9]  means cols 1 to 4 and 7 and 9).")
        exit()
    #array[1:2,np.r_[1:2,7:10,[13,14]]]
    #X = array[:,tuningInt[0]:tuningInt[1]+1]
    X = array[:,rangeT]
    #Y = array[:,countersInt[0]:countersInt[1]]  # Using profiling counter variables as dependent variables
    Y = array[:,rangeC]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(TEST_SIZE/100), random_state=SEED)
    columns_names = list(data.columns)
    columns_number = len(data.columns)
    # Estimate the score after iterative imputation of the missing values
    # with different estimators
    estimators = []
    #estimators.append(('DT-simple', DecisionTreeRegressor(max_features='sqrt', random_state=0)))
    estimators.append(('DT', ExtraTreesRegressor(n_estimators=10, random_state=0)))
    #estimators.append(('Knn10', KNeighborsRegressor(n_neighbors=10)))

    NumberOfModels = len(estimators)
    ExprimentStart = datetime.datetime.now()
    start = 0
    end = 0
    maxScore = 0
    Bestmodel = ''
    for name, imputer in estimators:
        start = datetime.datetime.now()
        imputer.fit(X_train, Y_train)
        #saving model to a file for later usages
        filename = str(bench) + "_" + str(name) + ".sav"
        pickle.dump(imputer, open(filename, 'wb'))
        #Computing the score of model
        scoreTrain = imputer.score(X_train, Y_train)
        #compute time elapsed
        end = datetime.datetime.now()
        durationTrain = end - start
        #Predicting with Test or X_test dataset and then writing the predicted results to .csv
        start = datetime.datetime.now()
        predicted = imputer.predict(X_test)
        scoreTest = imputer.score(X_test, Y_test)
        if (scoreTest >= maxScore):
            maxScore = scoreTest
            Bestmodel = name
        end = datetime.datetime.now()
        durationTest = end - start
        #Print some reports
        print("Training Result for ", name, " and ", bench, " is :")
        print("             Train Score is                : %", scoreTrain * 100)
        print("             Test Score is                 : %", scoreTest * 100)
        print("             Mean squared error is         : ", mean_squared_error(Y_test, predicted))
        print("=======================================================================")

        # Save metadata: tuning parameters, profiling counters
        metadata = {}
        metadata['cc'] = float(arguments['--cc'])
        metadata['tp'] = []
        metadata['pc'] = []
        for col in rangeT:
            metadata['tp'].append(str(data.columns[col]))
        for col in rangeC:
            metadata['pc'].append(str(data.columns[col]))
        print(metadata)
        filename = filename + ".metadata.json"
        with open(filename, 'w', ) as fp:
            json.dump(metadata, fp, indent=4)


    ExprimentEnd = datetime.datetime.now()
    TotalTime = ExprimentEnd - ExprimentStart
    print(Bestmodel, ' : For ', str(bench), ' and test size ', TEST_SIZE, ' === Maximum score between these models are:', maxScore)
    print("Total time that elapsed in this expriment is: ", TotalTime.total_seconds() * 1000)
    print("=======================================================================")
