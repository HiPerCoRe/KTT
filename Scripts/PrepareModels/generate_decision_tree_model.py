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
