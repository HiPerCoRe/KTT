#!/usr/bin/env python3
"""
Generate decision tree models to predict profiling counter values from tuning parameters.

This script consumes KTT's native JSON output formats (Legacy or T4) and produces
scikit-learn ExtraTreesRegressor models saved as .sav files with accompanying
.sav.metadata.json files.

## Input Format

Supports two JSON schemas:
- **T4 Format**: Lowercase keys (metadata, results, configuration, measurements)
- **Legacy Format**: PascalCase keys (Metadata, Results, Configuration)

Format is auto-detected. See KTT documentation for schema details.

## Output Files

1. **Model file** (.sav): Pickled scikit-learn ExtraTreesRegressor
   - Naming: `<input_basename>_DT.sav`

2. **Metadata file** (.sav.metadata.json): Training configuration
   ```json
   {
     "cc": 8.9,
     "tp": ["param1", "param2", ...],
     "pc": ["Global size", "Local size", "counter1", "counter2", ...,
           "Maximum work-group size", "Local memory size", "Private memory size",
           "Constant memory size", "Registers count"]
   }
   ```

## Usage Examples

Basic usage (auto-detect everything):
    python3 generate_decision_tree_model.py -i CoulombSumOutput.json

With output directory:
    python3 generate_decision_tree_model.py -i tuning.json --output-dir ./models

Override compute capability:
    python3 generate_decision_tree_model.py -i data.json --cc 7.5

Custom train/test split:
    python3 generate_decision_tree_model.py -i data.json --test-size 30 --random-seed 42

## CLI Reference

Usage:
  generate_decision_tree_model.py -i <input_json> [options]
  generate_decision_tree_model.py -h | --help

Options:
  -h --help                 Show this screen.
  -i <input_json>           Path to KTT JSON output file (Legacy or T4 format).
  --output-dir <dir>        Output directory for model files (default: input file's directory).
  --cc <capability>         Override compute capability detection (e.g., 8.9).
  --test-size <percent>     Test set size as percentage 0-100 [default: 50].
  --random-seed <seed>      Random seed for train/test split [default: 7].

## Requirements

- Python 3.6+
- scikit-learn
- pandas
- numpy
- docopt

## Device Support

Automatic compute capability detection for:
- RTX 5000 Ada (8.9)
- RTX 500 Ada (8.9)
- RTX 2080 (7.5)
- GTX 1070 (6.1)
- GTX 750 (5.0)
- GTX 680 (3.0)

Unknown devices require --cc override.
"""

import os
from docopt import docopt
import json
import pickle
import numpy as np
import pandas as pd
import datetime
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



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


def build_dataframe(results, parser):
    """
    Build pandas DataFrame from parsed results.
    
    Returns: (X, Y, tuning_param_names, target_names)
    """
    rows = []
    tuning_param_names = None
    target_names = None
    
    for result in results:
        if not parser.has_profiling_data(result):
            continue
        
        tuning_params = parser.extract_tuning_params(result)
        profiling_counters = parser.extract_profiling_counters(result)
        sizes = parser.extract_sizes(result)
        compilation_data = parser.extract_compilation_data(result)
        
        # Capture column names from first valid result
        if tuning_param_names is None:
            tuning_param_names = sorted(tuning_params.keys())
            profiling_counter_names = sorted(profiling_counters.keys())
        
        row = {}
        row.update(tuning_params)
        row.update(sizes)
        row.update(profiling_counters)
        row.update(compilation_data)
        rows.append(row)
    
    # Build DataFrame with deterministic column ordering
    size_columns = ["Global size", "Local size"]
    compilation_columns = ["Maximum work-group size", "Local memory size", "Private memory size",
                          "Constant memory size", "Registers count"]
    target_columns = size_columns + profiling_counter_names + compilation_columns
    ordered_columns = tuning_param_names + target_columns
    df = pd.DataFrame(rows, columns=ordered_columns)
    
    X = df[tuning_param_names].values
    Y = df[target_columns].values
    
    return X, Y, tuning_param_names, target_columns

if __name__ == '__main__':
    # Parse command line arguments
    arguments = docopt(__doc__)
    
    input_file = arguments['-i']
    output_dir = arguments['--output-dir']
    cc_override = float(arguments['--cc']) if arguments['--cc'] else None
    test_size = int(arguments['--test-size'])
    random_seed = int(arguments['--random-seed'])
    
    # Validate test size
    if test_size < 0 or test_size > 100:
        print("Error: --test-size must be between 0 and 100")
        exit(1)
    
    # Load JSON file
    try:
        with open(input_file, 'r') as f:
            json_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON: {e}")
        exit(1)
    
    # Detect format
    try:
        format_type = detect_format(json_data)
        print(f"Detected format: {format_type}")
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    
    # Create appropriate parser
    if format_type == "T4":
        parser = T4Parser()
        if "results" not in json_data:
            print("Error: Missing required field: 'results'")
            exit(1)
        results = json_data.get("results", [])
        metadata = json_data.get("metadata", {})
    else:  # Legacy
        parser = LegacyParser()
        if "Results" not in json_data:
            print("Error: Missing required field: 'Results'")
            exit(1)
        results = json_data.get("Results", [])
        metadata = json_data.get("Metadata", {})
    
    if not results:
        print("Error: No configurations found in input file")
        exit(1)
    
    # Extract device and compute capability
    device_name = parser.get_device_name(metadata)
    try:
        compute_capability = get_compute_capability(device_name, cc_override)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    
    # Validate kernel name for Legacy format (single kernel per file)
    if format_type == "Legacy":
        try:
            kernel_name = parser.get_kernel_name(results)
            print(f"Kernel: {kernel_name}")
        except ValueError as e:
            print(f"Error: {e}")
            exit(1)
    
    # Build DataFrame
    total_configs = len(results)
    try:
        X, Y, tuning_param_names, target_names = build_dataframe(results, parser)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    
    filtered_count = total_configs - X.shape[0]
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} configurations without profiling data")
    print(f"Training on {X.shape[0]} configurations")
    print(f"Features: {X.shape[1]} tuning parameters")
    print(f"Targets: {Y.shape[1]} outputs (2 sizes + {Y.shape[1] - 7} profiling counters + 5 compilation fields)")
    
    # Train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=(test_size / 100), random_state=random_seed
    )
    
    # Train model
    print("Training ExtraTreesRegressor model...")
    start_time = datetime.datetime.now()
    model = ExtraTreesRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, Y_train)
    end_time = datetime.datetime.now()
    training_duration = end_time - start_time
    
    # Evaluate model
    score_train = model.score(X_train, Y_train)
    score_test = model.score(X_test, Y_test)
    predicted = model.predict(X_test)
    mse = mean_squared_error(Y_test, predicted)
    
    print(f"Training completed in {training_duration.total_seconds():.2f} seconds")
    print(f"Train Score: {score_train * 100:.2f}%")
    print(f"Test Score: {score_test * 100:.2f}%")
    print(f"Mean Squared Error: {mse:.4f}")
    
    # Determine output directory and filename
    input_dir = os.path.dirname(os.path.abspath(input_file))
    output_dir_path = output_dir if output_dir else input_dir
    os.makedirs(output_dir_path, exist_ok=True)
    
    input_basename = os.path.splitext(os.path.basename(input_file))[0]
    
    # Output filename: <input_basename>_DT.sav
    model_filename = f"{input_basename}_DT.sav"
    model_path = os.path.join(output_dir_path, model_filename)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved: {model_path}")
    
    # Save metadata
    metadata_output = {
        "cc": compute_capability,
        "tp": tuning_param_names,
        "pc": target_names
    }
    
    metadata_path = model_path + ".metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata_output, f, indent=4)
    print(f"Metadata saved: {metadata_path}")
    
    print("Done!")
