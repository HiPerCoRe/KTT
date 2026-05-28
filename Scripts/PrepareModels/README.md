# Model Generation Scripts

Scripts for training machine learning models to predict GPU profiling counters from kernel tuning parameters.

## Scripts

### `generate_decision_tree_model.py`

Main script for generating ExtraTreesRegressor models from KTT JSON output.

**Requirements:**
- Python 3.6+
- scikit-learn
- pandas
- numpy
- docopt

**Input:** KTT JSON output (Legacy or T4 format)
**Output:** 
- `.sav` - Pickled scikit-learn model
- `.sav.metadata.json` - Training metadata (compute capability, features, targets)

**Usage:**
```bash
# Basic usage
python3 generate_decision_tree_model.py -i CoulombSumOutput.json

# With custom settings
python3 generate_decision_tree_model.py \
    -i data.json \
    --output-dir ./models \
    --cc 7.5 \
    --test-size 30 \
    --random-seed 42
```

**Arguments:**
- `--input, -i`: Path to KTT JSON output file (required)
- `--output-dir`: Output directory for models (default: same as input)
- `--cc`: Override compute capability detection (optional)
- `--test-size`: Test set percentage, 0-100 (default: 50)
- `--random-seed`: Random seed for train/test split (default: 7)

### `generate_models.py`

Batch model generation for multiple problems and GPUs.

**Usage:**
```bash
python3 generate_models.py \
    --problem coulomb \
    --input-dir ./tuning_data \
    --output-dir ./models
```

Expects input files at: `<input-dir>/<problem>/<gpu>-<problem>_output.json`

## Migration from CSV

**Breaking change:** CSV input support removed in favor of JSON-only workflow.

**Old workflow:**
```bash
python3 generate_decision_tree_model.py \
    -i data.csv -t 4:12 -c 13:50 --cc 7.5
```

**New workflow:**
```bash
python3 generate_decision_tree_model.py -i data.json
# Compute capability, tuning params, and counters auto-detected
```

## Supported Formats

### T4 Format
```json
{
  "metadata": {"device": "NVIDIA RTX 500 Ada", ...},
  "results": [
    {
      "configuration": {"param1": 32},
      "compilation_data": {
        "global_size": {"x": 1024, "y": 1, "z": 1},
        "local_size": {"x": 32, "y": 1, "z": 1}
      },
      "measurements": [
        {"name": "dram__throughput.avg", "value": 95.5}
      ]
    }
  ]
}
```

### Legacy Format
```json
{
  "Metadata": {"Device": "GeForce GTX 1070", ...},
  "Results": [
    {
      "Configuration": [{"Name": "param1", "Value": 32}],
      "ComputationResults": [{
        "KernelFunction": "vectorAdd",
        "GlobalSize": {"X": 1024, "Y": 1, "Z": 1},
        "LocalSize": {"X": 32, "Y": 1, "Z": 1},
        "ProfilingData": {
          "Counters": [{"Name": "dram__throughput.avg", "Value": 95.5}]
        }
      }]
    }
  ]
}
```

## Adding New Devices

Update the `DEVICE_COMPUTE_CAPABILITY` dictionary in `generate_decision_tree_model.py`:

```python
DEVICE_COMPUTE_CAPABILITY = {
    "RTX 5000 Ada": 8.9,
    "Your GPU Name": X.X,  # Add your device here
    ...
}
```

Device names use word-boundary matching, so "GTX 1070" will match "GeForce GTX 1070 Ti".

Another option is to specify compute capability explicitly with `--cc`.

## Model Details

**Algorithm:** ExtraTreesRegressor with 10 estimators

**Features (X):** Tuning parameters from configuration

**Targets (Y):** 
- `Global size` (product of x, y, z dimensions)
- `Local size` (product of x, y, z dimensions)
- Profiling counters (device-specific metrics)
- Compilation data (`Maximum work-group size`, `Local memory size`, `Private memory size`, `Constant memory size`, `Registers count`)

**Filtering:** Configurations without profiling data are automatically excluded

**Performance Metrics Excluded:** 
- T4: time, power_usage, energy_consumption, runtimes
- Legacy: Duration, PowerUsage, EnergyConsumption

**Ignored Measurements:**
- T4: temperature, sm_frequency, memory_frequency, fan_speed, duration_stdev
- Legacy: Temperature, SMFrequency, MemoryFrequency, FanSpeed, DurationStdev
