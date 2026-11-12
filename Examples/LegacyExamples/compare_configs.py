#!/usr/bin/env python3
"""
Compare Sort kernel configurations between original and refactored versions.

Usage:
    python3 compare_sort_configs.py <original_output.json> <refactored_output.json>

This script extracts all kernel configurations from both Output.json files
and verifies they searched the same configuration space.
"""

import json
import sys
from pathlib import Path


def extract_configurations(filepath):
    """Extract all configurations from Output.json"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    configs = []
    computation_results = []
    metadata = data.get('Metadata', {})

    print(f"Metadata:")
    print(f"  ComputeApi: {metadata.get('ComputeApi', 'N/A')}")
    print(f"  Device: {metadata.get('Device', 'N/A')[:60]}...")
    print(f"  KttVersion: {metadata.get('KttVersion', 'N/A')}")
    print()

    for result in data.get('Results', []):
        config = result.get('Configuration', [])
        kernel_name = result.get('KernelName', 'Unknown')
        status = result.get('Status', 'Unknown')

        # Convert to a hashable format for comparison
        config_dict = {c['Name']: c['Value'] for c in config}
        configs.append((kernel_name, config_dict, status))

        # Extract ComputationResults
        for comp_result in result.get('ComputationResults', []):
            kernel_func = comp_result.get('KernelFunction', 'Unknown')
            global_size = comp_result.get('GlobalSize', {})
            local_size = comp_result.get('LocalSize', {})
            compilation_data = comp_result.get('CompilationData', {})

            computation_results.append({
                'kernel_func': kernel_func,
                'global_size': global_size,
                'local_size': local_size,
                'compilation_data': compilation_data
            })

    return configs, computation_results


def computation_result_to_hashable(result):
    """Convert a computation result to a hashable format for comparison"""
    return (
        result['kernel_func'],
        (
            (result['global_size'].get('X', 0), result['global_size'].get('Y', 0), result['global_size'].get('Z', 0)),
            (result['local_size'].get('X', 0), result['local_size'].get('Y', 0), result['local_size'].get('Z', 0)),
            (
                result['compilation_data'].get('ConstantMemorySize', 0),
                result['compilation_data'].get('LocalMemorySize', 0),
                result['compilation_data'].get('PrivateMemorySize', 0),
                result['compilation_data'].get('RegistersCount', 0),
                result['compilation_data'].get('MaxWorkGroupSize', 0)
            )
        )
    )


def compare_configs(original_file, refactored_file):
    """Compare configurations between two output files"""

    print("=" * 60)
    print("Sort Refactor Configuration Comparison")
    print("=" * 60)
    print()

    print(f"Original file: {original_file}")
    print("-" * 40)
    original_configs, original_comp_results = extract_configurations(original_file)

    print(f"Refactored file: {refactored_file}")
    print("-" * 40)
    refactored_configs, refactored_comp_results = extract_configurations(refactored_file)

    # Create sets for configuration comparison (kernel_name, frozenset of params)
    original_config_set = set(
        (kernel, frozenset(params.items()))
        for kernel, params, _ in original_configs
    )
    refactored_config_set = set(
        (kernel, frozenset(params.items()))
        for kernel, params, _ in refactored_configs
    )

    # Create sets for computation results comparison
    original_comp_set = set(
        computation_result_to_hashable(r) for r in original_comp_results
    )
    refactored_comp_set = set(
        computation_result_to_hashable(r) for r in refactored_comp_results
    )

    print()
    print("Summary:")
    print(f"  Original configurations: {len(original_configs)}")
    print(f"  Refactored configurations: {len(refactored_configs)}")
    print(f"  Original computation results: {len(original_comp_results)}")
    print(f"  Refactored computation results: {len(refactored_comp_results)}")

    # Find differences in configurations
    config_missing = original_config_set - refactored_config_set
    config_extra = refactored_config_set - original_config_set

    # Find differences in computation results
    comp_missing = original_comp_set - refactored_comp_set
    comp_extra = refactored_comp_set - original_comp_set

    success = True

    if not config_missing and not config_extra and not comp_missing and not comp_extra:
        print()
        print("=" * 60)
        print("✓ SUCCESS: Both versions searched the same configurations!")
        print("=" * 60)
        return True

    print()
    if config_missing:
        print(f"✗ Missing configurations in refactored: {len(config_missing)}")
        for kernel, params in sorted(config_missing)[:5]:
            print(f"  [{kernel}] {dict(params)}")
        if len(config_missing) > 5:
            print(f"  ... and {len(config_missing) - 5} more")

    if config_extra:
        print(f"✗ Extra configurations in refactored: {len(config_extra)}")
        for kernel, params in sorted(config_extra)[:5]:
            print(f"  [{kernel}] {dict(params)}")
        if len(config_extra) > 5:
            print(f"  ... and {len(config_extra) - 5} more")

    if comp_missing:
        print(f"✗ Missing computation results in refactored: {len(comp_missing)}")
        for key in sorted(comp_missing)[:5]:
            kernel, (global_sz, local_sz, mem_sizes) = key
            print(f"  [{kernel}] Global:{global_sz} Local:{local_sz} Mem:{mem_sizes}")
        if len(comp_missing) > 5:
            print(f"  ... and {len(comp_missing) - 5} more")

    if comp_extra:
        print(f"✗ Extra computation results in refactored: {len(comp_extra)}")
        for key in sorted(comp_extra)[:5]:
            kernel, (global_sz, local_sz, mem_sizes) = key
            print(f"  [{kernel}] Global:{global_sz} Local:{local_sz} Mem:{mem_sizes}")
        if len(comp_extra) > 5:
            print(f"  ... and {len(comp_extra) - 5} more")

    print()
    print("=" * 60)
    print("✗ FAILURE: Configurations or computation results differ!")
    print("=" * 60)
    return False


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        print(f"\nUsage: {sys.argv[0]} <original.json> <refactored.json>")
        sys.exit(1)

    original_file = Path(sys.argv[1])
    refactored_file = Path(sys.argv[2])

    if not original_file.exists():
        print(f"Error: Original file not found: {original_file}")
        sys.exit(1)

    if not refactored_file.exists():
        print(f"Error: Refactored file not found: {refactored_file}")
        sys.exit(1)

    success = compare_configs(str(original_file), str(refactored_file))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
