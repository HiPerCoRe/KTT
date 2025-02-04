# Script converts json output from KTT format into common autotuning output format, schema available https://github.com/odgaard/TuningSchema/blob/T4/results-schema.json

import json
import sys

if len(sys.argv) != 3 :
    print("Error, run " + sys.argv[0] + ' ktt_json_file converted_json_filename')
    exit(1)

with open(sys.argv[1], 'r') as fp:
    ktt_output = json.load(fp)

ktt_result_status_mapping = {
        "Ok":"correct",
        "ComputationFailed":"runtime",
        "ValidationFailed":"correctness",
        "CompilationFailed":"compile",
        "DeviceLimitsExceeded":"runtime"
        # timeout is marked as ComputationFailed in KTT
        # constraints is marked as CompilationFailed in KTT
        }

converted_output = dict()

converted_output["schema_version"] = "1.0.0"
converted_output["results"] = []

for ktt_result in ktt_output["Results"]:
    converted_result = dict()
    converted_result["timestamp"] = ktt_output["Metadata"]["Timestamp"]
    # note that KTT outputs each run separately, it does not merge the output for the same configuration
    converted_result["configuration"] = dict()
    for tp in ktt_result["Configuration"]:
        converted_result["configuration"][tp["Name"]] = tp["Value"]
    # TODO PowerUsage also possible
    converted_result["objectives"] = ["TotalDuration"]
    converted_result["times"] = dict()
    converted_result["times"]["compilation_time"] = ktt_result["CompilationOverhead"]
    converted_result["times"]["runtimes"] = [ktt_result["TotalDuration"]]
    converted_result["times"]["framework"] = ktt_result["DataMovementOverhead"] + ktt_result["ProfilingOverhead"]
    converted_result["times"]["search_algorithm"] = ktt_result["SearcherOverhead"]
    converted_result["times"]["validation"] = ktt_result["ValidationOverhead"]
    # timeout, compile, runtime, correctness, constraints, correct
    converted_result["invalidity"] = ktt_result_status_mapping[ktt_result["Status"]]
    if ktt_result["Status"] == "ValidationFailed":
        converted_result["correctness"] = 0
    else:
        converted_result["correctness"] = 1
    converted_result["measurements"] = []
    converted_result["measurements"].append({
            "name": "TotalDuration",
            "value": ktt_result["TotalDuration"],
            "unit": ktt_output["Metadata"]["TimeUnit"]
            })
    # TODO what do we want here in case of multiple ComputationResults for multiple kernel functions?
    if "ProfilingData" in ktt_result["ComputationResults"][0]:
        for pc in ktt_result["ComputationResults"][0]["ProfilingData"]["Counters"]:
            converted_result["measurements"].append({"name":pc["Name"], "value":pc["Value"], "unit": ""})
    converted_output["results"].append(converted_result)


# dump converted file
with open(sys.argv[2], "w") as fp:
    json.dump(converted_output, fp, indent=4)
