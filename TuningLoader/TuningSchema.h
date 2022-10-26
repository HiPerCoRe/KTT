#pragma once

#include <json.hpp>

namespace ktt
{

inline const nlohmann::json TuningSchema =
R"(
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "https://github.com/HiPerCoRe/KTT/blob/development/TuningLoader/TuningFormatSchema.json",
    "title": "Tuning format",
    "description": "A description of a tuning problem which can be loaded by an autotuning framework",
    "type": "object",
    "required": [
        "ConfigurationSpace",
        "KernelSpecification"
    ],
    "properties": {
        "ConfigurationSpace": {
            "type": "object",
            "required": [
                "TuningParameters"
            ],
            "properties": {
                "TuningParameters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "Name",
                            "Type",
                            "Values"
                        ],
                        "properties": {
                            "Name": {
                                "type": "string"
                            },
                            "Type": {
                                "enum": [
                                    "int",
                                    "uint",
                                    "float",
                                    "bool",
                                    "string"
                                ]
                            },
                            "Values": {
                                "type": "string"
                            }
                        }
                    }
                },
                "Conditions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "Parameters",
                            "Expression"
                        ],
                        "properties": {
                            "Parameters": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "Expression": {
                                "type": "string"
                            }
                        }
                    }
                }
            }
        },
        "Search": {
            "type": "object",
            "required": [
                "Name"
            ],
            "properties": {
                "Name": {
                    "type": "string"
                },
                "Attributes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "Name",
                            "Value"
                        ],
                        "properties": {
                            "Name": {
                                "type": "string"
                            },
                            "Value": {
                                "type": "string"
                            }
                        }
                    }
                }
            }
        },
        "Budget": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "Type",
                    "BudgetValue"
                ],
                "properties": {
                    "Type": {
                        "enum": [
                            "TuningDuration",
                            "ConfigurationCount",
                            "ConfigurationFraction"
                        ]
                    },
                    "BudgetValue": {
                        "type": "number"
                    }
                }
            }
        },
        "General": {
            "type": "object",
            "properties": {
                "FormatVersion": {
                    "type": "integer"
                },
                "LoggingLevel": {
                    "enum": [
                        "Off",
                        "Error",
                        "Warning",
                        "Info",
                        "Debug"
                    ]
                },
                "TimeUnit": {
                    "enum": [
                        "Nanoseconds",
                        "Microseconds",
                        "Milliseconds",
                        "Seconds"
                    ]
                },
                "OutputFile": {
                    "type": "string",
                    "examples": [
                        "ReductionOutput",
                        "Results"
                    ]
                },
                "OutputFormat": {
                    "enum": [
                        "JSON",
                        "XML"
                    ]
                }
            }
        },
        "KernelSpecification": {
            "type": "object",
            "required": [
                "Language",
                "KernelName",
                "KernelFile",
                "GlobalSize",
                "LocalSize"
            ],
            "properties": {
                "Language": {
                    "enum": [
                        "OpenCL",
                        "CUDA",
                        "Vulkan"
                    ]
                },
                "CompilerOptions": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },
                "KernelName": {
                    "type": "string"
                },
                "KernelFile": {
                    "type": "string"
                },
                "KernelTypeNames": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },
                "GlobalSizeType": {
                    "enum": [
                        "OpenCL",
                        "CUDA",
                        "Vulkan"
                    ]
                },
                "SharedMemory": {
                    "type": "integer"
                },
                "SimulationInput": {
                    "type": "string"
                },
                "GlobalSize": {
                    "type": "object",
                    "required": [
                        "X"
                    ],
                    "properties": {
                        "X": {
                            "type": "string"
                        },
                        "Y": {
                            "type": "string"
                        },
                        "Z": {
                            "type": "string"
                        }
                    }
                },
                "LocalSize": {
                    "type": "object",
                    "required": [
                        "X"
                    ],
                    "properties": {
                        "X": {
                            "type": "string"
                        },
                        "Y": {
                            "type": "string"
                        },
                        "Z": {
                            "type": "string"
                        }
                    }
                },
                "Arguments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "Type",
                            "MemoryType"
                        ],
                        "properties": {
                            "Name": {
                                "type": "string"
                            },
                            "Type": {
                                "enum": [
                                    "bool",
                                    "int8",
                                    "uint8",
                                    "int16",
                                    "uint16",
                                    "int32",
                                    "uint32",
                                    "int64",
                                    "uint64",
                                    "half",
                                    "half2",
                                    "half4",
                                    "half8",
                                    "half16",
                                    "float",
                                    "float2",
                                    "float4",
                                    "float8",
                                    "float16",
                                    "double",
                                    "double2",
                                    "double4",
                                    "double8",
                                    "double16",
                                    "custom"
                                ]
                            },
                            "Size": {
                                "type": "integer",
                                "examples": [
                                    720,
                                    26000
                                ]
                            },
                            "TypeSize": {
                                "type": "integer",
                                "examples": [
                                    4,
                                    16
                                ]
                            },
                            "FillType": {
                                "enum": [
                                    "Constant",
                                    "Random",
                                    "Generator",
                                    "Script",
                                    "BinaryRaw",
                                    "BinaryHDF"
                                ]
                            },
                            "FillValue": {
                                "type": "number",
                                "examples": [
                                    40,
                                    1.0
                                ]
                            },
                            "DataSource": {
                                "type": "string"
                            },
                            "AccessType": {
                                "enum": [
                                    "ReadOnly",
                                    "WriteOnly",
                                    "ReadWrite"
                                ]
                            },
                            "MemoryType": {
                                "enum": [
                                    "Scalar",
                                    "Vector",
                                    "Local",
                                    "Symbol"
                                ]
                            }
                        }
                    }
                },
                "ReferenceData": {
                    "type": "object",
                    "required": [
                        "ReferenceArguments",
                        "ValidationPairs"
                    ],
                    "properties": {
                        "ReferenceArguments": {
                            "type": "array"
                        },
                        "ValidationPairs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": [
                                    "ReferenceName",
                                    "TargetName"
                                ],
                                "properties": {
                                    "ReferenceName": {
                                        "type": "string"
                                    },
                                    "TargetName": {
                                        "type": "string"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
)"_json;

} // namespace ktt
