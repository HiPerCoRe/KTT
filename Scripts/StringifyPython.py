#!/bin/env python3
from pathlib import Path
import sys


def stringify(input_file: str):
    output_file = f'{Path(input_file).stem}.h'

    with (
        open(input_file, 'r') as input,
        open(output_file, 'w') as output,
    ):
        output.write('std::string() +\n')

        for line in input:
            modifiedLine = line.replace('"', r'\"').rstrip()
            output.write(f'"{modifiedLine}\\n" +\n')

        output.write('"";')

    print(f'Stringified script stored into {output_file}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} [python-file-to-stringify]')
        exit(1)

    stringify(sys.argv[1])
