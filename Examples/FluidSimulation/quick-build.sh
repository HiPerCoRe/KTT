#!/bin/bash

if [ ! -d fluid-simulation ]; then
	git clone https://github.com/petbab/fluid-simulation.git || {
		echo "Failed clone" >&2; exit 1
	}
fi

cd fluid-simulation

if [ ! -f cmake/CMakeUserConfig.cmake ]; then
	sed 's/path\/to\/KTT/..\/..\/../g' cmake/CMakeUserConfig.cmake.in > cmake/CMakeUserConfig.cmake
fi

rm -rf build 2>/dev/null

cmake -B build || {
	echo "Failed CMake setup, make sure 'cmake/CMakeUserConfig.cmake' is configured" >&2; exit 1
}

cmake --build build -j $(nproc) || {
	echo "Failed build" >&2; exit 1
}

cd ..

echo "Run scene: ./fluid-simulation/build/bin/<scene>"
echo "Available scenes:"
ls fluid-simulation/build/bin

