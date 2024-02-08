#!/usr/bin/bash

CXX=hipcc cmake -DCMAKE_PREFIX_PATH="/opt/rocm" -DCMAKE_BUILD_TYPE=Release -DUSE_ROCM=ON -B build -S . 
cmake --build build -j$(nproc)
