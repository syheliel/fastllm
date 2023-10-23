set -e
rm -rf ./build 
# cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON  -DUSE_CUDA=ON -DUSE_ROCM=ON -DCMAKE_CUDA_ARCHITECTURES=75 -Bbuild -S. 
export CXX=hipcc
export CMAKE_CXX_FLAGS="-g3"
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON  -DUSE_ROCM=ON -Bbuild -S. 
cd build && make -j
