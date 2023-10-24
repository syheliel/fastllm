set -e
module load rocm
rm -rf ./build 
# cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON  -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -Bbuild -S. 
# export CXX=hipcc

CXX=hipcc cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Debug -DUSE_ROCM=ON -Bbuild -S. 
cd build && make -j
