set -e
rm -rf ./build 
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON  -Bbuild -S. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native 
cd build && make -j
