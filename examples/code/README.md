# build fastertransformer
mkdir -p FasterTransformerCode/build
cd FasterTransformerCode/build
cmake -DSM="7.5;8.6" -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=OFF ..
make