
#python -m tvm.driver.tvmc --help

python -m tvm.driver.tvmc compile \
--target "llvm" \
--tuning-records resnet50-v2-7-autotuner_records.json  \
--output resnet50-v2-7-tvm_autotuned.tar \
resnet50-v2-7.onnx