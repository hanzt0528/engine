python -m tvm.driver.tvmc tune \
--target "llvm -mcpu=skylake-avx512" \
--output resnet50-v2-7-autotuner_records.json \
resnet50-v2-7.onnx