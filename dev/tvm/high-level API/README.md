### Getting Starting using TVMC Python: a high-level API for TVM

~/.bashrc:
```
export TVM_HOME=/data/hanzt1/he/codes/tvm/

export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
```
```
source ~/.bashrc
mkdir myscripts
cd myscripts
wget https://github.com/onnx/models/raw/b9a54e89508f101a1611cd64f4ef56b9cb62c7cf/vision/classification/resnet/model/resnet50-v2-7.onnx
mv resnet50-v2-7.onnx my_model.onnx
touch tvmcpythonintro.py
```