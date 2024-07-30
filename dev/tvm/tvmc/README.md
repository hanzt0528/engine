### Python Package Installation
TVM package

Depending on your development environment, you may want to use a virtual environment and package manager, such as virtualenv or conda, to manage your python packages and dependencies.

The python package is located at tvm/python There are two ways to install the package:

Method 1
This method is recommended for developers who may change the codes.

Set the environment variable PYTHONPATH to tell python where to find the library. For example, assume we cloned tvm on the directory /path/to/tvm then we can add the following line in ~/.bashrc. The changes will be immediately reflected once you pull the code and rebuild the project (no need to call setup again)

```
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```
Method 2
Install TVM python bindings by setup.py:

# install tvm package for the current user
# NOTE: if you installed python via homebrew, --user is not needed during installaiton
#       it will be automatically installed to your user directory.
#       providing --user flag may trigger error during installation in such case.
export MACOSX_DEPLOYMENT_TARGET=10.9  # This is required for mac to avoid symbol conflicts with libstdc++
cd python; python setup.py install --user; cd ..

### check install tvmc
```
source ~/.bashrc
python -m tvm.driver.tvmc --help
```

### Compiling and Optimizing a Model with TVMC¶
wget https://github.com/onnx/models/raw/b9a54e89508f101a1611cd64f4ef56b9cb62c7cf/vision/classification/resnet/model/resnet50-v2-7.onnx

```
 pip3 install  onnx onnxoptimizer
```

### Compile

```
sh compile.sh
```

### Preprocess
make a input "imagenet_cat.npz"

```
python preprocess.py
```

### Run
```
sh run.sh
(py39) root@66d923040958:/data/hanzt1/he/codes/engine/tests/tvm/tvmc# sh run.sh 
2024-04-22 03:00:12.191 INFO load_module /tmp/tmpqcshun1p/mod.so
```

### PostProcess
```
python postprocess.py
(py39) root@66d923040958:/data/hanzt1/he/codes/engine/tests/tvm/tvmc# python postprocess.py
class='n02123045 tabby, tabby cat' with probability=0.610552
class='n02123159 tiger cat' with probability=0.367180
class='n02124075 Egyptian cat' with probability=0.019365
class='n02129604 tiger, Panthera tigris' with probability=0.001273
class='n04040759 radiator' with probability=0.000261
```

### auto tuning
```
pip install xgboost
sh tune.sh
(py39) root@66d923040958:/data/hanzt1/he/codes/engine/tests/tvm/tvmc# sh tune.sh 
[Task  3/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/40) | 0.00 s Done.
 Done.
[Task  3/25]  Current/Best:  165.99/ 816.86 GFLOPS | Progress: (40/40) | 22.11 s Done.
[Task  7/25]  Current/Best:  495.74/ 779.32 GFLOPS | Progress: (40/40) | 19.44 s Done.
[Task 10/25]  Current/Best:  794.29/1059.95 GFLOPS | Progress: (40/40) | 18.50 s Done.
[Task 12/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/40) | 0.00 s Done.
 Done.
 Done.
 Done.
 Done.
 Done.
[Task 13/25]  Current/Best:  379.48/ 802.83 GFLOPS | Progress: (40/40) | 22.77 s Done.
[Task 16/25]  Current/Best:  155.80/1051.64 GFLOPS | Progress: (40/40) | 20.58 s Done.
[Task 17/25]  Current/Best:  151.14/ 931.67 GFLOPS | Progress: (40/40) | 18.99 s Done.
[Task 19/25]  Current/Best:  603.97/ 995.37 GFLOPS | Progress: (40/40) | 22.77 s Done.
[Task 22/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/40) | 0.00 s Done.
 Done.
 Done.
 Done.
 Done.
[Task 22/25]  Current/Best:  703.68/ 996.88 GFLOPS | Progress: (40/40) | 19.62 s Done.
[Task 23/25]  Current/Best:  154.83/1489.02 GFLOPS | Progress: (40/40) | 20.34 s Done.
[Task 25/25]  Current/Best:   11.61/  59.50 GFLOPS | Progress: (40/40) | 23.82 s Done.
 Done.
 Done.
```

### run autotuned

```
sh run_autotuned.sh
py39) root@66d923040958:/data/hanzt1/he/codes/engine/tests/tvm/tvmc# sh run_autotuned.sh 
2024-04-22 07:53:12.537 INFO load_module /tmp/tmpl028cri6/mod.so
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
  20.9710      19.3385      70.8707      18.0113       5.9768                  
(py39) root@66d923040958:/data/hanzt1/he/codes/engine/tests/tvm/tvmc# 

```