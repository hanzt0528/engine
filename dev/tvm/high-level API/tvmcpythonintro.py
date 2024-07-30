from tvm.driver import tvmc
from tvm.contrib.download import download_testdata

import onnx
#model = tvmc.load('resnet50-v2-7.onnx')
#model = tvmc.load('resnet50-v2-7.onnx', shape_dict={'input1' : [1, 2, 3, 4], 'input2' : [1, 2, 3, 4]}) #Step 1: Load + shape_dict
#model.summary()
model_url = (
    "https://github.com/onnx/models/blob/main/Computer_Vision/resnet50_Opset16_timm/resnet50_Opset16.onnx"
)

#model_path = download_testdata(model_url, "resnet50_Opset16.onnx", module="onnx")
model_path = "resnet50_Opset16.onnx"
onnx_model = onnx.load(model_path)

