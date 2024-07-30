import os 
import numpy as np
import tvm 
from tvm import relay, topi, te 
from tvm.contrib import graph_executor

def test_relay_build():
    x1 = relay.var("x1", shape=(2,), dtype="float32")
    x2 = relay.var("x2", shape=(2,), dtype="float32")
    y = relay.add(x1, x2)
    func = relay.Function(relay.analysis.free_vars(y), y)
    module = tvm.IRModule.from_expr(func)
    module = relay.transform.InferType()(module)
    #print(module.astext)

    target = tvm.target.Target(target="llvm")
    device = tvm.device(target.kind.name, 0)
    
    lib = relay.build_module.build(module, target, params=None)
    
    module = graph_executor.GraphModule(lib["default"](device))

    # ====== test on graph executor ======
    x1 = np.random.uniform(-1, 1, size=(2,)).astype("float32")
    x2 = np.random.uniform(-1, 1, size=(2,)).astype("float32")
    print("x1:\n", x1)
    print("x2:\n", x2)
    # data = tvm.nd.array(data, device)
    module.set_input("x1", x1)
    module.set_input("x2", x2)
    module.run()
    out = module.get_output(0).asnumpy()
    
    print("out:\n", out)

if __name__ == '__main__':
  test_relay_build()