import numpy as np
import nnvm.compiler
import nnvm.testing
import tvm
from tvm.contrib import graph_runtime
import mxnet as mx
from mxnet import ndarray as nd
 
dtype = "float32"
ctx = tvm.cpu()
loaded_json = open("./deploy_graph.json").read()
loaded_lib = tvm.module.load("./deploy_lib.so")
loaded_params = bytearray(open("./deploy_param.params", "rb").read())
 
data_shape = (1, 3, 112, 112)
import cv2
face_1 = cv2.imread("./images/1_1.png")
face_2 = cv2.imread("./images/2_2.png")
print(data_shape)
face_1 = cv2.resize(face_1, (data_shape[2], data_shape[3]))
face_2 = cv2.resize(face_2, (data_shape[2], data_shape[3]))
face_1 = np.transpose(np.array(face_1), (2, 0, 1))
face_2 = np.transpose(np.array(face_2), (2, 0, 1))
 
input_face1 = tvm.nd.array(face_1.astype(dtype))
input_face2 = tvm.nd.array(face_2.astype(dtype))
module = graph_runtime.create(loaded_json, loaded_lib, ctx)
module.load_params(loaded_params)
 
 
module.run(data=input_face1)
v1 = module.get_output(0).asnumpy()[0]
module.run(data=input_face2)
v2 = module.get_output(0).asnumpy()[0]
num=float(np.sum(v1*v2))
denom=np.linalg.norm(v1)*np.linalg.norm(v2)
cos = num / denom
 
print("***************")
print(v1)
print("***************")
print(v2)
print("***************")
print("dist: ", cos)
