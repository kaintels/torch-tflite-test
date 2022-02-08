import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("./out_model.onnx")  # load onnx model

output = prepare(onnx_model)
output.export_graph("tf_pb.pb")