# torch-tflite-test


## How To Use

1. Export onnx file `python export_onnx.py`

2. Simplify onnx file `simplify.bat out_model.onnx out_model.onnx`

3. get pb files `python export_pb.py`

4. make directory ./tflite `mkdir tflite`

5. get tflite files `python export_tflite.py`

You can check dependancy lib from `requirements.txt`
