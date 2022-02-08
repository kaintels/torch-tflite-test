import tensorflow as tf

saved_model_dir = './tf_pb.pb'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]

converter.experimental_enable_resource_variables = True
tflite_model = converter.convert()
open('./tflite/converted_model.tflite', 'wb').write(tflite_model)