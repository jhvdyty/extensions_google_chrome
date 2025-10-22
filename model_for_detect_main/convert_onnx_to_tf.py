import onnx
from onnx_tf.backend import prepare

# Загружаем ONNX модель
onnx_model = onnx.load("model.onnx")

# Конвертируем в TensorFlow SavedModel
tf_rep = prepare(onnx_model)
tf_rep.export_graph("model_tf")

print("✅ Конвертация в TensorFlow завершена успешно!")
