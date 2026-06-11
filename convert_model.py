import tensorflow as tf
import tf2onnx
import numpy as np
import os
MODEL_PATH = "C:/Users/tempadmin/Downloads/python/mars_project/best_lcnn_final.keras"
ONNX_PATH  = "C:/Users/tempadmin/Downloads/python/mars_project/model.onnx"

@tf.keras.utils.register_keras_serializable()
def mfm(x):
    shape = tf.shape(x)
    x1 = x[:, :, :, :shape[3]//2]
    x2 = x[:, :, :, shape[3]//2:]
    return tf.maximum(x1, x2)

print("Loading model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"mfm": mfm}
)
print(" Model loaded successfully")
print(f"Input shape: {model.input_shape}")

print("\nConverting to ONNX...")
input_signature = [
    tf.TensorSpec(
        shape=(None, 64, 126, 3),
        dtype=tf.float32,
        name="input"
    )
]

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=13,
    output_path=ONNX_PATH
)

print(f" Saved as model.onnx")
size = os.path.getsize(ONNX_PATH) / 1024 / 1024
print(f"Model size: {size:.1f} MB")
print("\n Conversion complete!")