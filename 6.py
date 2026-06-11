import tensorflow as tf
import tf2onnx
import numpy as np
import json

@tf.keras.utils.register_keras_serializable()
def mfm(x):
    shape = tf.shape(x)
    x1 = x[:, :, :, :shape[3]//2]
    x2 = x[:, :, :, shape[3]//2:]
    return tf.maximum(x1, x2)

print("Loading model...")
model = tf.keras.models.load_model(
    "best_lcnn_final.keras",
    custom_objects={"mfm": mfm}
)
print(" Model loaded")

print("Converting to ONNX...")
input_signature = [
    tf.TensorSpec(shape=(None, 64, 126, 3),
                  dtype=tf.float32, name="input")
]

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=13,
    output_path="model.onnx"
)

print(" Saved as model.onnx")

import os
size = os.path.getsize("model.onnx") / 1024 / 1024
print(f"Model size: {size:.1f} MB")