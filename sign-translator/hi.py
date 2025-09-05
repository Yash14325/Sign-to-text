from tensorflow.keras.models import load_model

model = load_model("model/asl_model.h5")
print("Model output shape:", model.output_shape)
print(model.summary())
