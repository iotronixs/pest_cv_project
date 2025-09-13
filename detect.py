from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the full model
model = load_model("pest_model_final.keras")

# Class labels (example: adjust to match your dataset order!)
class_names = [
    "Maize","Tomato"
]

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return predicted_class, confidence

# Example usage
if __name__ == "__main__":
    result, conf = predict_image("static/uploads/9fe7e9d5-80bb-4d27-8d08-29facd0eb78b.jpg")
    print(f"Prediction: {result} ({conf*100:.2f}%)")
