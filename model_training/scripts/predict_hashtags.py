# predict_hashtags.py

import sys
import joblib

def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print("Error loading the model:", e)
        return None

def predict_hashtags(model, latitude, longitude):
    try:
        # Preprocess latitude and longitude if necessary
        # Example: You may need to convert latitude and longitude to floats
        latitude = float(latitude)
        longitude = float(longitude)

        # Make predictions using the loaded model
        hashtags = model.predict([[latitude, longitude]])
        return hashtags
    except Exception as e:
        print("Error predicting hashtags:", e)
        return None

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict_hashtags.py <latitude> <longitude>")
        sys.exit(1)

    latitude = sys.argv[1]
    longitude = sys.argv[2]

    # Load the trained model
    model_path = '/home/airbornharsh/Programming/internship/freelancer/social-media-hashtag-username/model_training/saved_models/train_location_model.pkl'  # Update with the actual path to your trained model
    model = load_model(model_path)

    if model is None:
        print("Failed to load the model.")
        sys.exit(1)

    # Make predictions
    hashtags = predict_hashtags(model, latitude, longitude)

    if hashtags is not None:
        print("\nPredicted Hashtags:")
        print(hashtags)
