# predict_hashtags.py

import pandas as pd
import joblib


def main(latitude, longitude):
    # Load the trained model
    model = joblib.load(
        "/home/airbornharsh/Programming/internship/freelancer/social-media-hashtag-username/model_training/saved_models/train_location_model.pkl"
    )

    # Transform latitude and longitude into a single string
    input_data = pd.DataFrame({"latitude": [latitude], "longitude": [longitude]})
    input_concat = input_data.apply(lambda x: " ".join(map(str, x)), axis=1)

    # Load the vectorizer used during training
    vectorizer = joblib.load(
        "/home/airbornharsh/Programming/internship/freelancer/social-media-hashtag-username/model_training/saved_models/train_location_model_vectorizer.pkl"
    )

    # Transform the input data using the same vectorizer
    input_vec = vectorizer.transform(input_concat)

    # Make predictions
    hashtags = model.predict(input_vec)
    return hashtags


if __name__ == "__main__":
    # Example usage
    latitude = 40.7128
    longitude = -74.0060
    predicted_hashtags = main(latitude, longitude)
    print(predicted_hashtags)
