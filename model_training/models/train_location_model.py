# model_training/train_hashtag_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os


def main():
    # Load data
    data = pd.read_csv("data/location_hashtag_data.csv")

    # Split data
    X = data[["latitude", "longitude"]]  # Features: latitude and longitude
    y = data["hashtags"]  # Target: relevant hashtags
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Concatenate latitude and longitude into a single string
    X_train_concat = X_train.apply(lambda x: " ".join(map(str, x)), axis=1)
    X_test_concat = X_test.apply(lambda x: " ".join(map(str, x)), axis=1)

    # Vectorize location data
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train_concat)
    X_test_vec = vectorizer.transform(X_test_concat)

    # Train model
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    # Evaluate model
    train_score = model.score(X_train_vec, y_train)
    test_score = model.score(X_test_vec, y_test)
    print("Train Score:", train_score)
    print("Test Score:", test_score)

    # Create directory if it doesn't exist
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")

    # Save model
    joblib.dump(model, "saved_models/train_location_model.pkl")
    joblib.dump(vectorizer, "saved_models/train_location_model_vectorizer.pkl")


if __name__ == "__main__":
    main()
