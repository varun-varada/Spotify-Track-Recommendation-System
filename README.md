🎵 Music Recommendation System
This project is a content-based music recommendation system that suggests similar songs based on audio features, artist, and genre. It uses Nearest Neighbors to find songs that are most alike to a given input track.

📌 Features
Data Preprocessing: Cleans and encodes song metadata for machine learning.

Feature Encoding: Converts artist names, genres, and other categorical features into numerical form using LabelEncoder.

Model Training: Uses K-Nearest Neighbors to find similar songs.

Model Saving: Stores the trained model and encoders using joblib for later use.

Song Recommendations: Given a song, returns a list of the most similar tracks from the dataset.

⚙️ How It Works
Load Dataset → Reads the music dataset (CSV).

Preprocess Data → Removes missing values, selects relevant features.

Encode Features → Converts categorical data into numbers.

Train Model → Fits a Nearest Neighbors model on the processed dataset.

Save Model → Saves model and encoders for quick future use.

Recommend Songs → Finds and displays the top N similar songs to the user’s input.
