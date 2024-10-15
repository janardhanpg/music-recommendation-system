# Music Recommender System

This project is a **Music Recommender System** built with Streamlit and Spotipy (Spotify API wrapper). The system recommends songs based on a selected song, retrieving related tracks and displaying album covers using Spotify's public API.

## Features

- **Song Recommendation**: Given a song, the system recommends 5 similar songs.
- **Album Cover Display**: Displays the album cover of the recommended songs using data from the Spotify API.
- **Interactive UI**: Built with Streamlit, allowing users to select a song from a dropdown and receive instant recommendations.

## Technologies Used

- **Streamlit**: For creating the interactive user interface.
- **Spotipy**: For fetching song details, including album covers from Spotify.
- **Pandas**: For data manipulation and analysis.
- **NLTK**: For natural language processing tasks, including tokenization and stemming.
- **Scikit-learn**: For calculating cosine similarity and building the recommendation model.
- **Pickle**: For saving the trained model and dataset.
- **Python**: The core language used for development.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/janardhanpg/music-recommendation-system.git
    ```
   
2. Navigate to the project directory:
    ```bash
    cd music-recommendation-system
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up your Spotify API credentials:
   - Create an application in the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/applications).
   - Copy the `CLIENT_ID` and `CLIENT_SECRET` values and update the script accordingly:
    ```python
    CLIENT_ID = "your_client_id"
    CLIENT_SECRET = "your_client_secret"
    ```

5. Download the dataset from Kaggle:
   - [Spotify Million Song Dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset).
   - Ensure the dataset (`spotify_millsongdata.csv`) is in the project directory.

## Model Training

The recommendation model is trained using a dataset containing song data. Here’s a brief overview of the steps involved in the training process:

1. **Data Loading**: The dataset is loaded from a CSV file using Pandas.

    ```python
    df = pd.read_csv('spotify_millsongdata.csv')
    ```

2. **Data Preprocessing**:
   - Randomly samples 5,000 rows from the dataset for training.
   - Drops the `link` column that is not needed for the recommendation.
   - Converts all text to lowercase and removes special characters and newline characters.

    ```python
    df['text'] = df['text'].str.lower().replace(r'\W\S', ' ').replace(r'\n', ' ', regex=True)
    ```

3. **Tokenization and Stemming**: The text data is tokenized, and stemming is applied using NLTK's PorterStemmer.

    ```python
    def tokenization(txt):
        tokens = nltk.word_tokenize(txt)
        stemming = [stemmer.stem(w) for w in tokens]
        return " ".join(stemming)

    df['text'] = df['text'].apply(lambda x: tokenization(x))
    ```

4. **Feature Extraction**: A TF-IDF vectorizer is used to convert the text data into numerical format suitable for machine learning.

    ```python
    tfid = TfidfVectorizer(analyzer="word", stop_words='english')
    matrix = tfid.fit_transform(df['text'])
    ```

5. **Cosine Similarity Calculation**: The cosine similarity between the songs is computed to find the related tracks.

    ```python
    similar = cosine_similarity(matrix)
    ```

6. **Saving the Model**: The trained model and dataset are saved using pickle for future use.

    ```python
    pickle.dump(similar, open('similarity.pkl', 'wb'))
    pickle.dump(df, open('df.pkl', 'wb'))
    ```

## Usage

1. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Open your browser and go to `http://localhost:8501`.

3. Select a song from the dropdown and click **Show Recommendation** to view similar song recommendations with album covers.

## Project Structure

