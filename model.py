import re
import string
import pickle
import pandas as pd
import numpy as np
import nltk

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords

# Download required NLTK assets
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')


class SentimentRecommenderModel:

    ROOT_PATH = "pickle/"
    RECOMMENDER = "user_final_rating.pkl"
    CLEANED_DATA = "cleaned-data.pkl"
    VECTORIZER = "tfidf-vectorizer.pkl"
    MODEL_NAME = "sentiment-classification-xg-boost-model.pkl"

    def __init__(self):
       # Load user–product collaborative filtering recommendation matrix
        self.user_final_rating = pickle.load(
            open(self.ROOT_PATH + self.RECOMMENDER, "rb")
        )

        # Load cleaned review dataset
        self.cleaned_data = pickle.load(
            open(self.ROOT_PATH + self.CLEANED_DATA, "rb")
        )

        # Load TF-IDF vectorizer
        self.vectorizer = pickle.load(
            open(self.ROOT_PATH + self.VECTORIZER, "rb")
        )

        # Load trained XGBoost sentiment classifier
        self.model = pickle.load(
            open(self.ROOT_PATH + self.MODEL_NAME, "rb")
        )

        # Lemmatization + Stopwords
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    # -------------------------------------------------------------
    # RECOMMENDATION PIPELINE: TOP 20 CF → SENTIMENT FILTER → TOP 5
    # -------------------------------------------------------------
    def getSentimentRecommendations(self, user):

        if user not in self.user_final_rating.index:
            print(f"The User {user} does not exist. Please provide a valid user id")
            return None

        # Step 1: Get top 20 recommended product IDs
        top20_recommended_products = list(
            self.user_final_rating.loc[user]
            .sort_values(ascending=False)
            .head(20)
            .index
        )

        # Step 2: Extract only these products from cleaned review dataset
        top20_products = self.cleaned_data[
            self.cleaned_data["id"].isin(top20_recommended_products)
        ].copy()

        # Step 3: Predict sentiment for each review
        X = self.vectorizer.transform(top20_products["cleaned_review"])
        top20_products["predicted_sentiment"] = self.model.predict(X)

        # Step 4: Aggregation (Count of all sentiments per product)
        grouped_pred = (
            top20_products[["name", "predicted_sentiment"]]
            .groupby("name", as_index=False)
            .count()
        )

        # Step 5: Count of only positive sentiments
        grouped_pred["pos_review_count"] = grouped_pred["name"].apply(
            lambda product_name: top20_products[
                (top20_products["name"] == product_name)
                & (top20_products["predicted_sentiment"] == 1)
            ]["predicted_sentiment"].count()
        )

        # Step 6: Total review count
        grouped_pred["total_review_count"] = grouped_pred["predicted_sentiment"]

        # Step 7: Positive sentiment percentage
        grouped_pred["pos_sentiment_percent"] = (
            grouped_pred["pos_review_count"] / grouped_pred["total_review_count"] * 100
        ).round(2)

        # Step 8: Return top 5 recommended products
        result = grouped_pred.sort_values(
            "pos_sentiment_percent", ascending=False
        ).head(5)

        return result


    # -------------------------------------------------------------
    #           SENTIMENT CLASSIFICATION (TEXT INPUT)
    # -------------------------------------------------------------
    def classify_sentiment(self, clean_text):
        X = self.vectorizer.transform([clean_text])
        pred = self.model.predict(X)[0]  
        return int(pred)

    # -------------------------------------------------------------
    #           TEXT PREPROCESSING PIPELINE
    # -------------------------------------------------------------
    def preprocess_text(self, text):
        text = text.lower().strip()

        # Remove text in brackets
        text = re.sub(r"\[\s*\w*\s*\]", "", text)

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Remove words with numbers
        text = re.sub(r"\S*\d\S*", "", text)

        # Lemmatize + remove stopwords
        return self.lemma_text(text)

    def get_wordnet_pos(self, tag):
        if tag.startswith("J"):
            return wordnet.ADJ
        elif tag.startswith("V"):
            return wordnet.VERB
        elif tag.startswith("N"):
            return wordnet.NOUN
        elif tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def remove_stopword(self, text):
        return " ".join(
            [w for w in text.split() if w.isalpha() and w not in self.stop_words]
        )

    def lemma_text(self, text):
        filtered = self.remove_stopword(text)
        pos_tags = nltk.pos_tag(word_tokenize(filtered))

        lemma_words = [
            self.lemmatizer.lemmatize(word, self.get_wordnet_pos(tag))
            for (word, tag) in pos_tags
        ]
        return " ".join(lemma_words)
