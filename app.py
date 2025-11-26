from flask import Flask, request, render_template
from model import SentimentRecommenderModel
import pandas as pd

app = Flask(__name__)

# Load model once
sentiment_model = SentimentRecommenderModel()


@app.route('/')
def home():
    return render_template(
        "index.html",
        column_names=None,
        row_data=None,
        sentiment=None,
        message=None
    )


@app.route('/predict', methods=['POST'])
def prediction():
    try:
        user = request.form.get('userName', '').strip()
        if not user:
            return render_template(
                "index.html",
                message="Please enter a valid user name!",
                column_names=None,
                row_data=None,
                sentiment=None
            )

        user = user.lower()
        result = sentiment_model.getSentimentRecommendations(user)
        print(result)

        if result is not None and not result.empty:
            return render_template(
                "index.html",
                column_names=result.columns.values.tolist(),
                row_data=result.values.tolist(),
                sentiment=None,
                message=None
            )
        else:
            return render_template(
                "index.html",
                message="User name doesn't exist or no product recommendations available!",
                column_names=None,
                row_data=None,
                sentiment=None
            )

    except Exception as e:
        print("ERROR in /predict:", e)
        return render_template(
            "index.html",
            message="An error occurred. Check server logs.",
            column_names=None,
            row_data=None,
            sentiment=None
        )


@app.route('/predictSentiment', methods=['POST'])
def predict_sentiment():
    try:
        review_text = request.form.get("reviewText", "").strip()

        if not review_text:
            return render_template(
                "index.html",
                sentiment="Please enter a review text.",
                column_names=None,
                row_data=None
            )

        # Clean and predict
        clean_text = sentiment_model.preprocess_text(review_text)
        print("Cleaned review text:", clean_text)

        pred_sentiment = sentiment_model.classify_sentiment(review_text)
        print("Model prediction:", pred_sentiment)

        return render_template(
            "index.html",
            sentiment=pred_sentiment,
            column_names=None,
            row_data=None,
            message=None
        )

    except Exception as e:
        print("ERROR in /predictSentiment:", e)
        return render_template(
            "index.html",
            sentiment="Error processing sentiment. Check logs.",
            column_names=None,
            row_data=None
        )


if __name__ == '__main__':
    app.run(debug=True)
