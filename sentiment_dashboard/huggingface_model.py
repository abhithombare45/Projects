from transformers import pipeline

sentiment_pipeline = pipeline(
    "sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment"
)
print(sentiment_pipeline("I love this project!"))
topic_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
text = "Apple releases new iPhone"
labels = ["technology", "politics", "sports"]
print(topic_pipeline(text, labels))
