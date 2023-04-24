from textblob import TextBlob

# Read the processed documents from the text file
with open("nba_all_star_posts.txt", "r", encoding="utf-8") as input_file:
    documents = input_file.readlines()

# Perform sentiment analysis and store the results
sentiment_scores = []
for doc in documents:
    sentiment_score = TextBlob(doc).sentiment.polarity
    sentiment_scores.append(sentiment_score)

# Calculate the average sentiment score
average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
print(f"Average sentiment: {average_sentiment}")

# Analyze individual documents' sentiment
for i, score in enumerate(sentiment_scores):
    sentiment = "Neutral"
    if score > 0.1:
        sentiment = "Positive"
    elif score < -0.1:
        sentiment = "Negative"
    print(f"Document {i + 1}: Sentiment - {sentiment} (Score: {score})")

# Calculate the average sentiment score
average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
print(f"Average sentiment: {average_sentiment}")

