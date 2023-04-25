import csv
from textblob import TextBlob

def classify_sentiment(score):
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

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

# Save document content, sentiment score, and sentiment to a CSV file
with open("nba_all_star_sentiments.csv", "w", encoding="utf-8", newline='') as output_file:
    csv_writer = csv.writer(output_file)
    # Write the header row
    csv_writer.writerow(["Document", "Sentiment Score", "Sentiment"])

    for i, (doc, score) in enumerate(zip(documents, sentiment_scores)):
        sentiment = classify_sentiment(score)
        csv_writer.writerow([doc.strip(), score, sentiment])

# Analyze individual documents' sentiment
for i, score in enumerate(sentiment_scores):
    sentiment = classify_sentiment(score)
    print(f"Document {i + 1}: Sentiment - {sentiment} (Score: {score})")
