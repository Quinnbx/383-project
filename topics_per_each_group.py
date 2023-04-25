import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

# Read the documents and their sentiment from the CSV file
documents = []
sentiments = []

with open("nba_all_star_sentiments.csv", "r", encoding="utf-8") as input_file:
    csv_reader = csv.reader(input_file)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        documents.append(row[0])
        sentiments.append(row[2])

# Group documents by sentiment
positive_docs = [doc for doc, sentiment in zip(documents, sentiments) if sentiment == "Positive"]
negative_docs = [doc for doc, sentiment in zip(documents, sentiments) if sentiment == "Negative"]
neutral_docs = [doc for doc, sentiment in zip(documents, sentiments) if sentiment == "Neutral"]

# Perform topic modeling for each group
no_features = 1000
no_topics = 5
no_top_words = 10

vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', random_state=0)

sentiment_groups = [("Positive", positive_docs), ("Negative", negative_docs), ("Neutral", neutral_docs)]

for sentiment, group_docs in sentiment_groups:
    if not group_docs:
        continue

    print(f"\nTopics for {sentiment} sentiment group:")
    X = vectorizer.fit_transform(group_docs)
    lda.fit(X)
    feature_names = vectorizer.get_feature_names()
    display_topics(lda, feature_names, no_top_words)
