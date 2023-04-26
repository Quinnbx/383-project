import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Read the processed documents from the text file
with open("nba_all_star_posts.txt", "r", encoding="utf-8") as input_file:
    documents = input_file.readlines()

# Create a document-term matrix using the CountVectorizer
vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000)
document_term_matrix = vectorizer.fit_transform(documents)

# Fit the LDA model
num_topics = 5
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda_model.fit(document_term_matrix)

# Function to display the top words for each topic
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

# Display the top words for each topic
num_top_words = 20
feature_names = vectorizer.get_feature_names_out()
display_topics(lda_model, feature_names, num_top_words)
