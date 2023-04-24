import praw
import datetime
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download the NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

reddit = praw.Reddit(
    client_id="WIfvDgkkbZhjCNlHihdPkw",
    client_secret="gG8njzu997dimnAivJ8x2bsOYirh4Q",
    user_agent="NBAAllStar2023Analysis/1.0 by YourUsername",
    username="Intelligent_Dot_5855",
    password="74Q7u/zm/!-%J)h",
)

target_keywords = ['2023 NBA All-star', '2023 NBA All-star game', '2023 NBA All-star voting']
subreddit = reddit.subreddit("nba")

# Define the time range for collecting data
start_time = int(datetime.datetime(2022, 10, 18).timestamp())
end_time = int(datetime.datetime(2023, 4, 9).timestamp())

# Collect posts with target keywords
all_posts = []
for keyword in target_keywords:
    posts = subreddit.search(
        query=keyword,
        time_filter="all",
        sort="relevance",
        limit=None,
        syntax="lucene"
    )

    for post in posts:
        post_creation_time = post.created_utc
        if start_time <= post_creation_time <= end_time:
            all_posts.append(post)

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove URLs and HTML tags
    text = re.sub(r'http\S+|www\S+|https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)

    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize the text
    word_tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in word_tokens if word not in stop_words]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Remove short words
    meaningful_tokens = [token for token in lemmatized_tokens if len(token) >= 3]

    return ' '.join(meaningful_tokens)

# Process data and store in a text file
with open("nba_all_star_posts.txt", "w", encoding="utf-8") as output_file:
    for post in all_posts:
        processed_text = preprocess_text(post.selftext)
        output_file.write(processed_text + "\n")
