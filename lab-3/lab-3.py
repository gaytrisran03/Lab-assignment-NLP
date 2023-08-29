from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample documents
document1 = "This is the first document containing some unique words."
document2 = "Here is another document with some different unique words."

# Tokenize documents and create sets of unique words
words1 = set(word_tokenize(document1.lower()))
words2 = set(word_tokenize(document2.lower()))

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_words1 = [word for word in words1 if word not in stop_words]
filtered_words2 = [word for word in words2 if word not in stop_words]

# Combine filtered words from both documents
all_unique_words_without_stopwords = set(filtered_words1).union(filtered_words2)
all_unique_words_with_stopwords = words1.union(words2)

# Create TF-IDF vectors
vectorizer_without_stopwords = TfidfVectorizer(vocabulary=all_unique_words_without_stopwords)
tfidf_matrix_without_stopwords = vectorizer_without_stopwords.fit_transform([document1, document2])

vectorizer_with_stopwords = TfidfVectorizer(vocabulary=all_unique_words_with_stopwords)
tfidf_matrix_with_stopwords = vectorizer_with_stopwords.fit_transform([document1, document2])

# Calculate cosine similarity
cosine_similarity_value_without_stopwords = cosine_similarity(tfidf_matrix_without_stopwords[0], tfidf_matrix_without_stopwords[1])[0][0]
cosine_similarity_value_with_stopwords = cosine_similarity(tfidf_matrix_with_stopwords[0], tfidf_matrix_with_stopwords[1])[0][0]

print(f"Cosine similarity between the two documents (without stopwords): {cosine_similarity_value_without_stopwords}")
print(f"Cosine similarity between the two documents (with stopwords): {cosine_similarity_value_with_stopwords}")
