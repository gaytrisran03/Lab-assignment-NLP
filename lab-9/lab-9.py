import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

def preprocess_text(text):
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = [word_tokenize(sentence) for sentence in sentences]

    # Remove punctuation and convert to lowercase
    table = str.maketrans('', '', string.punctuation)
    words = [[word.translate(table).lower() for word in sentence if word.isalpha()] for sentence in words]

    return sentences, words

def calculate_word_frequency(words):
    word_freq = {}
    for sentence in words:
        for word in sentence:
            if word not in stopwords.words('english'):
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1
    return word_freq

def calculate_sentence_scores(sentences, word_freq):
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                if sentence in sentence_scores:
                    sentence_scores[sentence] += word_freq[word]
                else:
                    sentence_scores[sentence] = word_freq[word]
    return sentence_scores

def extractive_summarization(text, num_sentences=2):
    sentences, words = preprocess_text(text)
    word_freq = calculate_word_frequency(words)
    sentence_scores = calculate_sentence_scores(sentences, word_freq)

    # Select the top sentences based on scores
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]

    return ' '.join(top_sentences)

# Input text
text = """
The Mars rover Perseverance has made a groundbreaking discovery on the Red Planet. NASA's Perseverance rover, which landed on Mars in February, has identified organic molecules in a rock sample it collected from the Jezero Crater. This finding raises the possibility that Mars may have once supported life or might still do so. Scientists are excited about the implications of this discovery and its potential impact on future missions to Mars.
"""

# Summarize the text
summary = extractive_summarization(text, num_sentences=2)

# Print the summary
print(summary)
