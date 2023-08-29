import string
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')

def beautified_print(label, value):
    print(f"{label:<25} {value}")

# Read the input text file
def read_text_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()


# Tokenize text into words
def tokenize_text(text):
    return word_tokenize(text.lower())


# Remove punctuation from a list of words
def remove_punctuation(word_list):
    return [word.strip(string.punctuation) for word in word_list]


# Remove stop words from a list of words using NLTK's predefined stop words
def remove_stop_words(word_list):
    stop_words = set(stopwords.words('english'))
    return [word for word in word_list if word not in stop_words]


# Calculate frequency of unique tokens
def calculate_token_frequency(word_list):
    return Counter(word_list)


# Analyze and print results with improved formatting
def analyze_results(token_frequencies, condition):
    print("\n" + "="*50)
    print(f"{condition}".center(45))
    print("="*50 + "\n")

    total_tokens = sum(token_frequencies.values())
    unique_tokens = len(token_frequencies)
    most_common_tokens = token_frequencies.most_common(10)

    beautified_print("Total Tokens ->", total_tokens)
    beautified_print("Unique Tokens ->", unique_tokens)

    print("\nMost Common Tokens:")
    print("\n")
    for token, frequency in most_common_tokens:
        beautified_print(f"{token.capitalize()} ->", frequency)

    


if __name__ == "__main__":
    filename = "lab-1/input.txt"  # Replace with your input file
    text = read_text_file(filename)

    words_with_punctuation = tokenize_text(text)
    words_without_punctuation = remove_punctuation(words_with_punctuation)
    words_without_stopwords = remove_stop_words(words_without_punctuation)

    freq_with_punctuation = calculate_token_frequency(words_with_punctuation)
    freq_without_punctuation = calculate_token_frequency(words_without_punctuation)
    freq_without_stopwords = calculate_token_frequency(words_without_stopwords)

    analyze_results(freq_with_punctuation, " a) including punctuation")
    analyze_results(freq_without_punctuation, " b) excluding punctuation")
    analyze_results(freq_without_stopwords, "c) excluding punctuation and stop words")
