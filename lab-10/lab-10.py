import nltk
from nltk import ngrams
from nltk.corpus import words
from tabulate import tabulate

nltk.download('words')

def build_ngram_set(n):
    english_corpus = set(words.words())
    ngram_set = set()

    for word in english_corpus:
        ngram_set.update(set(ngrams(word.lower(), n)))

    return ngram_set

def get_ngrams(word, n):
    return set(ngrams(word.lower(), n))

def calculate_ngram_similarity(input_word, candidate_word, n):
    input_ngrams = get_ngrams(input_word, n)
    candidate_ngrams = get_ngrams(candidate_word, n)
    common_ngrams = input_ngrams.intersection(candidate_ngrams)
    similarity = len(common_ngrams) / max(len(input_ngrams), len(candidate_ngrams))
    return similarity, common_ngrams

def spell_check(input_word, ngram_set, word_set, n):
    suggestions = sorted(word_set, key=lambda x: calculate_ngram_similarity(input_word, x, n), reverse=True)
    top_suggestions = suggestions[:3]
    return top_suggestions

if __name__ == "__main__":
    # Set the size of the n-grams (bigrams and trigrams in this case)
    bigram_set = build_ngram_set(2)
    trigram_set = build_ngram_set(3)
    word_set = set(words.words())

    # Input misspelled word
    misspelled_word = input("Enter a misspelled word: ")

    # Spell check using both bigrams and trigrams
    suggestions = spell_check(misspelled_word, trigram_set, word_set, 3)

    # Create a table with suggested words and the lengths of common bigrams and trigrams
    table_data = []
    for suggestion in suggestions:
        similarity, common_bigrams = calculate_ngram_similarity(misspelled_word, suggestion, 2)
        _, common_trigrams = calculate_ngram_similarity(misspelled_word, suggestion, 3)
        table_data.append([suggestion, len(common_bigrams), len(common_trigrams)])

    # Display the table
    headers = ["Word", "Length of Common Bigrams", "Length of Common Trigrams"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
