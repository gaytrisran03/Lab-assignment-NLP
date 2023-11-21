import nltk
from nltk import bigrams
from nltk.corpus import words

nltk.download('words')


def build_bigram_set():
    english_corpus = set(words.words())
    bigram_set = set()

    for word in english_corpus:
        bigram_set.update(set(bigrams(word.lower())))

    return bigram_set


def get_bigrams(word):
    return set(bigrams(word.lower()))


def calculate_bigram_similarity(input_word, candidate_word):
    input_bigrams = get_bigrams(input_word)
    candidate_bigrams = get_bigrams(candidate_word)
    common_bigrams = input_bigrams.intersection(candidate_bigrams)
    similarity = len(common_bigrams) / max(len(input_bigrams), len(candidate_bigrams))
    return similarity


def spell_check(input_word, bigram_set, word_set):
    suggestions = sorted(word_set, key=lambda x: calculate_bigram_similarity(input_word, x), reverse=True)
    top_suggestions = suggestions[:5]
    return top_suggestions


if __name__ == "__main__":
    # Build bigram set and word set
    bigram_set = build_bigram_set()
    word_set = set(words.words())

    # Input misspelled word
    misspelled_word = input("Enter a misspelled word: ")

    # Spell check using bigram similarity
    suggestions = spell_check(misspelled_word, bigram_set, word_set)

    # Display top 5 suggestions
    print("Top 5 suggestions:")
    for suggestion in suggestions:
        print(suggestion)