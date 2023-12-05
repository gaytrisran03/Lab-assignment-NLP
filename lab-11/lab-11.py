import nltk
from nltk import bigrams
from nltk.corpus import words
from nltk.metrics import edit_distance
from tabulate import tabulate

nltk.download('words')


def build_bigram_set():
    english_corpus = set(words.words())
    bigram_set = set()

    for word in english_corpus:
        bigram_set.update(set(bigrams(word.lower())))

    return bigram_set


def get_bigrams(word):
    return list(bigrams(word.lower()))


def calculate_bigram_similarity(input_word, candidate_word):
    input_bigrams = get_bigrams(input_word)
    candidate_bigrams = get_bigrams(candidate_word)

    common_bigrams = [bigram for bigram in input_bigrams if bigram in candidate_bigrams]

    if not common_bigrams:
        return 0.0, []

    # Calculate similarity based on the count and sequence of common bigrams
    count_similarity = len(common_bigrams) / max(len(input_bigrams), len(candidate_bigrams))

    # Weighted sum of positional differences
    position_diff_sum = sum(abs(input_bigrams.index(b) - candidate_bigrams.index(b)) for b in common_bigrams)
    sequence_similarity = 1 / (1 + position_diff_sum)

    # Combine count and sequence similarities (you can adjust the weights based on your preference)
    similarity = 0.7 * count_similarity + 0.3 * sequence_similarity

    return similarity, common_bigrams, count_similarity


def calculate_edit_distance(input_word, candidate_word):
    return edit_distance(input_word, candidate_word)


def spell_check(input_word, bigram_set, word_set):
    suggestions = sorted(word_set, key=lambda x: calculate_edit_distance(input_word, x))
    top_suggestions = suggestions[:3]
    return top_suggestions

def concatenate_bigrams(common_bigrams):
    return " ".join("".join(bigram) for bigram in common_bigrams)


if __name__ == "__main__":
    # Build bigram set and word set
    bigram_set = build_bigram_set()
    word_set = set(words.words())

    # Input misspelled word
    misspelled_word = input("Enter a misspelled word: ")

    # Spell check using edit distance
    suggestions = spell_check(misspelled_word, bigram_set, word_set)

    # Display top 3 suggestions with matching bigrams and their sequence in a table
    table_data = []
    for suggestion in suggestions:
        similarity, common_bigrams, count_similarity = calculate_bigram_similarity(misspelled_word, suggestion)
        table_data.append([suggestion, calculate_edit_distance(misspelled_word, suggestion), concatenate_bigrams(common_bigrams)])
        
    headers = ["Suggestion", "Edit Distance", "Common Bigrams"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
