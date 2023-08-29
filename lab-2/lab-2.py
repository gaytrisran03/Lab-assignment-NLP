from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Initialize the Porter Stemmer
porter = PorterStemmer()

text = "The quick brown foxes are jumping over the lazy dogs' tails."

words = word_tokenize(text)

print("Original Word\t\tStemmed Word")
print("----------------------------------------")
for word in words:
    stemmed_word = porter.stem(word)
    print(f"{word}\t\t\t{stemmed_word}")
