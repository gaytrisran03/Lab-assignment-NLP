import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Print NLTK version
print("NLTK version:", nltk.__version__)

# Initialize the Porter Stemmer
porter = PorterStemmer()

def custom_stem(word):
    if word.endswith("ies") and len(word) > 3 and word[-4] not in "aeiou":
        return word[:-3] + "y"
    if word.endswith("able") and len(word) > 5 and word[-5] in "aeiou":
        return word[:-4] + "e"
    if word.endswith("able") and len(word) > 5:
        return word[:-4]
    
    return porter.stem(word)

text = "tails and playing with babies parties companies studies breakable Readable Portable Enjoyable Adaptable Valuable Observable"

words = word_tokenize(text)

print("Original Word\t\tStemmed Word")
print("----------------------------------------")
for word in words:
    stemmed_word = custom_stem(word)
    print(f"{word}\t\t\t\t{stemmed_word}")
