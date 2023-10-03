import nltk
from nltk.corpus import wordnet
from nltk.wsd import lesk
import time

# Sample sentence and ambiguous word
input_sentence = "The company used advanced technology to monitor the environmental impact of its manufacturing process."
input_ambiguous_word = "monitor"


# Function to map Treebank POS tags to WordNet POS tags
def map_pos_to_wordnet(treebank_tag):
    if treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    elif treebank_tag.startswith('J'):
        return wordnet.ADJ
    else:
        return None


# Lesk Algorithm Variation
def Simplified_lesk(context_sentence, ambiguous_word):
    best_sense = None
    max_similarity = 0

    # Tokenize the context sentence
    context_tokens = set(nltk.word_tokenize(context_sentence.lower()))

    # Get the POS tag for the ambiguous word
    ambiguous_word_pos = map_pos_to_wordnet(nltk.pos_tag([ambiguous_word])[0][1])

    # Iterate over each sense of the ambiguous word in WordNet
    for sense in wordnet.synsets(ambiguous_word, pos=ambiguous_word_pos):
        signature = set(nltk.word_tokenize(sense.definition().lower()))
        for example in sense.examples():
            signature.update(set(nltk.word_tokenize(example.lower())))

        # Calculate Jaccard similarity between context and sense signature
        similarity = len(context_tokens.intersection(signature)) / len(context_tokens.union(signature))

        # Update the best sense if a higher similarity is found
        if similarity > max_similarity:
            max_similarity = similarity
            best_sense = sense

    return best_sense


# Perform WSD using Simplified Lesk Algorithm
start_time1 = time.time()
Simplified_lesk_result = Simplified_lesk(input_sentence, input_ambiguous_word)
Simplified_time = time.time() - start_time1

# Perform WSD using Lesk
start_time2 = time.time()
nltk_lesk_result = lesk(input_sentence, input_ambiguous_word)
nltk_time = time.time() - start_time2

# Print the results and time taken in a different format
print("Simplified Lesk Algorithm Result:")
if Simplified_lesk_result:
    print("    Synset:", Simplified_lesk_result)
    print("    Definition:", Simplified_lesk_result.definition())
else:
    print("    No sense found.")

print("\nLesk Algorithm Result:")
if nltk_lesk_result:
    print("    Synset:", nltk_lesk_result)
    print("    Definition:", nltk_lesk_result.definition())
else:
    print("    No sense found.")

print("\nTime Taken:")
print("    Simplified Lesk Algorithm:", Simplified_time, "seconds")
print("    Lesk Algorithm:", nltk_time, "seconds")