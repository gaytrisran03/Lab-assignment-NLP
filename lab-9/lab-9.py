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
Elephants are magnificent creatures that inhabit various forests across the globe. Their presence in these lush and diverse ecosystems is of utmost significance, as they play a pivotal role in maintaining the delicate balance of the forest's biodiversity. These intelligent and social animals, often referred to as the "keystone species" of the forest, have a profound impact on their habitat.

In the dense and sprawling forests they call home, elephants serve as nature's gardeners. They are known for their ability to shape the landscape through their feeding habits. By uprooting and munching on various plants and trees, they create clearings, which allow sunlight to reach the forest floor. This sunlight, in turn, promotes the growth of new plant species and enriches the forest's understory. These clearings also attract other herbivores, such as deer and antelope, which further contribute to the forest's overall health by aiding in seed dispersal.

Moreover, elephants are vital seed dispersers. After consuming fruits and vegetation, they traverse great distances, depositing seeds in their dung across the forest. This process contributes to the regrowth and diversification of the forest, ensuring a continuous cycle of renewal. Without elephants, many tree and plant species would struggle to propagate, potentially leading to imbalances and disruptions in the forest ecosystem.

The cultural and historical significance of elephants living in forests cannot be overstated. In various regions, these gentle giants are revered and even considered sacred animals. They have been intertwined with human history for millennia, serving as symbols of strength, wisdom, and majesty. Local communities have learned to coexist with elephants, forging unique and often deeply respectful relationships with these animals.

However, the peaceful coexistence of elephants and humans is becoming increasingly challenging due to habitat loss and human-wildlife conflicts. As forests are converted for agriculture and urban development, elephants often find themselves in proximity to human settlements, leading to conflicts that can have devastating consequences for both humans and elephants. Conservation efforts and responsible land-use planning are crucial to ensure the continued survival of elephants and the preservation of their forest homes.

In conclusion, elephants living in forests are not merely large, charismatic mammals; they are ecological architects, responsible for shaping and nurturing the ecosystems they inhabit. Their presence is a testament to the delicate balance of nature, and their conservation is not only an ethical imperative but also vital for the health and resilience of the world's forests. It is our collective responsibility to protect these magnificent creatures and the forests they call home, ensuring that future generations can witness the awe-inspiring beauty of elephants thriving in their natural habitat.
"""

# Summarize the text
summary = extractive_summarization(text, num_sentences=2)

# Print the summary
print(summary)
