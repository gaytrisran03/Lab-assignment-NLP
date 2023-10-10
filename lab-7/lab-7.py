import spacy

# Load the spaCy NER model
nlp = spacy.load("en_core_web_sm")

# Sample text for testing
sample_text = """
The United Nations is headquartered in New York City , and its Secretary-General, Gaytri Sran , will address the General Assembly on September 21, 2023  at 10:00 AM in the iconic United Nations Headquarters . The cost of the event is $100, which includes a 5% discount
"""

# Process the sample text with spaCy
doc = nlp(sample_text)

# Extract named entities
entities = []
for ent in doc.ents:
    entities.append((ent.text, ent.label_))

# Define entity types for 3-class, 4-class, and 7-class NER
entity_types_3class = ["ORG", "PERSON", "GPE"]
entity_types_4class = ["ORG", "PERSON", "GPE", "DATE"]
entity_types_7class = ["ORG", "PERSON", "GPE", "DATE", "TIME", "PERCENT", "MONEY"]

# Create dictionaries to hold entities for each class
classified_entities_3class = {label: [] for label in entity_types_3class}
classified_entities_4class = {label: [] for label in entity_types_4class}
classified_entities_7class = {label: [] for label in entity_types_7class}

# Categorize entities
for text, label in entities:
    if label in entity_types_3class:
        classified_entities_3class[label].append(text)
    if label in entity_types_4class:
        classified_entities_4class[label].append(text)
    if label in entity_types_7class:
        classified_entities_7class[label].append(text)

# Print the detected entities for each class
print("3-Class NER:")
for label, entities in classified_entities_3class.items():
    print(label, "entities:", entities)

print("\n4-Class NER:")
for label, entities in classified_entities_4class.items():
    print(label, "entities:", entities)

print("\n7-Class NER:")
for label, entities in classified_entities_7class.items():
    print(label, "entities:", entities)
