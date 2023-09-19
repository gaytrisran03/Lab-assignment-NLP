import nltk

# Define the grammar
grammar2 = nltk.CFG.fromstring("""
  S  -> NP VP | NP PP | PP PP
  NP -> Det Nom | PropN
  Nom -> Adj Nom | N
  VP -> V Adj | V NP | V S | V NP PP
  PP -> P NP
  PropN -> 'Buster' | 'Chatterer' | 'Ram'
  Det -> 'the' | 'a'
  N -> 'bear' | 'squirrel' | 'tree' | 'fish' | 'log'
  Adj  -> 'angry' | 'frightened' |  'little' | 'tall'
  V ->  'chased'  | 'saw' | 'said' | 'thought' | 'was' | 'put'
  P -> 'on'
  """)

# Create a RecursiveDescentParser
rd_parser = nltk.RecursiveDescentParser(grammar2)

# Sentence for NP VP
sent_np_vp = 'Ram saw the angry bear'

# Sentence for NP PP
sent_np_pp = 'Ram saw the bear on the log'

# Sentence for PP PP
sent_pp_pp = 'Ram on the log'

# Helper function to parse and draw a sentence
def parse_and_draw(sentence):
    sent_split = sentence.split()
    print(f"Sentence: {sentence}\n")
    for tree in rd_parser.parse(sent_split):
        tree.pretty_print()
        tree.draw()
    print("\n-----------------------------------\n")

# Parse and draw the sentences
parse_and_draw(sent_np_vp)
parse_and_draw(sent_np_pp)
parse_and_draw(sent_pp_pp)