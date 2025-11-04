# inclass-4-ready.py
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

# -------------------------
# Ensure required NLTK resources are downloaded
# -------------------------
resources = [
    'stopwords',
    'wordnet',
    'omw-1.4',
    'punkt',
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng'  # explicitly for Windows/Python 3.13
]

for r in resources:
    try:
        nltk.data.find(r)
    except LookupError:
        print(f"Downloading NLTK resource: {r} ...")
        nltk.download(r)

# -------------------------
# Initialize tools
# -------------------------
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
wnl = WordNetLemmatizer()

# -------------------------
# Sentences to process
# -------------------------
sentences = [
    "The duck will duck under the table.",
    "I will book a room to read a book."
]

# -------------------------
# Helper function: Map POS tags for WordNet lemmatizer
# -------------------------
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# -------------------------
# Process sentence function
# -------------------------
def process_sentence(sentence):
    tokens = word_tokenize(sentence)
    # Remove stopwords and non-alphabetic tokens
    filtered_tokens = [t for t in tokens if t.lower() not in stop_words and t.isalpha()]
    pos_tags = pos_tag(filtered_tokens)
    
    table = []
    for token, pos in pos_tags:
        stem = ps.stem(token)
        lemma = wnl.lemmatize(token, pos=get_wordnet_pos(pos))
        comment = ""
        if token.lower() in ['duck', 'book']:
            comment = f"Ambiguous POS; lemma may help disambiguate ({pos})"
        table.append([token, stem, lemma, pos, comment])
    return table

# -------------------------
# Print tables
# -------------------------
for s in sentences:
    print(f"\nSentence: {s}")
    print("{:<10} {:<10} {:<10} {:<6} {:<40}".format("Token","Stem","Lemma","POS","Comment"))
    print("-"*80)
    for row in process_sentence(s):
        print("{:<10} {:<10} {:<10} {:<6} {:<40}".format(*row))
