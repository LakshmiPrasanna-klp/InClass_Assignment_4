import spacy

# Load English model
nlp = spacy.load("en_core_web_sm")

# Sample paragraph (replace with exact slide paragraph)
paragraph = """
Christopher Robin was the son of Mr. Robin. Chris loved playing with Winnie the Pooh. 
He often visited his father, and Mr. Robin encouraged him. Chris wrote a poem about Pooh, 
and the book was later published.
"""

# -------------------------
# NER: Extract PERSON and ORG
# -------------------------
doc = nlp(paragraph)

entities = []
for ent in doc.ents:
    if ent.label_ in ["PERSON", "ORG"]:
        entities.append((ent.text, ent.label_))
print("Entities found:")
print(entities)

# -------------------------
# Coreference heuristic
# -------------------------

# Alias map for known aliases
alias_map = {
    "Mr. Robin": "Christopher Robin",
    "Chris": "Christopher Robin",
}

# Gendered pronouns
male_pronouns = ["he", "him", "his"]

# Store coref links
coref_links = {}

# Track last PERSON mention
last_person = None

normalized_tokens = []

for sent_id, sent in enumerate(doc.sents):
    for token in sent:
        token_lower = token.text.lower()
        if token.text in alias_map:
            antecedent = alias_map[token.text]
            coref_links[(token.text, sent_id)] = antecedent
            normalized_tokens.append(f"[{antecedent}]")
            last_person = antecedent
        elif token_lower in male_pronouns:
            if last_person:
                antecedent = last_person
                coref_links[(token.text, sent_id)] = antecedent
                normalized_tokens.append(f"[{antecedent}]")
            else:
                normalized_tokens.append(token.text)
        elif token.ent_type_ == "PERSON" and token.text not in alias_map.values():
            # Direct PERSON mention
            last_person = token.text
            normalized_tokens.append(f"[{token.text}]")
        else:
            normalized_tokens.append(token.text)

# Reconstruct normalized paragraph
normalized_paragraph = " ".join(normalized_tokens)

# -------------------------
# Output
# -------------------------
print("\nCoreference links (pronoun/alias → antecedent):")
for k,v in coref_links.items():
    print(f"{k} -> {v}")

print("\nNormalized paragraph:")
print(normalized_paragraph)

# -------------------------
# Failure case note
# -------------------------
failure_note = """
Failure case example: In "He often visited his father", the pronoun 'his' is linked to Christopher Robin 
by proximity, but semantically 'his father' refers to Mr. Robin. Without syntactic parsing or world knowledge, 
the heuristic may misassign 'his'. Additional signals such as dependency parse (subject-object relations) 
or discourse structure would improve accuracy.
"""
print("\nFailure case note:")
print(failure_note)
