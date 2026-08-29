import email_parser as ep
import spacy

nlp = spacy.load('en_core_web_sm')

random_text = "It was amazing! It is a great pleasure to be a part of this band."
random_text_doc = nlp(random_text)
print("Text".ljust(10), ' ', "Alpha", "Space", "Stop", "Punct")
for token in random_text_doc:
    print(token.text.ljust(10), ':', token.is_alpha, token.is_space, token.is_punct)