import re

class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[token] for token in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join(self.int_to_str[i] for i in ids)
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


def vocali( text):
  tokens = re.split(r'([,.?_!"()\']|--|\s)', text)
  tokens = [t.strip() for t in tokens if t.strip()]
  vocab = {token: i for i, token in enumerate(sorted(set(tokens)))}
  return vocab
    

text = '''"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride.'''

# Build vocabulary
vocab = vocali(text);

tokenizer = SimpleTokenizer(vocab)

ids = tokenizer.encode(text)
print(ids)

decoded = tokenizer.decode(ids)
print(decoded)