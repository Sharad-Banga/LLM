import re

class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int 
                        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[token] for token in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join(self.int_to_str[i] for i in ids)
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


def vocali( text):
  tokens = re.split(r'([,.?_!"()\']|--|\s)', text)
  tokens = [t.strip() for t in tokens if t.strip()]
  all_tokens_array = sorted(set(tokens));
  all_tokens_array.extend(["<|endoftext|>", "<|unk|>"])
  vocab = {token: i for i, token in enumerate(all_tokens_array)}
  return vocab
    

text = '''"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride .'''
# Build vocabulary / training
vocab = vocali(text);
# print(vocab.items())



text1 = '''"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride hollla hingga .'''

text2 = '''"It's painted, you know,"
Mrs. Gisburn said gaya hollla hingga .'''

text3= "<|endoftext|>".join((text1,text2))

tokenizer = SimpleTokenizer(vocab)

ids = tokenizer.encode(text3)
print(ids)

decoded = tokenizer.decode(ids)
print(decoded)