import re
text = "Hello, world. This, is a test?."

# seperating comas , dot , white space , words
result = re.split(r'([,.?]|\s)', text); 

# removing whitespaces from array
word_token = [item for item in result if item.strip()] 


# all words without duplicate and sorted order
all_words = sorted(set(word_token))


vocab_len = len(all_words)

dict = enumerate(all_words)

vocab = {a:b for b,a in  dict}

print(vocab)

# for i,item in vocab.items():
#   print(i,item)
