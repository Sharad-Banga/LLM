import tiktoken

# make tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

text = (
 "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
 "of someunknownPlace vbisrv ier."
)

ids = tokenizer.encode(text,allowed_special={"<|endoftext|>"})
print(ids)

text_res = tokenizer.decode(ids)
print(text_res)