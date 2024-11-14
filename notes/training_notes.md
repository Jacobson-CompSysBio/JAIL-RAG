
# Simple LLM Training
## Forward Pass (Training)

![Training](images/LLM%20Q_A%20Fine-Tune.png)

1. tokenize descriptions, questions, and labels
    * descriptions: textual graph desc
    * questions: given queries
    * labels: optimal outputs
2. Encode special tokens
3. For each batch:
    1. combine bos, desc, question, eos_user, labels, and eos token
    2. embed using llama trained embedding table
    3. append to total batch list
    5. add 1s to attention mask wherever for length of input tokens
    5. add ignore tokens where label tokens occur in the input sequence to create output labels
    6. pad all batches to be even size

## Inference

1. tokenize desc, questions (no labels given)
2. encode + append special tokens
3. for each batch:
    1. embed using llama trained embedding table
    2. add bos token emb
    3. pad tokens
4. give input embeddings + attn mask to model
5. decode the token output to produce a sentence
6. return dict