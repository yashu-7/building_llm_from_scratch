import os
import re
from tqdm import tqdm
from collections import Counter
from datasets import load_from_disk

dataset_dir = 'fineweb_10gb_dataset'

if os.path.exists(dataset_dir):
    data = load_from_disk(dataset_dir)
    # print(data[0])
    print(len(data))

    word_counter = Counter()
    all_words = []

    pattern = re.compile(r'[a-zA-Z]+|[0-9]+|[^a-zA-Z0-9\s]')

    for record in tqdm(data, desc='Processing data'):
        text = record['text']
        
        words = pattern.findall(text.lower())

        word_counter.update(words)

print(len(word_counter))
print(word_counter.most_common(10))
vocab = sorted(word_counter.keys())
print(f"VOCAB SIZE: {len(vocab)}")
print(vocab[:20])

stoi = {word:i for i, word in enumerate(vocab)}
itos = {i:word for word,i in stoi.items()}

encoder = lambda s_list: [stoi[s] for s in s_list]
decoder = lambda i_list: [itos[i] for i in i_list]

test_sentence = "Here is the Sample, to test the tokenizer!"

s_list = pattern.findall(test_sentence.lower())

encoded_text = encoder(s_list)
print("Encoded Text:\n",encoded_text)

decoded_text = decoder(encoded_text)
print("Decoded Text:\n",decoded_text)

reconstructed_text = " ".join(decoded_text)
print("Reconstructed Text:\n", reconstructed_text)
