import os
from tqdm import tqdm
from collections import Counter
from datasets import load_from_disk

dataset_dir = 'fineweb_10gb_dataset'

if os.path.exists(dataset_dir):
    data = load_from_disk(dataset_dir)
    print(data[0])
    print(len(data))

    word_counter = Counter()
    all_words = []

    for record in tqdm(data, desc='Processing data'):
        text = record['text']
        
        words = text.split(' ')

        word_counter.update(words)

print(len(word_counter))
# 17.45 M unique words in the dataset
