import json
from collections import Counter
from typing import Any
from vocab import Vocab
import torch
from torch.utils.data import Dataset
import json

class COVID19Dataset2(Dataset):
    def __init__(
            self, 
            raw_data, 
            tokenizer: Vocab, 
            tagger: Vocab,
            max_length: int = 100):
        self.raw_data = raw_data
        self.map_data = {}
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.tagger = tagger

    @property
    def tag_vocab_size(self):
        return self.tagger.vocab_size
    
    @property
    def pad_tag_id(self):
        return self.tagger.pad_token_id

    @property
    def outline_tag_id(self):
        return self.tagger.stoi('O')
    
    @property
    def tag2idx(self):
        return self.tagger.word2index
    
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, index):
        if index not in self.map_data:
            item = self.raw_data[index]
            input_ids = [self.tokenizer.stoi2(w.lower()) for w in item['words']][:self.max_length]
            while len(input_ids) < self.max_length: input_ids.append(self.tokenizer.pad_token_id)
            target_tags = [self.tagger.stoi(t) for t in item['tags']][:self.max_length]
            while len(target_tags) < self.max_length: target_tags.append(self.tagger.pad_token_id)
            attn_mask = [i != self.tokenizer.pad_token_id for i in input_ids]
            self.map_data[index] = (
                torch.tensor(input_ids),
                torch.tensor(attn_mask, dtype=torch.float),
                torch.tensor(target_tags)
            )
        return self.map_data[index]
    

if __name__ == "__main__": 
    from utils import build_text_vocab, get_pretrained_vocab
    raw_data = json.load(open('/workspace/nlplab/kienvt/PhoNER_COVID19_implement/data/syllable/train_syllable.json', 'r'))

    tokenizer = build_text_vocab(raw_data)
    tagger = get_pretrained_vocab('vocab_file/tagger.json')
    
    dataset = COVID19Dataset2(raw_data, tokenizer=tokenizer, tagger=tagger, max_length = 40)
    item = dataset.__getitem__(42)
    # for i in item:  print(i)
    print(item[0])
    print(item[-1])
    print(dataset.tokenizer.detokenize(item[0].tolist()))
    print(dataset.tagger.detokenize(item[-1].tolist()))
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=16, )