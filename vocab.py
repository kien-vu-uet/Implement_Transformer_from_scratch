import string
from typing import Optional
from collections import Counter
import json

def clean_word(word: str, special_sep='"&\'()*+-;?'):
    if word == '': return word
    integers = [str(_) for _ in range(10)]
    punctuation = list(string.punctuation) 
    prev = []
    while len(word) > 0:
        if word[0] in punctuation:
            prev.append(word[0])
            word = word[1:]
        elif word[0] in integers:
            if prev != [] and prev[-1][-1] in integers:
                prev[-1] += word[0]
            else:
                prev.append(word[0])
            word = word[1:]
        else: break

    post = []
    while len(word) > 0:
        if word[-1] in punctuation:
            post.append(word[-1])
            word = word[:-1]
        elif word[-1] in integers:
            if post != [] and post[-1][-1] in integers:
                post[-1] += word[-1]
            else:
                post.append(word[-1])
            word = word[:-1]
        else: break
    # word = list(word)
    # words = []
    # for i in word:
    #     if i not in special_sep and words != [] and words[-1][-1] not in special_sep:
    #             words[-1] += i
    #     else:
    #         words.append(i)
    return prev + [word] + post


def is_float(token: str):
    try:
        token = float(token.replace(',', '.'))
        return True
    except Exception:
        pass
    return False

def is_integer(token: str):
    try:
        token = token.replace(' ', '').replace('.', '')
        token = int(token)
        return True
    except Exception:
        return False
    
def is_date(token: str):
    try:
        tokens = token.split('/')
        return (is_integer(tokens[0]) and int(tokens[0]) in range(0, 32)) \
            and (is_integer(tokens[1]) and int(tokens[1]) in range(0, 13))
    except Exception:
        return False
    
def is_name(token: str):
    try: 
        while len(token) > 0:
            if ord(token[0]) not in range(ord('a'), ord('z') + 1) or token[1] != '.': 
                return False
            else:
                token = token[2:]
        return True
    except Exception:
        return False

def tokenize_type(token: str):
    if is_integer(token): return '<int>'
    if is_float(token): return '<float>'
    if is_date(token): return '<date>'
    if is_name(token): return '<name>'
    return token

class Vocab:
    def __init__(
            self,
            counter: Counter,
            pad_token: str = '<pad>',
            unk_token: Optional[str] = None,
            start_with_special_tokens: Optional[bool] = True,
            type_tokens: Optional[list] = None,
    ):
        word_start_idx = 2 if start_with_special_tokens else 0
        self.word2index = {k:v for k, v in zip(
                                        list(counter.keys()),
                                        range(word_start_idx, len(counter) + word_start_idx)
                                        )}
        self.pad_token = pad_token
        self.pad_token_id = 0 if start_with_special_tokens else len(self.word2index)
        self.word2index[pad_token] = self.pad_token_id
        if unk_token is not None:
            self.unk_token = unk_token
            self.unk_token_id = 1 if start_with_special_tokens else len(self.word2index)
            self.word2index[unk_token] = self.unk_token_id
        else:
            self.unk_token = ''
            self.unk_token_id = -1
        if type_tokens is not None:
            for t in type_tokens:
                self.word2index[t] = len(self.word2index)
        self.index2word = {v:k for k, v in self.word2index.items()}
        self.punc_tokens = [p for p in string.punctuation if p in self.word2index.keys()]

    def __len__(self):
        return len(self.word2index)
    
    @property
    def vocab_size(self):
        return len(self) + int(self.unk_token != '')
    
    @property
    def vocab(self):
        return self.word2index

    def stoi(self, w: str):
        if w not in self.word2index.keys():
            return self.unk_token_id
        return self.word2index[w]
    
    def stoi2(self, w: str):
        w = tokenize_type(w)
        if w not in self.word2index.keys():
            return self.unk_token_id
        return self.word2index[w]
    
    def itos(self, i: int):
        if i not in self.index2word.keys():
            return self.unk_token
        return self.index2word[i]
    

    def tokenize(
            self, 
            text: str or list(str), 
            max_length: Optional[int] = None, 
            do_lower: bool = True,
            do_clean: bool = True,
            return_dict: bool = False,
        ):
        if isinstance(text, str):
            while '  ' in text: text = text.replace('  ', ' ')
            words = text.strip().split()
        else:
            words = text
        words = [_ for _ in words if _ != '']
        if do_lower: words = [w.lower() for w in words]
        cleaned_words = []
        if do_clean:
            for w in words:
                cleaned_words += clean_word(w)
        else:
            cleaned_words = words
        token_ids = [self.stoi2(w) for w in cleaned_words]
        if return_dict:
            return [(k, v) for k, v in zip(cleaned_words, token_ids)]
        if max_length is not None:
            while len(token_ids) < max_length: token_ids.append(self.pad_token_id)
        return token_ids[:max_length]      
    

    def detokenize(
            self,
            ids: list[int],
            add_special_tokens: bool = True,
            unk_token_alternative: Optional[str] = None,
    ):  
        if add_special_tokens == False:
            ids = [i for i in ids if i not in [self.pad_token_id, self.unk_token_id]]
        text = [self.itos(_) for _ in ids]
        if unk_token_alternative is not None:
            for t in range(len(text)):
                if text[t] == self.unk_token:
                    text[t] = unk_token_alternative
        return text
    
    def save(self, fp):
        data = {
            'word2index' : self.word2index,
            'pad_token' : self.pad_token,
            'pad_token_id' : self.pad_token_id,
            'unk_token' : self.unk_token,
            'unk_token_id' : self.unk_token_id
        }
        json.dump(data, open(fp, 'w'), indent=4)

    def load(self, fp):
        data = json.load(open(fp, 'r'))
        self.word2index = data['word2index']
        self.index2word = {v:k for k, v in self.word2index.items()}
        self.pad_token = data['pad_token']
        self.pad_token_id = data['pad_token_id']
        self.unk_token = data['unk_token']
        self.unk_token_id = data['unk_token_id']
        return self


if __name__ == "__main__":
    import json
    from collections import Counter
    data = json.load(open('/workspace/nlplab/kienvt/PhoNER_COVID19_implement/data/syllable/train_syllable.json', 'r'))
    cnt = Counter()
    for item in data:
        tags = [_ for _ in item['tags']]
        cnt.update(tags)
    tagger = Vocab(cnt, pad_token='<pad>', unk_token=None, start_with_special_tokens=False)
    print(tagger.vocab)
    print(tagger.vocab_size)

    print(is_date('30/11'))