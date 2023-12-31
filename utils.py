import string
from vocab import Vocab
from collections import Counter
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def resplit_data(raw_data, test_size=0.1):
    np.random.shuffle(raw_data)
    test_size = int(len(raw_data) * test_size)
    return raw_data[test_size:], raw_data[:test_size]

def build_text_vocab(raw_data):
    cnt = Counter()
    for item in raw_data:
        words = [_.lower() for _ in item['words']]
        cnt.update(words)
    tokenizer = Vocab(cnt, pad_token='<pad>', unk_token='<unk>',\
                       type_tokens=['<int>', '<float>', '<date>', '<name>'])

    print('Word vocab size', tokenizer.vocab_size)
    return tokenizer

def build_tag_vocab(raw_data):
    tag_cnt = Counter()
    for item in raw_data:
        tags = [_ for _ in item['tags']]
        tag_cnt.update(tags)
    tagger = Vocab(tag_cnt, pad_token='<pad>', unk_token=None, \
                   start_with_special_tokens=False)
    print('Tag vocab size', tagger.vocab_size)
    return tagger

def get_pretrained_vocab(fp):
    vocab = Vocab(Counter()).load(fp)
    return vocab

def move_to_cuda(batch, device):
    return (t.to(device) for t in batch)

def update_confusion_matrix(cm, input, target, ignore_index, outline_index=0):
    assert input.size() == target.size()
    input[input == ignore_index] = outline_index
    input = input[target != ignore_index].view(-1).tolist()
    target = target[target != ignore_index].view(-1).tolist()
    for i in range(len(target)):
        cm[target[i], input[i]] += 1
    return cm

def plot_confusion_matrix(cm, class_names, fig_path='foo.png'):
    with plt.rc_context({'figure.facecolor':'white'}):
        plt.figure(figsize=[10, 8])

        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names).astype(int)
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="crest")

        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=10)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=10)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(fig_path)
        plt.close(fig_path)


def compute_score(cm, metric='accuracy', eps=1e-12):
    tp = torch.diag(cm)
    fp = torch.sum(cm, axis=0) - tp
    fn = torch.sum(cm, axis=1) - tp
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall)
    if metric == 'f1_macro': return torch.sum(f1) * 1.0 / f1.size(0)
    elif metric == 'f1_weighted': return torch.sum(f1 * torch.sum(cm, axis=1)) * 1.0 / torch.sum(cm)
    else: return torch.sum(tp) * 1.0 / torch.sum(cm)

if __name__ == "__main__":
    import json
    raw_data = json.load(open('/workspace/nlplab/kienvt/PhoNER_COVID19_implement/data/syllable/dev_syllable.json', 'r'))
    train, test = resplit_data(raw_data, 0.1)
    print(train)
    print(test)