import string
from vocab import Vocab
from collections import Counter
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def build_vocab(raw_data):
    cnt = Counter()
    tag_cnt = Counter()
    for item in raw_data:
        words = [_.lower() for _ in item['words']]
        tags = [_ for _ in item['tags']]
        cnt.update(words)
        tag_cnt.update(tags)
    tokenizer = Vocab(cnt, pad_token='<pad>', unk_token='<unk>')
    tagger = Vocab(tag_cnt, pad_token='<pad>', unk_token=None, start_with_special_tokens=False)
    print('Word vocab size', tokenizer.vocab_size)
    print('Tag vocab size', tagger.vocab_size)
    return tokenizer, tagger

def get_pretrained_vocab(tokenizer_fp, tagger_fp):
    tokenizer = Vocab(Counter()).load(tokenizer_fp)
    tagger = Vocab(Counter()).load(tagger_fp)
    return tokenizer, tagger

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