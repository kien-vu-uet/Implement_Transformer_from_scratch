
import torch
import torch.nn as nn
import os
import time
# from dataset import COVID19Dataset
from dataset import COVID19Dataset2
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from modules.bert import BERT
import json
from utils import move_to_cuda, update_confusion_matrix, \
    plot_confusion_matrix, compute_score, get_pretrained_vocab

def infer(model, loader, criterion, device):
    tag_vocab_size = loader.dataset.tag_vocab_size
    pad_tag_id = loader.dataset.pad_tag_id
    outline_tag_id = loader.dataset.outline_tag_id
    cls_names = list(loader.dataset.tag2idx.keys())

    # For evalution
    print('|-------------------------------------------------------------------------------------------|')

    test_loss = 0.      
    cm = torch.zeros((tag_vocab_size-1, tag_vocab_size-1), dtype=torch.long)

    for i, batch in enumerate(loader):
        input_ids, attn_mask, target_tags = move_to_cuda(batch, device)
        with torch.no_grad(): 
            logit_tags = model(input_ids, attn_mask)
            loss = criterion(logit_tags.view(-1, logit_tags.size(-1)), target_tags.view(-1))
            test_loss += loss
            cm = update_confusion_matrix(cm, logit_tags.argmax(dim=-1), target_tags, pad_tag_id, outline_tag_id)
            
    test_loss /= len(loader)
    
    print(f"Val:   loss         {test_loss:0.4f}\n"
            f"       f1_macro     {compute_score(cm, metric='f1_macro'):0.2f}\n"
            f"       f1_weighted  {compute_score(cm, metric='f1_weighted'):0.2f}\n"
            f"       accuracy     {compute_score(cm, metric='accuracy'):0.2f}\n")
    plot_confusion_matrix(cm, cls_names[:-1], fig_path=f'heatmap/test/test.png')

    print('|-------------------------------------------------------------------------------------------|')

if __name__ == "__main__":
    MAX_LENGTH = 60
    BATCH_SIZE = 128
    BEST_MODEL_PARAMS_PATH = "model_weights/best_model_params_word_large.pt"
    TEST_PATH = '/workspace/nlplab/kienvt/PhoNER_COVID19_implement/data/word/test_word.json'

    # Prepare data    
    tokenizer, tagger = get_pretrained_vocab('vocab_file/tokenizer_word.json',
                                              'vocab_file/tagger.json')

    test_raw_data = json.load(open(TEST_PATH, 'r'))
    test_set = COVID19Dataset2(test_raw_data, tokenizer, tagger, MAX_LENGTH)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    # Define model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BERT(
        num_classes=test_set.tag_vocab_size, 
        clf_hidden_dim=1000,
        vocab_size=test_set.tokenizer.vocab_size,
        pad_token_id=test_set.tokenizer.pad_token_id,
        num_encoder_blocks=12, 
        num_heads=16,
        embed_dim=1024,
        qkv_bias=True,
        ff_dim=4096,
        ff_activate_fn='gelu',
        )
    model.load_state_dict(torch.load(BEST_MODEL_PARAMS_PATH))
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=test_set.pad_tag_id)

    infer(model, test_loader, criterion, device)