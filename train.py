
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
from utils import build_vocab, move_to_cuda, update_confusion_matrix, \
    plot_confusion_matrix, compute_score, get_pretrained_vocab


def train(model, train_loader, val_loader, criterion, optimizer, epochs, device, model_path):
    tag_vocab_size = train_loader.dataset.tag_vocab_size
    pad_tag_id = train_loader.dataset.pad_tag_id
    outline_tag_id = train_loader.dataset.outline_tag_id
    cls_names = list(train_loader.dataset.tag2idx.keys())

    best_val_loss = 100.
    for ep in range(1, epochs + 1):
        avg_train_loss = 0.
        train_cm = torch.zeros((tag_vocab_size-1, tag_vocab_size-1), dtype=torch.long)

        for i, batch in enumerate(train_loader):
            input_ids, attn_mask, target_tags = move_to_cuda(batch, device)
            start = time.time()
            optimizer.zero_grad()
            logit_tags = model(input_ids, attn_mask, apply_tanh=True)
            loss = criterion(logit_tags.view(-1, logit_tags.size(-1)), target_tags.view(-1))
            loss.backward()
            optimizer.step()

            train_cm = update_confusion_matrix(train_cm, logit_tags.argmax(dim=-1), target_tags, pad_tag_id, outline_tag_id)
            
            avg_train_loss = (avg_train_loss * i + loss) / (i + 1)
            s_per_batch = time.time() - start
            print(f'| epoch {ep:3d} | | {(i+1):5d}/{len(train_loader):5d} batches | '
                f'| s/batch {s_per_batch:5.2f} | '
                f'| loss {int(loss):4d}.{int((loss - int(loss))*100):02d} | '
                f'| avg_loss {int(avg_train_loss):4d}.{int((avg_train_loss - int(avg_train_loss))*100):02d} |')
        # For evalution
        print('|-------------------------------------------------------------------------------------------|')

        print(f"\nTrain: f1_macro     {compute_score(train_cm, metric='f1_macro'):0.2f}\n"
              f"       f1_weighted  {compute_score(train_cm, metric='f1_weighted'):0.2f}\n"
              f"       accuracy     {compute_score(train_cm, metric='accuracy'):0.2f}\n")
        plot_confusion_matrix(train_cm, cls_names[:-1], fig_path=f'heatmap/train/ep_{ep}.png')

        val_loss = 0.
        start = time.time()        
        val_cm = torch.zeros_like(train_cm, dtype=torch.long)

        for i, batch in enumerate(val_loader):
            input_ids, attn_mask, target_tags = move_to_cuda(batch, device)
            start = time.time()
            with torch.no_grad(): 
                logit_tags = model(input_ids, attn_mask, apply_tanh=True)
                loss = criterion(logit_tags.view(-1, logit_tags.size(-1)), target_tags.view(-1))
                val_loss += loss
                val_cm = update_confusion_matrix(val_cm, logit_tags.argmax(dim=-1), target_tags, pad_tag_id, outline_tag_id)
               
        val_loss /= len(val_loader)
        val_f1_macro = compute_score(val_cm, metric='f1_macro')
        print(f"Val:   curr_loss    {val_loss:0.4f}\n"
              f"       global_loss  {best_val_loss:0.2f}\n"
              f"       f1_macro     {val_f1_macro:0.2f}\n"
              f"       f1_weighted  {compute_score(val_cm, metric='f1_weighted'):0.2f}\n"
              f"       accuracy     {compute_score(val_cm, metric='accuracy'):0.2f}\n")
        plot_confusion_matrix(val_cm, cls_names[:-1], fig_path=f'heatmap/val/ep_{ep}.png')


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print('Save model!\n')

        print('|-------------------------------------------------------------------------------------------|')


if __name__ == '__main__':
    TRAIN_PATH = '/workspace/nlplab/kienvt/PhoNER_COVID19_implement/data/word/train_word.json'
    VAL_PATH = '/workspace/nlplab/kienvt/PhoNER_COVID19_implement/data/word/dev_word.json'
    MAX_LENGTH = 60
    BATCH_SIZE = 64
    EPOCHS = 25
    LR = 3e-5
    BEST_MODEL_PARAMS_PATH = "model_weights/best_model_params_word_large.pt"
    TRAIN_NEW_MODEL = True

    # Prepare data
    train_raw_data = json.load(open(TRAIN_PATH, 'r'))
    val_raw_data = json.load(open(VAL_PATH, 'r'))
    
    # tokenizer, tagger = build_vocab(train_raw_data)
    tokenizer, tagger = get_pretrained_vocab('vocab_file/tokenizer_word.json',
                                              'vocab_file/tagger.json')

    
    train_set = COVID19Dataset2(train_raw_data, tokenizer, tagger, MAX_LENGTH)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    val_set = COVID19Dataset2(val_raw_data, tokenizer, tagger, MAX_LENGTH)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # Define model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BERT(
        num_classes=train_set.tag_vocab_size, 
        clf_hidden_dim=1000,
        vocab_size=train_set.tokenizer.vocab_size,
        pad_token_id=train_set.tokenizer.pad_token_id,
        num_encoder_blocks=12, 
        num_heads=16,
        embed_dim=1024,
        qkv_bias=True,
        ff_dim=4096,
        ff_activate_fn='gelu',
        )
    if TRAIN_NEW_MODEL: model._reset_parameters()
    else: model.load_state_dict(torch.load(BEST_MODEL_PARAMS_PATH))
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=train_set.pad_tag_id)

    # Optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = Adam(optimizer_grouped_parameters, lr=LR)

    print('START TRAINING')
    train(model, train_loader, val_loader, criterion, optimizer, EPOCHS, device, BEST_MODEL_PARAMS_PATH)
    
    