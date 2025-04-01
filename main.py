import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
from model.sasrec import *
from utils import *
from tqdm import tqdm
import os
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml-1m')
parser.add_argument('--use_rag', type=bool, default=False)
parser.add_argument('--embedding_model', type=str, default='bert')
args = parser.parse_args()



num_u, num_i = get_usr_itm_num(args.dataset)
train,valid,test = load_train_valid_test_data_num(ds_dict=load_txt_file(args.dataset), itemnum=num_i)


train_loader = DataLoader(train, batch_size = 256, shuffle = True, collate_fn = collate_train)
valid_loader  = DataLoader(valid, batch_size = 256, shuffle = False, collate_fn = collate_valid)
test_loader  = DataLoader(test, batch_size = 256, shuffle = False, collate_fn = collate_test)
print(num_u, num_i)

model = SASRec(user_num = num_u, item_num = num_i, maxlen = 200, num_blocks = 2, num_heads = 1, hidden_units = 50, dropout_rate = 0.2)
### DEVICE
model.load_state_dict(torch.load('checkpoint/ml-1m-base.pth', map_location=torch.device('cpu')), strict = True)
model = model.to('cpu')

### Valid Evaluation
model.eval()
recall_100, recall_50, recall_10, ndcg_10 = 0,0,0,0
with torch.no_grad():
    for valid_batch in valid_loader:
        u, seq, pos, test_items, mask = valid_batch
        batch_recall_100, batch_recall_50, batch_recall_10, batch_ndcg_10 = eval_step(model, u, seq, pos, test_items, mask)
        recall_100 += batch_recall_100
        recall_50 += batch_recall_50
        recall_10 += batch_recall_10
        ndcg_10 += batch_ndcg_10
print(f'valid_recall_100: {recall_100 / len(valid)}, valid_recall_50: {recall_50 / len(valid)}, valid_recall_10: {recall_10 / len(valid)}, valid_ndcg_10: {ndcg_10 / len(valid)}')


### Test Evaluation
model.eval()
recall_100, recall_50, recall_10, ndcg_10 = 0,0,0,0
with torch.no_grad():
    for test_batch in test_loader:
        u, seq, pos, test_items, mask = test_batch
        batch_recall_100, batch_recall_50, batch_recall_10, batch_ndcg_10 = eval_step(model, u, seq, pos, test_items, mask)
        recall_100 += batch_recall_100
        recall_50 += batch_recall_50
        recall_10 += batch_recall_10
        ndcg_10 += batch_ndcg_10
print(f'test_recall_100: {recall_100 / len(test)}, test_recall_50: {recall_50 / len(test)}, test_recall_10: {recall_10 / len(test)}, test_ndcg_10: {ndcg_10 / len(test)}')


### Fine-tuning Stage
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 25, eta_min = 1e-4)

for epoch in range(1):
    model.train()
    running_loss = 0
    for train_batch in train_loader:
        u, seq, pos, neg = train_batch
        batch_loss = train_step(model, u, seq, pos, neg, criterion, optimizer)
        running_loss += batch_loss
    
    scheduler.step()
    print(f'epoch: {epoch}, loss: {running_loss / len(train_loader)}')

    model.eval()
    valid_recall_100, valid_recall_50, valid_recall_10, valid_ndcg_10 = 0,0,0,0
    with torch.no_grad():
        for valid_batch in valid_loader:
            u, seq, pos, valid_items, mask = valid_batch
            valid_batch_recall_100, valid_batch_recall_50, valid_batch_recall_10, valid_batch_ndcg_10 = eval_step(model, u, seq, pos, valid_items, mask)
            valid_recall_100 += valid_batch_recall_100
            valid_recall_50 += valid_batch_recall_50
            valid_recall_10 += valid_batch_recall_10
            valid_ndcg_10 += valid_batch_ndcg_10
    print('validation')
    print(f'recall_100: {valid_recall_100 / len(valid)}, recall_50: {valid_recall_50 / len(valid)}, recall_10: {valid_recall_10 / len(valid)}, ndcg_10: {valid_ndcg_10 / len(valid)}')




print('testing')
model.eval()
test_recall_100, test_recall_50, test_recall_10, test_ndcg_10 = 0,0,0,0
with torch.no_grad():
    for test_batch in test_loader:
        u, seq, pos, test_items, mask = test_batch
        test_batch_recall_100, test_batch_recall_50, test_batch_recall_10, test_batch_ndcg_10 = eval_step(model, u, seq, pos, test_items, mask)
        test_recall_100 += test_batch_recall_100
        test_recall_50 += test_batch_recall_50
        test_recall_10 += test_batch_recall_10
        test_ndcg_10 += test_batch_ndcg_10
print(f'recall_100: {test_recall_100 / len(test)}, recall_50: {test_recall_50 / len(test)}, recall_10: {test_recall_10 / len(test)}, ndcg_10: {test_ndcg_10 / len(test)}')
    