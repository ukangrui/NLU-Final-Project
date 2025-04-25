import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
from model.sasrec import *
from utils import *
from tqdm import tqdm
import os
import pickle
import argparse
import wandb
import sys
# wandb.init(project="NLP_FINAL", entity="ukangrui")


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml-1m')
args = parser.parse_args()

num_u, num_i = get_usr_itm_num(args.dataset)
print(num_u, num_i)
train,valid,test = load_train_valid_test_data_num(ds_dict=load_txt_file(args.dataset), itemnum=num_i)

train_loader = DataLoader(train, batch_size = 128, shuffle = True, collate_fn = collate_train)
valid_loader  = DataLoader(valid, batch_size = 128, shuffle = False, collate_fn = collate_valid)
test_loader  = DataLoader(test, batch_size = 128, shuffle = False, collate_fn = collate_test)


model = SASRec(user_num = num_u, item_num = num_i, maxlen = 200, num_blocks = 2, num_heads = 1, hidden_units = 50, dropout_rate = 0.2)

model = model.to('cuda')
print(trainable_parameters(model))
sys.exit()
for name, param in model.named_parameters():
    try:
        torch.nn.init.xavier_normal_(param.data)
    except:
        pass

model.pos_emb.weight.data[0, :] = 0
model.item_emb.weight.data[0, :] = 0

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 200, eta_min = 1e-4)

best_recall100 = -float('inf')

for epoch in range(200):
    model.train()
    running_loss = 0
    for train_batch in train_loader:
        u, seq, pos, neg = train_batch
        batch_loss = train_step(model, u, seq, pos, neg, criterion, optimizer)
        running_loss += batch_loss
    
    scheduler.step()
    print(f'epoch: {epoch}, loss: {running_loss / len(train_loader)}')
    wandb.log({"epoch": epoch, "loss": running_loss / len(train_loader)})

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
    print(f'recall_100: {valid_recall_100 / len(valid)}, recall_50: {valid_recall_50 / len(valid)}, recall_10: {valid_recall_10 / len(valid)}, ndcg_10: {valid_ndcg_10 / len(valid)}')
    wandb.log({"valid_recall_100": valid_recall_100 / len(valid), "valid_recall_50": valid_recall_50 / len(valid), "valid_recall_10": valid_recall_10 / len(valid), "valid_ndcg_10": valid_ndcg_10 / len(valid)})


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
    wandb.log({"test_recall_100": test_recall_100 / len(test), "test_recall_50": test_recall_50 / len(test), "test_recall_10": test_recall_10 / len(test), "test_ndcg_10": test_ndcg_10 / len(test)})


    if valid_recall_100 / len(valid) > best_recall100:
        best_recall100 = valid_recall_100 / len(valid)
        torch.save(
            model.state_dict(),
            f'checkpoints/ml-1m-SASRec-best.pth'
        )
        print(f'➤ New best recall@100: {best_recall100:.5f} – saved checkpoint.')
    
wandb.finish()

    