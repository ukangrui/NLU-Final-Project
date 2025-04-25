import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from torch.utils.data.dataloader import DataLoader

def load_txt_file(ds_name = 'ml-1m'):
    f = open('data/%s/interaction.txt' % ds_name, 'r')
    ds_dict = {}
    for time_idx, line in enumerate(f): ### data is already sorted by timestamp
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        ds_dict[u] = ds_dict.get(u, []) + [i]
    return ds_dict

def get_usr_itm_num(ds_name = 'ml-1m'):
    users = []
    items = []
    f = open('data/%s/interaction.txt' % ds_name, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        users.append(int(u))
        items.append(int(i))
    users = pd.Series(users)
    items = pd.Series(items)
    return len(users.unique()), len(items.unique())

def pad_seq(seq, maxlen = 200):
    if len(seq) < maxlen:
        seq = [0] * (maxlen - len(seq)) + seq
    else:
        seq = seq[-maxlen:]
    return seq

def random_neg(pos, itemnum = 3416):
    neg = np.random.randint(1, itemnum + 1)
    while neg in pos:
        neg = np.random.randint(1, itemnum + 1)
    return neg


def load_train_valid_test_data_num(ds_dict, itemnum,  max_len = 200, skip_short = 4, num_valid = 1, num_test = 1):
    train_data = []
    valid_data = []
    test_data  = []
    for u in tqdm(ds_dict, desc = 'Processing Users'): # Iterate over users
        if len(ds_dict[u]) < skip_short:
            continue
        if len(ds_dict[u]) > max_len:
            ds_dict[u] = ds_dict[u][-max_len:]
        
        train_items = ds_dict[u][:-num_test-num_valid] ### 1 - 77
        valid_items = ds_dict[u][-num_test-num_valid:-num_test] ### 78
        test_items =  ds_dict[u][-num_test:] ### 79
        ### Gather training data
        train_seq = pad_seq(train_items[:-1], max_len)
        train_pos = pad_seq(train_items[1:], max_len)
        train_data.append((u, train_seq, train_pos))
        ### Gather test data
        for i in range(len(test_items)):
            test_seq = pad_seq(train_items + valid_items, max_len)
            test_pos = test_items[i]
            test_idxs = [item for item in list(range(1, itemnum + 1)) if item not in test_seq] + [test_pos]
            num_padding = itemnum - len(test_idxs)
            test_idxs = [0] * num_padding + test_idxs
            mask = num_padding
            test_data.append((u, test_seq, test_pos, test_idxs, mask))
        
        ### Gather valid data
        for i in range(len(valid_items)):
            valid_seq = pad_seq(train_items, max_len)
            valid_pos = valid_items[i]
            valid_idxs = [item for item in list(range(1, itemnum + 1)) if item not in valid_seq] + [valid_pos]
            num_padding = itemnum - len(valid_idxs)
            valid_idxs = [0] * num_padding + valid_idxs
            mask = num_padding
            valid_data.append((u, valid_seq, valid_pos, valid_idxs, mask))
        
    return train_data, valid_data, test_data


def collate_train(batch):
    u, seq, pos = zip(*batch)
    new_neg = []
    for pos_seq in pos:
        pos_set = set(p for p in pos_seq if p != 0)
        neg_samples = [random_neg(pos_set) for _ in range(len(pos_seq))]
        new_neg.append(pad_seq(neg_samples))
    return torch.LongTensor(u), torch.LongTensor(seq),torch.LongTensor(pos), torch.LongTensor(new_neg)


def collate_test(batch):
    u, seq, pos, idx, mask = zip(*batch)
    return torch.LongTensor(u), torch.LongTensor(seq), torch.LongTensor(pos), torch.LongTensor(idx), torch.LongTensor(mask)
def collate_valid(batch):
    u, seq, pos, idx, mask = zip(*batch)
    return torch.LongTensor(u), torch.LongTensor(seq), torch.LongTensor(pos), torch.LongTensor(idx), torch.LongTensor(mask)

def eval_step(model, u, seq, pos,test_items, mask):
    recall_100, recall_50, recall_10, ndcg_10 = 0,0,0,0
    predictions = -model.predict_batch(u, seq, test_items)
    for (usr_idx, pred,start_idx, test_idxs, label) in zip(u, predictions, mask, test_items, pos):
        pred = pred[start_idx:].detach().cpu().numpy()
        test_idxs = test_idxs[start_idx:]

        top_100idx = np.argsort(pred)[:100]
        top_100items = [int(test_idxs[idx]) for idx in top_100idx]

        top_50idx = np.argsort(pred)[:50]
        top_50items = [int(test_idxs[idx]) for idx in top_50idx]

        top_10idx = np.argsort(pred)[:10]
        top_10items = [int(test_idxs[idx]) for idx in top_10idx]


        label = int(label.detach().item())
        recall_100 += int(label in top_100items)
        recall_50 += int(label in top_50items)
        recall_10 += int(label in top_10items)
        if label in top_10items:
            ndcg_10 += 1/np.log2(1 + top_10items.index(label) + 1)
    return recall_100, recall_50, recall_10, ndcg_10


def train_step(model, u, seq, pos, neg, criterion, optimizer):
    u, seq, pos, neg = u.to(torch.device('cpu')), seq.to(torch.device('cpu')), pos.to(torch.device('cpu')), neg.to(torch.device('cpu'))
    pos_logits, neg_logits = model(u, seq, pos, neg)
    pos_labels, neg_labels = torch.ones(pos_logits.shape, device=torch.device('cuda')), torch.zeros(neg_logits.shape, device=torch.device('cuda'))
    optimizer.zero_grad()
    indices = np.where(pos != 0)
    loss = criterion(pos_logits[indices], pos_labels[indices]) + criterion(neg_logits[indices], neg_labels[indices])
    loss.backward()
    optimizer.step()
    return loss.detach().item()




import pandas as pd
import openai


class LlmModel():
    def __init__(self, 
                 model,
                 sys_prompt = '', 
                 api_key =None,
    ):
        super().__init__()
        self.model = model
        self.sys_prompt = sys_prompt
        self.client = openai.OpenAI(api_key)
    def forward(self, usr_prompt, template):
        if template is None:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.sys_prompt},
                    {"role": "user", "content": usr_prompt}
                ],
                max_tokens = 4096,
                temperature=0.001,
                n = 1
            )
            return [response.choices[i].message.content for i in range(len(response.choices))]
        else:
            response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": f'Output the description of each aspect for movie {usr_prompt}'}
            ],
            response_format=template,
            max_tokens = 4096,
            n = 1,
            temperature=0.001
            
        )
            return [response.choices[i].message.parsed.dict() for i in range(len(response.choices))]
    
    def __call__(self, *args):
        return self.forward(*args)
    



import os
from typing import List
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name

        if model_name == 'text-embedding-ada-002':
            self.api_key = None

        elif model_name == 'bert-base-uncased':
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()

        elif model_name == 'all-MiniLM-L6-v2':
            self.model = SentenceTransformer(model_name)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def embed(self, texts: List[str]) -> List[torch.Tensor]:
        if self.model_name == 'text-embedding-ada-002':
            client = openai.OpenAI(api_key=self.api_key)
            response = client.embeddings.create(input=texts, model="text-embedding-ada-002")
            return [torch.tensor(i.embedding) for i in response.data]  # Assuming the response contains a 'data' field with embeddings

        elif self.model_name == 'bert-base-uncased':
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state 

            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            summed = (hidden_states * mask).sum(1)
            counts = mask.sum(1)
            mean_pooled = summed / counts

            return [emb for emb in mean_pooled]

        elif self.model_name == 'all-MiniLM-L6-v2':
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            return [emb for emb in embeddings]

        else:
            raise ValueError(f"Unsupported model at embed time: {self.model_name}")

    def __call__(self, *args, **kwds):
        return self.embed(*args, **kwds)
    

def trainable_parameters(model):
    """
    Prints the percentage of trainable parameters in a PyTorch model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    return f"Trainable parameters: {total_params}"