import numpy as np
import torch
import torch.nn.functional as F

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.linear1 = torch.nn.Linear(hidden_units, hidden_units)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_units, hidden_units)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.linear2(self.relu(self.dropout1(self.linear1(inputs)))))
        outputs += inputs
        return outputs


class MSMRec(torch.nn.Module):
    def __init__(self, user_num, item_num, hidden_units, maxlen, num_blocks, num_heads, dropout_rate, num_modality, meta_emb):
        super(MSMRec, self).__init__()

        ### 
        self.num_modality = num_modality
        self.meta_emb = torch.nn.Embedding.from_pretrained(meta_emb, freeze=True)
        self.meta_prober = torch.nn.Sequential(
            torch.nn.Linear(self.meta_emb.weight.shape[1] // self.num_modality, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, hidden_units),
        )
        self.modality_emb = torch.nn.Embedding(self.num_modality, hidden_units)
        ### 256 x 200 x (3 x 768)
        ### NUM_BATCH x (NUM_MODALITY X EMBEDDING DIM)

        self.user_num = user_num
        self.item_num = item_num
        ### DEVICE
        self.dev = torch.device('cpu')
        self.maxlen = maxlen
        self.item_emb = torch.nn.Embedding(self.item_num+1, hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(maxlen+1, hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self.attention_layernorms = torch.nn.ModuleList() 
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer =  torch.nn.MultiheadAttention(hidden_units,
                                                            num_heads,
                                                            dropout_rate)
            self.attention_layers.append(new_attn_layer)
            new_fwd_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layer = PointWiseFeedForward(hidden_units, dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    


    def get_user_logits(self, log_seqs):
        ### USER REPR
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = torch.from_numpy(np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1]))
        poss *= (log_seqs != 0)
        seqs +=self.pos_emb(torch.tensor(poss, dtype=torch.long, device=self.dev))
        seqs = torch.mean(seqs, dim=1) 
        ### USER REPR (batch x emb_dim)

        ### MODALITY REPR
        modality_k_emb = self.modality_emb(torch.arange(self.num_modality).to(self.dev))
        modality_k_emb = modality_k_emb.unsqueeze(0).expand(seqs.shape[0], -1, -1)
        ### MODALITY REPR (batch x num_modality x emb_dim)
    
        return torch.softmax(torch.einsum('bi, bji->bj', seqs, modality_k_emb), dim=-1) ### (batch x num_modality)
    


    def log2feats(self, log_seqs): 
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = torch.from_numpy(np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1]))
        poss *= (log_seqs != 0)
        seqs +=self.pos_emb(torch.tensor(poss, dtype=torch.long, device=self.dev))
        seqs = self.emb_dropout(seqs)
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        ### META EMB
        meta_emb = self.meta_emb(torch.LongTensor(log_seqs).to(self.dev))
        meta_emb = meta_emb.reshape(meta_emb.shape[0], self.maxlen , self.num_modality, self.meta_emb.weight.shape[1] // self.num_modality) ### 256 x 200 x 3 x 768
        meta_emb = self.meta_prober(meta_emb) # 256 x 200 x 3 x 50

        user_repr = self.get_user_logits(log_seqs) # 256 x 3
        user_repr_expanded = user_repr.unsqueeze(1).unsqueeze(-1)  # Shape: [256, 1, 3, 1]
        meta_mix_emb = (meta_emb * user_repr_expanded).sum(dim=2)  # Shape: [256, 200, 50]
        meta_mix_emb *= self.item_emb.embedding_dim ** 0.5

        ### TODO Different Logits at different timestep

        ### INCORPORATE INTO SASREC
        seqs = seqs + meta_mix_emb
        ### END



        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
        log_feats = self.last_layernorm(seqs ) 
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): 
        log_feats = self.log2feats(log_seqs)
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits
    
    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits # preds # (U, I)
    
    def predict_batch(self, user_ids, log_seqs, item_indices):
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits =  torch.einsum('bi,bji->bj', final_feat, item_embs)
        return logits
