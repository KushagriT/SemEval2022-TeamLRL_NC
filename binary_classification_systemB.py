from transformers import (RobertaModel,
                          RobertaConfig,
                          RobertaTokenizerFast,
                          get_linear_schedule_with_warmup)
from tqdm.auto import tqdm
import gc
import torch
import random
from torch.utils.data import Sampler, Dataset, DataLoader
import numpy as np
import pandas as pd
import more_itertools

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

set_seed(28)

df_train = pd.read_csv("combined_training_task1.csv")
df_test = pd.read_csv("testdata.csv")

class SmartBatchingDataset(Dataset):
    def __init__(self, df, tokenizer):
        super(SmartBatchingDataset, self).__init__()
        self._data = (
            f"{tokenizer.bos_token} " + df.text + f" {tokenizer.eos_token}" 
        ).apply(tokenizer.tokenize).apply(tokenizer.convert_tokens_to_ids).to_list()
        self._targets = None
        if 'label' in df.columns:
            self._targets = df.label.tolist()
        self.sampler = None

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        if self._targets is not None:
            return self._data[item], self._targets[item]
        else:
            return self._data[item]

    def get_dataloader(self, batch_size, max_len, pad_id):
        self.sampler = SmartBatchingSampler(
            data_source=self._data,
            batch_size=batch_size
        )
        collate_fn = SmartBatchingCollate(
            targets=self._targets,
            max_length=max_len,
            pad_token_id=pad_id
        )
        dataloader = DataLoader(
            dataset=self,
            batch_size=batch_size,
            sampler=self.sampler,
            collate_fn=collate_fn
        )
        return dataloader

class SmartBatchingSampler(Sampler):
    def __init__(self, data_source, batch_size):
        super(SmartBatchingSampler, self).__init__(data_source)
        self.len = len(data_source)
        sample_lengths = [len(seq) for seq in data_source]
        argsort_inds = np.argsort(sample_lengths)
        self.batches = list(more_itertools.chunked(argsort_inds, n=batch_size))
        self._backsort_inds = None
    
    def __iter__(self):
        if self.batches:
            last_batch = self.batches.pop(-1)
            np.random.shuffle(self.batches)
            self.batches.append(last_batch)
        self._inds = list(more_itertools.flatten(self.batches))
        yield from self._inds

    def __len__(self):
        return self.len
    
    @property
    def backsort_inds(self):
        if self._backsort_inds is None:
            self._backsort_inds = np.argsort(self._inds)
        return self._backsort_inds

class SmartBatchingCollate:
    def __init__(self, targets, max_length, pad_token_id):
        self._targets = targets
        self._max_length = max_length
        self._pad_token_id = pad_token_id
        
    def __call__(self, batch):
        if self._targets is not None:
            sequences, targets = list(zip(*batch))
        else:
            sequences = list(batch)
        
        input_ids, attention_mask = self.pad_sequence(
            sequences,
            max_sequence_length=self._max_length,
            pad_token_id=self._pad_token_id
        )
        
        if self._targets is not None:
            output = input_ids.to('cuda'), attention_mask.to('cuda'), torch.tensor(targets).to('cuda')
        else:
            output = input_ids.to('cuda'), attention_mask.to('cuda')
        return output
    
    def pad_sequence(self, sequence_batch, max_sequence_length, pad_token_id):
        max_batch_len = max(len(sequence) for sequence in sequence_batch)
        max_len = min(max_batch_len, max_sequence_length)
        padded_sequences, attention_masks = [[] for i in range(2)]
        attend, no_attend = 1, 0
        for sequence in sequence_batch:
            # As discussed above, truncate if exceeds max_len
            new_sequence = list(sequence[:max_len])
            
            attention_mask = [attend] * len(new_sequence)
            pad_length = max_len - len(new_sequence)
            
            new_sequence.extend([pad_token_id] * pad_length)
            attention_mask.extend([no_attend] * pad_length)
            
            padded_sequences.append(new_sequence)
            attention_masks.append(attention_mask)
        
        padded_sequences = torch.tensor(padded_sequences)
        attention_masks = torch.tensor(attention_masks)
        return padded_sequences, attention_masks

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

class AttentionHead(torch.nn.Module):
    
    def __init__(self, input_dim=768, hidden_dim=512):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, last_hidden_state):
               
        linear1_output = self.linear1(last_hidden_state)  
        activation = torch.tanh(linear1_output)                   
        score = self.linear2(activation)                        
        weights = torch.softmax(score, dim=1)                          
        result = torch.sum(weights * last_hidden_state, dim=1)          
        return result

class RobertaFinetuneModel_AttnHead(torch.nn.Module):
            
    def __init__(self,reinit_n_layers=0):        
        super().__init__() 
        self.roberta_model = RobertaModel.from_pretrained(pretrained_path)
        self.attn_head = AttentionHead(768, 512)       
        self.classifier = torch.nn.Linear(768, 2)   
        self.reinit_n_layers = reinit_n_layers
        if reinit_n_layers > 0: self._do_reinit()            
            
    def _debug_reinit(self, text):
        print(f"\n{text}\nPooler:\n", self.roberta_model.pooler.dense.weight.data)        
        for i, layer in enumerate(self.roberta_model.encoder.layer[-self.reinit_n_layers:]):
            for module in layer.modules():
                if isinstance(module, torch.nn.Linear):
                    print(f"\n{i} nn.Linear:\n", module.weight.data) 
                elif isinstance(module, torch.nn.LayerNorm):
                    print(f"\n{i} nn.LayerNorm:\n", module.weight.data) 
        
    def _do_reinit(self):
        # Re-init pooler.
        self.roberta_model.pooler.dense.weight.data.normal_(mean=0.0, std=self.roberta_model.config.initializer_range)
        self.roberta_model.pooler.dense.bias.data.zero_()
        for param in self.roberta_model.pooler.parameters():
            param.requires_grad = True
        
        # Re-init last n layers.
        for n in range(self.reinit_n_layers):
            self.roberta_model.encoder.layer[-(n+1)].apply(self._init_weight_and_bias)

    def _init_weight_and_bias(self, module):                        
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.roberta_model.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)   
                     
    def forward(self, input_ids, attention_mask):       
        raw_output = self.roberta_model(input_ids,
                                        attention_mask,
                                        return_dict=True,
                                        output_hidden_states=True)        
        last_hidden_state = raw_output["last_hidden_state"] 
        attn = self.attn_head(last_hidden_state)            
        output = self.classifier(attn)                            
        return output 

def roberta_base_AdamW_grouped_LLRD(model):
        
    opt_parameters = []       # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters()) 
    
    # According to AAAMLP book by A. Thakur, we generally do not use any decay 
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    set_2 = ["layer.4", "layer.5", "layer.6", "layer.7"]
    set_3 = ["layer.8", "layer.9", "layer.10", "layer.11"]
    init_lr = 1e-5
    
    for i, (name, params) in enumerate(named_parameters):  
        
        weight_decay = 0.0 if any(p in name for p in no_decay) else 0.01
 
        if name.startswith("roberta_model.embeddings") or name.startswith("roberta_model.encoder"):            
            lr = init_lr       
            
            lr = init_lr * 2 if any(p in name for p in set_2) else lr
            
            lr = init_lr * 4 if any(p in name for p in set_3) else lr
            
            opt_parameters.append({"params": params,
                                   "weight_decay": weight_decay,
                                   "lr": lr})  
            
        # For classifier and pooler, set lr to 0.0000036 (slightly higher than the top layer).                
        if name.startswith("classifier") or name.startswith("roberta_model.pooler"):               
            lr = init_lr * 5
            
            opt_parameters.append({"params": params,
                                   "weight_decay": weight_decay,
                                   "lr": lr})    
    
    return torch.optim.AdamW(opt_parameters, lr=init_lr)

def get_scheduler(optimizer):
    scheduler = get_linear_schedule_with_warmup(                
                optimizer = optimizer,
                num_warmup_steps = 50,
                num_training_steps = num_train_steps
)
    return scheduler

pretrained_path = '/roberta_pretrained_for_semeval_task'
config = RobertaConfig.from_pretrained(pretrained_path) 
set_seed(28)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = RobertaFinetuneModel_AttnHead(reinit_n_layers=5)

epochs = 5
steps = 0
batch_size = 32
train_dataset = SmartBatchingDataset(df_train, tokenizer)
train_loader = train_dataset.get_dataloader(batch_size=batch_size, max_len=256, pad_id=tokenizer.pad_token_id)

num_train_steps = epochs*int(df_train.shape[0]/batch_size)

model.to(device)

optimizer = roberta_base_AdamW_grouped_LLRD(model)

scheduler = get_scheduler(optimizer)

progress_bar = tqdm(range(num_train_steps))
for epoch in range(epochs):
  train_loss = 0
  for i, batch in enumerate(train_loader):  
        model.train()                             
        optimizer.zero_grad()                  
        outputs = model(input_ids=batch[0],attention_mask=batch[1]).squeeze(-1) 
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(outputs, batch[2])  
        train_loss += loss.item()               
 
        loss.backward()                         
        optimizer.step()                        
        
        scheduler.step()                    
        steps+= 1
        if steps%100 == 0:
          torch.save(model.state_dict(), f"/binary_systemB_step_{steps}.pt")
          print(f"Epoch: {epoch}, Batch {i}, TRAIN_LOSS = {train_loss/(i+1)}")
        gc.collect()    
        try:
          del batch   
        except:
          pass
        progress_bar.update(1)

test_dataset = SmartBatchingDataset(df_test[['text']], tokenizer)
collate = SmartBatchingCollate(targets=None, max_length=256, pad_token_id=tokenizer.pad_token_id)

with torch.no_grad():
  predictions = []
  for i in tqdm(range(df_test.shape[0])):
    batch = collate([test_dataset[i]])
    outputs = model(input_ids=batch[0],attention_mask=batch[1]).squeeze(-1) 
    predictions.append(torch.argmax(outputs,1))

predictions = [int(pred[0].cpu().numpy()) for pred in predictions]

with open('/binary_systemB_predictions.txt','w') as f:
  for pred in predictions:
    f.write(str(pred))
    f.write('\n')