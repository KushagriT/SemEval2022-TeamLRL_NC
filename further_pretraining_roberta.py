from transformers import (
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    DataCollatorForLanguageModeling,
    get_scheduler
)
from tqdm.auto import tqdm
import torch
import random
from torch.utils.data import Sampler, Dataset, DataLoader
import numpy as np

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

class SmartBatchingSampler(Sampler):
    def __init__(self, data_source, batch_size,shuffle=False):
        super(SmartBatchingSampler, self).__init__(data_source)
        self.len = len(data_source)
        sample_lengths = [len(seq) for seq in data_source]
        argsort_inds = np.argsort(sample_lengths)
        self.batches = list(more_itertools.chunked(argsort_inds, n=batch_size))
        self._backsort_inds = None
        self.shuffle = shuffle
    def __iter__(self):
        if self.batches:
            last_batch = self.batches.pop(-1)
            if self.shuffle:
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

class MaskedLMDataset(Dataset):
    def __init__(self, file, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.lines = self.load_lines(file)
        self.ids = self.encode_lines(self.lines)
        self.sampler=None
    def load_lines(self,file):
        with open(file) as f:
            lines = [
                line
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]
        return lines
        
    def encode_lines(self,lines):
        batch_encoding = self.tokenizer(
            lines, add_special_tokens=True, truncation=True, max_length=256
        )

        return batch_encoding["input_ids"]
    
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return torch.tensor(self.ids[idx], dtype=torch.long)

    def get_dataloader(self, batch_size,shuffle=False):
        self.sampler = SmartBatchingSampler(
            data_source=self.ids,
            batch_size=batch_size,
            shuffle=shuffle
        )
        collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                     mlm=True,mlm_probability=0.15)
        dataloader = DataLoader(
            dataset=self,
            collate_fn=collate_fn
            , sampler=self.sampler
            , num_workers= 2
        )
        return dataloader

train_dataset = MaskedLMDataset('task_text.txt', tokenizer)
train_loader = train_dataset.get_dataloader(batch_size=64,shuffle=True)

num_training_steps = 30000
model = RobertaForMaskedLM.from_pretrained('roberta-base')
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.1)
lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=10,
    num_training_steps=num_training_steps
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


progress_bar = tqdm(range(num_training_steps))
n = 0
l_total = 0
num_epoch = 0
model.train()

while n<=num_training_steps:
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        l_total+= loss.item()
        if n>0 and n%500 ==0:
          print(f'train_loss {l_total/n}, stage {n}')
          torch.save(model.state_dict(), f"/content/roberta_base_step_{n}.pt") 
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        n +=1
        if n>num_training_steps:
          break
        num_epoch+= 1

model.save_pretrained('/roberta_pretrained_for_semeval_task')

