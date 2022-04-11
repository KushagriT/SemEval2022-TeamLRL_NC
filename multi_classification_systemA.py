import torch, os, random
import torch.nn as nn
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    RobertaModel,
    RobertaTokenizerFast,
    RobertaConfig
)
import numpy as np
import pandas as pd
from torch.utils.data import Sampler, Dataset, DataLoader
import more_itertools
from tqdm.auto import tqdm
from xgboost import XGBClassifier
import re, string, contractions
from ast import literal_eval
import csv

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def set_random_seed(seed):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_random_seed(28)

df_train = pd.read_csv("combined_training_task2.csv")
df_test = pd.read_csv("testdata.csv")

def preprocess(text):
    text = re.sub(re.compile('[",;-]'),' ',text)
    text = re.sub(r"( ')([a-zA-Z] |[a-zA-Z]{2} )",r"'\2",text)
    text = re.sub(r"(' )([a-zA-Z] |[a-zA-Z]{2} )",r"'\2",text)
    text = re.sub(r"([0-9])",r" ",text)
    text = contractions.fix(text)
    text = re.sub(re.compile('&amp|\\\w'),' ',text)
    text = "".join([w if w not in string.punctuation else ' ' for w in text])
    text = " ".join(text.split())
    return text

df_train.label = df_train.label.apply(literal_eval)
df_train.text = df_train.text.apply(preprocess)
df_test.text = df_test.text.apply(preprocess)

batch_size = 16
max_length = 256
num_epochs = 5
pretrained_path = '/roberta_pretrained_for_semeval_task'
# Pretrained on the whole data
num_labels = 7

num_train_steps = num_epochs*int(df_train.shape[0]/batch_size)

config = RobertaConfig.from_pretrained(pretrained_path)
hidden_size = config.hidden_size

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
            #print(self._targets[item])
            #print(torch.tensor(self._targets[item],dtype=torch.long))
            return self._data[item], torch.tensor(self._targets[item],dtype=torch.float)
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
            targets = list(targets)
            #print(targets)
        else:
            sequences = list(batch)
        
        input_ids, attention_mask = self.pad_sequence(
            sequences,
            max_sequence_length=self._max_length,
            pad_token_id=self._pad_token_id
        )
        
        if self._targets is not None:
            output = input_ids.to('cuda'), attention_mask.to('cuda'), torch.stack(targets).to('cuda')
            #print(torch.stack(targets).to('cuda').size())
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

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base',do_lower_case=True)
train_dataset = SmartBatchingDataset(df_train, tokenizer)
train_loader = train_dataset.get_dataloader(batch_size=batch_size,
                                            max_len=max_length,
                                            pad_id=tokenizer.pad_token_id)

class MLC_Finetuning_Model(nn.Module):
    def __init__(self,reinit_n_layers=0):
        super().__init__()
        self.roberta_model = RobertaModel.from_pretrained(pretrained_path)
        self.classifier = nn.Linear(hidden_size,num_labels)
        self.reinit_n_layers = reinit_n_layers
        if reinit_n_layers > 0: self._do_reinit()    
   
    def _debug_reinit(self, text):
        print(f"\n{text}\nPooler:\n", self.roberta_model.pooler.dense.weight.data)        
        for i, layer in enumerate(self.roberta_model.encoder.layer[-self.reinit_n_layers:]):
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    print(f"\n{i} nn.Linear:\n", module.weight.data) 
                elif isinstance(module, nn.LayerNorm):
                    print(f"\n{i} nn.LayerNorm:\n", module.weight.data) 

    def _do_reinit(self):
        # Re-init pooler.
        self.roberta_model.pooler.dense.weight.data.normal_(mean=0.0,
                                                            std=self.roberta_model.config.initializer_range)
        self.roberta_model.pooler.dense.bias.data.zero_()
        for param in self.roberta_model.pooler.parameters():
            param.requires_grad = True
        
        # Re-init last n layers.
        for n in range(self.reinit_n_layers):
            self.roberta_model.encoder.layer[-(n+1)].apply(self._init_weight_and_bias)

    def _init_weight_and_bias(self, module):                        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.roberta_model.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0) 

    def forward(self, input_ids, attention_mask):       
        raw_output = self.roberta_model(input_ids,
                                        attention_mask,
                                        return_dict=True,
                                        output_hidden_states=True)        
        last_hidden_state = raw_output.last_hidden_state
        cls_embeddings = last_hidden_state[:,0,:]           
        output = self.classifier(cls_embeddings)                            
        return output

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MLC_Finetuning_Model(reinit_n_layers=5)
model.to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=100,num_training_steps=25*int(df_train.shape[0]/batch_size))
save_chckpt_name = '/content/task2_finetune_2'
steps = 0
model.train()
epoch = 0

progress_bar = tqdm(range(num_train_steps))
while epoch < num_epochs:
    epoch += 1
    train_loss = 0
    for i, batch in enumerate(train_loader):
        model.train()
        optimizer.zero_grad() 
        outputs = model(input_ids=batch[0],attention_mask=batch[1]).squeeze(-1)
        loss = loss_fn(outputs, batch[2])
        train_loss += loss.item()
        loss.backward() 
        optimizer.step()
        scheduler.step()
        progress_bar.update(1) 
        steps+= 1
        if steps%50 == 0:
            file_name = save_chckpt_name + f'_step{steps}.pt'
            torch.save(model.state_dict(), file_name)          
            model.eval()
            print(f"Epoch: {epoch}, Batch {steps}, TRAIN_LOSS = {train_loss/(i+1)}")



booster='gbtree'
eta=0.5
min_split_loss=0.1
max_depth=6
min_child_weight=1
n_estimators = 100
max_delta_step=2
subsample=0.6
sampling_method='uniform'
reg_lambda=1
reg_alpha=0.5
grow_policy='depthwise'
tree_method='gpu_hist'
objective = 'binary:logistic'


estimators = [XGBClassifier(verbosity=2,
                               booster=booster,
                               eta=eta,
                               min_split_loss=min_split_loss,
                               max_depth=max_depth,
                               min_child_weight=min_child_weight,
                               max_delta_step=max_delta_step,
                               n_estimators = n_estimators,
                               subsample=subsample,
                               sampling_method=sampling_method,
                               reg_lambda=reg_lambda,
                               reg_alpha=reg_alpha,
                               grow_policy=grow_policy,
                               tree_method=tree_method,
                               objective=objective,
                               scale_pos_weight=12),
              XGBClassifier(verbosity=2,
                               booster=booster,
                               eta=eta,
                               min_split_loss=min_split_loss,
                               max_depth=max_depth,
                               min_child_weight=min_child_weight,
                               max_delta_step=max_delta_step,
                               n_estimators = n_estimators,
                               subsample=subsample,
                               sampling_method=sampling_method,
                               reg_lambda=reg_lambda,
                               reg_alpha=reg_alpha,
                               grow_policy=grow_policy,
                               objective=objective,
                               tree_method=tree_method,
                            scale_pos_weight=50),
              XGBClassifier(verbosity=2,
                               booster=booster,
                               eta=eta,
                               min_split_loss=min_split_loss,
                               max_depth=max_depth,
                               min_child_weight=min_child_weight,
                               max_delta_step=max_delta_step,
                               n_estimators = n_estimators,
                               subsample=subsample,
                               sampling_method=sampling_method,
                               reg_lambda=reg_lambda,
                               reg_alpha=reg_alpha,
                               grow_policy=grow_policy,
                               objective=objective, 
                               tree_method=tree_method,
                            scale_pos_weight=45),
              XGBClassifier(verbosity=2,
                               booster=booster,
                               eta=eta,
                               min_split_loss=min_split_loss,
                               max_depth=max_depth,
                               min_child_weight=min_child_weight,
                               max_delta_step=max_delta_step,
                               n_estimators = n_estimators,
                               subsample=subsample,
                               sampling_method=sampling_method,
                               reg_lambda=reg_lambda,
                               reg_alpha=reg_alpha,
                               grow_policy=grow_policy,
                               objective=objective,
                               tree_method=tree_method,
                            scale_pos_weight=45),
              XGBClassifier(verbosity=2,
                               booster=booster,
                               eta=eta,
                               min_split_loss=min_split_loss,
                               max_depth=max_depth,
                               min_child_weight=min_child_weight,
                               max_delta_step=max_delta_step,
                               n_estimators = n_estimators,
                               subsample=subsample,
                               sampling_method=sampling_method,
                               reg_lambda=reg_lambda,
                               reg_alpha=reg_alpha,
                               grow_policy=grow_policy,
                               objective=objective,
                               tree_method=tree_method,
                            scale_pos_weight=50),
              XGBClassifier(verbosity=2,
                               booster=booster,
                               eta=eta,
                               min_split_loss=min_split_loss,
                               max_depth=max_depth,
                               min_child_weight=min_child_weight,
                               max_delta_step=max_delta_step,
                               n_estimators = n_estimators,
                               subsample=subsample,
                               sampling_method=sampling_method,
                               reg_lambda=reg_lambda,
                               reg_alpha=reg_alpha,
                               grow_policy=grow_policy,
                               objective=objective,
                               tree_method=tree_method,
                            scale_pos_weight=20),
              XGBClassifier(verbosity=2,
                               booster=booster,
                               eta=eta,
                               min_split_loss=min_split_loss,
                               max_depth=max_depth,
                               min_child_weight=min_child_weight,
                               max_delta_step=max_delta_step,
                               n_estimators = n_estimators,
                               subsample=subsample,
                               sampling_method=sampling_method,
                               reg_lambda=reg_lambda,
                               reg_alpha=reg_alpha,
                               grow_policy=grow_policy,
                               objective=objective,
                               tree_method=tree_method,
                            scale_pos_weight=250)]

class Classifier:
    def __init__(self,estimators,num_labels):
        self.estimators = estimators
        self.num_labels = num_labels
    
    def fit(self,X,y):
        for i in range(self.num_labels):
            self.estimators[i].fit(X,y[:,i])
        return self
    
    def predict(self,X):
        m = X.shape[0]
        y_pred = np.empty((m,self.num_labels))
        for i in range(self.num_labels):
            y_pred[:,i] = self.estimators[i].predict(X)
        return y_pred

classifier = Classifier(estimators=estimators,num_labels=num_labels)

def get_features(dataloader):
    # Extract features
    model.eval()
    features = []
    targets = []
    with torch.no_grad():                           
        for batch in dataloader:                     
            outputs = model(input_ids=batch[0],attention_mask=batch[1]).squeeze(-1) 
            # Uncomment the next line for ablation variation
            # outputs = model.roberta_model(input_ids=batch[0],attention_mask=batch[1]).last_hidden_state[:,0,:]
            features.append(outputs)
            targets.append(batch[2])
            del outputs, batch
    features = torch.cat(features)
    targets = torch.cat(targets)
    return features, targets

X, y_true = get_features(train_loader)
classifier.fit(X.cpu(),y_true.cpu())

test_path = '/content/testdata.csv'
test_data = pd.read_csv(test_path)
collate = SmartBatchingCollate(targets=None, max_length=256, pad_token_id=tokenizer.pad_token_id)
test_dataset = SmartBatchingDataset(test_data, tokenizer)
with torch.no_grad():
  X_test = []
  for i in tqdm(range(test_data.shape[0])):
    batch = collate([test_dataset[i]])
    outputs = model.roberta_model(input_ids=batch[0],attention_mask=batch[1]).last_hidden_state[:,0,:]
    X_test.append(outputs)
  X_test = torch.cat(X_test)

preds = classifier.predict(X_test.cpu().numpy())

preds = [[str(int(float(i))) for i in x] for x in preds]

with open("/multi_systemA_predictions.txt", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(preds)