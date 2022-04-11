import torch, random
import torch.nn as nn
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    RobertaModel,
    RobertaTokenizerFast,
    RobertaConfig
)
import csv
import numpy as np
import pandas as pd
from torch.utils.data import Sampler, Dataset, DataLoader
import more_itertools
from tqdm.auto import tqdm
from ast import literal_eval
from fcmeans import FCM
import string,re,contractions
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
set_seed(28)  

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
num_epochs = 10
pretrained_path = '/content/drive/MyDrive/roberta_pretrained_for_task1'
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
        self.linear = nn.Linear(hidden_size,100)
        self.classifier = nn.Linear(100,num_labels)
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
        output = self.classifier(self.linear(cls_embeddings))                            
        return output

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MLC_Finetuning_Model(reinit_n_layers=5)
model.to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=100,num_training_steps=25*int(df_train.shape[0]/batch_size))
save_chckpt_name = '/content/task2_finetune_'
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

def get_og_features(dataloader):
    # Extract features
    model.eval()
    features = []
    targets = []
    with torch.no_grad():                           
        for batch in dataloader:                     
            outputs = model.linear(model.roberta_model(input_ids=batch[0],attention_mask=batch[1]).last_hidden_state[:,0,:]).squeeze(-1) 
            # Any other layer can also be used for getting the features
            features.append(outputs)
            targets.append(batch[2])
            del outputs, batch
    features = torch.cat(features)
    targets = torch.cat(targets)
    return features, targets

train_embeddings, y_train = get_og_features(train_loader)
train_embeddings, y_train = train_embeddings.cpu().numpy(),y_train.cpu().numpy()
collate = SmartBatchingCollate(targets=None, max_length=256, pad_token_id=tokenizer.pad_token_id)

test_dataset = SmartBatchingDataset(df_test[['text']], tokenizer)

with torch.no_grad():
  X_test = []
  for i in tqdm(range(df_test.shape[0])):
    batch = collate([test_dataset[i]])
    outputs = outputs = model.linear(model.roberta_model(input_ids=batch[0],
                                                         attention_mask=batch[1]).last_hidden_state[:,0,:]).squeeze(-1) 
    X_test.append(outputs)
    
test_embeddings = torch.cat(X_test).cpu().numpy()

def get_features(params):  
  X_demb = train_embeddings
  c = params['c']
  m = params['m']
  fcm = FCM(n_clusters=c,random_state=1,m=m)
  fcm.fit(X_demb)
  X_train = np.array(fcm.u)
  test_demb = test_embeddings
  X_test = np.array(fcm.soft_predict(test_demb))
  return X_train,X_test

def fuzzy_union(a, b):
  return a + b - a * b

def fuzzy_multi_union( memb_list):
        if len(memb_list) < 2:
            return memb_list[0]
        else:
            x = fuzzy_union(memb_list[0], memb_list[1])
            for i in range(2, len(memb_list)):
                x = fuzzy_union(memb_list[i], x)
            return x

def model_generation(k,X_train,y_train):
        """
        Model Generation: Getting cluster-label memberships and cluster-feature 
        association
        
        """
        # KMeans clustering
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=200, random_state=1)
        kmeans.fit(X_train)
        n = X_train.shape[1]
        p = y_train.shape[1]

        kmeans_grp = pd.DataFrame(
            np.append(y_train, kmeans.labels_.reshape(-1, 1), axis=1))
        kmeans_grp = kmeans_grp.rename(
            columns={kmeans_grp.columns[-1]: 'cluster_label'})
        
        # Membership functions
        mu_cluster_label = np.empty((k, p))
        for i in range(k):
            df = kmeans_grp[kmeans_grp['cluster_label'] == i]
            source = pd.DataFrame(df.iloc[:, :-1].
                                  apply(sum)).reset_index().rename(columns={0: 'count'})
            mu_cluster_label[i] = np.array(source['count']) / df.shape[0]

        # Cluster - feature degree of association
        kmeans_x = pd.DataFrame(np.hstack((X_train,
                                           kmeans.labels_.reshape(-1, 1))))
        kmeans_x = kmeans_x.rename(columns={kmeans_x.columns[-1]: 'cluster_label'})
        nu = np.empty((k, n))
        for i in range(k):
            x_ci = kmeans_x[kmeans_x['cluster_label'] == i].loc[:,
                   kmeans_x.columns !=
                   'cluster_label']
            nu[i] = np.mean(np.array(x_ci), axis=0)
        nu = np.transpose(nu)
        return mu_cluster_label, nu
 

def predict_prob(X_test,nu):
        
        
        # We get the projection R using nu and mu_cluster_genre
        R = np.dot(X_test, nu)
        sum_zeta = np.sum(nu, axis=0)
        for i in range(X_test.shape[0]):
            R[i] = np.divide(R[i], sum_zeta)
        
        y_pred = np.apply_along_axis(get_pred, 1, R)
        
        # We return the prediction array
        return y_pred

m = 1.6
c = 22
s = 4
alpha = 0.55
k = 38
params = {'m':m,'c':c}
X_train,X_test = get_features(params)
mu_cluster_label, nu = model_generation(k,X_train,y_train)
def get_pred(test_eg_R):
            get_ind_clus = test_eg_R.argsort()[-int(s):]
            mu_label = np.apply_along_axis(fuzzy_multi_union, 0,
                                           mu_cluster_label[list(get_ind_clus), :])
            return np.array(list(map(int, list(mu_label >= float(alpha)))))
pred_array = predict_prob(X_test,nu)

preds = [[int(x) for x in pred] for pred in pred_array.tolist()]

with open("/multi_systemB_predictions.txt", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(preds)