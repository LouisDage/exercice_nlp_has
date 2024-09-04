import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Union, List, Tuple, Optional
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from transformers.tokenization_utils import PreTrainedTokenizerBase,PreTrainedTokenizer,BatchEncoding
from torcheval.metrics.functional import multiclass_f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class DictAggregator:
    def __init__(self,name:str):
        self.name = name
        self._items = {}
    def reset(self):
        self._items = {}
    
    def add(self,outputs:dict):
        for k,v in outputs.items():
            if k in self._items:
                self._items[k].append(v)
            else:
                self._items[k]=[v]
    
    def dict(self):
        # Average all items from the inner items dict
        return {f'{self.name}/{k}':torch.mean(torch.tensor(v,dtype=torch.float32)) for k,v in self._items.items()}

class TransformerWithClassificationHead(nn.Module):
    def __init__(self, 
                 pretrained_model:Union[AutoModel,nn.Module], 
                 num_labels):
        super(TransformerWithClassificationHead, self).__init__()
        self.pretrained_model = pretrained_model
        # Define a classification head
        self.classifier = nn.Linear(pretrained_model.config.hidden_size, num_labels)

    def mean_pooling(self, model_output:torch.Tensor, attention_mask:torch.Tensor):
        # token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
        return torch.sum(model_output * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
    # def forward(self, encoded_input):

        # Forward pass through the pretrained model
        outputs = self.pretrained_model(input_ids=input_ids, 
                                        attention_mask=attention_mask, 
                                        token_type_ids=token_type_ids)
        # outputs=self.pretrained_model(**encoded_input)
        outputs=self.mean_pooling(outputs.last_hidden_state,attention_mask)
        # The output of the transformer is a tuple with the last hidden state as the first element
        # sequence_output = outputs[0]
        # Use the CLS token representation for classification (sequence_output[:, 0, :])
        # logits = self.classifier(sequence_output[:, 0, :])
        logits = self.classifier(outputs)
        return logits
    
    def predict(self,data_loader) -> List:
        predictions=[]
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                # labels = batch['labels'].to(device)

                outputs = self.forward(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

                _, preds = torch.max(outputs, dim=1)
                predictions+=preds.cpu().tolist()

        return predictions

class TextDataset(Dataset):
    def __init__(self, titles, resumeSiteWebs, labels, tokenizer, max_len):
        self.titles = titles
        self.resumeSiteWebs = resumeSiteWebs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        title = str(self.titles[idx])
        resumeSiteWeb = str(self.resumeSiteWebs[idx])
        label = self.labels[idx]

        # Encode the pair of texts
        encoding = self.tokenizer(
            title,
            resumeSiteWeb,
            add_special_tokens=True,
            # max_length=self.max_len,
            return_token_type_ids=True,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        # return encoding, label
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def custom_embedding_collate_fn(batch):

    # Use tokenizer pad_token_id
    # Pad all inputs to the max lenght of the batch
    max_lenght_token = max([b['input_ids'].shape[-1] for b in batch])

    bert_keys = ['input_ids','attention_mask','token_type_ids']

    final_batch = {}
    
    # for item in batch:

    token_batch = {k:[] for k in bert_keys}
    for item in batch:
        for key in bert_keys:
            tensor:torch.Tensor = item[key]
            pad =(0,max_lenght_token-tensor.shape[-1])
            padded_tensor = torch.nn.functional.pad(tensor,pad=pad,mode='constant',value=0)
            token_batch[key].append(padded_tensor)
    # token_batch = {k:torch.cat(v,dim=0) for k,v in token_batch.items()}
    token_batch = {k:torch.stack(v) for k,v in token_batch.items()}
    final_batch=BatchEncoding(data=token_batch)
    final_batch['labels']=torch.tensor([b['labels'] for b in batch])

    del batch
    return final_batch


def freeze_transformer_parameters(model:TransformerWithClassificationHead):
    for param in model.pretrained_model.parameters():
        param.requires_grad = False


def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    # train_outputs = DictAggregator('Train')
    tbar = tqdm(data_loader,desc='Training - Model',total=len(data_loader))
    for batch in tbar:
        batch_outputs = {}
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        # encodings,labels = batch

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # outputs= model(**encodings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_outputs['CrossEntropyLoss']=loss.detach().cpu().item()
        tbar.set_postfix(batch_outputs)
    return total_loss / len(data_loader)

def eval_epoch(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    batch_outputs = {}
    total_f1_score=0
    vbar = tqdm(data_loader,desc='Validation - Model',total=len(data_loader))
    with torch.no_grad():
        for batch in vbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            batch_outputs['CrossEntropyLoss']=loss.detach().cpu().item()

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            f1_score=multiclass_f1_score(preds,target=labels)
            total_f1_score+=f1_score.item()
            batch_outputs['correct_predictions']=correct_predictions
            batch_outputs['f1_score']=f1_score.detach().cpu().item()

            vbar.set_postfix(batch_outputs)

    return total_loss / len(data_loader), correct_predictions.double() / len(data_loader.dataset), total_f1_score/len(data_loader)

if __name__=="__main__":

    df=pd.read_json("data/has-publications-single/json/AllPublications.json")
    df_filtered=df[['class','title','resumeSiteWeb']]
    df_filtered.dropna(inplace=True)
    df_filtered=df_filtered.loc[df_filtered['resumeSiteWeb']!=""]
    df_filtered['resumeSiteWeb']=df_filtered['resumeSiteWeb'].apply(lambda x: x['markdown'])
    df_filtered.reset_index(inplace=True)
    df_filtered.drop('index',axis=1,inplace=True)
    le=LabelEncoder()
    df_filtered['Class_Int'] = le.fit_transform(df_filtered['class'])

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    pretrained_model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    model=TransformerWithClassificationHead(pretrained_model=pretrained_model, num_labels=df_filtered['Class_Int'].unique().shape[0])

    # Split the data into training and validation sets
    train_df, val_df = train_test_split(df_filtered, test_size=0.2, random_state=42)


    # Create the datasets
    train_dataset = TextDataset(
        titles=train_df['title'].to_numpy(),
        resumeSiteWebs=train_df['resumeSiteWeb'].to_numpy(),
        labels=train_df['Class_Int'].to_numpy(),
        tokenizer=tokenizer,
        max_len=128
    )

    val_dataset = TextDataset(
        titles=val_df['title'].to_numpy(),
        resumeSiteWebs=val_df['resumeSiteWeb'].to_numpy(),
        labels=val_df['Class_Int'].to_numpy(),
        tokenizer=tokenizer,
        max_len=128
    )

    train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=custom_embedding_collate_fn,shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=custom_embedding_collate_fn,shuffle=False)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 2
    f1_score_eval=0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs} ----------------------')
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, device)

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train loss: {train_loss:.4f}')
        print(f'Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}, Validation F1: {val_f1:.4f}')
    
        predictions=model.predict(val_loader)
        cm = confusion_matrix(val_df['Class_Int'], predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()




    
