import torch.nn as nn
from transformers import AutoModel
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
class BERTWithExtraFeatures(nn.Module):
    def __init__(self, num_labels, extra_feature_dim):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size + extra_feature_dim, num_labels)

    def forward(self, input_ids, attention_mask, extra_features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        combined = torch.cat((pooled_output, extra_features), dim=1)
        x = self.dropout(combined)
        return self.classifier(x)

class EmojiDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, sentiment_cols, emotion_cols):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sentiment_cols = sentiment_cols
        self.emotion_cols = emotion_cols

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['Text']
        labels = torch.tensor(row['emojis_vec_fixed'], dtype=torch.float) 
        
        # tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        # get extra features
        sentiment = torch.tensor(row[self.sentiment_cols].values.astype(float), dtype=torch.float)
        emotion = torch.tensor(row[self.emotion_cols].values.astype(float), dtype=torch.float)
        extra_features = torch.cat([sentiment, emotion], dim=0)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'extra_features': extra_features,
            'labels': labels
        }
class BERTWithExtraFeatures(nn.Module):
    def __init__(self, num_labels, extra_feature_dim, pretrained_model='bert-base-uncased'):
        super(BERTWithExtraFeatures, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size + extra_feature_dim, num_labels)

    
    def forward(self, input_ids, attention_mask, extra_features):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output  # shape: (batch_size, hidden_size)

        combined = torch.cat((pooled_output, extra_features), dim=1)  # shape: (batch_size, hidden + extra)
        x = self.dropout(combined)
        logits = self.classifier(x)

        return logits


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        try:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            extra_features = batch['extra_features'].to(device)
            labels = batch['labels'].to(device)
    
            optimizer.zero_grad()
    
            outputs = model(input_ids, attention_mask, extra_features)
            loss = criterion(outputs, labels)
    
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
        except Exception as e:
            print(f"Error encountered during training: {e}")
            continue

    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                extra_features = batch['extra_features'].to(device)
                labels = batch['labels'].to(device)
    
                outputs = model(input_ids, attention_mask, extra_features)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
            except Exception as e: 
                print(f"Error encountered during evaluation: {e}")
                continue

    return total_loss / len(dataloader)