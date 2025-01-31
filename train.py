import os
import warnings
import logging
import torch
import numpy as np
from eda import cargar_datos
from model import SentimentLSTM
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from collections import Counter
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def aplicar_smote(texts, labels):
    print("Distribución original:", Counter(labels))
    vectorizer = TfidfVectorizer(max_features=1000)
    X_vec = vectorizer.fit_transform([' '.join(str(text).split()[:100]) for text in texts])
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_vec, labels)
    balanced_texts = [' '.join(text) for text in vectorizer.inverse_transform(X_resampled)]  # Convertir a lista de strings
    print("Distribución después de SMOTE:", Counter(y_resampled))
    return balanced_texts, y_resampled

def create_dataloaders(texts, labels, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    dataset = TextDataset(texts, labels, tokenizer, max_length=128)
    
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    
    return train_loader, valid_loader, tokenizer.vocab_size

def train_sentiment_model():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Usando dispositivo: {device}")

        df = cargar_datos()
        texts = df['texto'].tolist()
        labels = df['etiqueta'].tolist()
        
        logging.info("Aplicando SMOTE...")
        texts_balanced, labels_balanced = aplicar_smote(texts, labels)
        
        logging.info("Creando dataloaders...")
        train_loader, valid_loader, vocab_size = create_dataloaders(texts_balanced, labels_balanced)
        
        model = SentimentLSTM(
            vocab_size=vocab_size,
            embedding_dim=768,
            hidden_dim=512,  # Aumentado
            n_layers=3,      # Aumentado
            n_classes=3,
            dropout=0.3
        ).to(device)
        
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=2e-5,
            epochs=10,
            steps_per_epoch=len(train_loader)
        )
        
        best_valid_loss = float('inf')
        patience = 5
        early_stopping_counter = 0
        
        for epoch in range(10):
            model.train()
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                predictions = model(input_ids, attention_mask)
                loss = criterion(predictions, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(predictions, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
            
            train_loss = total_loss / len(train_loader)
            train_accuracy = correct_predictions / total_predictions
            
            model.eval()
            valid_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            with torch.no_grad():
                for batch in valid_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    
                    predictions = model(input_ids, attention_mask)
                    loss = criterion(predictions, labels)
                    
                    valid_loss += loss.item()
                    _, predicted = torch.max(predictions, 1)
                    correct_predictions += (predicted == labels).sum().item()
                    total_predictions += labels.size(0)
            
            valid_loss = valid_loss / len(valid_loader)
            valid_accuracy = correct_predictions / total_predictions
            
            logging.info(f'Época: {epoch+1}/10')
            logging.info(f'\tPérdida de entrenamiento: {train_loss:.4f}')
            logging.info(f'\tPrecisión de entrenamiento: {train_accuracy:.4f}')
            logging.info(f'\tPérdida de validación: {valid_loss:.4f}')
            logging.info(f'\tPrecisión de validación: {valid_accuracy:.4f}')
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'best_model.pt')
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter >= patience:
                logging.info(f'Early stopping en época {epoch+1}')
                break
        
        return model
        
    except Exception as e:
        logging.error(f"Error durante el entrenamiento: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    print("Iniciando entrenamiento del modelo de sentimientos...")
    train_sentiment_model()