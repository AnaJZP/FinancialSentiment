import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import tensorflow as tf
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import nltk

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
    pass

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, n_classes, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, n_classes)  # * 2 for bidirectional
        
    def forward(self, text, attention_mask):
        embedded = self.embedding(text)
        
        # Secuencia de paquetes para manejar entradas de longitud variable
        packed_output, (hidden, cell) = self.lstm(embedded)
        
        # Concatenar los estados ocultos finales hacia adelante y hacia atrás
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        return torch.softmax(output, dim=1)

        pass

def prepare_data(texts, labels=None, max_length=None):
    """Prepara los datos para el entrenamiento, usando el 90% de los registros para determinar max_length"""
    if max_length is None:
        # Calcular longitud máxima usando el 90% de los datos
        text_lengths = [len(word_tokenize(text)) for text in texts]
        max_length = int(np.percentile(text_lengths, 90))
    
    return texts, labels, max_length

def create_word_embeddings(texts, embedding_size=100, window=5, min_count=1):
    """Crea embeddings usando Word2Vec"""
    # Tokenizar textos
    tokenized_texts = [word_tokenize(text.lower()) for text in texts]
    
    # Entrenar Word2Vec
    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=embedding_size,
        window=window,
        min_count=min_count,
        workers=4
    )
    
    return model

def train_model(model, train_loader, valid_loader, epochs=10, learning_rate=0.001):
    """Entrena el modelo LSTM"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_valid_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validación
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
        
        train_loss = total_loss / len(train_loader)
        valid_loss = valid_loss / len(valid_loader)
        accuracy = correct_predictions / total_predictions
        
        print(f'Época: {epoch+1}')
        print(f'\tPérdida de entrenamiento: {train_loss:.4f}')
        print(f'\tPérdida de validación: {valid_loss:.4f}')
        print(f'\tPrecisión: {accuracy:.4f}')
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pt')

def create_finbert_embeddings(texts, labels, max_length=128):
    """Crea embeddings usando FinBERT"""
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModel.from_pretrained("ProsusAI/finbert")
    
    # Crear dataset
    dataset = TextDataset(texts, labels, tokenizer, max_length)
    
    # Dividir en train y validation
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size]
    )
    
    # Crear dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=False
    )
    
    return train_loader, valid_loader, tokenizer.vocab_size

def main(texts, labels):
    """Función principal para ejecutar el pipeline completo"""
    # Preparar datos
    texts, labels, max_length = prepare_data(texts, labels)
    
    # Crear embeddings con FinBERT
    train_loader, valid_loader, vocab_size = create_finbert_embeddings(texts, labels)
    
    # Crear y entrenar modelo LSTM
    model = SentimentLSTM(
        vocab_size=vocab_size,
        embedding_dim=768,  # Para FinBERT
        hidden_dim=256,
        n_layers=2,
        n_classes=3
    )
    
    # Entrenar modelo
    train_model(model, train_loader, valid_loader)
    
    return model
