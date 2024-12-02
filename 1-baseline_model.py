import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, classification_report
from torch.cuda.amp import autocast, GradScaler
from requests.exceptions import ConnectionError

# Load tokenized data
train_encodings, train_labels = torch.load("train_data.pt")
test_encodings, test_labels = torch.load("test_data.pt")

# Define Dataset class
class HateSpeechDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Create Datasets and DataLoader
train_dataset = HateSpeechDataset(train_encodings, list(train_labels))
test_dataset = HateSpeechDataset(test_encodings, list(test_labels))

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Check GPU details
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9} GB")

# Load model
try:
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
except ConnectionError as e:
    print("Failed to download the model. Please check your internet connection or try downloading manually.")
    raise e
model.to(device)


# Set up DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Initialize GradScaler for mixed precision
scaler = GradScaler()

# Evaluation function
def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            labels = batch['labels'].to(device)
            
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    print("\nModel Evaluation:")
    print(classification_report(true_labels, predictions))
    return accuracy_score(true_labels, predictions)

# Training loop with mixed precision
epochs = 2
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        # Prepare inputs and labels
        inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
        labels = batch['labels'].to(device)
        
        # Mixed precision forward pass
        with torch.amp.autocast('cuda'):
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
        
        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    # Print epoch statistics
    print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(train_loader):.4f}")

# Evaluate the model
accuracy = evaluate_model(model, test_loader, device)
print(f"\nFinal Model Accuracy: {accuracy:.4f}")

# Save the trained model
model.save_pretrained("saved_model")  # Save entire model with config and tokenizer
torch.save(model.state_dict(), "baseline_model.pt")  # Save only model weights