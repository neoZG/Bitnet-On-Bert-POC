import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AdamW
from bitnet import replace_linears_in_hf  # Ensure BitNet is available
from sklearn.metrics import accuracy_score, classification_report
from torch.cuda.amp import autocast, GradScaler

# Load tokenized data (same way as before)
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

# Create Datasets and DataLoader for BitNet training
train_dataset = HateSpeechDataset(train_encodings, list(train_labels))
test_dataset = HateSpeechDataset(test_encodings, list(test_labels))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Check GPU details
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9} GB")

# Load the baseline model (DistilBERT)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Load pretrained weights from the baseline model
model.load_state_dict(torch.load("baseline_model.pt"))

# Apply BitNet quantization
replace_linears_in_hf(model)

# Move model to the correct device (GPU or CPU)
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Initialize GradScaler for mixed precision
scaler = GradScaler()

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
        with autocast():
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
        
        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    # Print epoch statistics
    print(f"Epoch {epoch + 1} (BitNet), Average Loss: {total_loss / len(train_loader):.4f}")

# Save the trained BitNet model
torch.save(model.state_dict(), "bitnet_model.pt")

# Evaluate the BitNet model
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
    
    print("\nModel Evaluation (BitNet):")
    print(classification_report(true_labels, predictions))
    return accuracy_score(true_labels, predictions)

# Evaluate BitNet model
accuracy = evaluate_model(model, test_loader, device)
print(f"\nFinal BitNet Model Accuracy: {accuracy:.4f}")






# # bitnet_model.py
# from transformers import AutoModelForSequenceClassification
# from bitnet import replace_linears_in_hf  # Ensure BitNet is available

# # Load baseline model
# model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
# model.load_state_dict(torch.load("baseline_model.pt"))

# # Apply BitNet
# replace_linears_in_hf(model)

# # Fine-tune BitNet model
# train_loader = torch.load("train_data.pt")
# optimizer = AdamW(model.parameters(), lr=5e-5)
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model.to(device)

# # Training loop
# epochs = 2
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     for batch in train_loader:
#         inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
#         labels = batch['labels'].to(device)
#         outputs = model(**inputs, labels=labels)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         total_loss += loss.item()
#     print(f"Epoch {epoch + 1} (BitNet), Loss: {total_loss / len(train_loader)}")

# # Save the quantized model
# torch.save(model.state_dict(), "bitnet_model.pt")
