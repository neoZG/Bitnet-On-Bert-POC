import torch
import time
import psutil
import GPUtil
from transformers import AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import codecarbon
from torch.utils.data import DataLoader, Dataset

# Ensure that the test_loader is correctly set up and loaded
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
    

# Assuming test_loader and models are prepared beforehand

class ModelEvaluator:
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device

    def evaluate_performance(self):
        """
        Comprehensive model evaluation
        
        Returns:
        - Detailed performance metrics dictionary
        """
        self.model.eval()
        predictions, true_labels = [], []
        
        # Performance tracking
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        # Track GPU metrics
        gpu_track = GPUtil.getGPUs()[0]
        gpu_memory_start = gpu_track.memoryUsed
        
        # Carbon tracking
        tracker = codecarbon.EmissionsTracker()
        tracker.start()
        
        with torch.no_grad():
            inference_start = time.time()
            for batch in self.test_loader:
                inputs = {key: val.to(self.device) for key, val in batch.items() if key != 'labels'}
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                
                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())
        
        # Inference time
        inference_time = time.time() - inference_start
        
        # Carbon emissions
        emissions = tracker.stop()
        
        # Memory usage
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        gpu_track = GPUtil.getGPUs()[0]
        gpu_memory_end = gpu_track.memoryUsed
        
        # Calculate metrics
        metrics = {
            # Classification Metrics
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, average='weighted'),
            'recall': recall_score(true_labels, predictions, average='weighted'),
            'f1_score': f1_score(true_labels, predictions, average='weighted'),
            
            # Computational Metrics
            'inference_time': inference_time,
            'total_runtime': time.time() - start_time,
            'memory_usage': {
                'process_memory_start_mb': start_memory,
                'process_memory_end_mb': end_memory,
                'memory_increase_mb': end_memory - start_memory,
                'gpu_memory_start_mb': gpu_memory_start,
                'gpu_memory_end_mb': gpu_memory_end,
                'gpu_memory_increase_mb': gpu_memory_end - gpu_memory_start
            },
            
            # Environmental Impact
            'carbon_emissions': {
                'emissions_g': emissions,
                # 'energy_consumed_kwh': emissions,
                # 'carbon_emissions_kg': emissions,
            },
            
            # Detailed Classification Report
            'classification_report': classification_report(true_labels, predictions),
            'confusion_matrix': confusion_matrix(true_labels, predictions)
        }
        
        return metrics

    def compare_models(self, other_model):
        """
        Compare current model with another model
        
        Returns:
        - Comparative metrics dictionary
        """
        current_metrics = self.evaluate_performance()
        
        # Switch model and re-evaluate
        original_model = self.model
        self.model = other_model
        other_metrics = self.evaluate_performance()
        
        # Restore original model
        self.model = original_model
        
        return {
            'current_model': current_metrics,
            'other_model': other_metrics,
            'performance_comparison': {
                'accuracy_diff': current_metrics['accuracy'] - other_metrics['accuracy'],
                'inference_time_diff': current_metrics['inference_time'] - other_metrics['inference_time'],
                'memory_usage_diff': current_metrics['memory_usage']['process_memory_end_mb'] - other_metrics['memory_usage']['process_memory_end_mb'],
                'carbon_emissions_diff': current_metrics['carbon_emissions']['emissions_g'] - other_metrics['carbon_emissions']['emissions_g']
            }
        }

# Usage example
def main():
    # Assuming models and test_loader are prepared:
    test_encodings, test_labels = torch.load("test_data.pt")
    test_dataset = HateSpeechDataset(test_encodings, list(test_labels))
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Set the device (CUDA or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    baseline_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    baseline_model.load_state_dict(torch.load("baseline_model.pt"))
    baseline_model.to(device)
    
    bitnet_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    bitnet_model.load_state_dict(torch.load("bitnet_model.pt"))
    bitnet_model.to(device)

    # Create evaluators
    baseline_evaluator = ModelEvaluator(baseline_model, test_loader, device)
    bitnet_evaluator = ModelEvaluator(bitnet_model, test_loader, device)

    # Evaluate models
    print("Baseline Model Metrics:")
    baseline_metrics = baseline_evaluator.evaluate_performance()
    print(baseline_metrics)

    print("\nBitNet Model Metrics:")
    bitnet_metrics = bitnet_evaluator.evaluate_performance()
    print(bitnet_metrics)

    # Compare models
    print("\nModel Comparison:")
    comparison = baseline_evaluator.compare_models(bitnet_model)
    print(comparison)

if __name__ == "__main__":
    main()









# from sklearn.metrics import accuracy_score, classification_report
# import torch

# def evaluate_model(model, test_loader):
#     model.eval()
#     predictions, true_labels = [], []
#     with torch.no_grad():
#         for batch in test_loader:
#             inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
#             labels = batch['labels'].to(device)
#             outputs = model(**inputs)
#             logits = outputs.logits
#             preds = torch.argmax(logits, dim=-1).cpu().numpy()
#             predictions.extend(preds)
#             true_labels.extend(labels.cpu().numpy())
#     return accuracy_score(true_labels, predictions), classification_report(true_labels, predictions)

# # Load test data
# test_loader = torch.load("test_data.pt")

# # Evaluate baseline model
# model.load_state_dict(torch.load("baseline_model.pt"))
# baseline_acc, baseline_report = evaluate_model(model, test_loader)
# print(f"Baseline Accuracy: {baseline_acc}")
# print(baseline_report)

# # Evaluate BitNet model
# model.load_state_dict(torch.load("bitnet_model.pt"))
# bitnet_acc, bitnet_report = evaluate_model(model, test_loader)
# print(f"BitNet Accuracy: {bitnet_acc}")
# print(bitnet_report)
