# **BitNet Experimentation**

This repository contains the code and results for a proof-of-concept experiment comparing DistilBERT and BitNet models for hate speech detection. The study evaluates model performance, memory usage, inference time, and energy consumption to determine the feasibility of using quantized models like BitNet in real-world scenarios.

The necessary model files (`.pt`) and datasets are available for download, so you can skip the training steps and go straight to the evaluation. You can find them in the following Google Drive folder:
[Download Model and Data Files](https://drive.google.com/drive/folders/1e_dK382XmkfVvE5LZIHr-_RV2g0mz8pQ?usp=sharing)

---

## **Project Structure**

| File                       | Description                                |
|----------------------------|--------------------------------------------|
| `0-dataset_preparation.py` | Prepares and tokenizes the dataset.        |
| `1-baseline_model.py`      | Trains the baseline DistilBERT model.      |
| `2-bitnet_model.py`        | Trains the quantized BitNet model.         |
| `3-evaluation.py`          | Evaluates and compares both models.        |

---

## **Setup Instructions**
Follow these steps to set up the project on your system.

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/bitnet_experimentation.git
cd bitnet_experimentation
```

### **2. Create a Virtual Environment**
#### **On Windows**:
```bash
py -3 -m venv venv
.\venv\Scripts\activate
```
#### **On macOS/Linux**:
```bash
python3 -m venv venv
source venv/bin/activate
```

### **3. Install Dependencies**
Create a `requirements.txt` to capture all dependencies:
```bash
pip freeze > requirements.txt
```
Then, install the required packages:
```bash
pip install -r requirements.txt
```

### **4. Deactivate the Virtual Environment**
To deactivate the environment:
```bash
deactivate
```

---

## **Execution Order**
Run the scripts in the following order to replicate the experiment:

1. **Dataset Preparation**:  
   ```bash
   python 0-dataset_preparation.py
   ```
   This script tokenizes the dataset and prepares it for training.

2. **Train Baseline Model**:  
   ```bash
   python 1-baseline_model.py
   ```
   Trains the DistilBERT model on the dataset.

3. **Train BitNet Model**:  
   ```bash
   python 2-bitnet_model.py
   ```
   Fine-tunes the BitNet model using the pre-trained weights from DistilBERT.

4. **Model Evaluation**:  
   ```bash
   python 3-evaluation.py
   ```
   Compares the models based on performance, memory usage, and inference time.

---

## **Key Results**

### **1. Classification Metrics**
| Metric          | Baseline Model | BitNet Model | Difference (Baseline - BitNet) |
|------------------|----------------|--------------|---------------------------------|
| Accuracy         | 0.9421         | 0.9415       | +0.0006                        |
| F1-Score (Weighted) | 0.9168         | 0.9131       | +0.0037                        |

---

### **2. Computational Metrics**
| Metric                    | Baseline Model | BitNet Model | Difference (Baseline - BitNet) |
|---------------------------|----------------|--------------|---------------------------------|
| Memory Increase (MB)      | 0.32           | 0.16         | +0.16                          |
| Inference Time (seconds)  | 18.57          | 18.53        | +0.04                          |

---

## **Observations**
- The **Baseline Model** slightly outperforms BitNet in accuracy and F1-score.  
- **BitNet** demonstrates better memory efficiency, using 50% less process memory during inference.  
- Inference time for both models is nearly identical, with BitNet being marginally faster.  
- These results confirm that BitNet is a viable alternative for classification tasks, especially in resource-constrained environments.

---

## **Future Directions**
This proof-of-concept establishes the groundwork for large-scale experimentation. Future experiments can focus on:
- Scaling to larger datasets.
- Increasing training epochs to test model convergence.
- Evaluating memory and energy efficiency on resource-intensive hardware setups.

