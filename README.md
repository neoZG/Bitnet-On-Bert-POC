open vscode or any IDE

create a virtual environment using (according to...)
python -m venv venv
python3 -m venv venv
py -3 -m venv venv

activate the venv (on windos)
.\venv\Scripts\Activate

this is how e create the requirements.txt
pip freeze > requirements.txt

but you need to use the following command to unstall all the dependencies
...

and this is how you deactivate the venv on windows
...


the order of execution of the files are the follwing:
0-dataset_preparation.py
1-baseline_model.py
2-bitnet_model.py
3-evaluation.py









DISTILBERT:
Epoch 1, Average Loss: 0.1794
Epoch 2, Average Loss: 0.1440

Model Evaluation:
              precision    recall  f1-score   support

           0       0.60      0.03      0.06       290
           1       0.94      1.00      0.97      4667

    accuracy                           0.94      4957
   macro avg       0.77      0.51      0.51      4957
weighted avg       0.92      0.94      0.92      4957


Final Model Accuracy: 0.9421


BITNET
Using device: cuda
GPU Name: NVIDIA GeForce RTX 3050 Laptop GPU
GPU Memory: 4.294443008 GB
Epoch 1 (BitNet), Average Loss: 0.6835
Epoch 2 (BitNet), Average Loss: 0.6751

Model Evaluation (BitNet):
              precision    recall  f1-score   support

           0       0.00      0.00      0.00       290
           0       0.00      0.00      0.00       290
           0       0.00      0.00      0.00       290
           1       0.94      1.00      0.97      4667

    accuracy                           0.94      4957
   macro avg       0.47      0.50      0.48      4957
weighted avg       0.89      0.94      0.91      4957


Final BitNet Model Accuracy: 0.9415