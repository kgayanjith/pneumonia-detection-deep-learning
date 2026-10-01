import numpy as np
from tensorflow.keras.models import load_model
from data_loader import get_generators
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

train_gen, val_gen, test_gen = get_generators("data")

model = load_model("pneumonia_model.keras")

test_gen.reset()
y_true = test_gen.classes
y_prob = model.predict(test_gen)
y_pred = (y_prob > 0.5).astype(int).flatten()

print(classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"]))
print(confusion_matrix(y_true, y_pred))
print("ROC-AUC:", roc_auc_score(y_true, y_prob))