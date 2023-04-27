import jsonlines
import numpy as np
from tabulate import tabulate
from sklearn.metrics import confusion_matrix

def confusionMatrix(fileName):
    true_labels = []
    predicted_labels = []
    with jsonlines.open(fileName) as r:
        for line in r:
            true_labels.append(line['label'])
            predicted_labels.append(line['predicted_label'])
    
    matrix = confusion_matrix(true_labels, predicted_labels)
    precision = np.round(np.diag(matrix) / np.sum(matrix, axis=0), 4)
    recall = np.round(np.diag(matrix) / np.sum(matrix, axis=1), 4)
    f1_scores = np.round(2 * precision * recall / (precision + recall), 4)

    labels = ['Entailment', 'Neutral', 'Contradiction']

    # Create a pandas DataFrame from the confusion matrix
    table = tabulate(matrix, headers=labels, showindex=labels)
    print('Confusion Matrix')
    print(table)
    print()
    
    data = np.vstack([precision, recall, f1_scores]).T.tolist()
    metrics = tabulate(data, headers=['Precision', 'Recall', 'F1 Score'], showindex=labels)
    print('Metrics')
    print(metrics)
    print()

    return matrix

baseModelConfusion = confusionMatrix('eval_predictions_baseModel.jsonl')