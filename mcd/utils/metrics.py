import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def get_confusionmatrix_fnd(preds, labels):
    # label_predicted = np.argmax(preds, axis=1)
    label_predicted = preds
    print(accuracy_score(labels, label_predicted))
    print(classification_report(labels, label_predicted, labels=[0.0, 1.0], target_names=['real', 'fake'], digits=4))
    print(confusion_matrix(labels, label_predicted, labels=[0, 1]))


def metrics(y_label, y_predict):
    scores = {}
    if y_predict is None or y_label is None:
        print(y_predict, y_label)
    scores['auc'] = round(roc_auc_score(y_label, y_predict, average='macro'), 5)
    y_predict = np.around(np.array(y_predict)).astype(int)
    scores['f1'] = round(f1_score(y_label, y_predict, average='macro'), 5)
    scores['recall'] = round(recall_score(y_label, y_predict, average='macro'), 5)
    scores['precision'] = round(precision_score(y_label, y_predict, average='macro'), 5)
    scores['acc'] = round(accuracy_score(y_label, y_predict), 5)

    return scores