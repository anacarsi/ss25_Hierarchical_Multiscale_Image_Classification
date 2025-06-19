# Metrics for evaluating the performance of classification models.


def accuracy(y_true, y_pred):
    """
    Calculate the accuracy of predictions.
    Parameters:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    Returns:
        float: Accuracy score.
    """
    return (y_true == y_pred).mean()


def precision(y_true, y_pred):
    """
    Calculate the precision of predictions.
    Parameters:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    Returns:
        float: Precision score of the positive class.
    """
    true_positive = ((y_true == 1) & (y_pred == 1)).sum()
    false_positive = ((y_true == 0) & (y_pred == 1)).sum()
    return (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )


def recall(y_true, y_pred):
    """
    Calculate the recall of predictions, also known as sensitivity or true positive rate.
    Parameters:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    Returns:
        float: Recall score of the positive class.
    """
    true_positive = ((y_true == 1) & (y_pred == 1)).sum()
    false_negative = ((y_true == 1) & (y_pred == 0)).sum()
    return (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )


def f1_score(y_true, y_pred):
    """
    Calculate the F1 score of predictions.
    Parameters:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    Returns:
        float: F1 score of the positive class, which is the harmonic mean of precision and recall.
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0


def confusion_matrix(y_true, y_pred):
    """
    Calculate the confusion matrix for binary classification.
    Parameters:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    Returns:
        dict: A dictionary containing the counts of true positives (TP), true negatives (TN),
              false positives (FP), and false negatives (FN).
    """
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
