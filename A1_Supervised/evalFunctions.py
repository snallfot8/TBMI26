import numpy as np
from sklearn import metrics

def calcAccuracy(LPred, LTrue):
    """Calculates prediction accuracy from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Retruns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    acc = np.sum(LPred==LTrue)/len(LPred)
    # ============================================
    return acc


def calcConfusionMatrix(LPred, LTrue):
    """Calculates a confusion matrix from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Returns:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    classes = np.unique(LTrue)
    
    cM = np.zeros((len(classes), len(classes)))

    for i in range(len(classes)):
        for j in range(len(classes)):
           cM[i, j] = np.sum((LTrue == classes[i]) & (LPred == classes[j]))
    
             
        
    #print(metrics.confusion_matrix(LTrue, LPred))
    # ============================================

    return cM


def calcAccuracyCM(cM):
    """Calculates prediction accuracy from a confusion matrix.

    Args:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.

    Returns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    true_values = 0
    for i in range(len(cM)):
        true_values += cM[i,i]
    cM = true_values/np.sum(cM)
    # ============================================
    
    return cM
