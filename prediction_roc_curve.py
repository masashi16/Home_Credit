import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, auc, roc_curve
from sklearn.model_selection import GridSearchCV


def predict(instance, X_test, y_test):

    prediction = instance.predict(X_test)
    prediction_prob = instance.predict_proba(X_test)
    conf_mat = confusion_matrix(y_test, prediction)

    print('accuracy: {}'.format(accuracy_score(y_test, prediction)))
    print('precision: {}'.format(precision_score(y_test, prediction)))
    print('recall: {}'.format(recall_score(y_test, prediction)))
    print('f_score: {}'.format(f1_score(y_test, prediction)))
    print('false positive rate: {}'.format(conf_mat[0][1] / sum(conf_mat[0])))
    print('confusion_matrix: \n{}'.format(conf_mat))

    # ROC曲線
    fpr, tpr, thresholds = roc_curve(y_test, prediction_prob[:,1])
    tpr_01 = tpr[list(fpr).index(fpr[fpr < 0.1][-1])]

    print('recall under FPR 10%: {}'.format(tpr_01))
    print('AUC score: {}'.format(auc(fpr, tpr)))
    plt.plot(fpr, tpr)
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

    return
