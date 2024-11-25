import numpy as np

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import cohen_kappa_score


def calculate_classification_metrics(y_pred_prob, y_true, average='macro'):
    """
    计算分类指标: 准确度、AUC、F1分数、精确度和召回率

    参数:
    y_true (np.ndarray): 真实标签
    y_pred_prob (np.ndarray): 模型预测的概率（对于多分类问题，应为二维数组）
    average (str): 用于计算多类指标的平均值方法，例如'macro', 'micro', 'weighted'等

    返回:
    dict: 包含以下键的字典：'accuracy', 'auc', 'f1', 'precision', 'recall'
    """
    # 计算准确度（假设y_pred_prob是类别预测，对于概率预测，需要转换为类别）
    y_pred_class = np.argmax(y_pred_prob, axis=1) if y_pred_prob.ndim > 1 else y_pred_prob
    accuracy = accuracy_score(y_true, y_pred_class)

    # average_accuracy = average_precision_score(y_true, y_pred_class)

    # # 如果是多分类问题，需要转换为one-hot编码以计算AUC
    # if y_true.ndim == 1:
    #     y_true_binary = label_binarize(y_true, classes=[i for i in range(y_pred_prob.shape[1])])
    #     # 计算AUC（对于多分类问题，通常计算每个类别的AUC然后取平均）
    #     auc = roc_auc_score(y_true_binary, y_pred_prob, average=average, multi_class='ovr')
    # else:
    #     # 对于二分类问题，直接使用roc_auc_score
    #     auc = roc_auc_score(y_true, y_pred_prob[:, 1])
    # auc = roc_auc_score(y_true, y_pred_prob[:, 1], multi_class='ovr', average='weighted')

    # 计算F1分数、精确度和召回率
    f1 = f1_score(y_true, y_pred_class, average=average)
    precision = precision_score(y_true, y_pred_class, average=average)
    recall = recall_score(y_true, y_pred_class, average=average)

    kappa = cohen_kappa_score(y_true, y_pred_class)

    # 返回结果字典
    return {
        'accuracy': accuracy,
        # 'Aver_acc': average_accuracy,
        # 'auc': auc,
        'f1-score': f1,
        'precision': precision,
        'recall': recall,
        'kappa': kappa
    }


# 示例用法：
# 假设 y_true 是真实标签，y_pred_prob 是模型预测的概率（对于多分类）
if __name__ == "__main__":
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred_prob = np.random.rand(len(y_true), 3)
    print(y_pred_prob)

    metrics = calculate_classification_metrics(y_pred_prob, y_true)
    print(metrics)