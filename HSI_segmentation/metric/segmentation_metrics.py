# -*- coding: utf-8 -*-
# @Time     : 2023/11/13  15:23
# @Author   : Geng Qin
# @File     : metrics.py
# @Software : Vscode
import numpy as np
from medpy import metric


def segmentation_accuracy(y_pred, y_true):
    # 确保y_true和y_pred具有相同的形状
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"

    # 将y_true和y_pred展平为一维数组
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # 计算正确预测的像素点数
    correct = np.sum(y_true_flat == y_pred_flat)

    # 计算总像素点数
    total = y_true_flat.shape[0]

    # 计算并返回分割准确度
    accuracy = correct / total
    return accuracy


def calculate_metrics_compose(prediction, ground_truth):
    """
    args:
    Dice相似系数(Dice Similarity Coefficient DSC)或
    Dice系数(Dice Coefficient DC)
    用于有效衡量分割算法预测标签与真实标注标签的空间重叠程度，
    其对应值越大表示分割精度越高。

    Jaccard相似系数(Jaccard Similarity Coefficient JSC)或
    Jaccard系数(Jaccard Coefficient JC)与 Dice 相似系数指标相似，
    也是一种衡量两幅图像相似程度的指标，
    其由实际分割结果与真实标签的交集同二者并集的比值得出(IOU)

    HD95: 计算两个图像中二值对象之间的（对称）豪斯多夫距离 HD 的第95个百分位数。
    与豪斯多夫距离相比，该指标对于小异常值稍微稳定一些，通常用于生物医学分割挑战
    """

    Dsc = metric.binary.dc(prediction, ground_truth)
    Jaccard = metric.binary.jc(prediction, ground_truth)
    hous_dis = metric.binary.hd(prediction, ground_truth, voxelspacing=None, connectivity=1, )
    hous_dis95 = metric.binary.hd95(prediction, ground_truth, voxelspacing=None, connectivity=1, )
    Recall = metric.binary.recall(prediction, ground_truth)
    Prec = metric.binary.precision(prediction, ground_truth)
    sen = metric.binary.sensitivity(prediction, ground_truth)
    spe = metric.binary.specificity(prediction, ground_truth)
    # tpr = metric.binary.true_positive_rate(prediction, ground_truth)
    # tnr = metric.binary.true_negative_rate(prediction, ground_truth)
    asd = metric.binary.asd(prediction, ground_truth)
    acc = segmentation_accuracy(prediction, ground_truth)

    return {
        # 'TPR': tpr,
        # 'TNR': tnr,
        'DICE': Dsc,
        'IOU': Jaccard,
        'ACC': acc,
        'Precision': Prec,
        'HD': hous_dis,
        'HD95': hous_dis95,
        'Recall': Recall,
        'sen': sen,
        'asd': asd
    }


def calculate_iou(prediction, ground_truth, K, ignored_index=None):
    assert prediction.shape == ground_truth.shape
    prediction = prediction.reshape(prediction.size).copy()
    ground_truth = ground_truth.reshape(ground_truth.size)
    # output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = prediction[np.where(prediction == ground_truth)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(prediction, bins=np.arange(K + 1))
    area_target, _ = np.histogram(ground_truth, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection

    return area_intersection, area_union, area_output, area_target


if __name__ == "__main__":

    array1 = np.random.rand(8, 256, 256)
    array2 = np.random.rand(8, 256, 256)
    array3 = []
    array3.append(array1)
    array3.append(array2)
    print(np.array(array3).shape)
    # arr = np.zeros((8, 256, 256), dtype=np.int8)
    # # 将数组中的所有元素随机设置为0或1
    # arr = np.random.randint(2, size=arr.shape, dtype=np.int8)
    label = np.zeros((8, 256, 256), dtype=np.int8)
    # 将数组中的所有元素随机设置为0或1
    label = np.random.randint(3, size=label.shape, dtype=np.int8)
    target = np.random.rand(2, 256, 256)

    dice = calculate_metrics_compose(array1, label)
    # dict = {}
    # dict['dsc'] = dsc
    # dict['iou'] = iou
    # dict['hd'] = hd
    # dict['hd95'] = hd95
    print(dict)
