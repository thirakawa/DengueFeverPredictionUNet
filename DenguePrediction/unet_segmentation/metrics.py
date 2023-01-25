#!/usr/bin/env python3

# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py


import numpy as np


class RunningScore(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    @staticmethod
    def _fast_hist(label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        # dice
        if self.n_classes == 2:
            dice = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 1] + 0.5 * (self.confusion_matrix[0, 1] + self.confusion_matrix[1, 0]))
        else:
            dice = 0.0

        return (
            {
                "OverallAcc": acc,
                "MeanAcc": acc_cls,
                "FreqWAcc": fwavacc,
                "MeanIoU": mean_iu,
                "Dice": dice,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))



if __name__ == "__main__":

    metrics = RunningScore(n_classes=2)
    label = np.concatenate((np.zeros(10, dtype=np.int), np.ones(10, dtype=np.int)))
    pred = np.random.randint(0, 2, 20)
    print(label)
    print(pred)
    metrics.update(label, pred)
    print(metrics.get_scores()) 
    print(metrics.confusion_matrix)
