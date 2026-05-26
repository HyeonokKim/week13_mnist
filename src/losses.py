# -*- coding: utf-8 -*-
"""손실 함수 모음."""

import numpy as np

from activations import Softmax


def cross_entropy_loss(y_pred, y_true):
    """
    Cross Entropy Error (배치 평균).
    y_pred: (batch_size, 10) 확률
    y_true: (batch_size,) 정수 레이블 0~9
    """
    # 힌트: np.clip으로 log(0)을 피하고, np.arange(batch_size)로 정답 위치를 고릅니다.
    y_pred = np.clip(y_pred, 1e-7, 1.0)
    batch_size = y_pred.shape[0]
    selected = y_pred[np.arange(batch_size), y_true]
    loss = -np.sum(np.log(selected)) / batch_size
    return loss


class SoftmaxWithLoss:
    """Softmax 출력 + Cross Entropy 손실을 한 묶음으로 처리하는 계층."""

    def __init__(self):
        self.loss = None
        self.y = None    # Softmax 출력 (확률)
        self.t = None    # 정답 레이블

    def forward(self, x, t):
        self.t = t
        self.y = Softmax().forward(x)
        self.loss = cross_entropy_loss(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        if self.y.size == self.t.size:
            # 원-핫 인코딩일 때
            dx = (self.y - self.t) / batch_size
        else:
            # 정수 레이블일 때
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx /= batch_size

        return dx
