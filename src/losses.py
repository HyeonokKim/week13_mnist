# -*- coding: utf-8 -*-
"""손실 함수 모음."""

import numpy as np


def cross_entropy_loss(y_pred, y_true):
    """
    Cross Entropy Error (배치 평균).
    y_pred: (batch_size, 10) 확률
    y_true: (batch_size,) 정수 레이블 0~9
    """
    batch_size = y_pred.shape[0]
    # log(0) 방지를 위해 확률을 [1e-7, 1] 범위로 clip
    y_pred_clipped = np.clip(y_pred, 1e-7, 1.0)
    # 각 샘플의 정답 클래스 확률만 추출 후 로그 적용
    correct_probs = y_pred_clipped[np.arange(batch_size), y_true]
    loss = -np.mean(np.log(correct_probs))
    return loss
