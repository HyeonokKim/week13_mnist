# -*- coding: utf-8 -*-
"""학습 루프, 평가, 시각화 함수 모음."""

import matplotlib.pyplot as plt
import numpy as np

from losses import cross_entropy_loss


def train(model, optimizer, x_train, y_train, epochs=20, batch_size=128):
    """
    미니배치 학습 루프.

    한 배치마다 Forward -> Loss -> Backward -> Optimizer 업데이트 순서로 진행합니다.
    교육생은 이 함수에서 "예측값을 만들고, 손실을 계산하고, gradient로 파라미터를 바꾸는"
    전체 흐름을 확인할 수 있습니다.

    Returns:
        loss_history: epoch별 평균 손실 리스트
    """
    n_samples = x_train.shape[0]
    loss_history = []

    for epoch in range(epochs):
        # 1) 인덱스 무작위 셔플
        perm = np.random.permutation(n_samples)
        x_shuffled = x_train[perm]
        y_shuffled = y_train[perm]

        epoch_losses = []
        # 2) 배치 단위 학습
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            x_batch = x_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            actual_batch = x_batch.shape[0]

            # Forward + 손실
            y_pred = model.forward(x_batch, train=True)
            loss = cross_entropy_loss(y_pred, y_batch)
            epoch_losses.append(loss)

            # Softmax + CrossEntropy 결합 gradient
            dout = y_pred.copy()
            dout[np.arange(actual_batch), y_batch] -= 1
            dout /= actual_batch

            # Backward + 파라미터 업데이트
            model.backward(dout)
            optimizer.update(model.params, model.grads)

        avg_loss = float(np.mean(epoch_losses))
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")

    return loss_history


def evaluate(model, x, y):
    """정확도(%)와 총 파라미터 수 반환."""
    y_pred = model.predict(x)
    accuracy = np.mean(np.argmax(y_pred, axis=1) == y) * 100
    total_params = sum(p.size for p in model.params.values())
    return accuracy, total_params


def plot_loss_history(loss_history):
    """손실 커브 그래프."""
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()
