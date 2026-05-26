# -*- coding: utf-8 -*-
"""
MNIST 분류용 신경망 조립 모듈.

개별 layer를 OrderedDict에 쌓아 forward/backward 순서를 명확히 유지합니다.
"""

from collections import OrderedDict

import numpy as np

from activations import ReLU, Softmax
from layers import Affine, BatchNorm, Dropout
from losses import cross_entropy_loss


class NeuralNetwork:
    """
    MNIST 분류용 신경망.
    입력 784 -> 은닉층(들) -> 출력 10 (Softmax).
    은닉층 구성: Affine -> BatchNorm -> ReLU -> Dropout (모두 필수)
    가중치 초기화: He 또는 Xavier 중 선택.
    """

    def __init__(self, use_batchnorm=True, use_dropout=True, dropout_ratio=0.5):
        """
        Args:
            use_batchnorm: 은닉층마다 BatchNorm을 넣을지 여부
            use_dropout: 은닉층마다 Dropout을 넣을지 여부
            dropout_ratio: Dropout에서 끌 뉴런 비율
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        # 구조: 784 -> 512 -> 256 -> 10
        sizes = [784, 512, 256, 10]
        self.params = {}
        self.grads = {}

        # He 초기화로 Affine 가중치 생성
        for i in range(len(sizes) - 1):
            in_dim, out_dim = sizes[i], sizes[i + 1]
            self.params[f"W{i+1}"] = np.random.randn(in_dim, out_dim).astype(np.float32) * np.sqrt(2.0 / in_dim)
            self.params[f"b{i+1}"] = np.zeros(out_dim, dtype=np.float32)

        # BatchNorm gamma, beta는 은닉층(1, 2)에만 적용
        if use_batchnorm:
            for i in range(1, len(sizes) - 1):
                self.params[f"gamma{i}"] = np.ones(sizes[i], dtype=np.float32)
                self.params[f"beta{i}"] = np.zeros(sizes[i], dtype=np.float32)

        # 모든 params와 같은 key로 grads dict 준비
        for key, val in self.params.items():
            self.grads[key] = np.zeros_like(val)

        # 레이어 조립
        self.layers = OrderedDict()
        # 은닉층 1
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        if use_batchnorm:
            self.layers["BatchNorm1"] = BatchNorm(self.params["gamma1"], self.params["beta1"])
        self.layers["ReLU1"] = ReLU()
        if use_dropout:
            self.layers["Dropout1"] = Dropout(dropout_ratio)

        # 은닉층 2
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])
        if use_batchnorm:
            self.layers["BatchNorm2"] = BatchNorm(self.params["gamma2"], self.params["beta2"])
        self.layers["ReLU2"] = ReLU()
        if use_dropout:
            self.layers["Dropout2"] = Dropout(dropout_ratio)

        # 출력층
        self.layers["Affine3"] = Affine(self.params["W3"], self.params["b3"])

        # Softmax (출력 확률 변환용)
        self.softmax = Softmax()

    def forward(self, x, train=True):
        """
        Args:
            x: (batch_size, 784) 정규화된 MNIST 이미지
            train: BatchNorm/Dropout의 학습 모드 여부

        Returns:
            (batch_size, 10) 각 숫자 클래스의 확률
        """
        out = x
        for name, layer in self.layers.items():
            if isinstance(layer, (BatchNorm, Dropout)):
                out = layer.forward(out, train=train)
            else:
                out = layer.forward(out)
        return self.softmax.forward(out)

    def backward(self, dout):
        """
        네트워크 전체 역전파를 수행하고 self.grads를 채웁니다.

        Args:
            dout: Softmax+CrossEntropy를 합친 출력층 gradient
        """
        # Softmax를 먼저 통과 (CE와 합쳐 미분되어 사실상 그대로 통과)
        dout = self.softmax.backward(dout)

        # layer를 역순으로 통과
        for name, layer in reversed(self.layers.items()):
            dout = layer.backward(dout)

        # 각 layer의 gradient를 self.grads로 모음
        for i in range(1, 4):
            self.grads[f"W{i}"] = self.layers[f"Affine{i}"].dW
            self.grads[f"b{i}"] = self.layers[f"Affine{i}"].db

        if self.use_batchnorm:
            for i in range(1, 3):
                self.grads[f"gamma{i}"] = self.layers[f"BatchNorm{i}"].dgamma
                self.grads[f"beta{i}"] = self.layers[f"BatchNorm{i}"].dbeta

    def loss(self, x, y):
        """현재 모델의 예측 확률을 만든 뒤 cross entropy loss를 반환합니다."""
        y_pred = self.forward(x, train=True)
        return cross_entropy_loss(y_pred, y)

    def predict(self, x):
        """추론 모드로 확률을 예측합니다. BatchNorm/Dropout은 train=False로 동작합니다."""
        return self.forward(x, train=False)
