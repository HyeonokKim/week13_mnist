# MNIST 숫자 인식 과제 보고서

## 0. 반·팀원

| 항목 | 내용 |
| --- | --- |
| 반 | 301 |
| 팀명 | 6조 |
| 팀원 | 김규태, 김세인, 김정환, 김현옥 |

---

## 1. 실험 목적

본 과제의 목적은 PyTorch, TensorFlow 같은 딥러닝 프레임워크를 사용하지 않고 NumPy만으로 MNIST 손글씨 숫자 분류 신경망을 구현하는 것이다.

현재 구현은 Affine 계층, ReLU, Softmax, Cross Entropy Loss, Batch Normalization, Dropout, Adam optimizer, 학습 루프, 평가 함수까지 직접 작성한 구조이다. `evaluate(model, x, y)`를 통해 테스트 정확도와 총 파라미터 수를 확인하며, 목표 정확도는 95% 이상이다.

---

## 2. 모델 구조

현재 `src/network.py`의 `NeuralNetwork` 구현 기준 모델 구조는 다음과 같다.

| 항목 | 내용 |
| --- | --- |
| 입력층 | 784차원 입력, 28x28 MNIST 이미지를 펼친 벡터 |
| 은닉층 수 | 2개 |
| 은닉층 차원 | 512, 256 |
| 출력층 | 10차원 출력, 숫자 0~9 분류 |
| 활성화 함수 | ReLU |
| BatchNorm | `use_batchnorm=True`일 때 각 은닉층 뒤에 적용 |
| Dropout | `use_dropout=True`일 때 각 은닉층 뒤에 적용 |
| Dropout 비율 | 기본값 `0.5` |
| 출력 처리 | 마지막 Affine 출력 뒤 Softmax 적용 |

```text
Input(784)
-> Affine(512) -> BatchNorm -> ReLU -> Dropout(0.5)
-> Affine(256) -> BatchNorm -> ReLU -> Dropout(0.5)
-> Affine(10)
-> Softmax
```

`predict()`에서는 `train=False`로 forward를 수행하여 BatchNorm과 Dropout이 추론 모드로 동작한다.

---

## 3. 학습 설정

현재 `mnist_lab.ipynb`, `src/training.py`, `src/optimizers.py` 기준 학습 설정은 다음과 같다.

| 항목 | 값 |
| --- | --- |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Adam beta1 | 0.9 |
| Adam beta2 | 0.999 |
| Adam eps | 1e-8 |
| Epochs | 20 |
| Batch size | 128 |
| 손실 함수 | Cross Entropy Loss |
| BatchNorm momentum | 0.9 |
| 가중치 초기화 | He initialization |
| Bias 초기화 | 0 |
| BatchNorm 초기화 | `gamma=1`, `beta=0` |

학습 루프는 매 epoch마다 학습 데이터를 무작위로 섞은 뒤 미니배치 단위로 forward, loss 계산, backward, optimizer update를 수행한다.

---

## 4. 실험 환경

| 항목 | 내용 |
| --- | --- |
| 실행 환경 | 로컬 또는 Google Colab CPU |
| Python | Python 3.11 기준 |
| 주요 라이브러리 | NumPy, Matplotlib |
| 데이터셋 | MNIST, `data/mnist.npz` |
| 학습 시간 | 현재 저장된 실행 결과 없음. Colab CPU 또는 로컬에서 실행 후 기록 필요 |

---

## 5. 결과

현재 코드에서 확정할 수 있는 결과와 실행 후 기록해야 하는 결과는 다음과 같다.

| 항목 | 값 |
| --- | --- |
| 총 파라미터 수 | 537,354 |
| Train loss | `train()` 실행 후 반환되는 `loss_history`로 확인 |
| Test accuracy | `evaluate(model, x_test, y_test)` 실행 후 확인 |
| 목표 정확도 | 95% 이상 |
| 권장 정확도 | 97% 이상 |

총 파라미터 수 계산은 다음과 같다.

| 파라미터 | 개수 |
| --- | ---: |
| W1, b1 | 784 x 512 + 512 = 401,920 |
| W2, b2 | 512 x 256 + 256 = 131,328 |
| W3, b3 | 256 x 10 + 10 = 2,570 |
| BatchNorm gamma/beta | 512 + 512 + 256 + 256 = 1,536 |
| 합계 | 537,354 |

### Loss curve

`plot_loss_history(loss_history)` 함수로 epoch별 손실 곡선을 시각화한다. 현재 노트북은 다음 흐름으로 손실 곡선을 출력하도록 구성되어 있다.

```python
loss_history = train(model, optimizer, x_train, y_train, epochs=20, batch_size=128)
plot_loss_history(loss_history)
```

### Learning rate 변경에 따른 정확도 비교

현재 최종 실행 코드는 `lr=0.001`을 사용한다. learning rate 비교 실험을 수행할 경우 아래 표에 결과를 기록한다.

| Learning rate | Test accuracy | 관찰 내용 |
| --- | ---: | --- |
| 0.01 | 실행 후 기록 | 손실 변동 여부 확인 |
| 0.001 | 실행 후 기록 | 현재 기본 설정 |
| 0.0001 | 실행 후 기록 | 수렴 속도 확인 |

### Dropout 적용 전후 정확도 비교

현재 최종 모델은 `use_dropout=True`, `dropout_ratio=0.5`를 사용한다. Dropout 비교 실험을 수행할 경우 아래 표에 결과를 기록한다.

| 설정 | Test accuracy | 관찰 내용 |
| --- | ---: | --- |
| Dropout 미적용 | 실행 후 기록 | 과적합 여부 확인 |
| Dropout 0.5 적용 | 실행 후 기록 | 현재 기본 설정 |

---

## 6. 회고

현재 구현은 NumPy만 사용해 신경망의 핵심 구성 요소를 직접 작성했다는 점에서 과제 요구사항에 맞는다. Affine 계층의 forward/backward, BatchNorm의 학습/추론 모드, Dropout의 학습/추론 모드, Softmax와 Cross Entropy 기반 gradient 흐름, Adam optimizer의 파라미터 업데이트 과정을 코드로 확인할 수 있다.

모델은 `784 -> 512 -> 256 -> 10` 구조로 구성되어 있으며, BatchNorm을 통해 학습 안정성을 높이고 Dropout을 통해 과적합을 줄이도록 설계했다. 최종 제출 전에는 노트북을 실행해 `loss_history`, `Test Accuracy`, 학습 시간을 실제 값으로 기록하고, 목표 정확도 95% 이상을 만족하는지 확인해야 한다.
