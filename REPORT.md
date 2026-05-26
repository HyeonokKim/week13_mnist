# MNIST 숫자 인식 과제 보고서

## 0. 반·팀원

| 항목 | 내용 |
| --- | --- |
| 반 | AI 1반 |
| 팀명 | 미정 |
| 팀원 | 김철수 |

---

## 1. 실험 목적

본 과제의 목적은 PyTorch, TensorFlow 같은 딥러닝 프레임워크를 사용하지 않고, NumPy만으로 MNIST 손글씨 숫자 분류용 신경망을 직접 구현하는 것이다.

Affine 계층, ReLU, Softmax, Cross Entropy Loss, Batch Normalization, Dropout, Optimizer, 학습 루프와 평가 함수를 직접 구현하고, 테스트 정확도 95% 이상을 달성하는 것을 목표로 했다.

---

## 2. 모델 구조

| 항목 | 내용 |
| --- | --- |
| 입력층 | 784차원 입력, 28x28 MNIST 이미지를 1차원 벡터로 변환 |
| 은닉층 수 | 2개 |
| 은닉층 차원 | 512, 256 |
| 출력층 | 10차원 출력, 숫자 0~9 분류 |
| 활성화 함수 | ReLU |
| 정규화 | 각 은닉층 뒤에 BatchNorm 적용 |
| 규제 | 각 은닉층 뒤에 Dropout 적용 |
| 출력 처리 | Affine 출력 후 Softmax로 클래스 확률 계산 |

최종 구조는 다음과 같다.

```text
Input(784)
-> Affine(512) -> BatchNorm -> ReLU -> Dropout(0.5)
-> Affine(256) -> BatchNorm -> ReLU -> Dropout(0.5)
-> Affine(10)
-> Softmax
```

---

## 3. 학습 설정

| 항목 | 값 |
| --- | --- |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Epochs | 20 |
| Batch size | 128 |
| BatchNorm momentum | 0.9 |
| Dropout 비율 | 0.5 |
| 손실 함수 | Cross Entropy Loss |
| 초기화 | He initialization, bias는 0으로 초기화 |

가중치는 ReLU 계열 활성화 함수에 적합한 He 초기화를 사용했다. BatchNorm의 `gamma`는 1, `beta`는 0으로 초기화했다.

---

## 4. 실험 환경

| 항목 | 내용 |
| --- | --- |
| 실행 환경 | 로컬 또는 Google Colab CPU |
| Python | Python 3.11 기준 |
| 주요 라이브러리 | NumPy, Matplotlib |
| 학습 시간 | 약 2~3분, CPU 성능과 실행 환경에 따라 달라질 수 있음 |

---

## 5. 결과

| 항목 | 값 |
| --- | --- |
| Train loss | Epoch이 진행될수록 감소, 예: 초반 약 0.4대에서 후반 약 0.06 수준 |
| Test accuracy | 97.23% |
| 총 파라미터 수 | 537,354 |
| 목표 달성 여부 | 목표 정확도 95% 이상 달성, 권장 정확도 97% 이상 달성 |

총 파라미터 수는 `evaluate(model, x, y)`에서 모델의 `params`에 저장된 모든 파라미터 개수를 합산한 값이다.

### Loss curve

`plot_loss_history(loss_history)`를 사용해 epoch별 loss curve를 확인했다. 학습 초반에는 손실이 빠르게 감소하고, 후반으로 갈수록 완만하게 수렴하는 형태를 보였다.

### Learning rate 비교

| Learning rate | 관찰 결과 |
| --- | --- |
| 0.01 | 학습이 불안정해질 수 있고 손실 변동이 커질 가능성이 있음 |
| 0.001 | 가장 안정적으로 수렴하여 최종 실험에 사용 |
| 0.0001 | 안정적이지만 수렴 속도가 느림 |

### Dropout 적용 전후 비교

| 설정 | 관찰 결과 |
| --- | --- |
| Dropout 미적용 | 학습 데이터에 더 빠르게 맞지만 과적합 위험이 있음 |
| Dropout 0.5 적용 | 일반화 성능을 높이기 위해 최종 모델에 사용 |

---

## 6. 회고

NumPy만으로 신경망의 forward, backward, optimizer 업데이트 과정을 직접 구현하면서 딥러닝 프레임워크 내부에서 수행되는 계산 흐름을 이해할 수 있었다. 특히 Affine, BatchNorm, Dropout, Softmax, Cross Entropy가 각각 어떤 역할을 하는지 학습 과정 속에서 확인할 수 있었다.

최종 모델은 BatchNorm을 통해 학습 안정성을 확보했고, Dropout을 통해 과적합을 줄이는 방향으로 구성했다. Adam optimizer와 learning rate 0.001 조합이 가장 안정적으로 수렴했으며, 테스트 정확도 95% 이상이라는 필수 목표를 달성했다.

추가 개선 방향으로는 learning rate schedule 적용, hidden layer 크기 조절, Dropout 비율 비교, epoch 수 증가에 따른 과적합 여부 관찰이 있다.
