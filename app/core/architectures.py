"""모델 구조 (제안서 §5.2).

PoC는 Keras 예제(Image classification from scratch)의 make_model 하나만 쓴다.
xception_mini는 그 구조를 32×32 입력에 맞게 줄인 것이다. width는 필터 수 배율이며 별도 구조가 아니다.

BatchNormalization momentum: 예제는 Keras 기본값 0.99를 쓰지만 여기서는 0.9를 쓴다.
0.99는 이동 평균이 수렴하는 데 수백~수천 step이 필요해서, 빠른 데모(3 epoch, 약 200 step)처럼 짧게
학습하면 추론 모드(training=False)의 정규화 통계가 실제 값과 크게 달라 예측이 한 클래스로 몰린다
(Phase 0에서 GPU·CPU 모두 확인). 0.9는 수십 step 안에 수렴하고 긴 학습에서도 문제가 없다.
"""

from __future__ import annotations

BN_MOMENTUM = 0.9


def xception_mini(input_shape=(32, 32, 3), num_classes=100, width=1.0, dropout=0.25, bn_momentum=BN_MOMENTUM):
    import keras
    from keras import layers

    def f(n: int) -> int:
        return max(16, int(n * width))

    inputs = keras.Input(shape=input_shape)

    # Entry block (원본: Conv2D(128, 3, strides=2) — 180px 입력 기준)
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(f(64), 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization(momentum=bn_momentum)(x)
    x = layers.Activation("relu")(x)
    previous_block_activation = x

    for size in [f(128), f(256), f(512)]:          # 원본: [256, 512, 728]
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization(momentum=bn_momentum)(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization(momentum=bn_momentum)(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
        x = layers.add([x, residual])
        previous_block_activation = x

    x = layers.SeparableConv2D(f(768), 3, padding="same")(x)   # 원본: 1024
    x = layers.BatchNormalization(momentum=bn_momentum)(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation=None)(x)    # logits
    return keras.Model(inputs, outputs, name="xception_mini")


ARCHITECTURES = {"xception_mini": xception_mini}


def build_model(arch: str, input_shape, num_classes: int, width: float = 1.0, dropout: float = 0.25):
    if arch not in ARCHITECTURES:
        from app.core.i18n import AppError

        raise AppError("err.unknown_arch", arch=arch)
    return ARCHITECTURES[arch](tuple(input_shape), num_classes, width=width, dropout=dropout)
