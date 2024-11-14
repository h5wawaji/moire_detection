import tensorflow as tf
import cv2
import numpy as np
import random
import pywt
import os

def build_model(h, w, c=1):
    inputs = {
        "LL": tf.keras.layers.Input(shape=(h, w, c)),
        "LH": tf.keras.layers.Input(shape=(h, w, c)),
        "HL": tf.keras.layers.Input(shape=(h, w, c))
    }

    # conv block 1
    x = tf.keras.layers.Conv2D(32, (7, 7), strides=(1, 1), padding='same')(inputs["LL"])
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
    ll = x

    x = tf.keras.layers.Conv2D(32, (7, 7), strides=(1, 1), padding='same')(inputs["LH"])
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
    lh = x

    x = tf.keras.layers.Conv2D(32, (7, 7), strides=(1, 1), padding='same')(inputs["HL"])
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
    hl = x

    # get maximum
    max_lh_hl = tf.keras.layers.Maximum()([lh, hl])

    # multiply
    mul_ll_max = tf.keras.layers.Multiply()([ll, max_lh_hl])
    mul_ll_max = tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same')(mul_ll_max)
    mul_ll_max = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(4, 4), padding='same')(mul_ll_max)

    mul_ll_max = tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same')(mul_ll_max)
    mul_ll_max = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(mul_ll_max)

    mul_ll_max = tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same')(mul_ll_max)
    mul_ll_max = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(mul_ll_max)

    flatten = tf.keras.layers.Flatten()(mul_ll_max)
    flatten = tf.keras.layers.Dropout(0.25)(flatten)
    flatten_32 = tf.keras.layers.Dense(32, activation='relu')(flatten)
    output = tf.keras.layers.Dense(2, activation='softmax')(flatten_32)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['acc'])
    return model

# 其他函数如 scale, DWT, preproces_data 等保持不变

# 使用 tf.data 或 tf.keras.utils.Sequence 来处理数据
# 这里需要根据您的具体数据和需求来实现数据加载和预处理

# 模型训练
model = build_model(482, 642, 1)
model.fit(your_data, your_labels, epochs=1000)  # 替换 your_data 和 your_labels 为您的数据和标签