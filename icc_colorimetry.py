#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compute colorimetry from ICC profile with calibration data."""

import argparse
import sys
import struct
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from matplotlib import pyplot as plt


class ColorTransformer(ABC):
    """Color transformation class."""

    @abstractmethod
    def transform(self, c: np.ndarray) -> np.ndarray:
        """Transform color."""
        raise NotImplementedError()


class GammaScaleTransform(ColorTransformer):
    """Transform with gamma and scale/"""

    def __init__(self, gain: np.ndarray, gamma: np.ndarray,
                 offset: np.ndarray) -> None:
        super().__init__()
        self.gain = gain
        self.gamma = gamma
        self.offset = offset

    def transform(self, c: np.ndarray) -> np.ndarray:
        """Transform color."""
        return (self.gain * c + self.offset) ** self.gamma


class ColorimetryLayer(layers.Layer):
    """Colorimetry Keras Layer"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.colorimetry = self.add_weight(shape=[8], initializer="zeros")
        self.colorimetry.assign(REC709)
        self.gain = self.add_weight(shape=(), initializer="ones")

    def call(self, inputs):
        rgb_to_xyz = compute_xyz_matrix(self.colorimetry)
        xyz_to_rgb = tf.linalg.inv(rgb_to_xyz)
        return tf.linalg.matvec(xyz_to_rgb, inputs) * self.gain


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=("Compute colorimetry from ICC profile " +
                     "with calibration data"))
    parser.add_argument("input_file",
                        help="Input profile",
                        type=str)
    return parser.parse_args()


def check_icc_signature(prof_data: bytes) -> bool:
    """Check profile signature."""
    prof_size, \
        prof_ver, \
        prof_dev, \
        prof_col, \
        prof_sig = struct.unpack_from(">I4xI4s4s16x4s", prof_data, 0)
    # Basic ICC header checks
    if prof_size != len(prof_data):
        return False
    if prof_size % 4 != 0:
        return False
    if prof_sig != b"acsp":
        return False
    if prof_ver > 0x04400000:
        return False
    # Only for monitor ICC
    if prof_dev != b"mntr":
        return False
    if prof_col != b"RGB ":
        return False
    return True


def parse_icc_tags(prof_data: bytes) -> dict:
    """Parse ICC tags."""
    tag_count = struct.unpack_from(">I", prof_data, 128)[0]
    result = {}
    for i in range(tag_count):
        tag_sig, \
            tag_offset, \
            tag_size = struct.unpack_from(">4sII", prof_data, 132 + 12 * i)
        result[tag_sig] = prof_data[tag_offset:tag_offset+tag_size]
    return result


def parse_vcgt(vcgt: bytes) -> ColorTransformer:
    """Parse VCGT section."""
    assert vcgt[:12] == b"vcgt\0\0\0\0\0\0\0\1"
    gains = []
    offsets = []
    gammas = []
    for i in range(1, 4):
        gamma, offset, gain = struct.unpack_from(">III", vcgt, 12 * i)
        gains.append(gain)
        offsets.append(offset)
        gammas.append(gamma)
    return GammaScaleTransform(
        gain=np.array(gains) / 65536,
        gamma=np.array(gammas) / 65535,
        offset=np.array(offsets) / 65535
    )


def compute_primary(xy: tf.Tensor) -> tf.Tensor:
    """Compute primary values."""
    return tf.convert_to_tensor([xy[0] / xy[1],
                                 1.0,
                                 (1.0 - xy[0] - xy[1]) / xy[1]])


def compute_xyz_matrix(colorimetry: tf.Tensor) -> tf.Tensor:
    """Compute RGB->XYZ matrix"""
    red = compute_primary(colorimetry[0:2])
    green = compute_primary(colorimetry[2:4])
    blue = compute_primary(colorimetry[4:6])
    white = compute_primary(colorimetry[6:8])
    scale = tf.linalg.matvec(
        tf.expand_dims(white, 0),
        tf.linalg.inv(tf.stack([red, green, blue], axis=1)))
    return tf.stack([red * scale[0], green * scale[1], blue * scale[2]],
                    axis=1)


REC709 = tf.constant([0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290])
REC709_MAT = compute_xyz_matrix(REC709)


def compute_colorimetry(transformer: ColorTransformer) -> list:
    """Compute colorimetry from transformer."""

    def generator():
        while True:
            input_val = np.random.uniform(0.0, 1.0, size=(1, 3))
            input_val = input_val.astype(np.float32)
            output_val = transformer.transform(input_val)
            yield tf.linalg.matvec(REC709_MAT, input_val), output_val

    def val_generator():
        input_vals = np.arange(0.0, 1.0, step=0.01)
        input_vals = np.tile(input_vals, 3).T.astype(np.float32)
        input_vals = input_vals.reshape(-1, 1, 3)
        output_vals = transformer.transform(input_vals)
        return tf.linalg.matvec(REC709_MAT, input_vals), output_vals

    dataset = tf.data.Dataset.from_generator(
        generator, output_signature=(
            tf.TensorSpec(shape=(1, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(1, 3), dtype=tf.float32),
        ))
    dataset = dataset.batch(16)

    val_dataset = tf.data.Dataset.from_tensors(val_generator())
    val_dataset.unbatch().batch(10)

    model = keras.models.Sequential([ColorimetryLayer(name="colorimetry")])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        dataset, epochs=20, steps_per_epoch=100,
        validation_data=val_dataset,
        callbacks=[keras.callbacks.EarlyStopping(
            restore_best_weights=True,
            patience=5,
        )]
    )
    colorimetry_layer = model.get_layer("colorimetry")
    return (colorimetry_layer.weights[0].numpy().tolist() +
            [colorimetry_layer.weights[1].numpy().tolist()])


def main(input_file: str) -> int:
    """Main function."""

    with open(input_file, "rb") as f:
        prof_data = f.read()
    if not check_icc_signature(prof_data):
        print("Invalid ICC profile")
        return 1
    tags = parse_icc_tags(prof_data)
    vcgt = tags.get(b"vcgt")
    if vcgt is None:
        print("Profile does not have VCGT")
        return 1
    transformer = parse_vcgt(vcgt)
    colorimetry = compute_colorimetry(transformer)
    print(colorimetry)
    rgb_to_xyz = compute_xyz_matrix(colorimetry[:8])
    xyz_to_rgb = tf.linalg.inv(rgb_to_xyz)
    transform_mat = (REC709_MAT @ xyz_to_rgb).numpy() * colorimetry[-1]
    x = np.arange(0.0, 1.0, 0.01)
    col = np.tile(x, 3).reshape(3, -1).T
    out_gt = transformer.transform(col)
    out_test = (transform_mat @ col.T).T
    plt.figure()
    plt.plot(col, out_gt[:, 0], "r")
    plt.plot(col, out_gt[:, 1], "g")
    plt.plot(col, out_gt[:, 2], "b")
    plt.savefig("out_icc.png")
    plt.figure()
    plt.plot(col, out_test[:, 0], "r")
    plt.plot(col, out_test[:, 1], "g")
    plt.plot(col, out_test[:, 2], "b")
    plt.savefig("out_colorimetry.png")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(**vars(parse_args())))
    except KeyboardInterrupt:
        sys.exit(1)
