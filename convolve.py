import cv2
import numpy as np

# some kernels
# fmt: off
identity = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
])
sobelX = np.array([
    [+1, 0, -1], 
    [+2, 0, -2], 
    [+1, 0, -1],
])
sobelY = np.array([
    [+1, +2, +1], 
    [+0, +0, +0], 
    [-1, -2, -1],
])
# fmt: on

kernels = {}
kernels["x"] = sobelX
kernels["y"] = sobelY

multipliers = {}
multipliers["x"] = 1.0
multipliers["y"] = 1.0

# convolve 2d input
def convolve(input, kernel, multiplier):
    shape = kernel.shape + tuple(np.subtract(input.shape, kernel.shape) + 1)
    stride = np.lib.stride_tricks.as_strided
    matrices = stride(input, shape=shape, strides=input.strides * 2)
    return np.einsum("ij,ijkl->kl", kernel * multiplier, matrices)


# convolve rgb input across each channel
def convolveRGB(rgb, kernel, multiplier):
    width, height, _ = rgb.shape
    b = rgb[:, :, 0]
    b = convolve(b, kernel, multiplier).astype("float32")
    b = cv2.resize(b, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
    rgb[:, :, 0] = b

    g = rgb[:, :, 1]
    g = convolve(g, kernel, multiplier).astype("float32")
    g = cv2.resize(g, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
    rgb[:, :, 1] = g

    r = rgb[:, :, 2]
    r = convolve(r, kernel, multiplier).astype("float32")
    r = cv2.resize(r, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
    rgb[:, :, 2] = r
    return rgb


convolveFn = "sqrt(x2+y2)"

# convolve based on given fn
def convolveWithFn(frame):
    cf = convolveRGB if len(frame.shape) == 3 else convolve
    match convolveFn:
        case "sqrt(x2+y2)":
            frameX = cf(np.copy(frame), kernels["x"], multipliers["x"])
            frameY = cf(np.copy(frame), kernels["y"], multipliers["y"])
            frame = np.sqrt(frameX**2 + frameY**2).astype("uint8")
        case "x":
            frame = cf(frame, kernels["x"], multipliers["x"])
        case "y":
            frame = cf(frame, kernels["y"], multipliers["y"])

    return frame
