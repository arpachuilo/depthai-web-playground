import numpy as np
import cv2
from convolve import *
import camera

tolerance = 100

previous_frame = None


def Zto255(frame):
    frame_downscaled = frame[::4]
    frame_downscaled_nz = frame_downscaled[frame_downscaled != 0]
    min_frame = (
        np.percentile(frame_downscaled_nz, 1) if frame_downscaled_nz.size != 0 else 0
    )
    max_frame = np.percentile(frame_downscaled, 95)
    return np.interp(frame, (min_frame, max_frame), (0, 255)).astype(np.uint8)


# using single previous frame
# TODO: average depth infill?
# TODO: improve this somehow
def depth_diff_generate_frame(rgb, depth, disparity):
    global previous_frame

    # gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    # gray = Zto255(depth)
    # gray = cv2.applyColorMap(Zto255(depth), camera.colormap)
    # gray = depth

    frame = np.copy(depth)
    if previous_frame is not None:
        depth_indices = np.where(depth == 0)
        depth[depth == 0] = previous_frame[depth_indices]

        frame = np.abs(previous_frame - frame)
        # frame = convolveWithFn(frame)
        frame[frame < tolerance] = 0
        frame[frame > tolerance] = 255

    previous_frame = np.copy(depth)
    frame = cv2.applyColorMap(Zto255(np.copy(frame)), camera.colormap)
    # frame = Zto255(frame)
    return frame


# rgb frame difference
def rgb_diff_generate_frame(rgb, depth, disparity):
    global previous_frame

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    frame = gray
    if previous_frame is not None:
        frame = previous_frame - frame
        # frame = convolveWithFn(frame)
        frame[frame > tolerance] = 0

    previous_frame = np.copy(gray)
    return frame


def generate_frame(rgb, depth, disparity):
    return rgb_diff_generate_frame(rgb, depth, disparity)
    # return depth_diff_generate_frame(rgb, depth, disparity)
