import importlib
import os
import logging

import numpy as np
import depthai as dai
import cv2
import uvicorn

from convolve import *
import hot

logger = logging.getLogger(uvicorn.__name__)
logger.info("test")
pipeline = dai.Pipeline()

# setup rgb camera
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setPreviewSize(640, 400)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# setup mono cameras for stereo
monoLeft = pipeline.create(dai.node.MonoCamera)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight = pipeline.create(dai.node.MonoCamera)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

# create stereo
stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

# setup depth
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(False)
stereo.setSubpixel(True)
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

config = stereo.initialConfig.get()
config.postProcessing.speckleFilter.enable = False
config.postProcessing.speckleFilter.speckleRange = 50
config.postProcessing.temporalFilter.enable = True
# config.postProcessing.temporalFilter.persistencyMode = (
#     dai.RawStereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.PERSISTENCY_OFF
# )
config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 2
config.postProcessing.spatialFilter.numIterations = 1
config.postProcessing.thresholdFilter.minRange = 400
config.postProcessing.thresholdFilter.maxRange = 10000
config.postProcessing.decimationFilter.decimationFactor = 1
stereo.initialConfig.set(config)

# link mono left/right into stereo
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# create output streams
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)

xoutDisparity = pipeline.create(dai.node.XLinkOut)
xoutDisparity.setStreamName("disparity")
stereo.disparity.link(xoutDisparity.input)

# start up pipeline
device = dai.Device()
device.startPipeline(pipeline)

rgbQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
disparityQueue = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

colormap = 20
mode = "depth"

# cached timestamp for hot.py changes
cached_hot_timestamp = None
import_error = False

def poll(cb):
    inRGB = rgbQueue.get()
    inDepth = depthQueue.get()
    inDisparity = disparityQueue.get()

    # return feed based on mode
    match mode:
        case "rgb":
            rgb = inRGB.getCvFrame()
            rgb = convolveWithFn(rgb)
            cb(rgb)

        case "gray":
            rgb = inRGB.getCvFrame()
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            gray = convolveWithFn(gray).astype("uint8")
            gray = cv2.applyColorMap(gray, colormap)
            cb(gray)

        case "depth":
            depth = inDepth.getFrame()
            depth = convolveWithFn(depth)
            depth_downscaled = depth[::4]
            depth_downscaled_nz = depth_downscaled[depth_downscaled != 0]
            min_depth = (
                np.percentile(depth_downscaled_nz, 1) if depth_downscaled_nz.size != 0 else 0
            )
            max_depth = np.percentile(depth_downscaled, 99)
            depth = np.interp(depth, (min_depth, max_depth), (0, 255)).astype(np.uint8)
            depth = cv2.applyColorMap(depth, colormap)
            cb(depth)

        case "disparity":
            disparity = inDisparity.getFrame()
            disparity = convolveWithFn(disparity)
            disparity = (disparity * (255 / stereo.initialConfig.getMaxDisparity())).astype(
                np.uint8
            )
            disparity = cv2.applyColorMap(disparity, colormap)
            cb(disparity)

        case "custom":
            rgb = inRGB.getCvFrame()
            depth = inDepth.getFrame()
            disparity = inDisparity.getFrame()
            
            ts = None
            try:
                ts = os.stat("./hot.py").st_mtime
            except Exception as error:
                logger.error(f'Error getting timestamp for hot.py {error}')

            try:
                # check for changes to hot.py
                global cached_hot_timestamp
                if ts != cached_hot_timestamp:
                    logger.info("Reloading hot.py")
                    importlib.reload(hot)

                if hasattr(hot, "generate_frame"):
                    cb(hot.generate_frame(rgb, depth, disparity))
            except Exception as error:
                logger.error(f'Error reloading hot.py {error}')
                pass
            finally:
                cached_hot_timestamp = ts

# run this to death
dead = False
def run(cb):
    while not dead:
        poll(cb)
    device.close()
