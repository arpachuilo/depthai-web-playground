import collections

import cv2

import camera

# queue frames from camera
frameQueueMax = 4
frameQueue = collections.deque(maxlen=frameQueueMax)

# encode frame to serve over api
def encode(frame):
    (flag, encodedImage) = cv2.imencode(".jpg", frame)
    # ensure the frame was successfully encoded
    if not flag:
        return None
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
    )


# encode and queue frame
def push_frame(frame):
    if len(frameQueue) == frameQueueMax - 1:
        frameQueue.pop()
    frameQueue.appendleft(encode(frame))


# fetch frame from queue
def fetch_frame():
    def frame_generator():
        while not camera.dead:
            yield frameQueue[0]

    return frame_generator()
