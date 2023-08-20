from threading import Thread
import signal
from contextlib import asynccontextmanager

import numpy as np

from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi import FastAPI
from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

import convolve
import camera
import camerafeed

# setup depthai camera on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup camera
    thread = Thread(target=camera.run, args=(camerafeed.push_frame,))

    # kill loop on sigint
    def handler(signum, frame):
        camera.dead = True

    signal.signal(signal.SIGINT, handler)
    thread.start()

    yield

    # kill camera
    camera.dead = True
    thread.join()


# setup api
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="./static")

# serve static
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})


# TODO: get working with multi client env?
# set current camera feed and stream it to client
@app.get("/video/{t}")
async def get_video(t):
    camera.mode = t  # set mode to requested type
    return StreamingResponse(
        camerafeed.fetch_frame(), media_type="multipart/x-mixed-replace;boundary=frame"
    )


# get current camera feed
@app.get("/mode", response_class=PlainTextResponse)
async def get_mode():
    return camera.mode


# get cv2 colormap in use
@app.get("/colormap")
async def get_colormap():
    return camera.colormap


# set cv2 colormap, used for everything but RGB
@app.post("/colormap/{color}")
async def set_colormap(color):
    try:
        colormap = int(color, 10)
        if colormap >= 0 and colormap <= 21:
            camera.colormap = colormap

        return True
    except:
        return False


# get convolve fn for mixing kernels
@app.get("/convolve", response_class=PlainTextResponse)
async def get_convolve_fn():
    return convolve.convolveFn


# set convolve fn for mixing kernels
@app.post("/convolve/{fn}")
async def set_convolve_fn(fn):
    convolve.convolveFn = fn
    return True


# set kernel for variable
@app.get("/kernel/{t}")
async def get_kernel(t):
    if t in convolve.kernels:
        return convolve.kernels[t].tolist()
    else:
        return convolve.identity


# set kernel for variable
@app.post("/kernel/{t}")
async def set_kernel(t, request: Request):
    values = np.array(await request.json())

    # check shape
    if values.shape != (3, 3) or np.isnan(values).any():
        return False

    convolve.kernels[t] = np.array(values)

    # ensure multiplier exist
    if t not in convolve.multipliers:
        convolve.multipliers[t] = 1.0

    return True


# get multiplier for variable
@app.get("/multiplier/{t}")
async def get_multiplier(t):
    if t in convolve.multipliers:
        return convolve.multipliers[t]
    else:
        return 1.0


# set multiplier for variable
@app.post("/multiplier/{t}/{x}")
async def set_multiplier(t, x):
    try:
        mult = float(x)
        convolve.multipliers[t] = mult

        # ensure kernel exist
        if t not in convolve.kernels:
            convolve.kernels[t] = convolve.identity

        return True
    except:
        return False


# start api server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
