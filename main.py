from pathlib import Path
from multiprocessing import Queue
import time
import os

import cv2
import numpy as np
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.helpers import Context
from tinygrad import Device, Tensor, dtypes
from tinygrad.features.jit import TinyJit
from tinygrad.nn.state import safe_load, load_state_dict, get_state_dict
from tinygrad import GlobalCounters

from model import Model
from smoother import Smoother

BASE_PATH = Path(os.environ.get("BASE_PATH", "./"))

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
  # initialize the dimensions of the image to be resized and
  # grab the image size
  dim = None
  (h, w) = image.shape[:2]

  # if both the width and height are None, then return the
  # original image
  if width is None and height is None:
    return image

  # check to see if the width is None
  if width is None:
    # calculate the ratio of the height and construct the
    # dimensions
    r = height / float(h)
    dim = (int(w * r), height)

  # otherwise, the height is None
  else:
    # calculate the ratio of the width and construct the
    # dimensions
    r = width / float(w)
    dim = (width, int(h * r))

  # resize the image
  resized = cv2.resize(image, dim, interpolation = inter)

  # return the resized image
  return resized

if __name__ == "__main__":
  Tensor.no_grad = True
  Tensor.training = False
  dtypes.default_float = dtypes.float16
  # Device[Device.DEFAULT].linearizer_opts = LinearizerOptions("HIP", supports_float4=False)

  model = Model()
  state_dict = safe_load(str(BASE_PATH / "model.safetensors"))
  load_state_dict(model, state_dict)
  for key, param in get_state_dict(model).items():
    if "norm" in key: continue
    if "bn" in key: continue
    if "stage1.2" in key: continue
    if "stage5.2" in key: continue
    param.assign(param.half()).realize()
  smoother_x, smoother_y = Smoother(), Smoother()

  @TinyJit
  def pred(img):
    cls, cam  = model(img)
    return cls.realize(), cam.realize()

  cap = cv2.VideoCapture("2743.mp4")
  # cap = cv2.VideoCapture(1)

  st = time.perf_counter()
  with Context(BEAM=0):
    while True:
      GlobalCounters.reset()
      # frame = cap_queue.get()

      ret, frame = cap.read()
      if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
      # convert to rgb
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frame = frame[-224-50:-50, 300:224+300]

      img = Tensor([frame], requires_grad=False, dtype=dtypes.uint8)
      cls, cam = pred(img)
      cls = cls.item()
      cam = cam.numpy()[0].astype(np.uint8)

      dt = time.perf_counter() - st
      st = time.perf_counter()

      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
      if cls != 0:
        cam = cv2.GaussianBlur(cam, (11, 11), 5.0, 0)
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        frame = np.uint8(0.6 * frame + 0.4 * heatmap)

        # contour based
        threshold = cv2.threshold(cam, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        hulls = list(map(cv2.convexHull, contours))
        if len(hulls) > 0:
          hulls = sorted(hulls, key=cv2.contourArea, reverse=True)
          for i in range(1):
            largest = hulls[-i]
            x, y = largest.mean(axis=0)[0]
            # x, y = smoother_x.update(x, dt), smoother_y.update(y, dt)
            cv2.drawContours(frame, [largest], -1, (0, 255, 0), 2)
            cv2.circle(frame, (int(x), int(y)), 6, (255, 0, 255), 3)

      cv2.putText(frame, f"{1/dt:.2f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 250, 55), 1)
      cv2.putText(frame, f"{cls:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 250, 55), 1)

      cv2.imshow("preview", frame)

      key = cv2.waitKey(1)
      if key == ord("q"): break
      time.sleep(0.02)
