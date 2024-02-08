from pathlib import Path
from multiprocessing import Queue
import time
import os

import cv2
import numpy as np
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.helpers import Context
from tinygrad import Device, Tensor, dtypes
from tinygrad.jit import TinyJit
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
  Device[Device.DEFAULT].linearizer_opts = LinearizerOptions("HIP", supports_float4=False)

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
  cap2 = cv2.VideoCapture("2744.mp4")
  # cap = cv2.VideoCapture(1)

  def get_patches(frame):
    def sliding_window(frame, height, width, step_size=224, window_size=224):
      y_flag = False
      for y in range(0, height, step_size):
        x_flag = False
        for x in range(0, width, step_size):
          x0 = x
          if width <= x + window_size: x0, x_flag = width - window_size, True
          y0 = y
          if height <= y + window_size: y0, y_flag = height - window_size, True
          yield frame[y0:y0+window_size, x0:x0+window_size], (x0, y0), (height, width)
          if x_flag: break
        if y_flag: break
    patches, xys, sizes = [], [], []
    for scale in [224 + 64 * i for i in range(1)]:
      scaled = image_resize(frame, width=scale)
      for patch, xy, size in sliding_window(scaled, scaled.shape[0], scaled.shape[1]):
        if patch.shape[0] != 224 or patch.shape[1] != 224: continue
        patches.append(patch)
        xys.append(xy)
        sizes.append(size)
    return patches, xys, sizes

  st = time.perf_counter()
  with Context(BEAM=0):
    while True:
      GlobalCounters.reset()
      # frame = cap_queue.get()

      ret, frame = cap.read()
      if not ret: break
      # ret, frame2 = cap2.read()
      # if not ret: break
      # convert to rgb
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      # frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
      # frame = np.concatenate((frame, frame2), axis=0)
      frame = frame[-224-50:-50, 300:224+300]
      patches, xys, sizes = get_patches(frame)

      cls, cam = pred(Tensor(patches))
      cls = cls.numpy()
      cam = cam.numpy()

      dt = time.perf_counter() - st
      st = time.perf_counter()

      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
      # build heatmap
      full_cam_imgs = []
      for i, (patch, (x, y), size) in enumerate(zip(patches, xys, sizes)):
        if cls[i] != 0:
          cam_img = cv2.resize(cam[i].astype(np.uint8), (224, 224), 0, 0, cv2.INTER_CUBIC)
          full_cam_img = np.zeros(size)
          full_cam_img[y:y+224, x:x+224] = cam_img
          full_cam_imgs.append(cv2.resize(full_cam_img, (frame.shape[1], frame.shape[0])))
      if len(full_cam_imgs) == 0: continue
      full_cam_img = np.array(full_cam_imgs).mean(axis=0).astype(np.uint8)
      full_cam_img = cv2.GaussianBlur(full_cam_img, (11, 11), 5.0, 0)
      heatmap = cv2.applyColorMap(full_cam_img, cv2.COLORMAP_JET)
      frame = np.uint8(0.6 * frame + 0.4 * heatmap)

      # contour based
      threshold = cv2.threshold(full_cam_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
      contours = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
      hulls = list(map(cv2.convexHull, contours))
      if len(hulls) > 0:
        hulls = sorted(hulls, key=cv2.contourArea)
        for i in range(2):
          largest = hulls[-i]
          x, y = largest.mean(axis=0)[0]
          x, y = smoother_x.update(x, dt), smoother_y.update(y, dt)
          cv2.drawContours(frame, [largest], -1, (0, 255, 0), 2)
          cv2.circle(frame, (int(x), int(y)), 6, (255, 0, 255), 3)

      cv2.putText(frame, f"{1/dt:.2f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 250, 55), 1)
      cv2.putText(frame, f"{cls.sum():.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 250, 55), 1)

      cv2.imshow("preview", frame)

      key = cv2.waitKey(1)
      if key == ord("q"): break
      time.sleep(0.02)
