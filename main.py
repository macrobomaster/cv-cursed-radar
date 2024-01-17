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

BASE_PATH = Path(os.environ.get("BASE_PATH", "./"))

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

  @TinyJit
  def pred(img):
    cls, cam = model(img)
    return cls.realize(), cam.realize()

  cap = cv2.VideoCapture("2743.mp4")
  # cap = cv2.VideoCapture(1)

  st = time.perf_counter()
  with Context(BEAM=4):
    while True:
      GlobalCounters.reset()
      # frame = cap_queue.get()

      ret, frame = cap.read()
      if not ret: break
      # convert to rgb
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frame = frame[-224-50:-50, 300:224+300]

      img = Tensor(frame).reshape(1, 224, 224, 3)
      cls, cam = pred(img)
      cls = cls.item()

      dt = time.perf_counter() - st
      st = time.perf_counter()

      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
      if cls != 0:
        cam_img = cv2.resize(cam.numpy(), (224, 224))
        heatmap = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
        frame = np.uint8(0.6 * frame + 0.4 * heatmap)

        x = np.argmax(cam_img) % cam_img.shape[1]
        y = np.argmax(cam_img) // cam_img.shape[1]
        cv2.circle(frame, (x, y), 10, (0, 255, 0), 2)

      cv2.putText(frame, f"{1/dt:.2f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 250, 55), 1)
      cv2.putText(frame, f"{cls:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 250, 55), 1)

      cv2.imshow("preview", frame)

      key = cv2.waitKey(1)
      if key == ord("q"): break

      time.sleep(0.05)
