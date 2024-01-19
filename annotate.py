import os
import glob
from pathlib import Path

import cv2

BASE_PATH = Path(os.environ.get("BASE_PATH", "./"))
frame_files = glob.glob(str(BASE_PATH / "preprocessed/*.png"))
print(f"there are {len(frame_files)} frames")

oframe, frame = None, None

i = 1498
flag = False
while not flag and i < len(frame_files):
  print(f"annotating frame {i} of {len(frame_files)}")
  frame_file = frame_files[i]
  oframe = frame = cv2.imread(frame_file)

  while True:
    cv2.imshow("preview", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
      flag = True
      break
    elif key == ord("n"):
      with open(Path(frame_file).with_suffix(".txt"), "w") as f:
        f.write("0")
      break
    elif key == ord("c"):
      with open(Path(frame_file).with_suffix(".txt"), "w") as f:
        f.write("1")
      break
    elif key == ord("d"):
      i -= 2
      break
  i += 1
