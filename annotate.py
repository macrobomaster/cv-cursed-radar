import os
import glob
from pathlib import Path

import cv2

BASE_PATH = Path(os.environ.get("BASE_PATH", "./"))
frame_files = glob.glob(str(BASE_PATH / "preprocessed/*.png"))
print(f"there are {len(frame_files)} frames")

oframe, frame = None, None

i = 0
flag = False
back = False
while not flag and i < len(frame_files):
  # check if the file has already been annotated
  if Path(frame_files[i]).with_suffix(".txt").exists() and not back:
    i += 1
    continue

  back = False
  print(f"annotating frame {i} of {len(frame_files)}")
  frame_file = frame_files[i]
  oframe = frame = cv2.imread(frame_file)

  while True:
    cv2.imshow("preview", frame)

    key = cv2.waitKey(1)
    if key == ord("z"):
      flag = True
      break
    elif key == ord("0"):
      with open(Path(frame_file).with_suffix(".txt"), "w") as f:
        f.write("0")
      break
    # red
    elif key == ord("1"):
      with open(Path(frame_file).with_suffix(".txt"), "w") as f:
        f.write("1")
      break
    elif key == ord("2"):
      with open(Path(frame_file).with_suffix(".txt"), "w") as f:
        f.write("2")
      break
    elif key == ord("3"):
      with open(Path(frame_file).with_suffix(".txt"), "w") as f:
        f.write("3")
      break
    elif key == ord("4"):
      with open(Path(frame_file).with_suffix(".txt"), "w") as f:
        f.write("4")
      break
    elif key == ord("5"):
      with open(Path(frame_file).with_suffix(".txt"), "w") as f:
        f.write("5")
      break
    elif key == ord("6"):
      with open(Path(frame_file).with_suffix(".txt"), "w") as f:
        f.write("6")
      break
    # blue
    elif key == ord("q"):
      with open(Path(frame_file).with_suffix(".txt"), "w") as f:
        f.write("7")
      break
    elif key == ord("w"):
      with open(Path(frame_file).with_suffix(".txt"), "w") as f:
        f.write("8")
      break
    elif key == ord("e"):
      with open(Path(frame_file).with_suffix(".txt"), "w") as f:
        f.write("9")
      break
    elif key == ord("r"):
      with open(Path(frame_file).with_suffix(".txt"), "w") as f:
        f.write("10")
      break
    elif key == ord("t"):
      with open(Path(frame_file).with_suffix(".txt"), "w") as f:
        f.write("11")
      break
    elif key == ord("y"):
      with open(Path(frame_file).with_suffix(".txt"), "w") as f:
        f.write("12")
      break
    # skip and back
    elif key == ord("s"):
      break
    elif key == ord("d"):
      i -= 2
      back = True
      break
  i += 1
