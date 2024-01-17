import glob

import albumentations as A
import cv2
from tqdm import tqdm
import numpy as np

from main import BASE_PATH

IMG_SIZE = 224
DUPE_COUNT = 10

PIPELINE = A.Compose([
  A.Perspective(p=0.25),
  A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.25),
  A.RandomCrop(IMG_SIZE, IMG_SIZE),
  A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
  A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=20, p=0.5),
  A.CLAHE(p=0.1),
  A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
  A.RandomGamma(gamma_limit=(80, 120), p=0.1),
  A.FancyPCA(alpha=0.1, p=0.5),
])

def add_to_batch(x_b, img):
  x_b.append(np.reshape(img, (IMG_SIZE, IMG_SIZE, 3)))

train_files = glob.glob(str(BASE_PATH / ".." / "cv-e2e-playground" / "annotated/*.png"))
print(f"there are {len(train_files)} frames")
def get_train_data():
  global non_detected_count
  for frame_file in train_files:
    # load x
    img = cv2.imread(frame_file)
    # convert to rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # add initial imgs
    x_b = []
    add_to_batch(x_b, img[:IMG_SIZE, :IMG_SIZE])
    add_to_batch(x_b, img[:IMG_SIZE, -IMG_SIZE:])
    add_to_batch(x_b, img[-IMG_SIZE:, :IMG_SIZE])
    add_to_batch(x_b, img[-IMG_SIZE:, -IMG_SIZE:])
    add_to_batch(x_b, img[img.shape[0] // 2 - (IMG_SIZE // 2) : img.shape[0] // 2 + (IMG_SIZE // 2), img.shape[1] // 2 - (IMG_SIZE // 2) : img.shape[1] // 2 + (IMG_SIZE // 2)])

    # augment
    for _ in range(DUPE_COUNT):
      transformed = PIPELINE(image=img)
      add_to_batch(x_b, transformed["image"])

    yield x_b

for i, x in enumerate(tqdm(get_train_data(), total=len(train_files))):
  for j, sx in enumerate(x):
    cv2.imwrite(str(BASE_PATH / f"preprocessed/{i:06}_{j:03}.png"), cv2.cvtColor(sx, cv2.COLOR_RGB2BGR))
