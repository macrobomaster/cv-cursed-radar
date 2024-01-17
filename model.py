from tinygrad import Tensor, dtypes
from tinygrad.nn import Linear

from shufflenet import ShuffleNetV2

class Model:
  def __init__(self):
    self.backbone = ShuffleNetV2()

    self.classifier = Linear(1024, 2, bias=False)

  def __call__(self, img: Tensor):
    # image normalization
    if Tensor.training: img = img.float()
    else: img = img.cast(dtypes.default_float)
    img = img.permute(0, 3, 1, 2) / 255

    x = self.backbone(img)
    x, x5 = x
    x = self.classifier(x)

    if Tensor.training: return x
    else:
      cls = x.argmax()

      # cam
      _, c, h, w = x5.shape
      cam = self.classifier.weight[cls] @ x5.reshape(c, h*w)
      cam = cam.reshape(h, w)
      cam = cam - cam.min()
      cam = cam / cam.max()

      return cls, (cam * 255).cast(dtypes.uint8)

