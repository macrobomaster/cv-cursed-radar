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
      cls = x.argmax(1)

      # cam
      bs, c, h, w = x5.shape
      cam = self.classifier.weight[cls].unsqueeze(1) @ x5.reshape(bs, c, h*w)
      cam = cam.reshape(bs, h, w)
      cam_min, cam_max = cam.min((1, 2), keepdim=True), cam.max((1, 2), keepdim=True)
      cam = (cam - cam_min) / (cam_max - cam_min)

      return cls, (cam * 255).cast(dtypes.uint8)

