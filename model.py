from tinygrad import Tensor, dtypes
from tinygrad.nn import Linear

from backbone import Backbone

def upsample(x: Tensor, scale: int):
  bs, c, py, px = x.shape
  return x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, scale, px, scale).reshape(bs, c, py * scale, px * scale)

class Model:
  def __init__(self):
    self.backbone = Backbone()

    self.classifier = Linear(2048, 13, bias=False)

  def __call__(self, img: Tensor) -> Tensor:
    # image normalization
    if Tensor.training: img = img.float()
    else: img = img.cast(dtypes.default_float)
    img = img.permute(0, 3, 1, 2) / 255

    x = self.backbone(img)
    x = self.classifier(x)
    return x

if __name__ == "__main__":
  from tinygrad.nn.state import get_parameters
  from tinygrad import GlobalCounters

  model = Model()
  print(f"model parameters: {sum(p.numel() for p in get_parameters(model))}")

  model(Tensor.zeros(1, 256, 256, 3)).realize()
  GlobalCounters.reset()
  model(Tensor.zeros(1, 256, 256, 3)).realize()
