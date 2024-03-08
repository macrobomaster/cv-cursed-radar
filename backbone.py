from typing import Tuple
from tinygrad import Tensor, nn, dtypes
from tinygrad.nn import Conv2d

class BatchNorm2d:
  def __init__(self, dim:int, eps=1e-6): self.n = nn.BatchNorm2d(dim, eps=eps)
  def __call__(self, x:Tensor) -> Tensor: return self.n(x.float()).cast(dtypes.default_float)

# def upsample(x:Tensor, scale:int):
#   bs, c, py, px = x.shape
#   return x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, scale, px, scale).reshape(bs, c, py * scale, px * scale)
#
# class DFCAttention:
#   def __init__(self, dim, *, attention_size=7):
#     self.cv = Conv2d(dim, dim, kernel_size=1, bias=False)
#     self.norm = BatchNorm2d(dim)
#
#     # horizontal fc
#     self.hcv = Conv2d(dim, dim, kernel_size=(1, attention_size), padding=(0, attention_size//2), groups=dim, bias=False)
#     self.hnorm = BatchNorm2d(dim)
#     # vertical fc
#     self.vcv = Conv2d(dim, dim, kernel_size=(attention_size, 1), padding=(attention_size//2, 0), groups=dim, bias=False)
#     self.vnorm = BatchNorm2d(dim)
#
#   def __call__(self, x:Tensor) -> Tensor:
#     assert x.shape[-1] % 2 == 0 and x.shape[-2] % 2 == 0, f"attention input must be divisible by 2, got {x.shape}"
#     # downsample
#     xx = x.avg_pool2d(2)
#     # attention map
#     xx = self.norm(self.cv(xx))
#     xx = self.hnorm(self.hcv(xx))
#     xx = self.vnorm(self.vcv(xx))
#     xx = xx.sigmoid()
#     # upsample
#     return upsample(xx, 2)
#
def channel_shuffle(x:Tensor) -> Tuple[Tensor, Tensor]:
  b, c, h, w = x.shape
  assert c % 4 == 0
  x = x.reshape(b * c // 2, 2, h * w).permute(1, 0, 2)
  x = x.reshape(2, -1, c // 2, h, w)
  return x[0], x[1]
#
# class Block:
#   def __init__(self, cin:int, cout:int, kernel_size:int=3, shuffle:bool=True, stride:int=1):
#     assert stride in [1, 2]
#     self.shuffle = shuffle
#     bin = cin // 2 if shuffle else cin
#     pad, bout, bmid = kernel_size // 2, cout - bin, cout // 2
#
#     self.attention = DFCAttention(cin)
#
#     # pw
#     self.cv1 = Conv2d(bin, bmid, 1, 1, 0, bias=False)
#     self.bn1 = BatchNorm2d(bmid)
#     # dw
#     self.cv2 = Conv2d(bmid, bmid, kernel_size, stride, pad, groups=bmid, bias=False)
#     self.bn2 = BatchNorm2d(bmid)
#     # pw
#     self.cv3 = Conv2d(bmid, bout, 1, 1, 0, bias=False)
#     self.bn3 = BatchNorm2d(bout)
#
#     if not self.shuffle:
#       # dw
#       self.cv4 = Conv2d(bin, bin, kernel_size, stride, pad, groups=bin, bias=False)
#       self.bn4 = BatchNorm2d(bin)
#       # pw
#       self.cv5 = Conv2d(bin, bin, 1, 1, 0, bias=False)
#       self.bn5 = BatchNorm2d(bin)
#
#   def __call__(self, x: Tensor) -> Tensor:
#     x = x * self.attention(x)
#     if self.shuffle:
#       x_proj, x = channel_shuffle(x)
#     else:
#       x_proj = self.bn4(self.cv4(x))
#       x_proj = self.bn5(self.cv5(x_proj)).hardswish()
#     x = self.bn1(self.cv1(x)).hardswish()
#     x = self.bn2(self.cv2(x))
#     x = self.bn3(self.cv3(x)).hardswish()
#     return x_proj.cat(x, dim=1)
#
# class Backbone:
#   def __init__(self):
#     stage_repeats = [3, 7, 3]
#     stage_out_channels = [24, 48, 96, 192, 1024]
#
#     self.stage1 = [Conv2d(3, stage_out_channels[0], 3, 2, 1, bias=False), BatchNorm2d(stage_out_channels[0]), Tensor.hardswish]
#     self.stage2 = [Block(stage_out_channels[0], stage_out_channels[1], shuffle=False, stride=2)]
#     self.stage2 += [Block(stage_out_channels[1], stage_out_channels[1]) for _ in range(stage_repeats[0])]
#     self.stage3 = [Block(stage_out_channels[1], stage_out_channels[2], shuffle=False, stride=2)]
#     self.stage3 += [Block(stage_out_channels[2], stage_out_channels[2]) for _ in range(stage_repeats[1])]
#     self.stage4 = [Block(stage_out_channels[2], stage_out_channels[3], shuffle=False, stride=1)]
#     self.stage4 += [Block(stage_out_channels[3], stage_out_channels[3]) for _ in range(stage_repeats[2])]
#     self.stage5 = [Conv2d(stage_out_channels[3], stage_out_channels[4], 1, 1, 0, bias=False), BatchNorm2d(stage_out_channels[4]), Tensor.hardswish]
#
#   def __call__(self, x:Tensor):
#     x = x.sequential(self.stage1).pad2d((1, 1, 1, 1)).max_pool2d(3, 2)
#     x2 = x.sequential(self.stage2)
#     x3 = x2.sequential(self.stage3)
#     x4 = x3.sequential(self.stage4)
#     x5 = x4.sequential(self.stage5)
#
#     x = x5.mean((2, 3)).flatten(1)
#     return x

def nonlinear(x: Tensor) -> Tensor: return x.hardswish()

class DWSepConv2d:
  def __init__(self, cin:int, cout:int, kernel_size:int, stride:int=1, padding:int=0, bias:bool=False):
    self.depthwise = Conv2d(cin, cin, kernel_size, stride, padding, groups=cin, bias=bias)
    self.dw_bn = BatchNorm2d(cin)
    self.pointwise = Conv2d(cin, cout, 1, 1, 0, bias=False)
    self.pw_bn = BatchNorm2d(cout)

  def __call__(self, x:Tensor) -> Tensor:
    x = self.dw_bn(self.depthwise(x))
    return self.pw_bn(self.pointwise(x))

class Block:
  def __init__(self, c:int):
    cmid = c // 2
    self.dw_3x3 = Conv2d(cmid, cmid, 3, 1, 1, groups=cmid, bias=False)
    self.dw_3x3_bn = BatchNorm2d(cmid)

    self.pw_e = Conv2d(cmid, c, 1, 1, 0, bias=False)
    self.pw_e_bn = BatchNorm2d(c)
    self.pw_s = Conv2d(c, cmid, 1, 1, 0, bias=False)
    self.pw_s_bn = BatchNorm2d(cmid)

  def __call__(self, x:Tensor) -> Tensor:
    x_proj, x = channel_shuffle(x)
    x_3x3 = self.dw_3x3_bn(self.dw_3x3(x))
    x = x + x_3x3

    x_pw = nonlinear(self.pw_e_bn(self.pw_e(x)))
    x_pw = self.pw_s_bn(self.pw_s(x_pw))
    x = x + x_pw
    return x_proj.cat(x, dim=1)

class Downsample:
  def __init__(self, cin:int, cout:int):
    self.block = Block(cin)
    self.dw = DWSepConv2d(cin, cout, 3, 2, 1, bias=False)
    self.dw_bn = BatchNorm2d(cout)

  def __call__(self, x:Tensor) -> Tensor:
    x = self.block(x)
    return self.dw_bn(self.dw(x))

class Backbone:
  def __init__(self):
    self.stem = [
      Conv2d(3, 24, 3, 2, 1, bias=False), BatchNorm2d(24), nonlinear,
      DWSepConv2d(24, 32, 3, 2, 1, bias=False), BatchNorm2d(32)
    ]

    self.stage2 = [Block(32) for _ in range(3)]
    self.stage2 += [Downsample(32, 64)]
    self.stage3 = [Block(64) for _ in range(7)]
    self.stage3 += [Downsample(64, 128)]
    self.stage4 = [Block(128) for _ in range(3)]
    self.stage5 = [Downsample(128, 32)]

  def __call__(self, x:Tensor) -> Tensor:
    x = x.sequential(self.stem)
    x = x.sequential(self.stage2)
    x = x.sequential(self.stage3)
    x = x.sequential(self.stage4)
    x = x.sequential(self.stage5)
    return x.flatten(1)

if __name__ == "__main__":
  from tinygrad.nn.state import get_parameters
  from tinygrad import GlobalCounters

  model = Backbone()
  print(f"model parameters: {sum(p.numel() for p in get_parameters(model))}")

  model(Tensor.zeros(64, 3, 256, 256)).realize()
  GlobalCounters.reset()
  model(Tensor.zeros(64, 3, 256, 256)).realize()

  print(f"model parameters: {sum(p.numel() for p in get_parameters(model))}")
