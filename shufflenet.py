from typing import Tuple
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn import Conv2d, BatchNorm2d, Linear

def channel_shuffle(x: Tensor) -> Tuple[Tensor, Tensor]:
  b, c, h, w = x.shape
  assert c % 4 == 0
  x = x.reshape(b * c // 2, 2, h * w).permute(1, 0, 2)
  x = x.reshape(2, -1, c // 2, h, w)
  return x[0], x[1]

class ShuffleV2Block:
  def __init__(self, inp: int, outp: int, c_mid: int, kernel_size: int, stride: int, shuffle: bool):
    assert stride in [1, 2]
    self.stride, self.inp, self.outp, self.c_mid, self.shuffle = stride, inp, outp, c_mid, shuffle
    pad, out = kernel_size // 2, outp - inp

    # pw
    self.cv1 = Conv2d(inp, c_mid, 1, 1, 0, bias=False)
    self.bn1 = BatchNorm2d(c_mid)
    # dw
    self.cv2 = Conv2d(c_mid, c_mid, kernel_size, stride, pad, groups=c_mid, bias=False)
    self.bn2 = BatchNorm2d(c_mid)
    # pw-linear
    self.cv3 = Conv2d(c_mid, out, 1, 1, 0, bias=False)
    self.bn3 = BatchNorm2d(out)

    if not self.shuffle:
      # dw
      self.cv4 = Conv2d(inp, inp, kernel_size, stride, pad, groups=inp, bias=False)
      self.bn4 = BatchNorm2d(inp)
      # pw-linear
      self.cv5 = Conv2d(inp, inp, 1, 1, 0, bias=False)
      self.bn5 = BatchNorm2d(inp)

  def __call__(self, x: Tensor) -> Tensor:
    if self.shuffle:
      x_proj, x = channel_shuffle(x)
    else:
      x_proj = self.bn4(self.cv4(x).float()).cast(dtypes.default_float)
      x_proj = self.bn5(self.cv5(x_proj).float()).cast(dtypes.default_float).relu()
    x = self.bn1(self.cv1(x).float()).cast(dtypes.default_float).relu()
    x = self.bn2(self.cv2(x).float()).cast(dtypes.default_float)
    x = self.bn3(self.cv3(x).float()).cast(dtypes.default_float).relu()
    return x_proj.cat(x, dim=1)

class ShuffleNetV2:
  def __init__(self):
    stage_repeats = [4, 8, 4]
    stage_out_channels = [24, 48, 96, 192, 1024]

    self.stage1 = [Conv2d(3, stage_out_channels[0], 3, 2, 1, bias=False), lambda x: x.float(), BatchNorm2d(stage_out_channels[0]), lambda x: x.cast(dtypes.default_float).relu()]
    self.stage2 = [ShuffleV2Block(stage_out_channels[0], stage_out_channels[1], stage_out_channels[1] // 2, kernel_size=3, stride=2, shuffle=False)]
    self.stage2 += [ShuffleV2Block(stage_out_channels[1] // 2, stage_out_channels[1], stage_out_channels[1] // 2, 3, 1, True) for _ in range(stage_repeats[0] - 1)]
    self.stage3 = [ShuffleV2Block(stage_out_channels[1], stage_out_channels[2], stage_out_channels[2] // 2, kernel_size=3, stride=2, shuffle=False)]
    self.stage3 += [ShuffleV2Block(stage_out_channels[2] // 2, stage_out_channels[2], stage_out_channels[2] // 2, 3, 1, True) for _ in range(stage_repeats[1] - 1)]
    self.stage4 = [ShuffleV2Block(stage_out_channels[2], stage_out_channels[3], stage_out_channels[3] // 2, kernel_size=3, stride=1, shuffle=False)]
    self.stage4 += [ShuffleV2Block(stage_out_channels[3] // 2, stage_out_channels[3], stage_out_channels[3] // 2, 3, 1, True) for _ in range(stage_repeats[2] - 1)]
    self.stage5 = [Conv2d(stage_out_channels[3], stage_out_channels[4], 1, 1, 0, bias=False), lambda x: x.float(), BatchNorm2d(1024), lambda x: x.cast(dtypes.default_float).relu()]

  def __call__(self, x: Tensor):
    x = x.sequential(self.stage1).pad2d((1, 1, 1, 1)).max_pool2d(3, 2)
    x2 = x.sequential(self.stage2)
    x3 = x2.sequential(self.stage3)
    x4 = x3.sequential(self.stage4)
    x5 = x4.sequential(self.stage5)

    x = x5.mean((2, 3)).flatten(1)
    return x, x5

if __name__ == "__main__":
  from tinygrad.nn.state import torch_load, safe_save, get_state_dict, get_parameters
  from tinygrad.helpers import get_child
  from tinygrad import dtypes

  net = ShuffleNetV2()

  state_dict = torch_load("../e2e_playground/cache/ShuffleNetV2.0.5x.pth.tar")["state_dict"]
  # modify state_dict to match our model
  for key in list(state_dict.keys()):
    if "num_batches_tracked" in key: state_dict[key] = Tensor([state_dict[key].numpy().item()])
  for key in list(state_dict.keys()):
    if "first_conv" in key:
      state_dict[key.replace("first_conv", "stage1")] = state_dict[key]
      del state_dict[key]
    if "conv_last" in key:
      state_dict[key.replace("conv_last", "stage5")] = state_dict[key]
      del state_dict[key]
  for key in list(state_dict.keys()):
    if "stage1" in key:
      index = int(key.split(".")[2])
      if index == 1:
        state_dict[key.replace("stage1.1", "stage1.2")] = state_dict[key]
        del state_dict[key]
    if "stage5" in key:
      index = int(key.split(".")[2])
      if index == 1:
        state_dict[key.replace("stage5.1", "stage5.2")] = state_dict[key]
        del state_dict[key]
  for key in list(state_dict.keys()):
    if "branch_main" in key:
      index = int(key.split(".")[4])
      if index == 0:
        state_dict[key.replace("branch_main.0", "cv1")] = state_dict[key]
        del state_dict[key]
      elif index == 1:
        state_dict[key.replace("branch_main.1", "bn1")] = state_dict[key]
        del state_dict[key]
      elif index == 3:
        state_dict[key.replace("branch_main.3", "cv2")] = state_dict[key]
        del state_dict[key]
      elif index == 4:
        state_dict[key.replace("branch_main.4", "bn2")] = state_dict[key]
        del state_dict[key]
      elif index == 5:
        state_dict[key.replace("branch_main.5", "cv3")] = state_dict[key]
        del state_dict[key]
      elif index == 6:
        state_dict[key.replace("branch_main.6", "bn3")] = state_dict[key]
        del state_dict[key]
    if "branch_proj" in key:
      index = int(key.split(".")[4])
      if index == 0:
        state_dict[key.replace("branch_proj.0", "cv4")] = state_dict[key]
        del state_dict[key]
      elif index == 1:
        state_dict[key.replace("branch_proj.1", "bn4")] = state_dict[key]
        del state_dict[key]
      elif index == 2:
        state_dict[key.replace("branch_proj.2", "cv5")] = state_dict[key]
        del state_dict[key]
      elif index == 3:
        state_dict[key.replace("branch_proj.3", "bn5")] = state_dict[key]
        del state_dict[key]
  for key in list(state_dict.keys()):
    if "features" in key:
      index = int(key.split(".")[2])
      if index in range(0, 4):
        state_dict[key.replace(f"features.{index}", f"stage2.{index}")] = state_dict[key]
        del state_dict[key]
      elif index in range(4, 12):
        state_dict[key.replace(f"features.{index}", f"stage3.{index - 4}")] = state_dict[key]
        del state_dict[key]
      elif index in range(12, 16):
        state_dict[key.replace(f"features.{index}", f"stage4.{index - 12}")] = state_dict[key]
        del state_dict[key]

  for key in list(state_dict.keys()):
    if "classifier" in key: continue
    print(f"Loading {key}...")
    get_child(net, key.replace("module.", "")).assign(state_dict[key].to(Device.DEFAULT)).realize()

  for param in get_parameters(state_dict):
    param.assign(param.cast(dtypes.float32)).realize()

  # save state_dict
  safe_save(get_state_dict(net), "./weights/shufflenetv2.safetensors")
