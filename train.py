import glob, math, random, sys, signal
import multiprocessing
from multiprocessing import Queue, Process

from tinygrad import dtypes, Tensor, GlobalCounters
from tinygrad.features.jit import TinyJit
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_load, safe_save
from tinygrad.helpers import Context
from tqdm import trange
import wandb
import cv2

from model import Model
from main import BASE_PATH

BS = 64
WARMUP_STEPS = 100
WARMPUP_LR = 0.0001
START_LR = 0.005
END_LR = 0.0005
STEPS = 2000

def loss_fn(pred: Tensor, y: Tensor): return pred.sparse_categorical_crossentropy(y)

@TinyJit
def train_step(x, y, lr):
  pred = model(x)
  loss = loss_fn(pred, y)

  optim.lr.assign(lr+1-1)
  optim.zero_grad()
  loss.backward()

  # calculate grad norm
  grad_norm = Tensor([0.], dtype=dtypes.float32)
  for p in get_parameters(model):
    if p.grad is not None:
      grad_norm.assign(grad_norm + p.grad.detach().pow(2).sum())

  optim.step()

  return loss.realize(), grad_norm.realize()

warming_up = True
def get_lr(i: int) -> float:
  global warming_up
  if warming_up:
    lr = START_LR * (i / WARMUP_STEPS) + WARMPUP_LR * (1 - i / WARMUP_STEPS)
    if i >= WARMUP_STEPS: warming_up = False
  else: lr = END_LR + 0.5 * (START_LR - END_LR) * (1 + math.cos(((i - WARMUP_STEPS) / (STEPS - WARMUP_STEPS)) * math.pi))
  return lr

preprocessed_train_files = glob.glob(str(BASE_PATH / "preprocessed/*.txt"))
def load_single_file(file):
  # read the image file
  img = cv2.imread(file.replace(".txt", ".png"))
  # read the annotation file
  with open(file, "r") as f:
    detected = int(f.readline())
  return img, detected

def minibatch_iterator(q: Queue):
  pool = multiprocessing.Pool(4)
  while True:
    random.shuffle(preprocessed_train_files)
    for i in range(0, len(preprocessed_train_files) - BS, BS):
      batched = pool.map(load_single_file, preprocessed_train_files[i : i + BS])
      x_b, y_b = zip(*batched)
      q.put((list(x_b), list(y_b)))

class ModelEMA:
  def __init__(self, model):
    self.model = Model()
    for ep, p in zip(get_state_dict(self.model).values(), get_state_dict(model).values()):
      ep.requires_grad = False
      ep.assign(p)

  @TinyJit
  def update(self, net, alpha):
    for ep, p in zip(get_state_dict(self.model).values(), get_state_dict(net).values()):
      ep.assign(alpha * ep.detach() + (1 - alpha) * p.detach()).realize()


if __name__ == "__main__":
  Tensor.no_grad = False
  Tensor.training = True
  dtypes.default_float = dtypes.float32

  wandb.init(project="mrm_cursed_radar")
  wandb.config.update({
    "warmup_steps": WARMUP_STEPS,
    "start_lr": START_LR,
    "end_lr": END_LR,
    "steps": STEPS,
  })

  model = Model()

  # sn_state_dict = safe_load("./weights/shufflenetv2.safetensors")
  # load_state_dict(model.backbone, sn_state_dict, strict=False)

  # state_dict = safe_load(str(BASE_PATH / "model.safetensors"))
  # load_state_dict(model, state_dict)

  # model_ema = ModelEMA(model)

  parameters = []
  for key, value in get_state_dict(model).items():
    parameters.append(value)
  optim = SGD(parameters, momentum=0.9, weight_decay=1e-4)
  # optim = AdamW(parameters, wd=1e-4)

  # start batch iterator in a separate process
  bi_queue = Queue(4)
  bi = Process(target=minibatch_iterator, args=(bi_queue,))
  bi.start()

  def sigint_handler(*_):
    print("SIGINT received, killing batch iterator")
    bi.terminate()
    sys.exit(0)
  signal.signal(signal.SIGINT, sigint_handler)

  for step in (t := trange(STEPS)):
    with Context(BEAM=0 if step == 0 else 2, WINO=1):
      GlobalCounters.reset()

      # train one step
      new_lr = get_lr(step)
      if step == 0:
        x, y = bi_queue.get()
        x, y = Tensor(x, dtype=dtypes.uint8), Tensor(y, dtype=dtypes.default_float)
      loss, grad_norm = train_step(x, y, Tensor([new_lr], dtype=dtypes.default_float))
      x, y = bi_queue.get()
      x, y = Tensor(x, dtype=dtypes.uint8), Tensor(y, dtype=dtypes.default_float)

      # update EMA
      # if step >= 400 and step % 5 == 0: model_ema.update(model, Tensor([0.998]))

      # sema
      # if step >= 600 and step % 200 == 0:
      #   for p, ep in zip(get_state_dict(model).values(), get_state_dict(model_ema.model).values()):
      #     p.assign(ep.detach()).realize()

      loss, grad_norm, lr = loss.item(), grad_norm.item(), optim.lr.item()
      t.set_description(f"loss: {loss:6.6f}, grad_norm: {grad_norm:6.6f}, lr: {lr:12.12f}")
      wandb.log({
        "loss": loss,
        "grad_norm": grad_norm,
        "lr": lr,
        "gflops": GlobalCounters.global_ops / GlobalCounters.time_sum_s,
        "gb/s": GlobalCounters.global_mem / GlobalCounters.time_sum_s,
      })

      if step % 10000 == 0 and step > 0: safe_save(get_state_dict(model), str(BASE_PATH / f"intermediate/model_{step}.safetensors"))
  safe_save(get_state_dict(model), str(BASE_PATH / f"model.safetensors"))

  bi.terminate()
