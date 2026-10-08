import argparse
import os
import numpy as np
import torch
from cs336_basics.model import Transformer_LM
from cs336_basics.train import (cross_entropy_loss, AdamW, Data_loading, Learning_rate_schedule,
                                Gradient_clipping, save_checkpoint, load_checkpoint)

parser = argparse.ArgumentParser()

parser.add_argument("--train_data", type=str, default=None)  
parser.add_argument("--val_data", type=str, default=None)

parser.add_argument("--vocab_size", type=int, default=100)
parser.add_argument("--context_length", type=int, default=16)
parser.add_argument("--d_model", type=int, default=64)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_heads", type=int, default=4)
parser.add_argument("--d_ff", type=int, default=128)
parser.add_argument("--rope_theta", type=float, default=10000)

parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--max_iters", type=int, default=300)
parser.add_argument("--lr_max", type=float, default=1e-3)
parser.add_argument("--lr_min", type=float, default=1e-4)
parser.add_argument("--warmup_iters", type=int, default=30)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--max_grad_norm", type=float, default=1.0)

parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--eval_interval", type=int, default=100)
parser.add_argument("--eval_batches", type=int, default=10)
parser.add_argument("--ckpt_path", type=str, default="checkpoint.pt")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()
torch.set_float32_matmul_precision("high")   # GPU 上矩阵乘法用 TF32:更快,精度对训练够用

torch.manual_seed(0)
np.random.seed(0)

if args.train_data is None:
    train_data = np.random.randint(0, args.vocab_size, size=10000)
    val_data = np.random.randint(0, args.vocab_size, size=2000)
else:
    train_data = np.load(args.train_data, mmap_mode="r")
    val_data = np.load(args.val_data, mmap_mode="r")

def get_batch(data):
    x, y = Data_loading(data, args.batch_size, args.context_length, args.device)
    return x.long(), y.long()     

model = Transformer_LM(args.vocab_size, args.context_length, args.d_model, args.num_layers,
                       args.num_heads, args.d_ff, args.rope_theta, device=args.device)
optimizer = AdamW(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay,
                  betas=(0.9, 0.999), eps=1e-8)

start_iter = 0
if args.resume and os.path.exists(args.ckpt_path):
    start_iter = load_checkpoint(args.ckpt_path, model, optimizer)
    print(f"resumed from step {start_iter}")


def evaluate():
    total = 0.0
    with torch.no_grad():
        for _ in range(args.eval_batches):
            x, y = get_batch(val_data)
            total += cross_entropy_loss(model(x), y).item()
    return total / args.eval_batches


for it in range(start_iter, args.max_iters):
    lr = Learning_rate_schedule(it, args.lr_max, args.lr_min, args.warmup_iters, args.max_iters)
    for group in optimizer.param_groups:
        group["lr"] = lr


    x, y = get_batch(train_data)
    loss = cross_entropy_loss(model(x), y)
    optimizer.zero_grad()
    loss.backward()

    Gradient_clipping(model.parameters(), args.max_grad_norm)

    optimizer.step()

    if it % args.log_interval == 0:
        print(f"step {it}  train loss {loss.item():.4f}  lr {lr:.2e}")


    if (it + 1) % args.eval_interval == 0 or it + 1 == args.max_iters:
        print(f"step {it}  val loss {evaluate():.4f}")
        save_checkpoint(model, optimizer, it + 1, args.ckpt_path)
