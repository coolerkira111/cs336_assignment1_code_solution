# 用训好的 checkpoint 生成文本
# 用法: python -m cs336_basics.gen --ckpt ts_ckpt.pt --vocab tinystories_train_vocab.pkl --merges tinystories_train_merges.pkl
#       OWT 模型再加 --vocab_size 32000,并换成 owt 的 pkl
import argparse
import torch
from cs336_basics.model import Transformer_LM, generate
from cs336_basics.tokenizer import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--vocab", type=str, required=True)
parser.add_argument("--merges", type=str, required=True)
parser.add_argument("--prompt", type=str, default="Once upon a time")
parser.add_argument("--max_new_tokens", type=int, default=256)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--num_samples", type=int, default=3)
# 模型结构要和训练时一样
parser.add_argument("--vocab_size", type=int, default=10000)
parser.add_argument("--context_length", type=int, default=256)
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--num_heads", type=int, default=16)
parser.add_argument("--d_ff", type=int, default=1344)
parser.add_argument("--rope_theta", type=float, default=10000)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

model = Transformer_LM(args.vocab_size, args.context_length, args.d_model, args.num_layers,
                       args.num_heads, args.d_ff, args.rope_theta, device=args.device)
# map_location: 在 GPU 上存的 checkpoint 也能在 CPU(比如你的 Mac)上读
checkpoint = torch.load(args.ckpt, map_location=args.device)
model.load_state_dict(checkpoint["model_weight"])
tokenizer = Tokenizer.from_files(args.vocab, args.merges, ["<|endoftext|>"])

for i in range(args.num_samples):
    text = generate(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature,
                    args.top_p, args.context_length)
    print(f"===== sample {i + 1} =====")
    print(text)
