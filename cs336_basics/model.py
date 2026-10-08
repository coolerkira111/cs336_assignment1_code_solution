 
import torch
from torch import nn
import math
import einops
from cs336_basics.tokenizer import Tokenizer
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.weight=nn.Parameter(torch.zeros(out_features,in_features,device=device,dtype=dtype))
        sigma=math.sqrt(2/(in_features+out_features))
        torch.nn.init.trunc_normal_(self.weight,mean=0,std=sigma,a=-3*sigma,b=3*sigma)
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return einops.einsum(self.weight,x,"out_features in_features, ... in_features -> ... out_features")
 
class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device=None, dtype=None):
        super().__init__()
        self.weight=nn.Parameter(torch.zeros(num_embeddings,embedding_dim,device=device,dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight,mean=0,std=1,a=-3,b=3)
    def forward(self, token_ids:torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))
        self.eps=eps
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        in_dtype=x.dtype
        x=x.to(torch.float32)
        rms_x=x**2
        rms_x=rms_x.mean(dim=-1,keepdim=True)
        rms_x=torch.add(rms_x,self.eps)
        rms_x=torch.sqrt(rms_x)
        x=torch.mul(x/rms_x,self.weight)
        x=x.to(in_dtype)
        return x

def SiLU(x:torch.Tensor) -> torch.Tensor:
    return torch.mul(x,torch.sigmoid(x))

class SwiGLU(nn.Module):
    def __init__(self, d_model:int, d_ff=None, device=None, dtype=None):
        super().__init__()
        self.d_model=d_model
        if not d_ff:
            self.d_ff=round((8*d_model/3)/64)*64
        else:
            self.d_ff=d_ff
        self.w1=Linear(self.d_model,self.d_ff,device=device,dtype=dtype)
        self.w2=Linear(self.d_ff,self.d_model,device=device,dtype=dtype)
        self.w3=Linear(self.d_model,self.d_ff,device=device,dtype=dtype)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        w1_x=self.w1.forward(x)
        w3_x=self.w3.forward(x)
        w2_x=self.w2.forward(torch.mul(SiLU(w1_x),w3_x))
        return w2_x

class RoPE(nn.Module):
    def __init__(self, theta:float, d_k:int, max_seq_len:int, device=None):
        super().__init__()
        speed_parameter=torch.arange(0,d_k,2,dtype=torch.float32,device=device)
        speed=1.0/(theta**(speed_parameter/d_k))
        position=torch.arange(0,max_seq_len,dtype=torch.float32,device=device)

        angle_table=einops.einsum(speed,position,"d_k_half, max_seq_len -> max_seq_len d_k_half")
        cos_table=torch.cos(angle_table)
        sin_table=torch.sin(angle_table)
        self.register_buffer("cos_table",cos_table,persistent=False)
        self.register_buffer("sin_table",sin_table,persistent=False)
    def forward(self, x:torch.Tensor, token_positions:torch.Tensor) -> torch.Tensor:
        pair_one=x[...,0::2]
        pair_two=x[...,1::2]
        cos_taken=self.cos_table[token_positions]
        sin_taken=self.sin_table[token_positions]
        embedded_pair_one=torch.mul(pair_one,cos_taken)-torch.mul(pair_two,sin_taken)
        embedded_pair_two=torch.mul(pair_two,cos_taken)+torch.mul(pair_one,sin_taken)
        restack=torch.stack([embedded_pair_one,embedded_pair_two],dim=-1)
        return einops.rearrange(restack,"... two n -> ... (two n)")

def softmax(x:torch.Tensor, dimention:int) -> torch.Tensor:
    top=torch.exp(x-x.amax(dim=dimention,keepdim=True))
    bottom=torch.sum(top,dim=dimention,keepdim=True)
    return top/bottom

def scaled_dot_product_attention(Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask=None) -> torch.Tensor:
    d_k=K.shape[-1]
    dot_product=einops.einsum(Q,K,"... n d_k, ... m d_k -> ... n m")/math.sqrt(d_k)
    if mask is not None:
        dot_product_masked=dot_product.masked_fill(~mask,float("-inf"))
    else:
        dot_product_masked=dot_product
    return einops.einsum(softmax(dot_product_masked,dimention=-1),V,"... n m, ... m d_v -> ... n d_v")


class Multi_Head_Self_Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()
        self.q_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.k_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.v_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.output_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.h=num_heads
        self.rope=None
        if theta is not None:
            self.rope=RoPE(theta,d_k=d_model//num_heads,max_seq_len=max_seq_len,device=device)
    def forward(self, x:torch.Tensor, token_positions=None):
        wq_x=self.q_proj.forward(x)
        wk_x=self.k_proj.forward(x)
        wv_x=self.v_proj.forward(x)
        headed_wq_x=einops.rearrange(wq_x,"b s (h d_k) ->b h s d_k ", h=self.h)
        headed_wk_x=einops.rearrange(wk_x,"b s (h d_k) ->b h s d_k ", h=self.h)
        headed_wv_x=einops.rearrange(wv_x,"b s (h d_k) ->b h s d_k ", h=self.h)
        if self.rope is not None:
            if token_positions is None:
                token_positions_to_use=torch.arange(x.shape[-2])
            else:
                token_positions_to_use=token_positions
            headed_token_positions=einops.rearrange(token_positions_to_use,"... s -> ... 1 s")
            headed_wq_x=self.rope.forward(headed_wq_x,headed_token_positions)
            headed_wk_x=self.rope.forward(headed_wk_x,headed_token_positions)
        mask=~torch.triu(torch.ones(x.shape[-2],x.shape[-2],dtype=torch.bool,device=x.device),diagonal=1)
        attention_result=scaled_dot_product_attention(headed_wq_x,headed_wk_x,headed_wv_x,mask=mask)
        attention_result=einops.rearrange(attention_result,"b h s d_k -> b s (h d_k)")
        return self.output_proj.forward(attention_result)

class Transformer_Block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta=None, device=None, dtype=None):
        super().__init__()
        self.ln1=RMSNorm(d_model,device=device,dtype=dtype)
        self.ln2=RMSNorm(d_model,device=device,dtype=dtype)
        self.attn=Multi_Head_Self_Attention(d_model,num_heads,max_seq_len,theta=theta,device=device,dtype=dtype)
        self.ffn=SwiGLU(d_model,d_ff,device=device,dtype=dtype)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y=x+self.attn.forward(self.ln1.forward(x))
        z=y+self.ffn.forward(self.ln2.forward(y))
        return z

class Transformer_LM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device=None, dtype=None):
        super().__init__()
        self.token_embeddings=Embedding(vocab_size,d_model,device=device,dtype=dtype)
        self.layers=nn.ModuleList([Transformer_Block(d_model,num_heads,d_ff,context_length,rope_theta,device=device,dtype=dtype) for _ in range(num_layers)])
        self.ln_final=RMSNorm(d_model,device=device,dtype=dtype)
        self.lm_head=Linear(d_model,vocab_size,device=device,dtype=dtype)
        self.num_layers=num_layers
    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        embedded_token=self.token_embeddings.forward(in_indices)
        input_transblock=embedded_token
        for transblock in self.layers:
            output_transblock=transblock.forward(input_transblock)
            input_transblock=output_transblock
        normed_output=self.ln_final.forward(output_transblock)
        output=self.lm_head.forward(normed_output)
        return output

def generate(model: Transformer_LM, tokenizer: Tokenizer, prompt, max_new_tokens, temperature, top_p, context_length):
    ids=tokenizer.encode(prompt)
    new_token_num=0
    with torch.no_grad():
        while new_token_num<max_new_tokens:
            model_input=ids
            if len(ids)>context_length:
                model_input=ids[-context_length:]
            logits=model.forward(torch.tensor([model_input]).to(next(model.parameters()).device))
            new_token_logits=logits[0,-1]
            new_token_logits=new_token_logits/temperature
            new_token_logits=softmax(new_token_logits,dimention=-1)
            sorted_logits, logits_label=torch.sort(new_token_logits,descending=True)
            cumsum=sorted_logits.cumsum(-1)
            for i in range(len(cumsum)):
                if cumsum[i]>=top_p:
                    break
            sorted_logits=sorted_logits[0:i+1]
            sorted_logits=sorted_logits/sorted_logits.sum()
            pos=torch.multinomial(sorted_logits,1)
            new_token_id=logits_label[pos.item()].item()
            ids.append(new_token_id)
            if new_token_id==tokenizer.bytes_to_id["<|endoftext|>".encode("utf-8")]:
                break
            new_token_num+=1
    return tokenizer.decode(ids)
        