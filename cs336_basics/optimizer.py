import torch
import numpy
import math
def cross_entropy_loss (inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    max_input=inputs.amax(dim=-1,keepdim=True)
    logsumexp_input=max_input+torch.log(torch.sum(torch.exp(inputs-max_input),dim=-1,keepdim=True))
    return torch.mean(logsumexp_input-inputs.gather(-1,targets.unsqueeze(-1)))

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, weight_decay: float, betas: tuple, eps: float):
        defaults={"lr":lr,"weight_decay":weight_decay,"betas":betas,"eps":eps}
        super().__init__(params, defaults)
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr=group["lr"]
            weight_decay=group["weight_decay"]
            beta1, beta2=group["betas"]
            eps=group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state[p]
                if len(state)==0:
                    state["m"]=torch.zeros_like(p)
                    state["v"]=torch.zeros_like(p)
                    state["t"]=0
                t=state["t"]
                m=state["m"]
                v=state["v"]
                t=t+1
                m=beta1*m+(1-beta1)*g
                v=beta2*v+(1-beta2)*(g**2)
                alpha_t=lr*math.sqrt(1-beta2**t)/(1-beta1**t)
                p.data=p.data-alpha_t*m/(torch.sqrt(v)+eps)
                p.data=p.data-lr*weight_decay*(p.data)
                state["t"]=t
                state["m"]=m
                state["v"]=v
        return loss

def Learning_rate_schedule(t: int, lr_max: float, lr_min: float, t_warm_up: int, t_cos: int) -> float:
    if t<t_warm_up:
        return (t/t_warm_up)*lr_max
    else:
        if t<=t_cos:
            progress=(t-t_warm_up)/(t_cos-t_warm_up)
            return lr_min+0.5*(1+math.cos(progress*math.pi))*(lr_max-lr_min)
        else:
            return lr_min

def Gradient_clipping(parameters: iter, max_l2_norm: float):
    parameters_copied=list(parameters)
    gradient_length=0
    for parameter in parameters_copied:
        if parameter.grad is None:
            continue
        gradient_length+=(parameter.grad**2).sum()
    gradient_length=math.sqrt(gradient_length)
    if gradient_length>max_l2_norm:
        clipping_scale=max_l2_norm/(gradient_length+1e-6)
        for parameter in parameters_copied:
            if parameter.grad is None:
                continue
            parameter.grad*=clipping_scale

def Data_loading(dataset: numpy.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    random_starting_point=numpy.random.randint(0,len(dataset)-context_length,size=batch_size)
    predict_list=[]
    target_list=[]
    for starting_point in random_starting_point:
        predict=dataset[starting_point:starting_point+context_length]
        target=dataset[starting_point+1:starting_point+context_length+1]
        predict_list.append(predict)
        target_list.append(target)
    
    p=torch.from_numpy(numpy.stack(predict_list))
    t=torch.from_numpy(numpy.stack(target_list))
    p=p.to(device=device)
    t=t.to(device=device)
    return (p,t)

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out) -> None:
    checkpoint={}
    checkpoint["model_weight"]=model.state_dict()
    checkpoint["optimizer_state"]=optimizer.state_dict()
    checkpoint["iteration"]=iteration
    torch.save(checkpoint,out)

def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    checkpoint=torch.load(src)
    model.load_state_dict(checkpoint["model_weight"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["iteration"]


