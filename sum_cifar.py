import torch
from torch.optim.optimizer import Optimizer, required


class SUM_Ind(Optimizer):
    
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, interp_factor=0,K=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if interp_factor < 0.0:
            raise ValueError("Invalid interp_factor value: {}".format(interp_factor))        
        if K < 0.0:
            raise ValueError("Invalid K value: {}".format(K))
            
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, interp_factor=interp_factor, K=int(K))
        
        super(SUM_Ind, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SUM_Ind, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, epoch, steplr, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        d_p_sum=0
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            interp_factor = group['interp_factor']
            init_lr = group['lr']
            K = group['K']
            lr=init_lr
            
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = -p.grad
                
                d_p_sum += torch.mean(torch.abs(d_p))
                
                param_state = self.state[p]
                if len(param_state) == 0:
                    param_state['step'] = 0
                    
                param_state['step'] += 1
                
                
                if epoch >= 0 and epoch <10:
                    lr =init_lr*0.1
                elif epoch >= 10 and epoch <100:
                    lr =init_lr
                elif epoch >= 100 and epoch <150:
                    lr =init_lr* 0.1
                elif epoch >= 150:
                    lr =init_lr* 0.01
                else:
                    lr =init_lr

                
                    
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=-weight_decay)
                    
                
                if param_state['step']>K:
                    lr=0.01*init_lr/(param_state['step']-K)
                    # print(lr)
     
                    
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=lr)
                        
                    if interp_factor != 0:
                        d_p = d_p.mul_(interp_factor*lr).add(buf, alpha=(1-interp_factor+momentum*interp_factor)) 
                    else:
                        d_p = buf
                else:
                    d_p = d_p.mul_(lr)
                        
                p.add_(d_p, alpha=1.0)

        return loss,d_p_sum
    