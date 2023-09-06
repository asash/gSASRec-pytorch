import torch
class gBCE(torch.nn.Module):
    def __init__(self, pool_size, negatives, t):
        super(gBCE, self).__init__()
        self.alpha = negatives/(pool_size - 1)
        self.beta = self.alpha * (t * (1 - 1/self.alpha) + 1/self.alpha)
        print(f"gbce beta: {self.beta}")

    def forward(self, logits, targets):
        logits = logits.to(torch.float64) #increase precision, as gBCE works with small numbers
        targets = targets.to(torch.float64)
        pos = targets*torch.nn.functional.softplus(-logits) 
        neg = (1.0 - targets)*torch.nn.functional.softplus(logits)
        loss = (self.beta*pos + neg)
        return loss
