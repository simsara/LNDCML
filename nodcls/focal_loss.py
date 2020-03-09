#Focal Loss
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        print ("FOCAL LOSS", gamma, alpha)

    def forward(self, input, target):
        target = target.float()
        if input.dim() == 1:
            input = input.unsqueeze(1)
        if target.dim() == 1:
            target = target.unsqueeze(1)

        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C

        #target = target.view(-1,1)
        target = target.float()
        pt = input * target + (1 - input) * (1 - target)
        logpt = pt.log()
        at = (1 - self.alpha) * target + (self.alpha) * (1 - target)
        logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()