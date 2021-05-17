import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryTverskyLoss(nn.Module):
    """Dice loss of binary
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1e-5
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, alpha=0.7, gamma=None ,smooth=1e-5, p=1, reduction='mean', k=80):
        super(BinaryTverskyLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        self.k = k

    def forward(self, predict, target):

        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        batch_size = predict.shape[0]
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        true_positive = torch.sum(torch.mul(predict, target), dim=1)
        false_positive = torch.sum(torch.mul((1 - target.pow(self.p)),predict.pow(self.p)), dim=1)
        false_negative = torch.sum(torch.mul(target.pow(self.p),(1 - predict.pow(self.p))), dim=1)
        
        loss = 1 - (true_positive + self.smooth)/ (true_positive + self.alpha*false_positive + (1 - self.alpha)*false_negative + self.smooth)

        if self.gamma != None:
            loss = loss.pow(self.gamma)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'topk':
            loss,_ = torch.topk(loss,int(batch_size * self.k / 100), sorted=False)
            return loss.mean()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class TverskyLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryTverskyLoss
    Return:
        same as BinaryTverskyLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(TverskyLoss, self).__init__()
        self.kwargs = kwargs
        self.class_weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        tversky = BinaryTverskyLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)
        
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                tversky_loss = tversky(predict[:, i], target[:, i])
                if self.class_weight is not None:
                    assert  self.class_weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1],  self.class_weight.shape[0])
                    tversky_loss *=  self.class_weight[i]
                total_loss += tversky_loss
        if self.ignore_index is not None:
            return total_loss/(target.shape[1] - 1)
        else:
            return total_loss/target.shape[1]