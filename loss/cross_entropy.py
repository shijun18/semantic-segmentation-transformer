import torch
import torch.nn.functional as F
import numpy as np


class CrossentropyLoss(torch.nn.CrossEntropyLoss):

    def forward(self, inp, target):
        if target.size()[1] > 1:
            target = torch.argmax(target,1)
        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2
        # this is ugly but torch only allows to transpose two axes at once
        while i1 < len(inp.shape): 
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)

        target = target.view(-1,)

        return super(CrossentropyLoss, self).forward(inp, target)




class TopKLoss(CrossentropyLoss):

    def __init__(self, weight=None, ignore_index=-100, k=10, reduction=None):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, False, reduction='none')

    def forward(self, inp, target):
        # target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()


class DynamicTopKLoss(CrossentropyLoss):

    def __init__(self, weight=None, ignore_index=-100, step_threshold=1000, min_k=20, reduction=None):
        self.k = 100
        self.step = 0
        self.min_k = min_k
        self.step_threshold = step_threshold
        super(DynamicTopKLoss, self).__init__(weight, False, ignore_index, False, reduction='none')
        
    def forward(self, inp, target):
        # target = target[:, 0].long()
        res = super(DynamicTopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        self.step += 1

        if self.step % self.step_threshold == 0 and self.k > self.min_k:
            self.k -= 1
        
        return res.mean()