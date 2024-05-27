import torch
import torch.nn as nn
import torch.nn.functional as F

"""Focal Loss"""
# from torchvision import ops

# class FocalLoss(nn.Module):
#     def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = 'none') -> torch.Tensor:
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#         self.loss_fn = ops.sigmoid_focal_loss

#     def __call__(self, inputs, targets):
#         loss = self.loss_fn(inputs=inputs, targets=targets, alpha=self.alpha, 
#                             gamma=self.gamma, reduction=self.reduction)
#         return loss


"""The code was taken from the below repository:
    https://github.com/shuaizzZ/Dice-Loss-PyTorch/blob/master/dice_loss.py
"""
class DiceLoss(nn.Module):
    """Dice Loss PyTorch
        Created by: Zhang Shuai
        Email: shuaizzz666@gmail.com
        dice_loss = 1 - 2*p*t / (p^2 + t^2). p and t represent predict and target.
    Args:
        weight: An array of shape [C,]
        predict: A float32 tensor of shape [N, C, *], for Semantic segmentation task is [N, C, H, W]
        target: A int64 tensor of shape [N, *], for Semantic segmentation task is [N, H, W]
    Return:
        diceloss
    """
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight / torch.sum(weight) # Normalized weight
        self.smooth = 1e-5

    def forward(self, predict, target):
        N, C = predict.size()[:2]
        predict = predict.view(N, C, -1) # (N, C, *)
        target = target.view(N, C, -1)

        predict = F.softmax(predict, dim=1) # (N, C, *) ==> (N, C, *)
        ## convert target(N, 1, *) into one hot vector (N, C, *)

        intersection = torch.sum(predict * target, dim=2)  # (N, C)
        union = torch.sum(predict.pow(2), dim=2) + torch.sum(target, dim=2)  # (N, C)
        ## p^2 + t^2 >= 2*p*t, target_onehot^2 == target_onehot
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)  # (N, C)

        if hasattr(self, 'weight'):
            if self.weight.type() != predict.type():
                self.weight = self.weight.type_as(predict)
                dice_coef = dice_coef * self.weight * C  # (N, C)
        dice_loss = 1 - torch.mean(dice_coef)  # 1

        return dice_loss
