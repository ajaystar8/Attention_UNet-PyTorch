import torch
import torch.nn as nn


def dice_coefficient(y_pred_logits: torch.Tensor, y_true: torch.Tensor):
    """
    The function calculates the dice coefficient aka F1 score given the input and target tensors
    :param y_pred_logits: the predicted logits of the model
    :param y_true: the ground truth
    :return: the dice coefficient
    """
    epsilon = 1.0e-6

    y_pred_logits = y_pred_logits.view(-1).to(torch.float)
    y_true = y_true.view(-1).to(torch.float)

    intersection = torch.sum(y_pred_logits * y_true)
    union = torch.sum(y_pred_logits) + torch.sum(y_true)

    DSC = 2.0 * (intersection / (union + epsilon))
    return DSC


class DiceLoss(nn.Module):
    """
    Dice Loss function
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(y_pred_logits: torch.Tensor, y_true: torch.Tensor):
        """
        Having soft labels in prediction tensor provides more information on the model's confidence on classifying a
        pixel. This will lead to more effective model training

        :param y_pred_logits: the predicted logits of the model (soft labelled)
        :param y_true: the ground truth  (hard labelled)
        :return: diceloss
        """

        epsilon = 1.0e-6

        y_pred_logits = y_pred_logits.view(-1).to(torch.float)
        y_true = y_true.view(-1).to(torch.float)

        intersection = torch.sum(y_pred_logits * y_true)
        union = torch.sum(y_pred_logits) + torch.sum(y_true)

        DSC = 2.0 * (intersection / (union + epsilon))
        diceloss = 1 - DSC
        return diceloss
