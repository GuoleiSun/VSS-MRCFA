import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


def cross_entropy(pred,
                  label,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100):
    """The wrapper function for :func:`F.cross_entropy`"""
    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_onehot_labels(labels, label_weights, target_shape, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)

    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1

    valid_mask = valid_mask.unsqueeze(1).expand(target_shape).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.unsqueeze(1).expand(target_shape)
        bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         ignore_index=255):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored. Default: 255

    Returns:
        torch.Tensor: The calculated loss
    """
    if pred.dim() != label.dim():
        assert (pred.dim() == 2 and label.dim() == 1) or (
                pred.dim() == 4 and label.dim() == 3), \
            'Only pred shape [N, C], label shape [N] or pred shape [N, C, ' \
            'H, W], label shape [N, H, W] are supported'
        label, weight = _expand_onehot_labels(label, weight, pred.shape,
                                              ignore_index)

    # weighted element-wise losses
    # print("label: ", label.min(), label.max())
    # exit()
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None,
                       ignore_index=None):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask'
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (None): Placeholder, to be consistent with other loss.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert ignore_index is None, 'BCE loss does not support ignore_index'
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction='mean')[None]


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls


## added by guolei

def binary_cross_entropy_affinity(pred,
                         label,
                         use_sigmoid=False,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None):
    """Calculate the binary CrossEntropy loss.
    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
    Returns:
        torch.Tensor: The calculated loss
    """
    if pred.dim() != label.dim():
        label, weight = _expand_onehot_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    if use_sigmoid:
        loss = F.binary_cross_entropy_with_logits(
            pred, label.float(), weight=class_weight, reduction='none')
    else:
        loss = F.binary_cross_entropy(
            pred, label.float(), weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss

@LOSSES.register_module()
class AffinityLoss(nn.Module):
    """CrossEntropyLoss.
    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(AffinityLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        # self.cls_criterion = binary_cross_entropy
        self.cls_criterion = binary_cross_entropy_affinity

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        unary_term = self.cls_criterion(
            cls_score,
            label,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        diagonal_matrix = (1 - torch.eye(label.size(1))).to(label.get_device())
        vtarget = diagonal_matrix * label

        recall_part = torch.sum(cls_score * vtarget, dim=2)
        denominator = torch.sum(vtarget, dim=2)
        denominator = denominator.masked_fill_(~(denominator > 0), 1)
        recall_part = recall_part.div_(denominator)
        recall_label = torch.ones_like(recall_part)
        recall_loss = self.cls_criterion(
            recall_part,
            recall_label,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        spec_part = torch.sum((1 - cls_score) * (1 - label), dim=2)
        denominator = torch.sum(1 - label, dim=2)
        denominator = denominator.masked_fill_(~(denominator > 0), 1)
        spec_part = spec_part.div_(denominator)
        spec_label = torch.ones_like(spec_part)
        spec_loss = self.cls_criterion(
            spec_part,
            spec_label,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        precision_part = torch.sum(cls_score * vtarget, dim=2)
        denominator = torch.sum(cls_score, dim=2)
        denominator = denominator.masked_fill_(~(denominator > 0), 1)
        precision_part = precision_part.div_(denominator)
        precision_label = torch.ones_like(precision_part)
        precision_loss = self.cls_criterion(
            precision_part,
            precision_label,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        global_term = recall_loss + spec_loss + precision_loss

        loss_cls = self.loss_weight * (unary_term + global_term)
        return loss_cls

@LOSSES.register_module()
class AffinityLoss2(nn.Module):
    """CrossEntropyLoss.
    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(AffinityLoss2, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        # self.cls_criterion = binary_cross_entropy
        self.cls_criterion = binary_cross_entropy_affinity

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        unary_term = self.cls_criterion(
            cls_score,
            label,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        # diagonal_matrix = (1 - torch.eye(label.size(1))).to(label.get_device())
        # vtarget = diagonal_matrix * label

        # recall_part = torch.sum(cls_score * vtarget, dim=2)
        # denominator = torch.sum(vtarget, dim=2)
        # denominator = denominator.masked_fill_(~(denominator > 0), 1)
        # recall_part = recall_part.div_(denominator)
        # recall_label = torch.ones_like(recall_part)
        # recall_loss = self.cls_criterion(
        #     recall_part,
        #     recall_label,
        #     reduction=reduction,
        #     avg_factor=avg_factor,
        #     **kwargs)

        # spec_part = torch.sum((1 - cls_score) * (1 - label), dim=2)
        # denominator = torch.sum(1 - label, dim=2)
        # denominator = denominator.masked_fill_(~(denominator > 0), 1)
        # spec_part = spec_part.div_(denominator)
        # spec_label = torch.ones_like(spec_part)
        # spec_loss = self.cls_criterion(
        #     spec_part,
        #     spec_label,
        #     reduction=reduction,
        #     avg_factor=avg_factor,
        #     **kwargs)

        # precision_part = torch.sum(cls_score * vtarget, dim=2)
        # denominator = torch.sum(cls_score, dim=2)
        # denominator = denominator.masked_fill_(~(denominator > 0), 1)
        # precision_part = precision_part.div_(denominator)
        # precision_label = torch.ones_like(precision_part)
        # precision_loss = self.cls_criterion(
        #     precision_part,
        #     precision_label,
        #     reduction=reduction,
        #     avg_factor=avg_factor,
        #     **kwargs)

        # global_term = recall_loss + spec_loss + precision_loss

        # loss_cls = self.loss_weight * (unary_term + global_term)
        loss_cls = self.loss_weight * (unary_term )
        return loss_cls


@LOSSES.register_module()
class mseloss(nn.Module):
    """CrossEntropyLoss.
    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(mseloss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        # self.cls_criterion = binary_cross_entropy
        # self.cls_criterion = binary_cross_entropy_affinity

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        return self.loss_weight*F.mse_loss(cls_score,label)