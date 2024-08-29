from fastai.vision.all import *
from torch.distributions.dirichlet import Dirichlet
import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight

# can deal with balanced datasets, no label_smoothing technique
class EnsembleClassificationLoss(Module):
    def __init__(self, train=True, reduction='mean', is_logits=True, num_classes=-1):
        store_attr()

    def __call__(self, *args, **kwargs):
        _pred = args[0]
        _tgts = args[1]

        # there should be no need to return both the logits and softmax-activated outputs (as in the original code)
        if self.is_logits:
            _loss = torch.stack([F.cross_entropy(p, _tgts, reduction=self.reduction) for p in _pred])
        else:
            # should be equivalent, but more elegant
            _loss = torch.stack([F.nll_loss(torch.log(p+1e-15), _tgts, reduction=self.reduction) for p in _pred])

            if self.reduction != 'mean':
                warn(f"In EnsembleClassificationLoss: Unsupported reduction method for is_logits={self.is_logits} ('{self.reduction}')!")
            #_loss = (-(F.one_hot(_tgts, num_classes=self.num_classes).expand_as(_pred)*torch.log(_pred+1e-15))).max(2).values.sum(1)/_tgts.shape[0]

            if ~_loss.isfinite().all():
                print(f"some elements of the loss are not finite: {_loss}\n preds: {_pred}")

        return _loss
    
    def decodes(self, out):
        return out.argmax(dim=-1, keepdim=False)

    def activation(self, out):
        return F.softmax(out, dim=-1)


class CrossEntropyClassificationLoss(Module):
    def __init__(self, train=True, reduction='mean', is_logits=True):
        store_attr()

    def __call__(self, *args, **kwargs):
        _pred = args[0]
        _tgts = args[1]

        if self.is_logits:
            _loss = F.cross_entropy(_pred, _tgts, reduction=self.reduction)
        else:
            _loss = F.nll_loss(torch.log(_pred+1e-15), _tgts, reduction=self.reduction)

            if self.reduction != 'mean':
                warn(f"In CrossEntropyClassificationLoss: Unsupported reduction method for is_logits={self.is_logits} ('{self.reduction}')!")

            if ~_loss.isfinite().all():
                print(f"some elements of the loss are not finite: {_loss}\n preds: {_pred}")

        return _loss
    
    def decodes(self, out):
        return out.argmax(dim=-1, keepdim=False)

    def activation(self, out):
        return F.softmax(out, dim=-1)
    

class UCELoss(Module):
    def __init__(self, output_dim:int, regr:float=1e-5, train=True, reduction='none'):
        store_attr()

    def __call__(self, *args, **kwargs):
        _alpha = args[0]
        _tgts = args[1]

        alpha_0 = _alpha.sum(1).unsqueeze(-1).repeat(1, self.output_dim)
        entropy_reg = Dirichlet(_alpha).entropy()
        UCE_loss = torch.sum(_tgts * (torch.digamma(alpha_0) - torch.digamma(_alpha))) - self.regr * torch.sum(entropy_reg)

        return UCE_loss

    def decodes(self, out):
        return out.argmax(dim=-1, keepdim=False)

    def activation(self, out):
        return F.softmax(out, dim=-1)


#weighted version, can deal with balanced datasets, with label_smoothing technique
class WeightedEnsembleClassificationLoss(Module):
    def __init__(self, train=True, reduction='mean', is_logits=True, num_classes=-1, label_smoothing=0.1, weight=None):
        store_attr()

    def __call__(self, *args, **kwargs):
        _pred = args[0]
        _tgts = args[1]

        if self.is_logits:
            if self.label_smoothing > 0:
                smooth_targets = torch.zeros_like(_pred[0]).scatter_(1, _tgts.unsqueeze(1), 1)
                smooth_targets = smooth_targets * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
                _loss = torch.stack([F.cross_entropy(p, smooth_targets, weight=self.weight, reduction=self.reduction) for p in _pred])
            else:
                _loss = torch.stack([F.cross_entropy(p, _tgts, weight=self.weight, reduction=self.reduction) for p in _pred])
        else:
            if self.label_smoothing > 0:
                raise ValueError("Label smoothing should be applied to logits, not softmax outputs.")
            _loss = torch.stack([F.nll_loss(torch.log(p + 1e-15), _tgts, weight=self.weight, reduction=self.reduction) for p in _pred])

        if self.reduction != 'mean':
            warn(f"In WeightedEnsembleClassificationLoss: Unsupported reduction method for is_logits={self.is_logits} ('{self.reduction}')!")

        if ~_loss.isfinite().all():
            print(f"Some elements of the loss are not finite: {_loss}\nPreds: {_pred}")

        return _loss

    def decodes(self, out):
        return out.argmax(dim=-1, keepdim=False)

    def activation(self, out):
        return F.softmax(out, dim=-1)


class WeightedCrossEntropyClassificationLoss(Module):
    def __init__(self, train=True, reduction='mean', is_logits=True, num_classes=-1, label_smoothing=0.0, weight=None):
        store_attr()

    def __call__(self, *args, **kwargs):
        _pred = args[0]
        _tgts = args[1]

        if self.is_logits:
            if self.label_smoothing > 0:
                # create label_smoothing
                smooth_targets = torch.zeros_like(_pred).scatter_(1, _tgts.unsqueeze(1), 1)
                smooth_targets = smooth_targets * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
                _loss = F.cross_entropy(_pred, smooth_targets, weight=self.weight, reduction=self.reduction)
            else:
                _loss = F.cross_entropy(_pred, _tgts, weight=self.weight, reduction=self.reduction)
        else:
            if self.label_smoothing > 0:
                raise ValueError("Label smoothing should be applied to logits, not softmax outputs.")
            _loss = F.nll_loss(torch.log(_pred+1e-15), _tgts, weight=self.weight, reduction=self.reduction)

        if self.reduction != 'mean':
            warn(f"In WeightedCrossEntropyClassificationLoss: Unsupported reduction method for is_logits={self.is_logits} ('{self.reduction}')!")

        if not _loss.isfinite().all():
            print(f"some elements of the loss are not finite: {_loss}\n preds: {_pred}")

        return _loss

    def decodes(self, out):
        return out.argmax(dim=-1, keepdim=False)

    def activation(self, out):
        return F.softmax(out, dim=-1)

class WeightedUCELoss(Module):
    def __init__(self, output_dim:int, regr:float=1e-5, train=True, reduction='none', label_smoothing=0.0, weight=None):
        # set reduction = "none" , since additional complexity penalty terms need to be introduced into the loss function,
        # help to maintain the independence and detail of each sample loss.
        store_attr()
        self.weight = weight if weight is not None else torch.ones(output_dim)
        if not isinstance(self.weight, torch.Tensor):
            self.weight = torch.tensor(self.weight)
        self.weight = self.weight / self.weight.sum()

    def __call__(self, *args, **kwargs):
        _alpha = args[0]
        _tgts = args[1]

        if self.label_smoothing > 0:
            _tgts = _tgts * (1 - self.label_smoothing) + self.label_smoothing / self.output_dim

        alpha_0 = _alpha.sum(1).unsqueeze(-1).repeat(1, self.output_dim)
        entropy_reg = Dirichlet(_alpha).entropy()

        weighted_loss = self.weight * (_tgts * (torch.digamma(alpha_0) - torch.digamma(_alpha)))
        UCE_loss = torch.sum(weighted_loss) - self.regr * torch.sum(entropy_reg)

        return UCE_loss

    def decodes(self, out):
        return out.argmax(dim=-1, keepdim=False)

    def activation(self, out):
        return F.softmax(out, dim=-1)


# Focal loss version, another weighted version, with label_smoothing technique
class FocalEnsembleClassificationLoss(Module):
    def __init__(self, train=True, reduction='mean', is_logits=True, num_classes=-1, label_smoothing=0.1, gamma=2.0, alpha=None):
        store_attr()
        self.alpha = alpha if alpha is not None else torch.ones(num_classes)

    def focal_loss(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

    def __call__(self, *args, **kwargs):
        _pred = args[0]
        _tgts = args[1]

        if self.is_logits:
            if self.label_smoothing > 0:
                smooth_targets = torch.zeros_like(_pred[0]).scatter_(1, _tgts.unsqueeze(1), 1)
                smooth_targets = smooth_targets * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
                _loss = torch.stack([self.focal_loss(p, smooth_targets) for p in _pred])
            else:
                _loss = torch.stack([self.focal_loss(p, _tgts) for p in _pred])
        else:
            if self.label_smoothing > 0:
                raise ValueError("Label smoothing should be applied to logits, not softmax outputs.")

            # Apply focal loss concept to nll_loss for non-logits input
            nll_losses = [F.nll_loss(torch.log(p + 1e-15), _tgts, reduction='none', weight=self.alpha) for p in _pred]
            _loss = torch.stack([(1 - torch.exp(-n)) ** self.gamma * n for n in nll_losses])

            if self.reduction == 'mean':
                _loss = _loss.mean()
            elif self.reduction == 'sum':
                _loss = _loss.sum()

        if self.reduction != 'mean' and self.reduction != 'sum':
            warn(f"In FocalEnsembleClassificationLoss: Unsupported reduction method for is_logits={self.is_logits} ('{self.reduction}')!")

        if ~_loss.isfinite().all():
            print(f"Some elements of the loss are not finite: {_loss}\nPreds: {_pred}")

        return _loss

    def decodes(self, out):
        return out.argmax(dim=-1, keepdim=False)

    def activation(self, out):
        return F.softmax(out, dim=-1)

class FocalCrossEntropyClassificationLoss(Module):
    def __init__(self, train=True, reduction='mean', is_logits=True, num_classes=-1, label_smoothing=0.0, gamma=2.0, alpha=None):
        store_attr()
        self.alpha = alpha if alpha is not None else torch.ones(num_classes)

    def focal_loss(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

    def __call__(self, *args, **kwargs):
        _pred = args[0]
        _tgts = args[1]

        if self.is_logits:
            if self.label_smoothing > 0:
                smooth_targets = torch.zeros_like(_pred).scatter_(1, _tgts.unsqueeze(1), 1)
                smooth_targets = smooth_targets * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
                _loss = self.focal_loss(_pred, smooth_targets)
            else:
                _loss = self.focal_loss(_pred, _tgts)
        else:
            if self.label_smoothing > 0:
                raise ValueError("Label smoothing should be applied to logits, not softmax outputs.")

            # Apply focal loss concept to nll_loss for non-logits input
            nll_loss = F.nll_loss(torch.log(_pred + 1e-15), _tgts, reduction='none', weight=self.alpha)
            pt = torch.exp(-nll_loss)
            _loss = (1 - pt) ** self.gamma * nll_loss

        if self.reduction == 'mean':
            _loss = _loss.mean()
        elif self.reduction == 'sum':
            _loss = _loss.sum()
        else:
            warn(f"In FocalCrossEntropyClassificationLoss: Unsupported reduction method for is_logits={self.is_logits} ('{self.reduction}')!")

        if not _loss.isfinite().all():
            print(f"some elements of the loss are not finite: {_loss}\n preds: {_pred}")

        return _loss

    def decodes(self, out):
        return out.argmax(dim=-1, keepdim=False)

    def activation(self, out):
        return F.softmax(out, dim=-1)

class FocalUCELoss(Module):
    def __init__(self, output_dim:int, regr:float=1e-5, train=True, reduction='mean', label_smoothing=0.0, gamma=2.0, alpha=None):
        super().__init__()
        self.output_dim = output_dim
        self.regr = regr
        self.train = train
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.gamma = gamma
        self.alpha = alpha if alpha is not None else torch.ones(output_dim)
        self.alpha = self.alpha / self.alpha.sum()

    def forward(self, _alpha, _tgts):
        if self.label_smoothing > 0:
            _tgts = _tgts * (1 - self.label_smoothing) + self.label_smoothing / self.output_dim

        alpha_0 = _alpha.sum(1, keepdim=True)
        entropy_reg = Dirichlet(_alpha).entropy()

        log_probs = torch.digamma(alpha_0) - torch.digamma(_alpha)
        probs = _alpha / alpha_0
        focal_weights = (1 - probs) ** self.gamma

        weighted_loss = self.alpha * focal_weights * (_tgts * log_probs)
        UCE_loss = weighted_loss.sum(dim=1) - self.regr * entropy_reg

        if self.reduction == 'mean':
            return UCE_loss.mean()
        elif self.reduction == 'sum':
            return UCE_loss.sum()
        elif self.reduction == 'none':
            return UCE_loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}. Supported modes are 'mean', 'sum', and 'none'.")

    def decodes(self, out):
        return out.argmax(dim=-1, keepdim=False)

    def activation(self, out):
        return F.softmax(out, dim=-1)


# get weights for each datasets
def calculate_weights(labels):
    labels_np = labels.numpy()
    class_weights = compute_class_weight('balanced', classes=np.unique(labels_np), y=labels_np)
    return torch.FloatTensor(class_weights)

def get_labels_and_weights(ds_name: str, data: DataLoaders) -> tuple:
    labels = torch.tensor([data.train_ds[i][1] for i in range(len(data.train_ds))])
    weights = calculate_weights(labels)
    return labels, weights

# Weights for each dataset
# the current weights maybe too extreme, you can also update with more soft weights calculation method
cifar10_weights = tensor([0.9953, 1.0000, 0.9983, 0.9983, 0.9965, 1.0005, 1.0005, 0.9973, 1.0050, 1.0086]) # can be seen as balanced datasets, "None"
cifar10_2_weights = tensor([0.9975, 1.0076, 1.0323, 0.9709, 1.0230, 1.0101, 0.9901, 0.9877, 0.9913, 0.9926]) # can be seen as balanced datasets, "None"
dr_weights = tensor([0.2720, 2.8703, 1.3397, 7.9267, 9.7232])
domainnet_weights = None # balanced datasets
ham_weights = tensor([ 4.2777,  2.7909,  1.3403, 12.7143,  1.2803,  0.2127,  9.9503])
class MixUpLoss(nn.Module):
    "Adapts the loss function to go with mixup."

    def __init__(self, crit):
        super().__init__()
        self.crit = crit

    def forward(self, output, target):
        if not len(target.size()) == 2:
            return self.crit(output, target).mean()
        loss1, loss2 = self.crit(output, target[:, 0].long()), self.crit(output, target[:, 1].long())
        return (loss1 * target[:, 2] + loss2 * (1 - target[:, 2])).mean()
