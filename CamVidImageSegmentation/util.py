import torch
import torch.nn as nn
import torch.nn.functional as F
'''

def dice_coef(input_, target):
    input_ = torch.sigmoid(input_)
    smooth = 1e-5

    iflat = input_.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

def dice_loss(y_true, y_pred, numLabels):
    dice=0
    for index in range(numLabels):
        dice += dice_coef(y_pred[:,:,:,index], y_true[:,:,:,index] )
    return dice/numLabels

def dice_coef_metric(inputs, target):

	intersection = 2.0 * (target * inputs).sum()
	union = target.sum() + inputs.sum()
	if target.sum() == 0 and inputs.sum() == 0:
		return 1.0

	return intersection / union

def dice_coef_loss(inputs, target):
    num = target.size(0)
    inputs = inputs.reshape(num, -1)

    target = target.reshape(num, -1)

    smooth = 1.0
    intersection = (inputs * target)
    dice = (2. * intersection.sum(dim = 1) + smooth) / (inputs.sum(dim = 1) + target.sum(dim = 1) + smooth)
    dice = 1 - dice.sum() / num
    return dice

def bce_dice_loss(inputs, target):
    dicescore = dice_coef_loss(inputs, target)
    bcescore = nn.BCELoss()
    bceloss = bcescore(inputs, target)
    return bceloss + dicescore

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)
        #print(inputs.shape, targets.shape)      
        
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

ALPHA = 0.5
BETA = 0.5

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky
'''
def IoU(input_, target):
    #print(input_.shape, target.shape)
    ind_i = input_.argmax(dim=1)
    ep = 1e-5
    input_ = torch.zeros_like(input_)
    for i in range(input_.shape[1] -1):
        x = torch.where(ind_i == i)
        input_[x[0], i, x[1], x[2]] = 1
    input_ = input_.view(input_.shape[0], input_.shape[1], -1)
    target = target.permute(0,3,1,2).reshape(input_.shape[0], input_.shape[1], -1)
	#print(input_.shape, target.shape)

    TP = (input_ * target).sum(dim = 2)
    FP = input_.sum(dim = 2) - TP
    FN = target.sum(dim = 2) - TP
    return (TP + ep)/(TP + FP + FN + ep)
