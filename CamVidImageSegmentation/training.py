import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import dset
import modelUNET
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import GPUtil
from util import IoU, bce_dice_loss,dice_coef_metric,DiceBCELoss,TverskyLoss
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment16')
torch.cuda.empty_cache()
model = modelUNET.UNET(3, 32, 4).to(device = 'cuda:0')
#PATH = 'model_parameter/trial11'
#model.load_state_dict(torch.load(PATH))
#model = copiedUNET.UNet(3, 32,5, 6, padding = True).to(device = 'cuda')

cvd = dset.CamVidDataset()
cvd_dataloader = DataLoader(cvd, batch_size = 1, shuffle = True)
#optimizer = optim.SGD(model.parameters(), lr =0.015, momentum = 0.9)

cvd_val = dset.CamVidDataset('val')
cvd_val_dataloader = DataLoader(cvd_val, batch_size = 1, shuffle = True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
GPUtil.showUtilization()
n_epoch = 300
torch.cuda.empty_cache()
for epoch in range(n_epoch):
	total_loss = 0.0
	total_accuracy =0.0

	for data in cvd_dataloader:
		input_, _, target = data

		input_ = input_.to(device = 'cuda:0')
		target = target.to(device= 'cuda:0', dtype = torch.long)
		mask = _.to(device= 'cuda:0', dtype = torch.long)

		output = model(input_)

		optimizer.zero_grad()
		loss = loss_fn(output, target)
		loss.backward()
		optimizer.step()

		total_loss += loss.item()
		with torch.no_grad():
			if epoch == 0 or epoch % 10 == 0:
				accuracy = IoU(output.detach(), mask.detach()).mean()
				total_accuracy += accuracy
		torch.cuda.empty_cache()
	'''
	with torch.no_grad():
		for data in cvd_val_dataloader:
			inputs, mask, target = data
			inputs = inputs.to(device = 'cuda')
			mask = mask.to(device = 'cuda', dtype = torch.float32)

			output = model(inputs)
			loss = bce_dice_loss(output, mask.permute(0,3,1,2))
			total_val_loss += loss.item()
			if epoch == 0 or epoch % 10 == 0:
				accuracy = dice_coef_metric(output.detach(), mask.detach().permute(0,3,1,2))
				total_val_accuracy += accuracy

	'''

	print(f"epoch: {epoch} , total_loss: {total_loss/len(cvd_dataloader)}")
	
	writer.add_scalar('Loss/train', total_loss/len(cvd_dataloader), epoch)
	#writer.add_scalar('Loss/val', total_val_loss/len(cvd_val_dataloader),epoch)
	

	PATH = 'model_parameter/trial11'
	if epoch == 0 or epoch % 9 == 0:
		torch.save(model.state_dict(), PATH)
		print(f"epoch: {epoch} , total_accuracy: {total_accuracy/len(cvd_dataloader)}")
		writer.add_scalar('acc/train', total_accuracy/len(cvd_dataloader), epoch)
		#writer.add_scalar('acc/val', total_val_accuracy/len(cvd_val_dataloader),epoch)

