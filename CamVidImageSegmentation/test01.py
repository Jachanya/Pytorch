import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import dset
import modelUNET
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import GPUtil
from util import IoU

torch.cuda.empty_cache()
GPUtil.showUtilization()
model = modelUNET.UNET(3, 32, 4).to(device = 'cuda:0')
PATH = 'model_parameter/trial11'
model.load_state_dict(torch.load(PATH))
model.eval()

with torch.no_grad():
	cvd = dset.CamVidDataset('val')
	torch.cuda.empty_cache()
	input_, _, target = cvd[56]
	input_ = input_.to(device = 'cuda:0')
	print(target.shape)
	out = model(input_.unsqueeze(0))

	data = CamVidDataset.indextocolor(out.cpu())
	print(f'accuracy {IoU(out.cpu(),_.unsqueeze(0)).mean()}')
	#print(_.shape)
	#target = CamVidDataset.indextocolor(target.unsqueeze(0).permute(2,0,1).cpu())
	print(target.unsqueeze(0).permute(1,2,0).shape)
	model = torch.argmax(out, dim = 1)
	model = model.to(device = 'cpu')
	


	plt.subplot(311)
	plt.imshow(model.permute(1,2,0))
	plt.title('Output')
	plt.subplot(312)
	plt.title('Target')
	plt.imshow(target.unsqueeze(0).permute(1,2,0))
	plt.subplot(313)
	plt.imshow(data.astype(np.uint8))
	plt.title('Output')
	plt.show()