import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import GPUtil
#GPUtil.showUtilization()
class UNET(nn.Module):
	def __init__(self, input_feature, n_classes, no_of_block):
		super(UNET, self).__init__()
		self.BN1 = nn.BatchNorm2d(input_feature)
		
		self.DownConv = nn.ModuleList()
		self.output_feature = 64
		self.convKernel = 3
		self.no_of_block = no_of_block

		#DownConvolution
		self.DownConv.append(BlockConv(input_feature, self.output_feature))
		for n in range(no_of_block):
			input_feature = self.output_feature
			self.output_feature = self.output_feature * 2
			self.DownConv.append(BlockConv(input_feature, self.output_feature))

		self.input_feature = self.output_feature
		self.output_feature = self.input_feature // 2

		#UpConvolutions
		self.UpConv = nn.ModuleList()
		for n in range(no_of_block):
			self.UpConv.append(nn.ConvTranspose2d(self.input_feature, self.output_feature, 2, 2))
			self.UpConv.append(BlockConv(self.input_feature, self.output_feature))
			self.input_feature = self.output_feature
			self.output_feature = self.input_feature // 2
		#Output
		self.output = nn.Conv2d(self.input_feature, n_classes, 1)

		self.init_weights()

	def init_weights(self):
		for m in self.modules():
			if type(m) in {nn.Conv2d,nn.ConvTranspose2d,}:
				nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu',)
				if m.bias is not None:
					fan_in, fan_out = \
						nn.init._calculate_fan_in_and_fan_out(m.weight.data)
					bound = 1 / math.sqrt(fan_out)
					nn.init.normal_(m.bias, -bound, bound)


	def forward(self, x):
		x = self.BN1(x)
		concat_path = []
		#DownConvolutions
		#GPUtil.showUtilization()
		for i,op in enumerate(self.DownConv):
			x = op(x)
			if i < self.no_of_block:
				concat_path.append(x)
				x = F.max_pool2d(x, 2)
		i = 0
		#GPUtil.showUtilization()
		for op in self.UpConv[::2]:
			x = op(x)
			
			ct = self.center_crop(concat_path.pop(),x.shape[2:])
			x = torch.cat((x, ct), 1)
			x = self.UpConv[2*i + 1](x)
			i += 1
		x = self.output(x)
		return x

	def center_crop(self, layer, target_size):
		_, _, layer_height, layer_width = layer.size()
		diff_y = (layer_height - target_size[0]) // 2
		diff_x = (layer_width - target_size[1]) // 2
		return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

class BlockConv(nn.Module):
	def __init__(self, input_feature, output_feature):
		super(BlockConv, self).__init__()
		self.convKernel = 3
		self.conv1 = nn.Conv2d(input_feature, output_feature, self.convKernel, padding = 1)
		self.BN1 = nn.BatchNorm2d(output_feature)
		#self.dropout1 = nn.Dropout(0.8)
		self.conv2 = nn.Conv2d(output_feature, output_feature, self.convKernel, padding = 1)
		self.BN2 = nn.BatchNorm2d(output_feature)
		#self.dropout2 = nn.Dropout(0.8)

	def forward(self, x):
		#GPUtil.showUtilization()
		x = F.relu(self.conv1(x))
		x = self.BN1(x)
		x = F.relu(self.conv2(x))
		x = self.BN2(x)

		return x
    
if __name__ == "__main__":
	model = UNET(3, 32,5).to(device = 'cuda')
	cvd = CamVidDataset.CamVidDataset()
	image, _, _ = cvd[0]
	print(image.shape)
	x = torch.randn((1,3, 250,250)).to(device = 'cuda')
	y = model(x)
	print(y.shape)