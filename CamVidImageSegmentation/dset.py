from torch.utils.data import Dataset
import glob
from PIL import Image
from torchvision.transforms import transforms
import pandas as pd
import numpy as np
import torch
import random

def getdata(train_mode):
    """
    train_mode(str): 'test', 'train', 'val'
    """
    path = f'data/{train_mode}/*'
    labels_path = f'data/{train_mode}_labels/*'
    data = glob.glob(path)
    labels = glob.glob(labels_path)
    return data, labels

def Color2index(y):

    path = f'data/class_dict.csv'
    df = pd.read_csv(path)
    y = np.array(y)
    arr_col = np.array(df[['r', 'g', 'b']])

    y_ind = np.zeros((y.shape[0], y.shape[1]))
    y_cat = np.zeros((y.shape[0], y.shape[1], len(arr_col)))
    i = 1
    for i_color in arr_col:
        ind_i = np.where(
            (y[:,:, 0] == i_color[0]) 
            & (y[:,:, 1] == i_color[1])
            & (y[:,:, 2] == i_color[2])
        )
        y_ind[ind_i[0], ind_i[1]] = i - 1
        y_cat[ind_i[0], ind_i[1], i-1] = 1
        i += 1
    
    return y_cat, y_ind

def indextocolor(y):

    path = f'data/class_dict.csv'
    df = pd.read_csv(path)
    arr_col = np.array(df[['r', 'g', 'b']])
    y = torch.argmax(y, axis = 1)
    y = y.detach().numpy()
    y_img = np.zeros((y.shape[1], y.shape[2], 3))
    i = 0
    for i_color in arr_col:
        ind_i = np.where(y[0,:,:] == i)
        y_img[ind_i[0], ind_i[1], 0] = i_color[0]
        y_img[ind_i[0], ind_i[1], 1] = i_color[1]
        y_img[ind_i[0], ind_i[1], 2] = i_color[2]
        i += 1

    return y_img


class CamVidDataset(Dataset):
    def __init__(self, train_mode = 'train', transform = None):

        self.train_mode = train_mode
        self.path, self.labels_path = getdata(train_mode)
        self.transform = transform

    def __len__(self):
        return len(self.path)

    def __getitem__(self, ndx):

        data = Image.open(self.path[ndx])
        label = Image.open(self.labels_path[ndx])

        if self.transform:
            data = self.transform(data)
            label = self.transform(label)

        label_one_hot, label_ind = Color2index(label)
        t = transforms.ToTensor()
        data = t(data)
        label_ind = torch.from_numpy(label_ind)
        label_one_hot = torch.from_numpy(label_one_hot).permute(2,0,1)
        if self.train_mode == 'train':
            data, label_one_hot, label_ind = self.random_crop(data, label_one_hot, label_ind, 512, 512)
        return data, label_one_hot, label_ind

    def random_crop(self, img, label, ind, width, height):
        assert img.shape[1] >= height
        assert img.shape[2] >= width
        assert img.shape[1] == label.shape[1]
        assert img.shape[2] == label.shape[2]
        assert img.shape[1] == ind.shape[0]
        assert img.shape[2] == ind.shape[1]

        x = random.randint(0, img.shape[2] - width)
        y = random.randint(0, img.shape[1] - height)

        img = img[:, y:y+height, x:x+width]
        label = label[:, y:y + height, x:x+width]
        ind = ind[y:y + height, x:x+width]
        return img, label, ind

if __name__ == "__main__":
    
    print('Here')
    #transform = transforms.Compose([transforms.RandomCrop(572, 572)])
    cvd = CamVidDataset()
    img, label, ind = cvd[0]
    print(label.shape)