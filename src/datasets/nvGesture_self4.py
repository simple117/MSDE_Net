import copy
import os

import torch

from torch.utils.data.dataset import Dataset

import numpy as np
import cv2

from src.datasets.utils.read_data import load_split_nvgesture, load_data_from_file
from src.datasets.utils.normals import normals_multi
from src.datasets.utils.normalize import normalize

from pathlib import Path

from tqdm import tqdm
import imgaug.augmenters as iaa

class NVGesture(Dataset):
    """NVGesture Dataset class"""
    def __init__(self, configer, path, split="train", data_type="depth", transforms=None, n_frames=40, optical_flow=False):
        """Constructor method for NVGesture Dataset class

        Args:
            configer (Configer): Configer object for current procedure phase (train, test, val)
            split (str, optional): Current procedure phase (train, test, val)
            data_type (str, optional): Input data type (depth, rgb, normals, ir)
            transform (Object, optional): Data augmentation transformation for every data
            n_frames (int, optional): Number of frames selected for every input clip
            optical_flow (bool, optional): Flag to choose if calculate optical flow or not

        """
        super().__init__()

        print("Loading NVGestures {} dataset...".format(split.upper()), end=" ")

        self.dataset_path = Path(path)
        self.split = split
        self.data_type = data_type
        self.transforms = transforms
        self.optical_flow = optical_flow
        if self.data_type in ["normal", "normals"] and self.optical_flow:
            raise NotImplementedError("Optical flow for normals image is not supported.")

        # 加载视频流
        self.data_list=list()
        if self.split == "train":
            path_open=os.path.join(path,self.split)
            files=os.listdir(path_open)
            for i in range(len(files)):
                for x in tqdm(os.listdir(os.path.join(path_open,files[i]))):
                    self.data_list.append((os.path.join(path_open,files[i],x),i))
                # break
        elif self.split == "test" or self.split == "val":
            path_open=os.path.join(path,'test')
            files=os.listdir(path_open)
            for i in range(len(files)):
                for x in tqdm(os.listdir(os.path.join(path_open,files[i]))):
                    self.data_list.append((os.path.join(path_open,files[i],x),i))
                # break

        # 手动更改
        skeleton_path=r'D:\datasets\NVGesture_skeleton'

        # 加骨骼点流
        self.skeleton_list=list()
        if self.split == "train":
            path_open=os.path.join(skeleton_path,self.split)
            files=os.listdir(path_open)
            for i in range(len(files)):
                for x in tqdm(os.listdir(os.path.join(path_open,files[i]))):
                    self.skeleton_list.append((os.path.join(path_open,files[i],x),i))
                # break
        elif self.split == "test" or self.split == "val":
            path_open=os.path.join(skeleton_path,'test')
            files=os.listdir(path_open)
            for i in range(len(files)):
                for x in tqdm(os.listdir(os.path.join(path_open,files[i]))):
                    self.skeleton_list.append((os.path.join(path_open,files[i],x),i))

        # self.data_list+=[[] for i in range(len(self.data_list))]

        if self.data_type in ["depth_z", "depth", "normal", "normals"]:
            self.sensor = "depth"
        elif self.data_type == "wrapped":
            self.sensor = "wrapped"
        elif self.data_type in ["rgb", "color"]:
            self.sensor = "color"
        elif self.data_type == "ir":
            self.sensor = "duo_left"
        else:
            raise NotImplementedError
        print("done.")



    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        np_path,label=self.data_list[idx]
        flag=False
        if idx>=len(self.data_list)//2:
            idx-=len(self.data_list)//2
            flag=True

        skeleton_path,label=self.skeleton_list[idx]

        data=np.load(np_path)
        skeleton_data=np.load(skeleton_path)

        if flag:
            skeleton_data=skeleton_data[[*range(1, skeleton_data.shape[0], 2)],... ]
        else:
            skeleton_data=skeleton_data[[*range(0, skeleton_data.shape[0], 2)],... ]

        
        data = torch.from_numpy(data)
        skeleton_data=torch.from_numpy(skeleton_data)



        label = torch.LongTensor(np.asarray([label]))
        # data:[40,192,256]
        return data.float(), skeleton_data, label