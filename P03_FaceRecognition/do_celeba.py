#Dataset is available at
#http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets

def show_landmark(image, landmark):
    """
    to display keypoints
    """
    # tensor is 4 dimensional (N x C x H x W), get rid of the 0th dimension (N)
    image = image.numpy().squeeze(0)
    # tensor is now (Channel，Height，Width)
    # change to (Height，Width，Channel)
    image = np.transpose(image, (1, 2, 0))

    # display
    plt.imshow(image)
    plt.scatter(landmark[0, :, 0], landmark[0, :, 1], s=10, marker='.', c='r')
    # pause, wait for update
    plt.pause(0.001)

    return None

class ColorAugment(object):
    def __init__(self, color_range=0.2):
        self.range = color_range
    
    def __call__(self, sample):
        image = sample['image']
        landmark = sample['landmark']
        # gamma correction (value_out = value_in ** gamma)
        image = transforms.functional.adjust_gamma(image, gamma=1)
        # saturation
        sat_factor = 1 + (self.range - self.range * 2 * np.random.rand())
        image = transforms.functional.adjust_saturation(image, sat_factor)

        return {'image': image, 'landmark': landmark}

class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmark = sample['image'], sample['landmark']
        h = image.height
        w = image.weight
        new_h, new_w = self.output_size
        # random vertical coordinate
        top = np.random.randint(0, h - new_h)
        # random horizontal coordinate
        left = np.random.randint(0, w - new_w)
        # crop with new coordinates
        image = image.crop((left, top, left + new_w, top + new_h))
        # keypoints adjustment
        landmark = landmark - [left, top]

        return {'image': image, 'landmark': landmark}

class ToTensor(object):
    def __call__(self, sample):
        image, landmark = sample['image'], sample['landmark']
        # turn to tensor
        image = transforms.ToTensor()(image)
        landmark = torch.from_numpy(landmark)
        return {'image': image, 'landmark': landmark}


class CelebADataset(Dataset):
    def __init__(self, anno, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # open anno file
        anno_file = open(anno, 'r')
        # read the file as list
        self.anno = anno_file.readlines()
        # remove the first 2 lines
        self.anno.pop(0)
        self.anno.pop(0)
        
        # save the paths into image_list, save the labels into landmark_list
        self.image_list = []
        self.landmark_list = []
        for item in self.anno:
            subitem = item.strip().split()
            self.image_list.append(subitem[0])
            landmark = list(map(int, subitem[1:]))
            self.landmark_list.append(landmark)

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        # combine the paths of root_dir and image
        image_name = os.path.join(self.root_dir, self.image_list[index])
        # read image
        image = Image.open(image_name)
        # get the corresponding label
        landmark = self.landmark_list[index]
        # turn to numpy array
        landmark = np.array(landmark)
        # reshape to matrix of (5,2), 5 key points
        landmark = landmark.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmark': landmark}

        if self.transform:
            sample = self.transform(sample)

        return sample

def dataset_main(args):
    # create a list of multiple transforms
    composed_transforms = transforms.Compose([RandomCrop(150),
                                            ColorAugment(),
                                            ToTensor()])
    # create CelebA dataset
    dataset = CelebADataset('list_landmarks_align_celeba.txt',
                            'img_align_celeba',
                            composed_transforms)
    # creat DataLoader of CelebA dataset, batch_size = 128
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    epoches = 1
    for epoc in range(0, epoches):
        for iteration, sample in enumerate(dataloader):
            # image
            image = sample['image']
            # keypoints
            landmark = sample['landmark']
            
            plt.figure()
            show_landmark(image, landmark)
            plt.show()
            plt.close('all')
    
    return None

if __name__ == "__init__":
    args = 0
    dataset_main(args)