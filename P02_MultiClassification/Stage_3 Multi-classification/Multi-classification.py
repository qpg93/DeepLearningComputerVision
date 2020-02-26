import os
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import random

from Multi_Network import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch import optim
from torch.optim import lr_scheduler
import copy

CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(os.path.dirname(CURRENT_DIR), 'Dataset')
TRAIN_DIR = os.path.join(ROOT_DIR, 'train')
VAL_DIR = os.path.join(ROOT_DIR, 'val')
TRAIN_ANNO = os.path.join(CURRENT_DIR, 'Multi_train_annotation.csv')
VAL_ANNO = os.path.join(CURRENT_DIR, 'Multi_val_annotation.csv')
CLASSES = ['Mammals', 'Birds']
SPECIES = ['rabbits', 'rats', 'chickens']

class MyDataset():

    def __init__(self, root_dir, annotations_file, transform=None):
        self.root_dir = root_dir
        self.annotations_file = annotations_file
        self.transform = transform
    
        if not os.path.isfile(self.annotations_file):
            print(self.annotations_file + ' does not exist!')
        self.file_info = pd.read_csv(annotations_file, index_col=0)
        self.size = len(self.file_info)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.file_info['path'][idx]
        if not os.path.isfile(image_path):
            print(image_path + ' does not exist!')
            return None
        
        image = Image.open(image_path).convert('RGB')
        label_class = int(self.file_info.iloc[idx]['classes'])
        label_species = int(self.file_info.iloc[idx]['species'])
        sample = {'image':image, 'classes':label_class, 'species':label_species}
        
        if self.transform:
            sample['image'] = self.transform(image)
    
        return sample


def getloader():
    train_transforms = transforms.Compose([transforms.Resize((500, 500)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])

    test_transforms = transforms.Compose([transforms.Resize((500, 500)),
                                        transforms.ToTensor()])

    train_dataset = MyDataset(root_dir=ROOT_DIR,
                            annotations_file=TRAIN_ANNO,
                            transform=train_transforms)

    test_dataset = MyDataset(root_dir=ROOT_DIR,
                            annotations_file=VAL_ANNO,
                            transform=test_transforms)

    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset)
    data_loaders = {'train':train_loader, 'val':test_loader}

    return data_loaders

def visualize_dataset(train_loader):
    print(len(train_loader.dataset))
    idx = random.randint(0, len(train_loader.dataset))
    sample = train_loader.dataset[idx]
    print('index: ' + idx,
            'size: ' + sample['image'].shape,
            'classes: '+ CLASSES[sample['classes'],
            'species: ' + SPECIES[sample['species']]])
    img = sample['image']
    plt.imshow(transforms.ToPILImage()(img))
    plt.show()


def train_model(model, data_loaders, method="average", num_epoches=30,
                save_model_name="Best_model_weight1.5_lr0.01.pt",
                save_fig_name="Accuracy_vs_epoch_weight1.5_lr0.01.png"):
    # Only accuracy is stored for simplicity
    Accuracy_list_classes = {'train':[], 'val':[]}
    Accuracy_list_species = {'train':[], 'val':[]}

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    criterion_classes = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).cuda())
    criterion_species = nn.CrossEntropyLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epoches):
        print('Epoch {}/{}'.format(epoch + 1, num_epoches))
        print('-*'*10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            correct_classes = 0.0
            correct_species = 0.0

            for idx, data in enumerate(data_loaders[phase]):
                inputs = data['image'].cuda()
                label_classes = torch.tensor(data['classes']).cuda()
                label_species = torch.tensor(data['species']).cuda()
                optimizer.zero_grad()
                weight = torch.ones(len(inputs)).cuda()

                with torch.set_grad_enabled(phase == 'train'):
                    x_classes, x_species = model(inputs)
                    x_classes = x_classes.view(-1, 2)
                    x_species = x_species.view(-1, 3)
                    
                    _, preds_classes = torch.max(x_classes, 1)
                    _, preds_species = torch.max(x_species, 1)

                    correct_classes += torch.sum(preds_classes == label_classes)
                    correct_species += torch.sum(preds_species == label_species)

                    if phase == 'val':
                        print(list(zip(preds_classes, label_classes)))
                    
                    if phase == 'train':
                        loss1 = criterion_classes(x_classes, label_classes)
                        loss2 = criterion_species(x_species, label_species)
                        loss = loss1 + loss2 * 1.5
                        #loss = loss1 + loss2
                        print('Epoch {}, phase {}, batch {}, loss_c={}, loss_s={}'.format(
                            epoch, phase, idx, loss1, loss2
                        ))
                        loss.backward()
                        optimizer.step()
                        #scheduler.step()

            correct_classes = 100 * correct_classes.double() / len(data_loaders[phase].dataset)
            correct_species = 100 * correct_species.double() / len(data_loaders[phase].dataset)
            Accuracy_list_classes[phase].append(correct_classes)
            Accuracy_list_species[phase].append(correct_species)
            print('Epoch {}, phase={}, classes acc={}, species acc={}'.format(
                epoch, phase, correct_classes, correct_species
            ))
            epoch_acc = correct_classes + correct_species

            if phase == 'val' and (epoch_acc > best_acc):
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_model_wts)
    torch.save(model, CURRENT_DIR + '/' + save_model_name)
    
    # Save figure
    x = range(num_epoches)
    plt.plot(x, Accuracy_list_classes['train'], linestyle="-", marker=".", linewidth=1, label="train classes")
    plt.plot(x, Accuracy_list_classes['val'], linestyle="-", marker=".", linewidth=1, label="test classes")
    plt.plot(x, Accuracy_list_species['train'], linestyle="-", marker=".", linewidth=1, label="train species")
    plt.plot(x, Accuracy_list_species['val'], linestyle="-", marker=".", linewidth=1, label="test species")
    plt.legend()
    plt.savefig(CURRENT_DIR + '/' + save_fig_name)
    plt.close('all')


def train_model_twostep(model, data_loaders, num_epoches=30, num_epoches1 = 20,
                        save_model_name="Best_model_twostep_20_10.pt",
                        save_fig_name="Accuracy_vs_epoch_twostep_20_10.png"):
    """
    Only accuracy is stored for simplicity
    This does species first for num_epoches1, then classes
    """
    Accuracy_list_classes = {'train':[], 'val':[]}
    Accuracy_list_species = {'train':[], 'val':[]}

    criterion_classes = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0])).cuda()
    criterion_species = torch.nn.CrossEntropyLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0

    for epoch in range(num_epoches):
        print('Epoch {}/{}'.format(epoch + 1, num_epoches))
        print('-*'*10)

        if epoch == 0:
            model.fc2_classes.weight.requires_grad = False
            model.fc2_classes.bias.requires_grad = False
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            #scheduler = lr_scheduler.StepLR(optimer, step_size=1, gamma=0.5)
        
        if epoch == num_epoches1:
            for para in model.parameters():
                para.requires_grad = False
            model.fc2_classes.weight.requires_grad = True
            model.fc2_classes.bias.requires_grad = True
            optimizer = optim.SGD(model.fc2_classes.parameters(), lr=0.01, momentum=0.9)
            #scheduler = lr_scheduler.StepLR(optimer, step_size=1, gamma=0.5)

        # Each epoch has its training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()
            for layer in model.children():
                print(layer)
                print(list(layer.parameters()))
            
            correct_classes = 0
            correct_species = 0

            for idx, data in enumerate(data_loaders[phase]):
                inputs = data['image'].cuda()
                label_classes = torch.tensor(data['classes']).cuda()
                label_species = torch.tensor(data['species']).cuda()
                optimizer.zero_grad()
                weight = torch.ones(len(label_classes))

                with torch.set_grad_enabled(phase == 'train'):
                    x_classes, x_species = model(inputs)
                    x_classes = x_classes.view(-1, 2)
                    x_species = x_species.view(-1, 3)

                    _, preds_classes = torch.max(x_classes, 1)
                    _, preds_species = torch.max(x_species, 1)

                    correct_classes += torch.sum(preds_classes == label_classes)
                    correct_species += torch.sum(preds_species == label_species)

                    if phase == 'val':
                        print(list(zip(preds_classes, label_classes)))
                    
                    if phase == 'train':
                        loss1 = criterion_classes(x_classes, label_classes)
                        loss2 = criterion_species(x_species, label_species)

                        # If we want faster speed, just keep one loss
                        if epoch < num_epoches1:
                            loss = loss2
                        else:
                            loss = loss1

                        print('Epoch {}, phase {}, batch {}, loss_c={}, loss_s={}'.format(
                            epoch, phase, idx, loss1, loss2
                        ))
                        loss.backward()
                        optimizer.step()
                        #scheduler.step()

            correct_classes = 100 * correct_classes.double() / len(data_loaders[phase].dataset)
            correct_species = 100 * correct_species.double() / len(data_loaders[phase].dataset)
            Accuracy_list_classes[phase].append(correct_classes)
            Accuracy_list_species[phase].append(correct_species)
            print('Epoch {}, phase={}, classes acc={}, species acc={}'.format(
                epoch, phase, correct_classes, correct_species
            ))
            epoch_acc = correct_classes + correct_species

            if phase == 'val' and (epoch_acc > best_acc):
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_model_wts)
    torch.save(model, CURRENT_DIR + '/' + save_model_name)
    
    # Save figure
    x = range(num_epoches)
    plt.plot(x, Accuracy_list_classes['train'], linestyle="-", marker=".", linewidth=1, label="train classes")
    plt.plot(x, Accuracy_list_classes['val'], linestyle="-", marker=".", linewidth=1, label="test classes")
    plt.plot(x, Accuracy_list_species['train'], linestyle="-", marker=".", linewidth=1, label="train species")
    plt.plot(x, Accuracy_list_species['val'], linestyle="-", marker=".", linewidth=1, label="test species")
    plt.legend()
    plt.savefig(CURRENT_DIR + '/' + save_fig_name)
    plt.close('all')


if __name__ == "__main__":
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    """

    data_loaders = getloader()
    model = Net().cuda()
    
    """
    train_model(model, data_loaders,
                save_model_name="Best_model_weight1.5_lr0.01.pt",
                save_fig_name="Accuracy_vs_epoch_weight1.5_lr0.01.png")
    """
    
    print("30 epoches in total: 25 then 5")
    train_model_twostep(model, data_loaders, num_epoches=30, num_epoches1=12,
                        save_model_name="Best_model_twostep_12_18.pt",
                        save_fig_name="Accuracy_vs_epoch_twostep_12_18.png")
    