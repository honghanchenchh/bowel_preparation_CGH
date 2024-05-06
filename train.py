import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.utils.data import Dataset,DataLoader
from torchvision.models import resnet50, ResNet50_Weights

import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

cudnn.benchmark = True
#plt.ion()   # interactive mode

train_transforms=transforms.Compose([
        transforms.Resize(576),
        transforms.RandomRotation(degrees=(0,30)),
        transforms.RandomResizedCrop((512,512)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
val_transforms=transforms.Compose([
        transforms.Resize((512,512)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class cleanClassification(Dataset):
    def __init__(self, data_filelist, transform=None):
        self.transform = transform
        self.data_filelist=data_filelist
        self.info=[]
        self.img_paths=[]
        self.labels=[]
        self.transform=None
        if transform!=None:
            self.transform=transform

        # create the csv file if necessary
        with open(self.data_filelist, "r", encoding="utf-8") as f:
            self.info = f.readlines()
        for img_info in self.info:
            img_path, label = img_info.strip().split('\t')
            self.img_paths.append(img_path)
            self.labels.append(int(label))

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path)

        if img.mode != 'RGB':
            img = img.convert('RGB')
        '''
        img = img.resize((224, 224), Image.BILINEAR)
        img = np.array(img).astype('float32')
        img = img.transpose((2, 0, 1)) / 255
        '''
        img = self.transform(img)
        label = self.labels[index]
        label = np.array(label, dtype="int64")
        ret_path = os.path.basename(img_path)
        return img, label, ret_path

    def print_sample(self, index: int = 0):
        print("文件名", self.img_paths[index], "\t标签值", self.labels[index])

    def __len__(self):
        return len(self.img_paths)

train_set=cleanClassification('/home/hogan/data1/chh/clean_exp/code_classification/data_list/train_1.txt',transform=train_transforms)
train_loader=DataLoader(train_set,shuffle=True,batch_size=32)
train_set.print_sample(10)
val_set=cleanClassification('/home/hogan/data1/chh/clean_exp/code_classification/data_list/val_1.txt',transform=val_transforms)
val_loader=DataLoader(val_set,shuffle=False,batch_size=32)
val_set.print_sample(10)
dataloaders={
    'train':train_loader,
    'val':val_loader
}
dataset_sizes={
    'train':len(train_loader),
    'val':len(val_loader)
}

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    tempdir='/home/hogan/data1/chh/clean_exp/code_classification'
    best_model_params_path = os.path.join(tempdir, 'resnet50_1.pth')

    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels,paths in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path))
    return model



device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model_ft = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.01)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=50)
