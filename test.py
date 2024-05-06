import os
import datetime
import io
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch
from sklearn.metrics import classification_report,accuracy_score,recall_score,roc_curve,auc
from torcheval.metrics import BinaryAccuracy,BinaryRecall

transform = transforms.Compose([transforms.Resize(448),
                                        #transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
#####alex
#model_ft = models.alexnet(models.AlexNet_Weights.IMAGENET1K_V1)
#num_in_features=model_ft.classifier[6].in_features
#num_classes=2
#model_ft.classifier[6]=nn.Linear(num_in_features, num_classes,bias=True)
#####vgg19
#model_ft=models.vgg19(models.VGG19_Weights.IMAGENET1K_V1)
#num_in_features=model_ft.classifier[6].in_features
#num_classes=2
#model_ft.classifier[6]=nn.Linear(num_in_features, num_classes,bias=True)
#####ResNet50
#model_ft=models.resnet50(models.ResNet50_Weights.IMAGENET1K_V1)
#####ResNet101
#model_ft=models.resnet101(models.ResNet101_Weights.IMAGENET1K_V1)
#num_in_features=model_ft.fc.in_features
#num_classes=2
#model_ft.fc=nn.Linear(num_in_features, num_classes,bias=True)
#####efficientnet
#model_ft = models.efficientnet_v2_l(weights='IMAGENET1K_V1')
#num_in_features=model_ft.classifier[1].in_features
#num_classes=2
#model_ft.classifier[1]=nn.Linear(num_in_features, num_classes,bias=True)

#####ShuffleNet_V2
model_ft = models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1)
num_in_features=model_ft.fc.in_features
num_classes=2
model_ft.fc=nn.Linear(num_in_features, num_classes,bias=True)

model_ft.load_state_dict(torch.load('/home/hogan/data1/chh/clean_exp/code_classification/shufflenet_3.pth'))
# Since we are using our model only for inference, switch to `eval` mode:
model_ft.eval()

test_file='/home/hogan/data1/chh/clean_exp/code_classification/data_list/test_3.txt'
imgs_path=[]
true_labels=[]
with open(test_file, "r", encoding="utf-8") as f:
    info = f.readlines()
    for img_info in info:
        img_path, label = img_info.strip().split('\t')
        imgs_path.append(img_path)
        true_labels.append(int(label))

pred_labels=[]
for i_img, i_label in zip(imgs_path,true_labels):
    #print(i_img,i_label)
    image = Image.open(i_img)
    im_tensor=transform(image).unsqueeze(0)
    outputs=model_ft(im_tensor)
    class_probs = torch.softmax(outputs, dim=1)
    class_prob, topclass = torch.max(class_probs, dim=1)
    pred_labels.append(topclass.item())
    print(class_prob,topclass.item())
    #print(topclass.item(),type(topclass.item()))

target_names = ['clean', 'feces']
print(classification_report(true_labels, pred_labels, target_names=target_names))
#metric = BinaryAccuracy()
#metric.update(torch.tensor(pred_labels),torch.tensor(true_labels))
#print(metric.compute())
#metric2 = BinaryRecall()
#metric2.update(torch.tensor(pred_labels),torch.tensor(true_labels))
#print(metric2.compute())
fpr,tpr,thresholds=roc_curve(true_labels,pred_labels,pos_label=1)
print(auc(fpr,tpr))




'''
def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return y_hat
'''

