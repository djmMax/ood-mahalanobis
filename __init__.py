import torch.nn as nn
import torch.nn.functional as F
import torchvision

# load model
def get_resnet18(): 
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(in_features=512, out_features=2)

    self = model
    def feature_list(x):
        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = F.adaptive_avg_pool2d(out, output_size=(1, 1))
        out = out.view(out.size(0), -1)
        y = self.fc(out)
        return y, out_list
    
    model.feature_list = feature_list

    return model

def get_densenet121():
    model = torchvision.models.densenet121()
    num_features = model.classifier.in_features
    features = list(model.classifier.children())[:-1]
    features.extend([nn.Linear(num_features,  out_features=2)])
    model.classifier = nn.Sequential(*features)
    return model

def get_vgg16():
    model = torchvision.models.vgg16()
    features = list(model.classifier.children())[:-1]
    features.extend([nn.Linear(num_features,  out_features=2)])
    model.classifier = nn.Sequential(*features)
    return model

def get_vgg19():
    model = torchvision.models.vgg19()
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]
    features.extend([nn.Linear(num_features,  out_features=2)])
    model.classifier = nn.Sequential(*features)
    return model

#MODELS = ['resnet18', 'densenet121', 'vgg16', 'vgg19']
def get_model(model):
    if model == 'resnet18':
        return get_resnet18()
    if model == 'densenet121':
        return get_densenet121()
    if model == 'vgg16':
        return get_vgg16()
    if model == 'vgg19':
        return get_vgg19()
    