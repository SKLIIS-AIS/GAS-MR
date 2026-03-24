import torch.nn as nn
import torch
import os
import json
import torchvision.models as models
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter

class SimpleCNN(nn.Module):
    def __init__(self, channel, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN, self).__init__()
        self.base = FE(channel, input_dim, hidden_dims)
        self.classifier = Classifier(hidden_dims[1], output_dim)

    def forward(self, x):
        return self.classifier(self.base(x))

class BNCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(BNCNN, self).__init__()
        self.base = BNFE(input_dim, hidden_dims)
        self.classifier = Classifier(hidden_dims[1], output_dim)

    def forward(self, x):
        return self.classifier(self.base(x))

class FE(nn.Module):
    def __init__(self, channel, input_dim, hidden_dims):
        super(FE, self).__init__()
        self.conv1 = nn.Conv2d(channel, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

class BNFE(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(BNFE, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

class Classifier(nn.Module):
    def __init__(self, hidden_dims, output_dim=10):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(hidden_dims, output_dim)

    def forward(self, x):
        return self.fc(x)

class CIFAR100_FE(nn.Module):
    def __init__(self):
        super(CIFAR100_FE, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(256, 128)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return x

class CIFAR100_Classifier(nn.Module):
    def __init__(self):
        super(CIFAR100_Classifier, self).__init__()
        self.fc = nn.Linear(128, 100)

    def forward(self, x):
        return self.fc(x)

class CIFAR100_CNN(nn.Module):
    def __init__(self):
        super(CIFAR100_CNN, self).__init__()
        self.base = CIFAR100_FE()
        self.classifier = CIFAR100_Classifier()

    def forward(self, x):
        return self.classifier(self.base(x))

def simplecnn(n_classes, channel=3, image_size=32):
    if image_size == 32:
        input_dim = 16 * 5 * 5
    elif image_size == 28:
        input_dim = 16 * 4 * 4
    else:
        raise ValueError(f"Unsupported image_size={image_size}")
    return SimpleCNN(channel=channel, input_dim=input_dim, hidden_dims=[120, 84], output_dim=n_classes)

def simplecnn_mnist(n_classes):
    return SimpleCNN(channel=1, input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=n_classes)

def simplecnn_fmnist(n_classes):
    return SimpleCNN(channel=1, input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=n_classes)

class TextCNN_FE(nn.Module):
    def __init__(self, vocab_size, embeddings, emb_size):
        super(TextCNN_FE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.convs = nn.ModuleList([nn.Conv2d(1, 100, (size, emb_size)) for size in [3, 4, 5]])
        self.relu = nn.ReLU()

    def forward(self, text):
        embeddings = self.embedding(text).unsqueeze(1)
        conved = [self.relu(conv(embeddings)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conved]
        flattened = torch.cat(pooled, dim=1)
        return flattened

class TextCNN(nn.Module):
    def __init__(self, n_classes, vocab_size, embeddings, emb_size):
        super(TextCNN, self).__init__()
        self.base = TextCNN_FE(vocab_size, embeddings, emb_size)
        self.classifier = Classifier(300, n_classes)

    def forward(self, x):
        return self.classifier(self.base(x))

def textcnn(n_classes):
    return TextCNN(n_classes, vocab_size=20000, embeddings=None, emb_size=256)

class ResNet18_FE(nn.Module):
    def __init__(self, conv3=False, gn=False):
        super(ResNet18_FE, self).__init__()
        basemodel = models.resnet18(pretrained=False)
        if conv3:
            basemodel.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            basemodel.maxpool = nn.Identity()
        self.fe = nn.Sequential(*list(basemodel.children())[:-1])

    def forward(self, x):
        out = self.fe(x)
        return out.view(out.size(0), -1)

class ResNet18(nn.Module):
    def __init__(self, n_classes, conv3=False, gn=False):
        super(ResNet18, self).__init__()
        self.base = ResNet18_FE(conv3, gn)
        self.classifier = nn.Linear(512, n_classes)

    def forward(self, x):
        return self.classifier(self.base(x))

def resnet18(n_classes):
    return ResNet18(n_classes)

def resnet18_3(n_classes):
    return ResNet18(n_classes, conv3=True)

def cifar100_cnn(n_classes):
    return CIFAR100_CNN()

def get_model(args, cfg):
    if args.model == 'simplecnn':
        return lambda n_classes: simplecnn(n_classes, channel=args.channel, image_size=cfg["image_size"])
    elif args.model == 'simplecnn-mnist':
        model = simplecnn_mnist
    elif args.model == 'simplecnn-fmnist':
        model = simplecnn_fmnist
    elif args.model == 'bncnn':
        model = bncnn
    elif args.model == 'textcnn':
        model = textcnn
    elif args.model == 'resnet18':
        model = resnet18
    elif args.model == 'resnet18-3':
        model = resnet18_3
    elif args.model == 'cifar100-cnn':
        model = cifar100_cnn
    return model