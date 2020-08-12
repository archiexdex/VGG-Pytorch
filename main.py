import os, shutil
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import argparse
import random
from datetime import datetime
import numpy as np
import nvidia_smi

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", "-e", type=int, default=10)
parser.add_argument("--batch_size", "-b", type=int, default=64)
parser.add_argument("--ngpu", type=int, default=1)
parser.add_argument('--lms', type=int, default=0, help='0 for not usig large model support, 1 for using')
parser.add_argument('--seed', type=int, default=0, help='it is just seed')
parser.add_argument('--nst', type=int, default=0, help='it is for nv-nsight-cu-cli')
parser.add_argument('--cdn', type=int, default=1, help='it is for cudnn enable')

args = parser.parse_args()

print(args)

if args.ngpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
same_seeds(args.seed)
if args.lms == 1:
    torch.cuda.set_enabled_lms(True)
    if args.cdn == 0:
        torch.backends.cudnn.enabled = False 

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=os.cpu_count())

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=os.cpu_count())

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model 
net = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True).cuda()

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def train():
    net.train()
    if args.nst == 0:
        nvidia_smi.nvmlInit()
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + argsimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if args.nst == 1:
            break

    if args.nst == 0:
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print(f"nvidia-smi: {info.used//1024//1024}")
        # For second gpu
        if args.ngpu > 1:
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(1)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            print(f"nvidia-smi: {info.used//1024//1024}")
        nvidia_smi.nvmlShutdown()

def test():
    net.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if args.nst == 1:
                break

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

print("Start training")
st_time = datetime.now()
for e in range(args.epoch):  # loop over the dataset multiple times
    print(f"Epoch: {e}")
    train()
    test()

print(f"Cost time: {(datetime.now()-st_time).seconds}")
