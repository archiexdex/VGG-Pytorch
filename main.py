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
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler



parser = argparse.ArgumentParser()
parser.add_argument("--epoch", "-e", type=int, default=10)
parser.add_argument("--batch_size", "-b", type=int, default=64)
parser.add_argument("--ngpu", type=int, default=2)
parser.add_argument('--lms', type=int, default=0, help='0 for not usig large model support, 1 for using')
parser.add_argument('--seed', type=int, default=0, help='it is just seed')
parser.add_argument('--nst', type=int, default=0, help='it is for nv-nsight-cu-cli')
parser.add_argument('--cdn', type=int, default=1, help='it is for cudnn enable')
parser.add_argument('--local_rank', type=int, help='Local rank. Necessary for using the torch.distributed.launch utility.')


args = parser.parse_args()

#print(args)

#if args.ngpu == 1:
#    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#if args.lms == 1:
#    torch.cuda.set_enabled_lms(True)
#    if args.cdn == 0:
#        torch.backends.cudnn.enabled = False 

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    cur_rank = dist.get_rank()
    group = dist.get_world_size()
    print(group)
    print(f"rank: {rank}, local_rank: {cur_rank}")

def cleanup():
    dist.destroy_process_group()

def data_setup(rank):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    train_sample = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=args.ngpu, rank=rank)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, pin_memory=True, sampler=train_sample, num_workers=os.cpu_count(), drop_last=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    #test_sample = torch.utils.data.distributed.DistributedSampler(testset, num_replicas=args.ngpu, rank=rank)

    #testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, pin_memory=True, sampler=test_sample, num_workers=os.cpu_count())
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=os.cpu_count())

    return trainloader, testloader


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    trainloader, testloader = data_setup(rank)

    # create model and move it to GPU with id rank
    net = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=False).to(rank)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = DDP(net, device_ids=[rank], output_device=rank)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    def train():
        net.train()
        mean_loss = 0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(rank)
            labels = labels.to(rank)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            mean_loss += loss.item()
        mean_loss /= len(trainloader)
        return mean_loss
    
    def test():
        net.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs = inputs.to(rank)
                labels = labels.to(rank)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    loss_list = []
    best_loss = 1234567890
    cnt = 0
    for e in range(args.epoch):
        #print("Local Rank: {rank}, Epoch: {e}, Training ...")
        loss = train()
        if rank == 0:
            if best_loss > loss:
                best_loss = loss
                cnt = 0
            else:
                cnt += 1
            if cnt > 10:
                print("early stop")
                break
            loss_list.append(loss)
            print(f"Epoch [{e}/{args.epoch}]: loss: {loss}")

    if rank == 0:
        accuracy = test()
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * accuracy))
    
    np.save("train_loss.npy", loss_list)
    cleanup()
    

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

def main():
    # Parameters
    rank = args.local_rank
    device = torch.device("cuda:{}".format(rank))

    torch.distributed.init_process_group(backend="nccl")

    net = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=False).to(rank)
    #net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = DDP(net, device_ids=[rank], output_device=rank)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_sampler = DistributedSampler(dataset=trainset)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, pin_memory=True, sampler=train_sampler, num_workers=os.cpu_count(), drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=os.cpu_count())

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    def train():
        net.train()
        mean_loss = 0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            mean_loss += loss.item()
            if args.nst == 1:
                break
        mean_loss /= len(trainloader)
        return mean_loss
    
    def test():
        net.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    loss_list = []
    best_loss = 1234567890
    cnt = 0
    for e in range(args.epoch):
        print(f"Local Rank: {rank}, Epoch: {e}, Training ...")
        loss = train()
        if best_loss > loss:
            best_loss = loss
            cnt = 0
        else:
            cnt += 1
        if cnt > 10:
            print("early stop")
            break
        loss_list.append(loss)
        if rank == 0:
            print(f"Epoch [{e}/{args.epoch}], loss: {loss}")
        if args.nst == 1:
            break

    if rank == 0:
        accuracy = test()
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * accuracy))
        #np.save("train_loss.npy", loss_list)
    cleanup()


if __name__ == "__main__":

    same_seeds(args.seed)
    st_time = datetime.now()
    main()
    print(f"Cost time: {datetime.now()-st_time}")
    #n_gpus = torch.cuda.device_count()
    #run_demo(demo_basic, args.ngpu)
