'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import datetime
import pandas as pd
import numpy as np
import sum_mnist as optim2
from numpy import *

class LeNet(nn.Module):
    def __init__(self,p =0.3):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1)
        self.dropout1c=nn.Dropout(p)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.dropout2c=nn.Dropout(p)
        self.fc1 = nn.Linear(256, 120)
        self.dropout1=nn.Dropout(p)
        self.fc2 = nn.Linear(120, 84)
        self.dropout2=nn.Dropout(p)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        x = F.relu(self.dropout1c(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.dropout2c(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.dropout1(self.fc1(x)))
        x = F.relu(self.dropout2(self.fc2(x)))
        x = self.fc3(x)
        output = x
        return output
    
def set_seed(seed):  
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init_fun(worker_id, seed):
    np.random.seed(seed + worker_id)



# Training
def train(net, optimizer, train_loader, device, steplr, epoch, criterion,opt):
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    loader_num=len(train_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs_dev, targets_dev = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs_dev)
        loss = criterion(outputs, targets_dev)          
        loss.backward()
        optimizer.step(epoch=epoch, steplr=steplr)      
        train_loss+=loss.item() 
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets_dev).sum().item()
        

    train_loss /= loader_num  
    tain_acc=100.*correct / len(train_loader.dataset)
    
    print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(inputs), len(train_loader.dataset),tain_acc, train_loss))     
    
    return train_loss,tain_acc
      

def test(net, test_loader, device, epoch, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    loader_num=len(test_loader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs_dev, targets_dev = inputs.to(device), targets.to(device)
            outputs = net(inputs_dev)
            loss = criterion(outputs, targets_dev)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets_dev.size(0)
            correct += predicted.eq(targets_dev).sum().item()
                 

    test_loss /= loader_num 
    test_acc=100.*correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),test_acc))
    
    return test_loss,test_acc 


def create_data(worker_id, fname, seed):

    if fname == 'cifar-10':
        print('==> Preparing data cifar10..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2, worker_init_fn=worker_init_fun(worker_id, seed))
        
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2, worker_init_fn=worker_init_fun(worker_id, seed)) 
    
    elif fname == "mnist":
        print('==> Preparing data mnist..')
        train_set=torchvision.datasets.MNIST('./data', train=True, download=True,
            transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        

        
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fun(worker_id, seed))       
            
        test_set=torchvision.datasets.MNIST('./data', train=False, download=True,
            transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
                           ]))        
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=100, shuffle=False, worker_init_fn=worker_init_fun(worker_id, seed))       
                                 
    else:        
          raise ValueError("Invalid cifar")
          
    return train_loader, test_loader
 

def create_model(fname, device):
    if fname == "mnist":
        net = LeNet(p=p)
    else:
        raise ValueError("Invalid model")
    
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
    # Model
    print('==> Building model..')
      
    return net

def create_optimizer(opt,net, weight_decay, T):
    if opt == 'SUM-0.0':
        optimizer= optim2.SUM_Ind(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=beta1, interp_factor=0.0, K=K*T)
    elif opt == 'SUM-0.5':
        optimizer= optim2.SUM_Ind(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=beta1, interp_factor=0.5, K=K*T)
    elif opt == 'SUM-1.0':
        optimizer= optim2.SUM_Ind(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=beta1, interp_factor=1.0, K=K*T)
    elif opt == 'SUM-5.0':
        optimizer= optim2.SUM_Ind(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=beta1, interp_factor=5.0, K=K*T)
    elif opt == 'SUM-10.0':
        optimizer= optim2.SUM_Ind(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=beta1, interp_factor=10.0, K=K*T)

    else:
        print("no algorithm")
    
    return optimizer


def train_and_save(trial_idx,start_epoch):
    for opt in algorithms: 
        train_loader, test_loader = create_data(worker_id, fname, seed)
        T = len(train_loader)*(epochs)
        net = create_model(fname, device)
        criterion = nn.CrossEntropyLoss() 
        
        optimizer = create_optimizer(opt, net, weight_decay, T)
        
        paras = ('dsets=%s trial_idx=%s opt=%s epochs=%s batch_size=%s lr=%s K=%s p=%s seed=%s' %
                (fname, trial_idx, opt, epochs, batch_size, lr, K, p, seed))
        history_file = os.path.join(history_folder,
                                    paras+'.csv')
        ckpt_path = './checkpoint/' + paras + '.pth'
        if os.path.exists(history_file):
            print("The same csv file already exists in the tests directory. Back it up to avoid overwriting it. If it has been backed up, delete it!")
            break
        
        if resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(ckpt_path)
            net.load_state_dict(checkpoint['net'])
            info = checkpoint['info']
            start_epoch = checkpoint['epoch']
            print("current checkpoint info：", info)
            
        M_columns = ['train_loss_'+opt, 'train_acc_'+opt, 'test_loss_'+opt, 'test_acc_'+opt]
        columns_total = ['epoch'] + M_columns
        dfhistory = pd.DataFrame(columns=columns_total)
           
        start = datetime.datetime.now()
        for epoch in range(start_epoch, start_epoch+epochs):
            M_infos = tuple()

            if not debug:
                tr_loss, tr_metric=train(net, optimizer, train_loader, device, steplr, epoch, criterion, opt)
                te_loss, te_metric = test(net, test_loader, device, epoch, criterion)
            else:
                tr_loss, tr_metric=0.1,99
                te_loss, te_metric=1.0,66
                
            M_infos += (tr_loss, tr_metric, te_loss, te_metric)    
            info = (int(epoch),) + M_infos
            dfhistory.loc[int(epoch)] = info

            state = {
                'net': net.state_dict(),
                'acc': te_metric,
                'epoch': epoch,
                'info': paras
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, ckpt_path) 
                
            dfhistory.to_csv(history_file)
            torch.cuda.empty_cache()
            
            end = datetime.datetime.now()
            print('time%.3f:'%(((end-start).seconds)/60))
    
if __name__ == '__main__':
    
################################################
    start_idx =1
    trial_num =5
    seed = 1
    worker_id = 0
    epochs = 100
    p=0.3
    lr= 0.01
    beta1=0.9
    beta2=0.999
    interp_factor=1.0
    K=0.9
    batch_size=128
    start_epoch = 0
    weight_decay = 0
    steplr = 0.8*epochs
    resume = False
    fname = "mnist"    #cifar-10 
    debug= False
    #algorithms = ['SUM-0.0','SUM-0.5','SUM-1.0','SUM-5.0','SUM-10.0']
    algorithms = ['SUM-10.0'] #one by one
#################################################
    

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    ROOT_PATH = os.path.abspath('./')
    history_folder = os.path.join(ROOT_PATH, 'tests/Logistic_History')  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for trial_idx in range(start_idx,trial_num+start_idx):
        seed = trial_idx*5
        set_seed(seed)
        train_and_save(trial_idx,start_epoch)    
