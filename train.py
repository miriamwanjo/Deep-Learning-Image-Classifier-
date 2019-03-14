import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Training model")
    parser.add_argument('--data_dir', type=str, default='flowers', help='directory of flowers')
    parser.add_argument('--gpu', type=bool, dest='cuda', default=True, help='device gpu or cpu')
    parser.add_argument('--arch', type=str, help='pretrained model')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=256, help='hidden layers')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--save_dir' , type=str, default='my_chekpoint_cmd.pth', dest='save_dir')

    args = parser.parse_args()

def load_model(arch='densenet121', num_labels=102, hidden_units=256):
    if args.arch =='densenet121':
        model = models.densenet121(pretrained=True)
    elif args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad=False   
    features = list(model.classifier.children())[:-1]   
    num_filters = model.classifier[len(features)].in_features
   
    features.extend([ nn.Dropout(),
                     nn.Linear(num_filters, hidden_units),
                     nn.ReLU(True),
                     nn.Dropout(),
                     nn.Linear(hidden_units, hidden_units),
                     nn.ReLU(True),
                     nn.Linear(hidden_units, num_labels)
                    ])
                                 

    model.classifier = nn.Sequential(*features)
    return model
                     
    
def train_model(model, trainloader, testloader, epochs=10):
    cuda= torch.cuda.is_available()
    if gpu and cuda:
        model.cuda()
    else:
        model.cpu()
    start=time.time()
    epochs = 10
    print_every = 5
    steps = 0
    model.to(device)

    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    
                    for inputs, labels in validloader:
                    
                        inputs, labels = inputs.cuda(), labels.cuda()
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps).data
                        equals = (labels.data == ps.max(1)[1])
                        accuracy += equals.type_as(torch.FloatTensor()).mean()
                    
                        print(f"Epoch {epoch+1}/{epochs}.. "
                              f"Train loss: {running_loss/print_every:.3f}.. "
                              f"Valid loss: {valid_loss/len(validloader):.3f}.."
                              f"Valid accuracy: {accuracy/len(validloader):.3f}")
                    running_loss = 0
                model.train()

    end= time.time()
    elapsed_time = start-end
    print('Elapsed Time: {:.0f}m {:.0f}s'.format(elapsed_time//60, elapsed_time % 60))                       
                          
    
def main(): 
    args = parse_args()
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test' 

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                        [0.229, 0.224, 0.225])]),
    
    test_transforms = transforms.Compose([
                                    
                                  transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])]),

    validation_transforms = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                       	transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])])
    trainset = datasets.ImageFolder(train_dir,transform=train_transforms)
    testset = datasets.ImageFolder(test_dir,transform=test_transforms)
    validset = datasets.ImageFolder(valid_dir, transform=validation_transforms)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle=True),
    testloader = torch.utils.data.DataLoader(testset, batch_size = 32, shuffle=True),
    validloader = torch.utils.data.DataLoader(validset, batch_size = 32, shuffle = True)
    
    model = load_model(num_labels=102, hidden_units=args.hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    train_model(model, criterion, optimizer, epochs, gpu)
    epochs = int(args.epochs)
    model.class_to_idx = trainset.class_to_idx
    checkpoint = {'arch': args.arch,
                        'classifier': model.classifier,
                        'learning_rate' : args.lr,
                        'optimizer_state': optimizer.state_dict,
                        'class_to_idx': trainset.class_to_idx,
                        'state_dict':model.state_dict(),
                        'epochs': model.epochs}

# save the checkpoint
    torch.save(checkpoint, args.save_dir)
          
if __name__ == "__main__":
    main()