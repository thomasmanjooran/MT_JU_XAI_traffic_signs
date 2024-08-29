from ATSDS import ATSDS

import torch
from torchvision import transforms as transforms
## Standard libraries
import os
import json
import math
import random
import numpy as np 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
CUDA_LAUNCH_BLOCKING=1
from collections import defaultdict 
import argparse

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim


transform_train = transforms.Compose(
    [transforms.Resize(256),
    transforms.RandomCrop(size=(224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
transform_test = transforms.Compose(
    [transforms.Resize(256),
    transforms.CenterCrop(size=(224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
DATASET_PATH = "/data/horse/ws/juul507e-ju_streetsigns/data/"

CHECKPOINT_PATH = "model/"



# Used for reproducability to fix randomness in some GPU calculations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

def test_model(model, testloader,criterion):
    model.eval()
    num_classes = len(testloader.dataset.get_classes())
    correct = torch.zeros(num_classes, dtype=torch.int64, device=device)
    correct_top5 = torch.zeros(num_classes, dtype=torch.int64, device=device)
    total = torch.zeros(num_classes, dtype=torch.int64, device=device)
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)  # Move images and labels to the GPU
            outputs = model(images)
            
            loss = criterion(outputs, labels)  # Calculate the loss
            test_loss += loss.item()  # Accumulate the loss

            _, predicted = torch.max(outputs, 1)
            _, predicted_top5 = torch.topk(outputs, 5, 1)


            for i in range(len(predicted)):
                correct[labels[i]] += (predicted[i] == labels[i])
                correct_top5[labels[i]] += (labels[i] in predicted_top5[i])
                total[labels[i]] += 1
                

    accuracy_per_class = (correct.float() / total.float())
    top5_accuracy_per_class = (correct_top5.float() / total.float())
    test_loss /= len(testloader)  # Calculate the average test loss

    print(f'Test Total Accuracy: {accuracy_per_class.mean():.2%}')
    print(f'Test Total Top-5 Accuracy: {top5_accuracy_per_class.mean():.2%}')

    #for i in range(num_classes):
    #    print(f'Class {i} Test Accuracy: {accuracy_per_class[i]:.2%}')
    #    print(f'Class {i} Test Top-5 Accuracy: {top5_accuracy_per_class[i]:.2%}')

    model.train()  # Set the model back to training mode

    return correct.cpu().numpy(), correct_top5.cpu().numpy(), total.cpu().numpy(), test_loss
    
def save_model(model,optimizer,scheduler,trainstats,epoch,filepath="model/current/model.tar"):
    torch.save({'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),
                'trainstats':trainstats,
                'epoch':epoch},filepath)



from model import get_model




def start_training(model_name, model_number, dataset_type, initial_learning_rate, batch_size, weight_decay, max_epochs):

  
    trainset = ATSDS(root=DATASET_PATH, dataset_type=dataset_type, split="train", transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = 2)
    testset = ATSDS(root=DATASET_PATH, dataset_type=dataset_type, split="test", transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 1, shuffle = True, num_workers = 2)
    
    model = get_model(model_name = model_name, n_classes = trainset.get_num_classes())
    model = model.to(device)
    
    
    print(model)
    loss_criterion = nn.CrossEntropyLoss()
    
    loss_criterion = loss_criterion.to(device)
    optimizer = optim.AdamW(model.parameters(),lr=initial_learning_rate,weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max_epochs)
    
    
    train_loss = 0.0
    total = 0
    correct = 0
    save_osc = 1
    epoch = 0
    trainlosses = []
    testlosses = []
    
    correct_train_s = []
    correct_top5_train_s = []
    total_train_s = []
    
    correct_test_s = []
    correct_top5_test_s = []
    total_test_s = []
    
    num_classes = testset.get_num_classes()
    
    
    while epoch <max_epochs:
        
        #correct_test,correct_top5_test,total_test = test_model(model,testloader)
        
        correct = torch.zeros(num_classes, dtype=torch.int64, device=device)
        correct_top5 = torch.zeros(num_classes, dtype=torch.int64, device=device)
        total = torch.zeros(num_classes, dtype=torch.int64, device=device)
            
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            if (device == torch.device("cuda:0")):
                inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = loss_criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            # print statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            _, predicted_top5 = torch.topk(outputs, 5, 1)
    
    
            for i in range(len(predicted)):
                correct[labels[i]] += (predicted[i] == labels[i])
                correct_top5[labels[i]] += (labels[i] in predicted_top5[i])
                total[labels[i]] += 1
                
        accuracy_per_class = (correct.float() / total.float())
        top5_accuracy_per_class = (correct_top5.float() / total.float())
    
        print(f'Epoch: {epoch} Train Accuracy: {accuracy_per_class.mean():.2%}')
        print(f'Train Top-5 Accuracy: {top5_accuracy_per_class.mean():.2%}')
    
        #for i in range(num_classes):
        #    print(f'Class {i} Train Accuracy: {accuracy_per_class[i]:.2%}')
        #    print(f'Class {i} Train Top-5 Accuracy: {top5_accuracy_per_class[i]:.2%}')    
            
        correct_train_s.append(accuracy_per_class)
        correct_top5_train_s.append(top5_accuracy_per_class)
        total_train_s.append(total)
        
        trainlosses.append(train_loss)
        train_loss = 0.0
        correct = 0
        total = 0
        save_osc +=1
        save_osc %= 2
        correct_test,correct_top5_test,total_test, test_loss = test_model(model,testloader,loss_criterion)
        testlosses.append(test_loss)
        correct_test_s.append(correct_test)
        correct_top5_test_s.append(correct_top5_test)
        total_test_s.append(total_test)
        save_model(model,optimizer,scheduler,[trainlosses,testlosses,[correct_train_s,correct_top5_train_s,total_train_s],[correct_test_s,correct_top5_test_s,total_test_s]],epoch,"Code/model/" + model_name + "_" + str(model_number) + "_" + str(save_osc) + ".tar")
        scheduler.step()
        epoch += 1
    #testmodel = BaseCNN()
    #load_model(testmodel,CHECKPOINT_PATH+"/model.tar")
    
    
def main(model_name, model_number, dataset_type, learning_rate, batch_size, weight_decay, max_epochs, random_seed):
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)  
    start_training(model_name, model_number, dataset_type, learning_rate, batch_size, weight_decay, max_epochs)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="vgg16", help="Name of the model")
    parser.add_argument("--model_number", type=int, default=99, help="Model number")
    parser.add_argument("--dataset_type", type=str, default="atsds_large", help="Type of dataset")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=200, help="Maximum number of epochs")
    parser.add_argument("--random_seed", type=int, default=1500, help="Random Seed for training")

    args = parser.parse_args()
    main(args.model_name, args.model_number, args.dataset_type, args.learning_rate, args.batch_size, args.weight_decay, args.max_epochs, args.random_seed)
    