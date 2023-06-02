
import torch
from torchvision import models,transforms,datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import time
import copy
import datetime


def train_model(device,model, pic_size,data_loaders,criterion, optimizer, scheduler, num_epochs=50):
    model = model.to(device)
    Loss_list = {'train': [], 'val': []}
    Accuracy_list_species = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        st = time.time()
        print('epoch {}/{}开始训练！'.format(epoch,num_epochs - 1))
        print('-*' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            corrects_species = 0
            
            for idx,data in enumerate(data_loaders[phase]):
                inputs,labels_species = data
                inputs = inputs.to(device)
                labels_species = labels_species.to(device)
                # print(inputs)
                # print("-----------")
                # print(labels_species)
                # print("-----------")
                # # print(inputs,labels_species)
                # exit()



                #print(phase+' processing: {}th batch.'.format(idx))

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    x_species = model(inputs)
        
                    _, preds_species = torch.max(x_species, 1)

                    loss = criterion(x_species, labels_species)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)

                corrects_species += torch.sum(preds_species == labels_species)
            
            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            Loss_list[phase].append(epoch_loss)

            epoch_acc_species = corrects_species.double() / len(data_loaders[phase].dataset)
            epoch_acc = epoch_acc_species.item()

            Accuracy_list_species[phase].append(100 * epoch_acc)
            et = time.time()
            print('{} Loss: {:.4f}  Acc_species: {:.2%} cost_time:{}'.format(phase, epoch_loss,epoch_acc_species,et-st))

            if phase == 'val' and epoch_acc > best_acc:

                best_acc = epoch_acc_species
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Best val species Acc: {:.2%}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), f'best_model-{pic_size}-{datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")}.pt')
    print('Best val species Acc: {:.2%}'.format(best_acc))
    return model, Loss_list,Accuracy_list_species
