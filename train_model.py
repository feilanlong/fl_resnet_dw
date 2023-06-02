import torch
from torchvision import models,transforms,datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import time
import copy
import datetime
import os
import json
import logging
#  定义模型训练函数
def train_model(model, criterion, optimizer, scheduler,data_loaders,pic_size, device,logger,dt,batch_size=16,num_epochs=50):
    model = model.to(device)
    
    loss_dic = {'train': [], 'val': []}
    acc_dic = {'train': [], 'val': []}
    time_dic = {'train': [], 'val': []}

    best_model_state_dict = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 1
    for epoch in range(num_epochs):
        
        logger.info('epoch {}/{}开始训练！'.format(epoch+1,num_epochs))
        logger.info('=*' * 30)
        
        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            epcoh_sum_loss = 0.0
            epoch_sum_correct = 0
            
            st = time.time()
            
            for idx,data in enumerate(data_loaders[phase]):
                images,labels = data
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    _, prec = torch.max(outputs ,1)
                    
                    # print(prec)
                    # print(labels)

                    loss = criterion(outputs, labels)#多分类采用交叉熵损失函数
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                epcoh_sum_loss += loss.item() * images.size(0)
                epoch_sum_correct += torch.sum(prec == labels)
            
            epoch_loss = epcoh_sum_loss / len(data_loaders[phase].dataset)
            loss_dic[phase].append(epoch_loss)

            epoch_acc = epoch_sum_correct.double() / len(data_loaders[phase].dataset)
            epoch_acc = epoch_acc.item()

            acc_dic[phase].append(100 * epoch_acc)
            et = time.time()
            cost_time = et - st
            time_dic[phase].append(cost_time)
            logger.info('{} epoch_loss: {:.4f}  epoch_acc: {:.2%} cost_time:{}'.format(phase, epoch_loss,epoch_acc,cost_time))
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_state_dict = copy.deepcopy(model.state_dict())
                    best_epoch = epoch+1
                logger.info('目前验证集精度最好的是第{}轮，精度是: {:.2%}'.format(best_epoch,best_acc))

    model.load_state_dict(best_model_state_dict)
    
    torch.save(model.state_dict(), f'resnet18_no_pre_{pic_size}_{batch_size}_{num_epochs}epoch_best_model_{dt}.pt')
    logger.info('【训练结束】目前验证集精度最好的是第{}轮，精度是: {:.2%}'.format(best_epoch,best_acc))
    return model, loss_dic,acc_dic,time_dic