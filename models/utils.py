import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.evaluation import calculate_auc, calculate_metrics

from importlib import import_module
from pprint import pprint

def standard_train(opt, network, optimizer, loader, _criterion, wandb):
    """Train the model for one epoch"""
    tol_output, tol_target, tol_sensitive, tol_index, tol_uncertainty = [], [], [], [], []
    train_loss, no_iter = 0., 0.
    metrics = {'Training AUC': 0.,'Training ACC': 0.,'Training UNC': 0.,}
    for i in range(opt['sens_classes']):
        metrics[f'Training AUC A{i}'] = 0.0
        metrics[f'Training ACC A{i}'] = 0.0
        metrics[f'Training UNC A{i}'] = 0.0
    for iter, (images, targets, sensitive_attr, index) in enumerate(loader):
        images, targets, sensitive_attr = images.to(opt['device']), targets.to(opt['device']), sensitive_attr.to(opt['device'])
        optimizer.zero_grad()
        outputs, features = network(images)
        #print(outputs.shape, features.shape)
        if outputs.shape[-1] > 1:
            loss = F.cross_entropy(F.softmax(outputs), targets.long().squeeze())
            uncertainty = F.softmax(outputs).max(1)[0]
        else:
            loss = F.binary_cross_entropy_with_logits(outputs, targets)
            #loss = _criterion(outputs, targets)
            uncertainty = F.sigmoid(outputs)
            uncertainty = torch.where(uncertainty >= 0.5, uncertainty, 1.0-uncertainty)
        loss.backward()
        optimizer.step()

        if outputs.shape[-1] > 1:
            tol_output.extend(F.softmax(outputs)[:, 1].flatten().cpu().data.numpy().tolist())
        else:
            tol_output.extend(F.sigmoid(outputs.squeeze()).flatten().cpu().data.numpy().tolist())
        tol_target.extend(targets.squeeze().cpu().data.numpy().tolist())
        tol_sensitive.extend(sensitive_attr.squeeze().squeeze().cpu().data.numpy().tolist())
        tol_index.extend(index.squeeze().cpu().numpy().tolist())
        tol_uncertainty.extend(uncertainty.squeeze().detach().cpu().numpy().tolist())

        if iter == 0:
            print('\nTraining Y0/Y1', outputs[:, 1][targets.squeeze()==0].shape, outputs[:, 1][targets.squeeze()==1].shape)
            for i in range(opt['sens_classes']):
                print(f'A{i} Total/Y0/Y1:', outputs[sensitive_attr.squeeze()==i].shape, outputs[torch.logical_and((sensitive_attr.squeeze()==i),(targets.squeeze()==0))].shape,
                                                                              outputs[torch.logical_and((sensitive_attr.squeeze()==i),(targets.squeeze()==1))].shape)
        train_loss += loss.item()
        no_iter += 1

        #if opt['log_freq'] and (i % opt['log_freq'] == 0):
        wandb.log({'Training loss': train_loss / no_iter,})
    tol_output = np.array(tol_output)
    tol_target = np.array(tol_target)
    tol_sensitive = np.array(tol_sensitive)
    tol_uncertainty = np.array(tol_uncertainty)
    print(np.round(tol_output)==tol_target, tol_output.shape, tol_target.shape)
    metrics[f'Training AUC'] += calculate_auc(tol_output, tol_target)
    metrics[f'Training ACC'] += len(tol_output[np.round(tol_output)==tol_target])/len(tol_output)
    tol_output_y0 = tol_output[tol_target==0]
    tol_output_y1 = tol_output[tol_target==1]
    class_balancer = np.array([(tol_target == 0).sum(), (tol_target == 1).sum()])/len(tol_target)
    metrics[f'Training Class-Reweighted ACC'] = class_balancer[1] * (np.round(tol_output_y0) == 0).sum()/len(tol_output_y0) + class_balancer[0] * (np.round(tol_output_y1) == 1).sum()/len(tol_output_y1)
    metrics[f'Training UNC'] += np.mean(tol_uncertainty)
    for i in range(opt['sens_classes']):
        oi = tol_output[tol_sensitive==i]
        if len(oi) > 0:        
            ti = tol_target[tol_sensitive==i]
            metrics[f'Training AUC A{i}'] += calculate_auc(oi, ti)
            metrics[f'Training ACC A{i}'] += len(oi[np.round(oi)==ti])/len(oi)
            metrics[f'Training UNC A{i}'] += np.mean(tol_uncertainty[tol_sensitive==i])
            oi_y0 = oi[ti==0]
            oi_y1 = oi[ti==1]
            class_balancer = np.array([(ti == 0).sum(), (ti == 1).sum()])/len(ti)
            metrics[f'Training Class-Reweighted ACC A{i}'] = class_balancer[1] * (np.round(oi_y0) == 0).sum()/len(oi_y0) + class_balancer[0] * (np.round(oi_y1) == 1).sum()/len(oi_y1)
        else:
            metrics[f'Training AUC A{i}'] += 0.
            metrics[f'Training ACC A{i}'] += 0.
            metrics[f'Training UNC A{i}'] += 0.

    wandb.log(metrics)
    pprint(metrics)
    train_loss /= no_iter
    return metrics['Training AUC'], train_loss


def standard_val(opt, network, loader, _criterion, sens_classes, wandb):
    """Compute model output on validation set"""
    tol_output, tol_target, tol_sensitive, tol_index, tol_uncertainty = [], [], [], [], []
    val_loss, no_iter = 0., 0.
    metrics = {'Validation AUC': 0.,'Validation ACC': 0.,'Validation UNC': 0.,}
    for i in range(opt['sens_classes']):
        metrics[f'Validation AUC A{i}'] = 0.0
        metrics[f'Validation ACC A{i}'] = 0.0
        metrics[f'Validation UNC A{i}'] = 0.0

    with torch.no_grad():
        for iter, (images, targets, sensitive_attr, index) in enumerate(loader):
            images, targets, sensitive_attr = images.to(opt['device']), targets.to(opt['device']), sensitive_attr.to(
                opt['device'])
            outputs, features = network.forward(images)
            

            if outputs.shape[-1] > 1:
                loss = F.cross_entropy(F.softmax(outputs), targets.long().squeeze())
                uncertainty = F.softmax(outputs).max(1)[0]
            else:
                loss = F.binary_cross_entropy_with_logits(outputs, targets)
                #loss = _criterion(outputs, targets)
                uncertainty = F.sigmoid(outputs)

            try:
                val_loss += loss.item()
            except:
                val_loss += loss.mean().item()
            if outputs.shape[-1] > 1:
                tol_output.extend(F.softmax(outputs)[:, 1].flatten().cpu().data.numpy().tolist())
            else:
                tol_output.extend(F.sigmoid(outputs.squeeze()).flatten().cpu().data.numpy().tolist())
            tol_target.extend(targets.squeeze().cpu().data.numpy().tolist())
            tol_sensitive.extend(sensitive_attr.squeeze().squeeze().cpu().data.numpy().tolist())
            tol_index.extend(index.squeeze().cpu().numpy().tolist())
            tol_uncertainty.extend(uncertainty.squeeze().detach().cpu().numpy().tolist())

            if iter == 0:
                print('\nValidation Y0/Y1', outputs[:, 1][targets.squeeze()==0].shape, outputs[:, 1][targets.squeeze()==1].shape)
                for i in range(opt['sens_classes']):
                    print(f'A{i} Total/Y0/Y1:', outputs[sensitive_attr.squeeze()==i].shape, outputs[torch.logical_and((sensitive_attr.squeeze()==i),(targets.squeeze()==0))].shape,
                                                                              outputs[torch.logical_and((sensitive_attr.squeeze()==i),(targets.squeeze()==1))].shape)
            no_iter += 1
            wandb.log({'Validation loss': val_loss / no_iter})

    tol_output = np.array(tol_output)
    tol_target = np.array(tol_target)
    tol_sensitive = np.array(tol_sensitive)
    tol_uncertainty = np.array(tol_uncertainty)
    metrics[f'Validation AUC'] += calculate_auc(tol_output, tol_target)
    metrics[f'Validation ACC'] += len(tol_output[np.round(tol_output)==tol_target])/len(tol_output)
    tol_output_y0 = tol_output[tol_target==0]
    tol_output_y1 = tol_output[tol_target==1]
    class_balancer = np.array([(tol_target == 0).sum(), (tol_target == 1).sum()])/len(tol_target)
    metrics[f'Validation Class-Reweighted ACC'] = class_balancer[1] * (np.round(tol_output_y0) == 0).sum()/len(tol_output_y0) + class_balancer[0] * (np.round(tol_output_y1) == 1).sum()/len(tol_output_y1)
    metrics[f'Validation UNC'] += np.mean(tol_uncertainty)
    for i in range(opt['sens_classes']):
        oi = tol_output[tol_sensitive==i]
        if len(oi) > 0:
            ti = tol_target[tol_sensitive==i]
            metrics[f'Validation AUC A{i}'] += calculate_auc(oi, ti)
            metrics[f'Validation ACC A{i}'] += len(oi[np.round(oi)==ti])/len(oi)
            metrics[f'Validation UNC A{i}'] += np.mean(tol_uncertainty[tol_sensitive==i])
            oi_y0 = oi[ti==0]
            oi_y1 = oi[ti==1]
            class_balancer = np.array([(ti == 0).sum(), (ti == 1).sum()])/len(ti)
            metrics[f'Validation Class-Reweighted ACC A{i}'] = class_balancer[1] * (np.round(oi_y0) == 0).sum()/len(oi_y0) + class_balancer[0] * (np.round(oi_y1) == 1).sum()/len(oi_y1)
        else:
            metrics[f'Validation AUC A{i}'] += 0.
            metrics[f'Validation ACC A{i}'] += 0.
            metrics[f'Validation UNC A{i}'] += 0.
    wandb.log(metrics)
    pprint(metrics)
    val_loss /= no_iter

    log_dict, t_predictions, pred_df = calculate_metrics(tol_output, tol_target, tol_sensitive, tol_index, sens_classes)
    wandb.log(log_dict)
    pred_df['sensitive'] = tol_sensitive
    return metrics['Validation AUC'], val_loss, log_dict, pred_df


def standard_test(opt, network, loader, _criterion, wandb):
    """Compute model output on testing set"""
    tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []

    with torch.no_grad():
        for i, (images, targets, sensitive_attr, index) in enumerate(loader):
            images, targets, sensitive_attr = images.to(opt['device']), targets.to(opt['device']), sensitive_attr.to(
                opt['device'])
            outputs, features = network.forward(images)

            tol_output += F.sigmoid(outputs).flatten().cpu().data.numpy().tolist()
            tol_target += targets.cpu().data.numpy().tolist()
            tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()
            tol_index += index.numpy().tolist()
            
    return tol_output, tol_target, tol_sensitive, tol_index
