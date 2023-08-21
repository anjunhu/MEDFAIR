import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint
from torch.utils.data.dataloader import default_collate
from importlib import import_module

from PIL import Image, ImageFilter

from models import basemodels
from models.resampling import resampling
from utils import basics
import pandas as pd
from datasets.utils import get_dataset
from utils.evaluation import calculate_auc, calculate_metrics, calculate_FPR_FNR
from models.basenet import BaseNet
from sklearn.metrics import accuracy_score

import torchvision
from torchvision.models import resnet18, resnet50, resnet101
from torchvision import transforms

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean, std=std)

CE = nn.CrossEntropyLoss(reduction='none')
BCE = nn.BCEWithLogitsLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()

def entropy(p, axis=1):
    return -torch.sum(p * torch.log2(p+1e-5), dim=axis)

def get_distances(X, Y, dist_type="cosine"):
    if dist_type == "euclidean":
        distances = torch.cdist(X, Y)
    elif dist_type == "cosine":
        distances = 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
    else:
        raise NotImplementedError(f"{dist_type} distance not implemented.")

    return distances

@torch.no_grad()
def soft_k_nearest_neighbors(features, features_bank, probs_bank):
    pred_probs = []
    pred_probs_all = []

    for feats in features.split(64):
        distances = get_distances(feats, features_bank)
        _, idxs = distances.sort()
        idxs = idxs[:, : 10]
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
        # (64, num_nbrs, num_classes)
        probs_all = probs_bank[idxs, :]
        pred_probs_all.append(probs_all)

    pred_probs_all = torch.cat(pred_probs_all)
    pred_probs = torch.cat(pred_probs)
    
    _, pred_labels = pred_probs.max(dim=1)
    # (64, num_nbrs, num_classes), max over dim=2
    _, pred_labels_all = pred_probs_all.max(dim=2)
    #First keep maximum for all classes between neighbors and then keep max between classes
    _, pred_labels_hard = pred_probs_all.max(dim=1)[0].max(dim=1)

    return pred_labels, pred_probs, pred_labels_all, pred_labels_hard

def refine_predictions(
    features,
    probs,
    banks):
    feature_bank = banks["features"]
    probs_bank = banks["probs"]
    pred_labels, probs, pred_labels_all, pred_labels_hard = soft_k_nearest_neighbors(
        features, feature_bank, probs_bank
    )

    return pred_labels, probs, pred_labels_all, pred_labels_hard

def contrastive_loss(logits_ins, pseudo_labels, mem_labels):
    # labels: positive key indicators
    labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).to(logits_ins.device)

    mask = torch.ones_like(logits_ins, dtype=torch.bool)
    mask[:, 1:] = torch.all(pseudo_labels.unsqueeze(1) != mem_labels.unsqueeze(0), dim=2) 
    logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).to(logits_ins.device))

    loss = F.cross_entropy(logits_ins, labels_ins)

    return loss

@torch.no_grad()
def update_labels(banks, idxs, features, logits):
    if logits.shape[-1] > 1:
        probs = F.softmax(logits, dim=1)
    else:
        probs = F.sigmoid(logits)
    start = banks["ptr"]
    end = start + len(idxs)
    idxs_replace = torch.arange(start, end).to(logits.device) % len(banks["features"])
    banks["features"][idxs_replace, :] = features
    banks["probs"][idxs_replace, :] = probs
    banks["ptr"] = end % len(banks["features"])

def div(logits, epsilon=1e-8):
    if logits.shape[-1] > 1:
        probs = F.softmax(logits, dim=1)
    else:
        probs = F.sigmoid(logits)
    probs_mean = probs.mean(dim=0)
    loss_div = -torch.sum(-probs_mean * torch.log(probs_mean + epsilon))
    return loss_div

def nl_criterion(output, y, nc):
    output = torch.log( torch.clamp(1.-F.softmax(output, dim=1), min=1e-5, max=1.) )
    labels_neg = ( (y.unsqueeze(-1).repeat(1, 1) + torch.LongTensor(len(y), 1).random_(1, nc).to(y.device)) % nc ).view(-1)
    l = F.nll_loss(output, labels_neg, reduction='none')
    return l

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

class PseudoLabelsWrapper(torch.utils.data.Dataset):
    def __init__(self, base_ds):
        self.base_ds = base_ds
        self.weak_augmentation =  self.base_ds.transform
        self.strong_augmentation = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                    p=0.8,  # not strengthened
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        imtensor, label, sensitive, idx = self.base_ds.__getitem__(idx)
        item = self.base_ds.dataframe.iloc[idx % len(self)]
        img = Image.fromarray(item['image']).convert('RGB')
        strong_augmented = self.strong_augmentation(img)
        strong_augmented2 = self.strong_augmentation(img)
        return imtensor, label, sensitive, idx, strong_augmented, strong_augmented2


class AdaMoCo(nn.Module):
    def __init__(self, src_model, momentum_model, features_length, num_classes, dataset_length, temporal_length, device):
        super(AdaMoCo, self).__init__()

        self.m = 0.999

        self.first_update = True

        self.src_model = src_model
        self.momentum_model = momentum_model

        self.momentum_model.requires_grad_(False)

        self.queue_ptr = 0
        self.mem_ptr = 0

        self.T_moco = 0.07

        #queue length
        self.K = min(16384, dataset_length)
        self.memory_length = temporal_length

        self.register_buffer("features", torch.randn(features_length, self.K))
        self.register_buffer(
            "labels", torch.randint(0, num_classes, (self.K,))
        )
        self.register_buffer(
            "idxs", torch.randint(0, dataset_length, (self.K,))
        )
        self.register_buffer(
            "mem_labels", torch.randint(0, num_classes, (dataset_length, self.memory_length))
        )

        self.register_buffer(
            "real_labels", torch.randint(0, num_classes, (dataset_length,))
        )

        self.features = F.normalize(self.features, dim=0)
        self.device = device
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.mem_labels = self.mem_labels.to(device)
        self.real_labels = self.real_labels.to(device)
        self.idxs = self.idxs.to(device)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        # encoder_q -> encoder_k
        for param_q, param_k in zip(
            self.src_model.parameters(), self.momentum_model.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def update_memory(self, epoch, idxs, keys, pseudo_labels, real_label):
        #print(pseudo_labels, real_label)
        start = self.queue_ptr
        end = start + len(keys)
        idxs_replace = torch.arange(start, end).to(real_label.device) % self.K
        self.features[:, idxs_replace] = keys.T
        self.labels[idxs_replace] = pseudo_labels
        self.idxs[idxs_replace] = idxs
        self.real_labels[idxs_replace] = real_label.long()
        self.queue_ptr = end % self.K

        self.mem_labels[idxs, self.mem_ptr] = pseudo_labels
        self.mem_ptr = epoch % self.memory_length

    @torch.no_grad()
    def get_memory(self):
        return self.features, self.labels

    def forward(self, im_q, im_k=None, cls_only=False):
        # compute query features
        logits_q, feats_q = self.src_model(im_q)

        if cls_only:
            return logits_q, feats_q

        q = F.normalize(feats_q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            _, k = self.momentum_model(im_k)
            k = F.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.features.clone().detach()])

        # logits: Nx(1+K)
        logits_ins = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits_ins /= self.T_moco

        # dequeue and enqueue will happen outside
        return logits_q, feats_q, logits_ins, k


class PseudoLabels(resampling):
    def __init__(self, opt, wandb):
        super(PseudoLabels, self).__init__(opt, wandb)
        wandb.run.log_code(".")
        self.set_data(opt)
        self.set_network(opt)
        self.set_optimizer(opt)
        self.banks = None
        self._val(self.val_loader)
        self.opt = opt

    def set_data(self, opt):
        def seed_worker(worker_id):
            np.random.seed(opt['random_seed'] )
            random.seed(opt['random_seed'])
        """Set up the dataloaders"""
        self.train_data, self.val_data, self.test_data, _, _, _, self.val_meta, self.test_meta = get_dataset(opt)

        if opt['resample_which']:
            g = torch.Generator()
            g.manual_seed(opt['random_seed'])
            weights = self.train_data.get_weights(resample_which = opt['resample_which'])
            sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True, generator=g)
        else:
            sampler = None

        self.train_loader = train_loader = torch.utils.data.DataLoader(
                            PseudoLabelsWrapper(self.train_data), batch_size=opt['batch_size'],
                            sampler=sampler,
                            shuffle=(not opt['resample_which']),
                            num_workers=8, #collate_fn=lambda x: default_collate(torch.Tensor(x)).to(opt['device']),
                            worker_init_fn=seed_worker, generator=g, pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(
                          PseudoLabelsWrapper(self.val_data), batch_size=opt['batch_size'],
                          sampler=sampler, shuffle=False, num_workers=8, worker_init_fn=seed_worker, generator=g, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(
                           PseudoLabelsWrapper(self.test_data), batch_size=opt['batch_size'],
                           shuffle=True, num_workers=8, worker_init_fn=seed_worker, generator=g, pin_memory=True)

    def set_network(self, opt):
        if self.is_3d:
            mod = import_module("models.basemodels_3d")
            cusModel = getattr(mod, self.backbone)
            self.network = cusModel(n_classes=self.output_dim, pretrained = self.pretrained).to(self.device)
            self.momentum_net = cusModel(n_classes=self.output_dim, pretrained = self.pretrained).to(self.device)
        elif self.is_tabular:
            mod = import_module("models.basemodels_mlp")
            cusModel = getattr(mod, self.backbone)
            self.network = cusModel(n_classes=self.output_dim, in_features= self.in_features, hidden_features = 1024).to(self.device)
            self.momentum_net = cusModel(n_classes=self.output_dim, pretrained=self.pretrained).to(self.device)
        else:
            mod = import_module("models.basemodels")
            cusModel = getattr(mod, self.backbone)
            self.network = cusModel(n_classes=self.output_dim, pretrained=self.pretrained).to(self.device)
            self.momentum_net = cusModel(n_classes=self.output_dim, pretrained=self.pretrained).to(self.device)
        state_dict = torch.load(opt['source_model'])
        self.network.load_state_dict(state_dict['model'])
        self.momentum_net.load_state_dict(state_dict['model'])
        self.moco_model = AdaMoCo(src_model = self.network, momentum_model = self.momentum_net,
                                  features_length=self.network.bottleneck_dim*2, num_classes=opt['num_classes'],
                                  dataset_length=len(self.train_data), temporal_length=5, device=opt['device'])
        self.device = opt['device']

    def set_optimizer(self, opt):
        optimizer_setting = opt['optimizer_setting']
        self.optimizer = optimizer_setting['optimizer'](
            params=self.network.parameters(),
            lr=optimizer_setting['lr'],
            weight_decay=optimizer_setting['weight_decay']
        )

    def state_dict(self):
        state_dict = {
            'moco_model': self.moco_model.state_dict(),
            'network': self.network.state_dict(),
            'momentum_net': self.momentum_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch
        }
        return state_dict

    def _train(self, loader):
        """Train the model for one epoch"""

        self.network.train()
        self.moco_model.train()

        train_loss = 0
        auc = 0.
        no_iter = 0

        tol_output, tol_target, tol_sensitive, tol_index, tol_uncertainty = [], [], [], [], []
        train_loss, no_iter = 0., 0.
        metrics = {} #{'Training AUC': 0.,'Training ACC': 0.,'Training Loss': 0.,}
        for sa in range(self.opt['sens_classes']):
            #metrics[f'Training AUC A{sa}'] = 0.0
            #metrics[f'Training ACC A{sa}'] = 0.0
            metrics[f'Training Loss Cls A{sa}'] = 0.0

        for i, (images, targets, sensitive_attr, index, strong_augmented, strong_augmented2) in enumerate(loader):

            if i == 0:
                for sa in range(self.opt['sens_classes']): #torch.unique(sensitive_attr):
                    print(f'Train A{sa} Total/Y0/Y1:', images[sensitive_attr==sa].shape, images[torch.logical_and((sensitive_attr==sa),(targets.squeeze()==0))].shape,
                                                                              images[torch.logical_and((sensitive_attr==sa),(targets.squeeze()==1))].shape)

            weak_x, y, sensitive_attr, idxs = images.to(self.device), targets.to(self.device), sensitive_attr.to(self.device), index.to(self.device)
            strong_x, strong_x2 = strong_augmented.to(self.device), strong_augmented2.to(self.device)

            self.optimizer.zero_grad()

            logits_w, feats_w = self.moco_model(weak_x, cls_only=True)
            #print(logits_w.shape)
            if self.opt['label_refinement']:
                with torch.no_grad():
                    if logits_w.shape[-1] > 1:
                        probs_w = F.softmax(logits_w, dim=1)
                    else:
                        probs_w = F.sigmoid(logits_w)
                    pseudo_labels_w, probs_w, _, _ = refine_predictions(feats_w, probs_w, self.banks)
            else:
                if logits_w.shape[-1] > 1:
                    probs_w = F.softmax(logits_w, dim=1)
                else:
                    probs_w = F.sigmoid(logits_w)
                pseudo_labels_w = probs_w.max(1)[1]

            logits_q, _, logits_ctr, keys = self.moco_model(strong_x, strong_x2)
            loss_ctr = contrastive_loss(
                    logits_ins=logits_ctr,
                    pseudo_labels=self.moco_model.mem_labels[idxs],
                    mem_labels=self.moco_model.mem_labels[self.moco_model.idxs]
                )

            # update key features and corresponding pseudo labels
            self.moco_model.update_memory(self.epoch, idxs, keys.to(self.device), pseudo_labels_w.to(self.device), y.squeeze())

            with torch.no_grad():
                #CE weights
                max_entropy = torch.log2(torch.tensor(min(2, self.opt['num_classes'])))
                w = entropy(probs_w)
                w = w / max_entropy
                w = torch.exp(-w)

            #Standard positive learning
            if self.opt['negative_learning']:
                #Standard negative learning
                loss_cls = ( nl_criterion(logits_q, pseudo_labels_w, self.opt['num_classes'])).mean()
                if self.opt['uncertainty_reweighting']:
                    loss_cls = (w * nl_criterion(logits_q, pseudo_labels_w)).mean()
            else:
                #print(logits_q, pseudo_labels_w)
                if self.opt['uncertainty_reweighting']:
                    #loss_cls = (w * CE(logits_q.squeeze(), pseudo_labels_w.long()))
                    if logits_w.shape[-1] > 1:
                        loss_cls = (w * BCE(logits_q[:, 1], pseudo_labels_w.float()))
                    else:
                        loss_cls = (w * BCE(logits_q.squeeze(), pseudo_labels_w.squeeze().float()))

            for sa in range(self.opt['sens_classes']): #torch.unique(sensitive_attr):
                #metrics[f'Training AUC A{sa}'] += 0.0
                #metrics[f'Training ACC A{sa}'] += 0.0
                metrics[f'Training Loss Cls A{sa}'] += loss_cls[sensitive_attr==sa].mean()

            loss_div = div(logits_w) + div(logits_q)

            loss = loss_cls.mean() + self.opt['loss_ctr_weight']*loss_ctr + self.opt['loss_div_weight']*loss_div

            update_labels(self.banks, idxs, feats_w, logits_w)

            loss.backward()
            self.optimizer.step()

            if logits_w.shape[-1] > 1:
                probs_w = probs_w[:, 1]
            auc += calculate_auc(probs_w.squeeze().cpu().data.numpy(), y.squeeze().cpu().data.numpy())
            train_loss += loss.item()
            no_iter += 1
            
            if self.log_freq and (i % self.log_freq == 0):
                self.wandb.log({'Training loss': train_loss / (i+1), 'Training AUC': auc / (i+1),
                                'Training loss_cls': loss_cls, 'Training loss_ctr': loss_ctr, 'Training loss_dic': loss_div,})
                for sa in range(self.opt['sens_classes']):
                    self.wandb.log({f'Training Loss Cls A{sa}': metrics[f'Training Loss Cls A{sa}'] / (i+1)})
        auc = 100 * auc / no_iter
        train_loss /= no_iter
        for sa in range(self.opt['sens_classes']):
            loss_sa = metrics[f'Training Loss Cls A{sa}'] / no_iter
            print('Training epoch {}: Loss Cls A{}: {}'.format(self.epoch, sa, loss_sa))
        print('Training epoch {}: AUC: {}'.format(self.epoch, auc))
        print('Training epoch {}: Loss: {}'.format(self.epoch, train_loss))

        self.epoch += 1

    @torch.no_grad()
    def _val(self, loader):
        print("Evaluating Dataset + Updating PseudoLabels!")

        self.network.eval()
        self.moco_model.eval()

        tol_logits, tol_probs, tol_index, tol_target = [], [], [], []
        tol_sensitive, features = [], []


        for i, (images, targets, sensitive_attr, idxs, strong_augmented, strong_augmented2) in enumerate(loader):
            images, targets, sensitive_attr = images.to(self.device), targets.to(self.device), sensitive_attr.to(self.device)

            if i == 0:
                print('\nVal Y0/Y1', targets[targets==0].shape, targets[targets==1].shape)
                for sa in range(self.opt['sens_classes']): #torch.unique(sensitive_attr):
                    sa = int(sa)
                    print(f'A{sa} Total/Y0/Y1:', images[sensitive_attr==sa].shape, images[torch.logical_and((sensitive_attr==sa),(targets.squeeze()==0))].shape,
                                                                              images[torch.logical_and((sensitive_attr==sa),(targets.squeeze()==1))].shape)

            logits_cls, feats = self.moco_model(images, cls_only=True)
            if logits_cls.shape[-1] > 1:
                probs = F.softmax(logits_cls, dim=1)
            else:
                probs = F.sigmoid(logits_cls)
            tol_logits.append(logits_cls)
            features.append(feats)
            tol_target.append(targets)
            tol_probs.append(probs)
            tol_index.append(idxs)
            tol_sensitive.append(sensitive_attr)

        features = torch.cat(features)
        tol_target = torch.cat(tol_target)
        tol_logits = torch.cat(tol_logits)
        tol_probs = torch.cat(tol_probs)
        tol_index = torch.cat(tol_index)
        tol_sensitive = torch.cat(tol_sensitive)

        rand_idxs = torch.randperm(len(features)).to(features.device)
        self.banks = {
            "features": features[rand_idxs][: 16384],
            "probs": tol_probs[rand_idxs][: 16384],
            "ptr": 0,
        }

        # refine predicted labels
        #pred_labels, _, _, _ = refine_predictions(features, tol_probs, self.banks)
        pred_labels = torch.argmax(tol_probs, 1)

        if tol_probs.shape[-1] > 1: tol_probs = tol_probs[:, 1]

        #print(tol_target.shape, tol_index.shape, tol_sensitive.shape, tol_probs.shape)
        log_dict, t_predictions, pred_df = calculate_metrics(tol_probs.squeeze().detach().cpu().data.numpy(),
                         tol_target.squeeze().detach().cpu().data.numpy(), tol_sensitive.detach().cpu().data.numpy(),
                         tol_index.detach().cpu().data.numpy(), self.opt['sens_classes'])
        pred_df['sensitive'] = tol_sensitive.detach().cpu().data.numpy()
        log_dict['Pseudo-Real Agreement'] = accuracy_score(tol_target.squeeze().detach().cpu().data.numpy(), pred_labels.squeeze().detach().cpu().data.numpy())
        #log_dict['Prob-Pseudo Agreement'] = accuracy_score(torch.argmax(tol_probs, 1).squeeze().detach().cpu().data.numpy(), pred_labels.squeeze().detach().cpu().data.numpy())
        log_dict['Pred-Pseudo Agreement'] = accuracy_score(torch.round(tol_probs).squeeze().detach().cpu().data.numpy(), pred_labels.squeeze().detach().cpu().data.numpy())
        log_dict['Pred-Real Agreement'] = accuracy_score(torch.round(tol_probs).squeeze().detach().cpu().data.numpy(), tol_target.squeeze().detach().cpu().data.numpy())
        self.wandb.log(log_dict)
        pprint(log_dict)
        return log_dict['Overall AUC'], log_dict['Overall AUC'], log_dict, pred_df

    def test(self):
        pass

    def _test(self, loader):
        return {}
