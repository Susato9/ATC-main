import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

from datasets.imagenet import ImageNet
import clip
from utils import *
import time
import numpy as np
from clip.model import linearCLIP,ATC
import torch


def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config',help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()
    return args



def ATC(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, clip_model, train_loader_F,test_loader,imagenet,params):
    
    year,month,day,hour,minute,second= time.localtime(time.time())[0:6]
    if cfg['backbone']=='RN50':
        embed_dim=1024
    else:
        embed_dim=512
    guide_net=nn.LSTM(embed_dim,embed_dim,3,batch_first=True).to(clip_model.dtype).cuda()
    
    
    opimizer2=torch.optim.AdamW(guide_net.parameters(), lr=0.0015,eps=1e-3)
    scheduler2=torch.optim.lr_scheduler.MultiStepLR(opimizer2, milestones=[15,20], gamma=0.5)
    beta, alpha = cfg['init_beta'], cfg['init_alpha']

    best_acc, best_epoch = 0.0, 0
    log_name="acc"+str(cfg['backbone']).replace("/","_") + "_" + str(str(cfg['shots']))+"_shots_" + str(year) + "_" + str(month) + "_" + str(day) + "_" + str(hour) + "_" + str(minute) + "_" + str(second) + ".txt"
    for train_idx in range(cfg['train_epoch']):
        guide_net.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features= clip_model.encode_image(images)  #ATC_txt:(batch_size,512)
                image_features = image_features/image_features.norm(dim=-1, keepdim=True)#(batch_size,1024)
           
            txt_emb,_=guide_net(image_features)
            txt_emb=(txt_emb/txt_emb.norm(dim=-1, keepdim=True)).mean(dim=0).unsqueeze(0)
            mask_txt=txt_emb.repeat(1000,1).permute(1,0)
            
            
    
            clip_logits = 100. * (image_features)@ (clip_weights+mask_txt)
            affinity=image_features@cache_keys
           
            cache_logits=beta*affinity@cache_values
            
            tip_logits = clip_logits + cache_logits * alpha
            loss = F.cross_entropy(tip_logits, target)

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100.*len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            opimizer2.zero_grad()
            loss.backward()
            opimizer2.step()
            
        # scheduler1.step()
        scheduler2.step()
        current_lr = scheduler2.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))
        
        # adapter.eval()
        clip_model.eval()
        guide_net.eval()
        affinity_list=[]
        correct_samples, all_samples = 0, 0
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(test_loader)):
                images, target = images.cuda(), target.cuda()
                image_features= clip_model.encode_image(images)
                image_features =image_features/ image_features.norm(dim=-1, keepdim=True)
                
                txt_emb,_=guide_net(image_features)
                txt_emb=(txt_emb/txt_emb.norm(dim=-1, keepdim=True)).mean(dim=0).unsqueeze(0)
                mask_txt=txt_emb.repeat(1000,1).permute(1,0)
                
                clip_logits = 100. * (image_features) @ (clip_weights+mask_txt)
    
                affinity=image_features@cache_keys
    
                cache_logits=beta*affinity@cache_values
                tip_logits = clip_logits + cache_logits * alpha
                acc= cls_acc(tip_logits, target)
                correct_samples += acc /100.* len(tip_logits)
                all_samples += len(tip_logits)
                
    
        acc_all=correct_samples*100./all_samples
        if acc_all > best_acc:
            best_acc = acc_all
            best_epoch = train_idx     
            torch.save(clip_model.state_dict(), cfg['workdir'] + "/best_F_" + str(cfg['backbone']).replace("/","_")+str(cfg['shots']) + "shots_clip_model.pt")
            torch.save(guide_net.state_dict(), cfg['workdir'] + "/best_F_" +cfg['backbone'].replace("/","_")+ str(cfg['shots']) + "shots_guide_net.pt")
            
            best_mask=mask_txt
            best_image_mask=0
        
        print("**** Tip-Adapter-F's test accuracy: {:.4f}. ****\n".format(acc_all))
        with open(cfg['workdir']+"/"+log_name,"a+") as f:
            f.write(str(acc_all)+"\n")
            
    clip_model.load_state_dict(torch.load(cfg['workdir'] + "/best_F_" +cfg['backbone'].replace("/","_")+ str(cfg['shots']) + "shots_clip_model.pt"))
    print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)
    # Search Hyperparameters
    # affinity=test_features@(cache_keys+best_image_mask.half().cuda())
    affinity=test_features@cache_keys
    _ = search_hp(cfg, affinity, cache_values, test_features, test_labels, clip_weights,best_mask,best_image_mask)
    

def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    workdir = os.path.join('work_dirs', cfg['dataset'])+"/"+str(cfg['shots'])+"shots"
    os.makedirs(workdir, exist_ok=True)
    cfg['workdir'] = workdir

    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    model=ATC(cfg,clip_model)

    print(model)
    print("learnable  parameters ####################################")
    
    for name, param in model.named_parameters():
        if "image_mask" not in name:
            param.requires_grad_(False)
        else:
            print(name)
    params = filter(lambda p: p.requires_grad, model.parameters()) 
    
    for name, value in model.named_parameters():
        print(name, value.requires_grad)  

    random.seed(1)
    torch.manual_seed(1)
    
    print("Preparing ImageNet dataset.")
    imagenet = ImageNet(cfg['root_path'], cfg['shots'], preprocess)
    
    test_loader = torch.utils.data.DataLoader(imagenet.test, batch_size=64, num_workers=8, shuffle=False)
    train_loader_cache = torch.utils.data.DataLoader(imagenet.train, batch_size=64, num_workers=8, shuffle=False)
    train_loader_F = torch.utils.data.DataLoader(imagenet.train, batch_size=64, num_workers=8, shuffle=True)

    # Textual features
    print("Getting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(imagenet.classnames, imagenet.template, model)
    
  
    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(cfg, model, train_loader_cache)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", model, test_loader)


    # ------------------------------------------ ATC ------------------------------------------
    ATC(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, model, train_loader_F,test_loader,imagenet,params)
           

if __name__ == '__main__':
    main()

