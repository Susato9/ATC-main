from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

import clip
import time
import numpy as np


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):#计算出所有类名的embedding
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()  #把句子补充成句子
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)#对文本进行编码
            class_embeddings =class_embeddings/ class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding =class_embedding/ class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights

def clip_classifier_train(classnames, template, clip_model):#计算出所有类名的embedding
    clip_weights = []
    for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()  #把句子补充成句子
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)#对文本进行编码
            class_embeddings =class_embeddings/ class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding =class_embedding/ class_embedding.norm()
            clip_weights.append(class_embedding)

    clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def build_cache_model(cfg, clip_model, train_loader_cache):

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []#清空缓存

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for (images, target) in tqdm(train_loader_cache):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)#用CLIP的视觉编码器对图片进行编码256*1024
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)#因为每次训练的顺序是一样的，所以这里的target也是一样的，所以只需要在第一次训练的时候添加就可以了
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))#把每个augment_epoch的tensor都放到一个list里面，最后把这个list转化为tensor，然后把这个tensor的shape变成[10, 1600, 1024]，其中10是augment_epoch的数量，1600是训练集的数量，1024是视觉编码器的输出维度
        #经过数据增强之后，得到了cache_keys和cache_values，cache_keys是一个list，长度为10，里面有augment_epoch个tensor，每个tensor的shape是[1, 1600, 1024]，cache_values是一个list，里面有augment_epoch个tensor，每个tensor的shape是[256]
        # print("cache_keys的shape：",cache_keys.size)   
        
        
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)#考虑一下此处可不可以用聚类的方法来做1600*1024，考虑用聚类转化为100*1024
        
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        
        # temp_tensor=torch.zeros(1000,1024)
        # temp_cache_values=torch.zeros(1000).long()
        # for i in range(1000):
        #     temp_tensor[i]=cache_keys[i*16:(i+1)*16].mean(dim=0)  #100*1024
        #     temp_cache_values[i]=torch.cat(cache_values,dim=0)[i*16]
        # cache_keys_proto=temp_tensor.cuda().half()
        # cache_values_proto=temp_cache_values.cuda()
        
        # cache_values_proto=F.one_hot(cache_values_proto).half()
        # cache_keys_proto = cache_keys_proto.permute(1, 0)
        
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values,dim=0)).half()
        

        torch.save(cache_keys, cfg['workdir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['workdir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['workdir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['workdir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values


def pre_load_features(cfg, split, clip_model, loader):

    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():#不需要计算梯度
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features= clip_model.encode_image(images)#64*1024
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['workdir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['workdir'] + "/" + split + "_l.pt")
   
    else:
        features = torch.load(cfg['workdir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['workdir'] + "/" + split + "_l.pt")
    
    return features, labels


def search_hp(cfg, affinity, cache_values, features, labels, clip_weights,mask_txt,image_mask):#寻找最佳的超参数
    year,month,day,hour,minute,second= time.localtime(time.time())[0:6]
    log_name="search"+str(cfg['backbone']).replace("/","_")+ "_" + str(str(cfg['shots']))+"_shots_" + str(year) + "_" + str(month) + "_" + str(day) + "_" + str(hour) + "_" + str(minute) + "_" + str(second) + ".txt"
    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                
                # affinity = features @ (cache_keys)
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                # cache_logits=affinity @ cache_values
                clip_logits = 100. * (features) @ (clip_weights+mask_txt)
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)
                
                with open(cfg["workdir"]+"/"+log_name,"a+") as f:
                        f.write("beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}\n".format(best_beta, best_alpha, best_acc))
                        
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha
                    with open(cfg["workdir"]+"/"+log_name,"a+") as f:
                        f.write("beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}*****************\n".format(best_beta, best_alpha, best_acc))
                    

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))
        
    return best_beta, best_alpha

def pairwise_distances(x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str) -> torch.Tensor:
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.
    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'l2':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances
    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise(ValueError('Unsupported similarity function'))




def attention_weights(test_feature,support_feature):
    Q=test_feature.unsqueeze(1)
    k=support_feature.repeat(test_feature.shape[0],1,1)
    v=support_feature.repeat(test_feature.shape[0],1,1)
    attention = torch.bmm(Q, k)
    weight_att=torch.bmm(attention,v.permute(0,2,1)).squeeze(1).softmax(dim=1)
    return weight_att
    

if __name__ == '__main__':
    x=torch.randn(1,1024)
    y=torch.randn(16000,1024)
    distances=pairwise_distances(x,y,'cosine')
    attention = (-distances).softmax(dim=1)
    print(attention.shape)