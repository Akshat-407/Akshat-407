import torch
import pandas as pd
import numpy as np
import random
from sklearn.utils.class_weight import compute_class_weight

def balanced_class_distribution(class_weights, dataset, device):
    sample_weights = [0] * len(dataset)
    for idx, (data, label) in enumerate(dataset):
        label = label.to(dtype=torch.long, device=device)
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight
    return sample_weights


def balanced_class_distribution_multi(class_weights1, class_weights2, dataset, device):
    sample_weights1 = [0] * len(dataset)
    sample_weights2 = [0] * len(dataset)
    for idx, (data, label1, label2) in enumerate(dataset):
        label1 = label1.to(dtype=torch.long, device=device)
        class_weight1 = class_weights1[label1]
        sample_weights1[idx] = class_weight1

        label2 = label2.to(dtype=torch.long, device=device)
        class_weight2 = class_weights2[label2]
        sample_weights2[idx] = class_weight2

    return sample_weights1, sample_weights2


def count_class_distribution(file, which_labels):
    read_file = pd.read_excel(file)
    labels = read_file[which_labels].tolist()
    values, counts = np.unique(labels, return_counts=True)
    return counts[::-1]


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def labels_for_classification(labels):
    if labels[0].detach().cpu().numpy() == 0:
        title = 'HRS Positive'
    elif labels[0].detach().cpu().numpy() == 1:
        title = 'HRS Negative'
    return title

def labels_for_classification2(labels):
    if labels[0].detach().cpu().numpy() == 0:
        title = 'Follow Up'
    elif labels[0].detach().cpu().numpy() == 1:
        title = 'No Follow Up'
    return title
    
    
def label_encoder(labels, class_type = 'subtype_class'):

    if class_type == 'subtype_class':
        if labels == 'Luminal A' or labels == 'Luminal B':
            label = 0
        elif labels == 'HER2-enriched' or labels == 'triple negative':
            label = 1
    elif class_type == 'subtype':
        if labels == 'Luminal A':
            label = 0
        elif labels == 'Luminal B':
            label = 1
        elif labels == 'HER2-enriched':
            label = 2
        elif labels == 'triple negative':
            label = 3
    elif class_type=='CESM':
        if labels == 'Normal':
            label = 0
        elif labels == 'Benign':
            label = 1
        elif labels == 'Malignant':
            label = 2 
    else:
    
        if labels == 'mass_MALIGNANT':
            label = 0
        elif labels == 'mass_BENIGN':
            label = 1
        elif labels == 'mass_BENIGN_WITHOUT_CALLBACK':
            label = 2
        elif labels == 'calcification_MALIGNANT':
            label = 3
        elif labels == 'calcification_BENIGN':
            label = 4
        elif labels == 'calcification_BENIGN_WITHOUT_CALLBACK':
            label = 5
    return label
    
def compute_sample_weights(csv_file, class_type):
    read_csv = pd.read_csv(csv_file)
    
    if class_type == 'subtype_class' or class_type=='subtype':
        print('here')
        labels = read_csv['subtype']
        print(labels)
        
    elif class_type == 'CESM':
        labels = read_csv['Pathology Classification/ Follow up']
    else:
        abnorm = read_csv['abnormality type']
        pathologies = read_csv['pathology']
        labels = abnorm + '_' + pathologies
    targets = labels.tolist()
    tar = [label_encoder(labels, class_type = class_type) for labels in targets]
    target = np.array(tar)
    class_sample_count = np.unique(target, return_counts=True)[1]
    print('Class Sample Count : ', class_sample_count)
    class_weights=compute_class_weight(class_weight ='balanced', classes = np.unique(tar),y = tar)
    class_weights_torch=torch.tensor(class_weights,dtype=torch.float)

    
    weight = 1. / class_sample_count
    print('Weights: ', class_weights)
    samples_weight = class_weights[target]
    return class_weights_torch, samples_weight
