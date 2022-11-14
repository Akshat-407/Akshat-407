import torch
from torch.utils.data import WeightedRandomSampler
from src.data_loader.mammo_data_cluster import MammographyCBISDDSM
from src.data_loader.segmentation_transforms import SegmentationPresetEval
from torch import nn, optim
from src.gain.train_gain import train_gain_criterion
from src.gain.train_gain_mask import train_gain_mask_criterion
from src.models.gain import GAIN


def main():
    #wandb.login()  ##
    #wandb.init(project="GAIN_New")

    num_epoch = 100
    learning_rate = 0.00005
    weight_decay = 0.005
    target_names = ['BENIGN_WITHOUT_CALLBACK', 'BENIGN', 'MALIGNANT']
    num_classes = len(target_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu" )
    image_height = 1156
    image_width = 892
    batch_size = 4
    out_folder = '/cluster/me66vepi/GAIN_working/GAIN/Output'
    task = 'GAIN'  # Options: 'GradCAM'(Baseline), 'GAIN', 'GAIN_Mask'
    print('Device : ', device)
    transforms = SegmentationPresetEval(base_size=image_height)

    dataset_train = MammographyCBISDDSM("/cluster/me66vepi/GAIN_working/GAIN/data/train_CV1.csv",
                                        transform=transforms,
                                        image_height=1156,
                                        image_width=892,
                                        category='Classification')

    dataset_valid = MammographyCBISDDSM("/cluster/me66vepi/GAIN_working/GAIN/data/valid_CV1.csv",
                                       transform=transforms,
                                       image_height=1156,
                                       image_width=892,
                                       category='Classification')

    dataloader = {
        'train': torch.utils.data.DataLoader(dataset_train,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             sampler=None,
                                             num_workers=0,
                                             pin_memory=True,
                                             drop_last=True),

        'valid': torch.utils.data.DataLoader(dataset_valid,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             sampler=None,
                                             num_workers=0,
                                             pin_memory=True,
                                             drop_last=False)
    }


    gain = GAIN(grad_layer='_features.7', num_classes=3, device = device).to(device)
    gain_criterion = nn.CrossEntropyLoss().to(device)
    soft_mask_criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(gain.parameters(),
                              lr=learning_rate,
                              weight_decay=weight_decay)

    if torch.cuda.device_count() > 1:
	    #print("**********Let's use", torch.cuda.device_count(), "GPUs!")
        gain = nn.DataParallel(gain)
	

    #gain = gain.to(device)

    if task == 'GAIN':
        train_gain_criterion(num_epoch,
                             optimizer,
                             gain_criterion,
                             out_folder,
                             device,
                             gain,
                             dataloader_train=dataloader['train'],
                             dataloader_valid=dataloader['valid'])



    elif task == 'GAIN_Mask':
        train_gain_mask_criterion(num_epoch,
                                  optimizer,
                                  gain_criterion,
                                  soft_mask_criterion,
                                  out_folder,
                                  device,
                                  gain,
                                  dataloader_train=dataloader['train'],
                                  dataloader_valid=dataloader['valid'])

    elif task == 'GradCAM':
        ### Create simple classification training loop
        ### Check save_gradcam file and create gradcam
        pass

    else:
        raise NotImplementedError('Select from: GAIN, GAIN_Mask, GradCAM')
if __name__ == "__main__":
    main()


