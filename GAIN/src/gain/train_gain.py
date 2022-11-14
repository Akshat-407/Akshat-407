import torch
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score
import torch.nn.functional as F
from src.gain.save_gradcam import GAINSaveHeatmap
from src.utils.pytorchtools import EarlyStopping


def train_gain_criterion(num_epoch,
                         optimizer,
                         gain_criterion,
                         out_folder,
                         device,
                         gain,
                         dataloader_train,
                         dataloader_valid):

    '''Adapted from callbacks.py GAINCriterionCallback'''


    metric_collection_train = MetricCollection([
        Accuracy().to(device),
        Precision(num_classes=3, average='micro').to(device),
        Recall(num_classes=3, average='micro').to(device),
        F1Score(num_classes=3, average='micro').to(device),
        #ConfusionMatrix(num_classes=3).to(device),
    ])
    metric_collection_valid = MetricCollection([
        Accuracy().to(device),
        Precision(num_classes=3, average='micro').to(device),
        Recall(num_classes=3, average='micro').to(device),
        F1Score(num_classes=3, average='micro').to(device),
        #ConfusionMatrix(num_classes=3).to(device),
    ])
    early_stopping = EarlyStopping(patience=10, verbose=True)
    for epoch in range(num_epoch):
        train_loss_all = []
        loss = 0.0
        for images, labels, segmentations,_ in dataloader_train:
            images = images.to(device)
            labels = labels.type(torch.LongTensor).to(device)
            optimizer.zero_grad()
            logits, logits_am, heatmap = gain(images, labels)
            loss = gain_criterion(logits, labels) * 0.8
            loss_am = F.softmax(logits_am)
            loss_am, _ = loss_am.max(dim=1)
            loss_am = loss_am.sum() / loss_am.size(0)
            loss += loss_am * 0.2
            optimizer.step()
            train_loss_all.append(loss.cpu().data.item())
            acc_train = metric_collection_train(logits, labels)

        with torch.no_grad():
            valid_losses_all = []
            for images_v, labels_v,segmentations_v, filenames_v in dataloader_valid:
                images_v = images_v.to(device)
                labels_v = labels_v.type(torch.LongTensor).to(device)
                logits_v, logits_am_v, heatmap_v = gain(images_v, labels_v)
                loss_v = gain_criterion(logits_v, labels_v) * 0.8 # keep
                loss_am_v = F.softmax(logits_am_v)
                loss_am_v, _v = loss_am_v.max(dim=1)
                loss_am_v = loss_am_v.sum() / loss_am_v.size(0)
                loss_v += loss_am_v * 0.2
                valid_losses_all.append(loss_v.cpu().data.item())
                acc_valid = metric_collection_valid(logits_v, labels_v)
                if epoch%20==0:
                    gains = GAINSaveHeatmap(heatmap_v,filenames_v,images_v,segmentations_v, epoch, out_folder)
                    gains.on_batch_end()
        train_loss_avg = (sum(train_loss_all) / len(train_loss_all))
        valid_loss_avg = (sum(valid_losses_all) / len(valid_losses_all))
        early_stopping(valid_loss_avg, gain)
        acc_train = metric_collection_train.compute()
        acc_valid = metric_collection_valid.compute()
        print('Epoch: ', epoch)
        print('Train Loss : ',train_loss_avg)
        print('Train Metrics', acc_train)
        print('')
        print('Valid Metrics', acc_valid)
        print('Validation Loss: ', valid_loss_avg)
        print('')
        metric_collection_train.reset()
        metric_collection_valid.reset()
        if early_stopping.early_stop:
            print("Early stopping")
            break