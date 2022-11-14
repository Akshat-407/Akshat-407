import torch
import math
import random




class BaseDataset(torch.utils.data.IterableDataset):
    def __init__(self, imgs,labels,transforms) -> None:
        self.imgs = imgs
        self.labels=labels
        self.transforms = transforms

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        random.shuffle(self.imgs)
        if worker_info is None:
            local_imgs = self.imgs
        else:
            per_worker = int(math.ceil(len(self.imgs) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = 0 + worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.imgs))

            local_imgs = self.imgs[iter_start:iter_end]

        return iter(self.image_yielder(local_imgs, self.labels))

    def image_yielder(self, imgs, labels):
        pass