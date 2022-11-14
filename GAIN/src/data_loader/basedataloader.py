import pytorch_lightning as pl

from torch.utils.data import DataLoader


class BaseDataloader(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        pass

    def prepare_data(self):
        pass

    def setup(self, stage):
        pass


    def train_dataloader(self):
        return DataLoader(self.dataloader_train, batch_size=self.batch_size, num_workers=0, pin_memory=True, prefetch_factor=4)

    def val_dataloader(self):
        return DataLoader(self.dataloader_eval, batch_size=self.batch_size, num_workers=0, pin_memory=True, prefetch_factor=4)