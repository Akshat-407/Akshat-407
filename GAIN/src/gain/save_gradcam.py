import cv2
import os
import numpy as np
from matplotlib import pyplot as plt


class GAINSaveHeatmap:
    def __init__(self,
        heatmaps,
        image_names,
        images,
        mask,
        epoch,
        out_folder,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.heatmaps = heatmaps
        self.image_names = image_names
        self.images = images
        self.mask = mask
        self.mean = mean
        self.std = std
        self.out_folder = out_folder
        self.epoch = epoch
        os.makedirs(self.out_folder, exist_ok=True)

    def denorm(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

    def _combine_heatmap_with_image(self, image, heatmap):
        heatmap = heatmap - np.min(heatmap)
        if np.max(heatmap) != 0:
            heatmap = heatmap / np.max(heatmap)
        heatmap = np.float32(cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET))

        scaled_image = self.denorm(image) * 255
        scaled_image = scaled_image.cpu().numpy().astype(np.uint8).transpose((1, 2, 0))

        cam = heatmap + np.float32(scaled_image)
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)

        heat_map = cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)
        return heat_map

    def on_batch_end(self):

        if (self.epoch%10==0):
            outdir = os.path.join(self.out_folder, f"epoch{self.epoch}")
            os.makedirs(outdir, exist_ok=True)

        # rand_wandb_images = np.random.randint(0, len(image_names), 2)

        for i, (image, ac,mask,  image_name) in enumerate(zip(self.images, self.heatmaps, self.mask, self.image_names)):
            ac = ac.data.cpu().numpy()[0]

            heat_map = self._combine_heatmap_with_image(
                image=image,
                heatmap=ac
            )
            image = image.data.cpu().numpy()[0]
            mask = mask.data.cpu().numpy()[0]
            plt.subplot(1, 3, 1), plt.imshow(image, 'gray')
            plt.subplot(1, 3, 2), plt.imshow(heat_map, 'gray')
            plt.subplot(1, 3, 3), plt.imshow(mask, 'gray')
            file = os.path.basename(image_name)
            filename = os.path.join(outdir, file)
            plt.savefig(filename)