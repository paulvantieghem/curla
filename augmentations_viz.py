import os
import cv2
import torch
from augmentations import make_augmentor
import matplotlib.pyplot as plt
import numpy as np

# Input image
image = cv2.imread(os.path.join('_out', 'im_2_133.png'))

# Create a frame stack
image = np.transpose(image, (2, 0, 1))
frames = [image, image, image]
frame_stack = np.concatenate(frames, axis=0).astype(np.float32)
frame_batch = np.expand_dims(frame_stack, axis=0)
frame_batch = np.concatenate([frame_batch, frame_batch], axis=0).astype(np.float32)

# Loop over augmentations and visualize
augmentations = ['identity', 'random_crop', 'color_jiggle','noisy_cover']
fig, axs = plt.subplots(len(augmentations), 2, figsize=(10, 15), dpi=300)
for aug in augmentations:
    frame_stack = np.concatenate(frames, axis=0).astype(np.float32)
    frame_batch = np.expand_dims(frame_stack, axis=0)
    frame_batch = np.concatenate([frame_batch, frame_batch], axis=0).astype(np.float32)
    augmentor = make_augmentor(aug, (90, 160))
    if aug == 'color_jiggle' or aug == 'noisy_cover':
        frame_stack = torch.as_tensor(frame_stack, device='cuda')
        frame_batch = torch.as_tensor(frame_batch, device='cuda')
    anchor_aug = augmentor.anchor_augmentation(frame_stack)
    target_aug = augmentor.target_augmentation(frame_batch)[0]
    if aug == 'color_jiggle' or aug == 'noisy_cover':
        anchor_aug = anchor_aug.detach().cpu().numpy()
        target_aug = target_aug.detach().cpu().numpy()
    anchor_aug = np.transpose(anchor_aug, (1, 2, 0))
    target_aug = np.transpose(target_aug, (1, 2, 0))
    anchor_aug = anchor_aug[:, :, 0:3]/255.0
    target_aug = target_aug[:, :, 0:3]/255.0
    anchor_aug = cv2.cvtColor(anchor_aug, cv2.COLOR_BGR2RGB)
    target_aug = cv2.cvtColor(target_aug, cv2.COLOR_BGR2RGB)
    axs[augmentations.index(aug), 0].imshow(anchor_aug)
    axs[augmentations.index(aug), 1].imshow(target_aug)
    axs[augmentations.index(aug), 0].set_title(f'{aug} anchor augmentation')
    axs[augmentations.index(aug), 1].set_title(f'{aug} target augmentation')
    axs[augmentations.index(aug), 0].axis('off')
    axs[augmentations.index(aug), 1].axis('off')
fig.tight_layout()
plt.savefig(os.path.join('augmentations.png'))
