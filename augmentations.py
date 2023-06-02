import numpy as np
from skimage.util.shape import view_as_windows
import torch
import random
from kornia import augmentation as K

class IdentityAugmentation:
    def __init__(self, input_shape):
        assert len(input_shape) == 2, "Input shape must be 2D"
        self.input_shape = input_shape
        self.output_shape = input_shape

    def evaluation_augmentation(self, image):
        return image

    def training_augmentation(self, image_batch):
        return image_batch


class RandomCrop(IdentityAugmentation):
    def __init__(self, input_shape):
        super().__init__(input_shape)
        self.cropping_factor = 0.84
        self.output_shape = tuple(int(np.ceil(x*self.cropping_factor)) for x in self.input_shape)

    def evaluation_augmentation(self, image):
        '''
        Performs a center crop on the input image to the output shape

        Args:
            image: Image with shape (channels*frame_stack, height, width)
        Returns:
            cropped_image: Center cropped image with shape (channels*frame_stack, *self.output_shape)
        '''

        # Get image shapes and crop sizes
        h, w = self.input_shape
        new_h, new_w = self.output_shape
        top = (h - new_h)//2
        left = (w - new_w)//2

        # Perform center crop
        cropped_image = image[:, top:top + new_h, left:left + new_w]

        return cropped_image

    def training_augmentation(self, image_batch):
        '''
        Performs random cropping on a batch of images in a vectorized 
        way using sliding windows and picking out random ones

        Args:
            image_batch: Batch of images with shape (batch_size, channels*frame_stack, height, width)
        Returns:
            augmented_batch: Batch of randomly cropped images with shape (batch_size, channels*frame_stack, *self.output_shape)
        '''

        # Batch size
        n = image_batch.shape[0]

        # Determine cropping possibilities
        img_shape = image_batch.shape[2:4]
        crop_max_h = img_shape[0] - self.output_shape[0]
        crop_max_w = img_shape[1] - self.output_shape[1]
        image_batch = np.transpose(image_batch, (0, 2, 3, 1))
        h1 = np.random.randint(0, crop_max_h, n)
        w1 = np.random.randint(0, crop_max_w, n)

        # Creates all sliding windows combinations of size (output_size)
        windows = view_as_windows(image_batch, (1, self.output_shape[0], self.output_shape[1], 1))[..., 0,:,:, 0] # @TODO: Check correctness!

        # Selects a random window for each batch element
        augmented_batch = windows[np.arange(n), h1, w1]

        return augmented_batch
    

class ColorJiggle(IdentityAugmentation):

    def __init__(self, input_shape):
        super().__init__(input_shape)
        self.output_shape = self.input_shape
        
        # Define the ColorJiggle augmentation with 85% probability
        self.aug = K.ColorJiggle(brightness=0.0, 
                                 contrast=0.2, 
                                 saturation=0.5, 
                                 hue=0.5, 
                                 same_on_batch=False, 
                                 p=0.85, 
                                 keepdim=True)


    def evaluation_augmentation(self, image):
        '''
        Returns the original image

        Args:
            image: Image with shape (channels*frame_stack, height, width)
        Returns:
            image: Image with shape (channels*frame_stack, height, width)
        '''

        return image
    
    def training_augmentation(self, image_batch):
        '''
        Applies a random transformation to the brightness, contrast, saturation 
        and hue of the image batch

        Args:
            image_batch: Batch of images with shape (batch_size, channels*frame_stack, height, width)
        Returns:
            image_batch: Batch of color jiggled images with shape (batch_size, channels*frame_stack, height, width)
        '''

        # Normalize image batch to [0, 1]
        image_batch /= 255.0

        # Each image in the batch is actually a frame stack of `frame_stack` images,
        # resulting in a tensor of shape (batch_size, channels*frame_stack, height, width),
        # which is not compatible with the augmentation function. Therefore, we reshape
        # the batch to (batch_size*frame_stack, channels, height, width)
        frame_stack = image_batch.shape[1]//3
        image_batch = image_batch.reshape(-1, 3, *self.input_shape)

        # Perform color jiggling augmentation on batch
        image_batch = self.aug(image_batch)

        # Reshape batch back to original shape
        image_batch = image_batch.reshape(-1, 3*frame_stack, self.input_shape[0], self.input_shape[1])

        # Denormalize image batch back to [0, 255]
        image_batch *= 255.0

        return image_batch
    
class NoisyCover(IdentityAugmentation):
        
    def __init__(self, input_shape):
        super().__init__(input_shape)
        self.output_shape = self.input_shape
        top_ratio = 0.31
        bottom_ratio = 0.20
        self.h = self.input_shape[0]
        self.top = int(np.ceil(self.h * top_ratio))
        self.bottom = int(np.ceil(self.h * bottom_ratio))

        # Indexes of rows of the image that should get covered
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cover_indexes = torch.tensor(np.concatenate((np.arange(0, self.top), 
                                                          np.arange(self.h-self.bottom, self.h))), 
                                                          dtype=torch.int64, 
                                                          device=device)
        
        # Define the RandomGaussianNoise augmentation with 100% probability
        self.aug = K.RandomGaussianNoise(mean=0.0, std=10.0, p=1.0)


    def evaluation_augmentation(self, image):
        '''
        Returns the original image

        Args:
            image: Image with shape (channels*frame_stack, height, width)
        Returns:
            image: Image with added Gaussian noise with shape (channels*frame_stack, height, width)
        '''

        return image
    
    def training_augmentation(self, image_batch):
        '''
        Applies a random transformation to the brightness, contrast, saturation 
        and hue of the image batch

        Args:
            image_batch: Batch of images with shape (batch_size, channels*frame_stack, height, width)
        Returns:
            image_batch: Batch of partially covered (by blocks of a random color), noisy images 
            with shape (batch_size, channels*frame_stack, height, width)
        '''

        # Each image in the batch is actually a frame stack of `frame_stack` images,
        # resulting in a tensor of shape (batch_size, channels*frame_stack, height, width),
        # which is not compatible with the augmentation function. Therefore, we reshape
        # the batch to (batch_size*frame_stack, channels, height, width)
        frame_stack = image_batch.shape[1]//3
        image_batch = image_batch.reshape(-1, 3, *self.input_shape)

        # Cover top and bottom of image with random color
        image_batch[:, 0, :, :].index_fill_(1, self.cover_indexes, np.random.randint(0, 255))
        image_batch[:, 1, :, :].index_fill_(1, self.cover_indexes, np.random.randint(0, 255))
        image_batch[:, 2, :, :].index_fill_(1, self.cover_indexes, np.random.randint(0, 255))

        # Add gaussian noise to the image
        image_batch = self.aug(image_batch)

        # Reshape batch back to original shape
        image_batch = image_batch.reshape(-1, 3*frame_stack, self.input_shape[0], self.input_shape[1])

        # Clip values to [0, 255] with torch.clamp
        image_batch = torch.clamp(image_batch, 0, 255)

        return image_batch
    

def make_augmentor(name, input_shape):
    print(f'CHOSEN AUGMENTATION: {name}')
    augmentor = None
    if name == 'identity':
        augmentor = IdentityAugmentation(input_shape)
    elif name == 'random_crop':
        augmentor = RandomCrop(input_shape)
    elif name == 'color_jiggle':
        augmentor = ColorJiggle(input_shape)
    elif name == 'noisy_cover':
        augmentor = NoisyCover(input_shape)
    else:
        raise ValueError('augmentation is not supported: %s' % name)
    return augmentor
