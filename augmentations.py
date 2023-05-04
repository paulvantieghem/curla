import numpy as np
from skimage.util.shape import view_as_windows
from kornia import augmentation as K
import torch

class IdentityAugmentation:
    def __init__(self, input_shape):
        assert len(input_shape) == 2, "Input shape must be 2D"
        self.input_shape = input_shape
        self.output_shape = input_shape

    def anchor_augmentation(self, image):
        return image

    def target_augmentation(self, image_batch):
        return image_batch


class RandomCrop(IdentityAugmentation):
    def __init__(self, input_shape):
        super().__init__(input_shape)
        self.cropping_factor = 0.84
        self.output_shape = tuple(int(np.ceil(x*self.cropping_factor)) for x in self.input_shape)

    def anchor_augmentation(self, image):
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

    def target_augmentation(self, image_batch):
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

        # Define the ColorJiggle augmentation with 100% probability
        self.aug1 = K.ColorJiggle(brightness=0.2, 
                                 contrast=0.2, 
                                 saturation=0.2, 
                                 hue=0.2, 
                                 same_on_batch=False, 
                                 p=1.0, 
                                 keepdim=True)
        
        # Define the RandomGrayscale augmentation with 10% probability
        self.aug2 = K.RandomGrayscale(same_on_batch=False, 
                                      p=0.1,
                                      keepdim=True)
        
        # Define the device to be used
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def anchor_augmentation(self, image):
        '''
        Returns the original image

        Args:
            image: Image with shape (channels*frame_stack, height, width)
        Returns:
            image: Image with shape (channels*frame_stack, height, width)
        '''

        return image
    
    def target_augmentation(self, image_batch):
        '''
        Applies a random transformation to the brightness, contrast, saturation 
        and hue of the image batch

        Args:
            image_batch: Batch of images with shape (batch_size, channels*frame_stack, height, width)
        Returns:
            augmented_batch: Batch of color jiggled images with shape (batch_size, channels*frame_stack, height, width)
        '''

        # Convert batch to Pytorch tensor
        tensor_batch = torch.from_numpy(image_batch.astype(np.float32) / 255.0)
        tensor_batch.to(self.device)

        # Each image in the batch is actually a frame stack of `frame_stack` images,
        # resulting in a tensor of shape (batch_size, channels*frame_stack, height, width),
        # which is not compatible with the color jiggling augmentation. Therefore, we
        # split the tensor into `frame_stack` tensors of shape (batch_size, channels, height, width),
        # apply the augmentation to each tensor and concatenate the resulting tensors back
        frame_stack = image_batch.shape[1]//3
        tensor_batch = torch.cat(torch.split(tensor_batch, frame_stack, dim=1), dim=0)

        # Perform color jiggling on batch with 100% probability
        tensor_batch = self.aug1(tensor_batch)

        # Perform random grayscale on batch with 10% probability
        tensor_batch = self.aug2(tensor_batch)

        # Reshape batch back to original shape
        tensor_batch = torch.cat(torch.split(tensor_batch, tensor_batch.shape[0]//frame_stack, dim=0), dim=1)

        # Convert batch back to numpy array
        augmented_batch = tensor_batch.numpy()*255.0
        augmented_batch = augmented_batch.astype(np.uint8)


        return augmented_batch
    

def make_augmentor(name, input_shape):
    augmentor = None
    if name == 'identity':
        augmentor = IdentityAugmentation(input_shape)
    elif name == 'random_crop':
        augmentor = RandomCrop(input_shape)
    elif name == 'color_jiggle':
        augmentor = ColorJiggle(input_shape)
    else:
        raise ValueError('augmentation is not supported: %s' % name)
    return augmentor