import numpy as np
from skimage.util.shape import view_as_windows

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
            cropped_image_batch: Batch of randomly cropped images with shape (batch_size, channels*frame_stack, *self.output_shape)
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
        cropped_image_batch = windows[np.arange(n), h1, w1]

        return cropped_image_batch
    

def make_augmentor(name, input_shape):
    augmentor = None
    if name == 'identity':
        augmentor = IdentityAugmentation(input_shape)
    elif name == 'random_crop':
        augmentor = RandomCrop(input_shape)
    else:
        raise ValueError('augmentation is not supported: %s' % name)
    return augmentor