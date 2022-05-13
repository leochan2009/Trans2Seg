"""Pascal Transparent Semantic Segmentation Dataset."""
import os
import logging
import torch, cv2
import numpy as np

from PIL import Image
from .seg_data_base import SegmentationDataset


class TransparentUroDataSegmentationBoundary(SegmentationDataset):
    """ADE20K Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to ADE20K folder. Default is './datasets/ade'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = TransparentUroDataSegmentationBoundary(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = ['GLDataset']#['Nov_ModifiedLVEdata', 'GLDataset']
    NUM_CLASS = 2

    def __init__(self, root='datasets/transparent', split='test', mode=None, transform=None, **kwargs):
        super(TransparentUroDataSegmentationBoundary, self).__init__(root, split, mode, transform, **kwargs)
        self.images, self.masks = _get_trans10k_pairs(self.root, self.BASE_DIR, split)
        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        logging.info('Found {} images in the folder {}'.format(len(self.images), root))

    def _mask_transform(self, mask):
        mask_np = np.array(mask)
        if len(mask_np.shape) ==3:
            return torch.LongTensor((mask_np / mask_np.max().astype('int32'))[:,:,0])
        else:
            return torch.LongTensor(mask_np/mask_np.max().astype('int32'))
        #return torch.LongTensor(np.array(mask).astype('int32'))

    def _val_sync_transform_resize(self, img, mask):
        short_size = self.crop_size
        img = img.resize(short_size, Image.BILINEAR)
        mask = mask.resize(short_size, Image.NEAREST)

        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def get_boundary(self, mask, thicky=8):
        tmp = mask.data.numpy().astype('uint8')
        contour, _ = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boundary = np.zeros_like(tmp)
        boundary = cv2.drawContours(boundary, contour, -1, 1, thicky)
        return boundary

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])#.convert("P")
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask, resize=True)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform_resize(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._val_sync_transform_resize(img, mask)
        # general resize, normalize and to Tensor
        if self.transform is not None:
            img = self.transform(img)
        boundary = self.get_boundary(mask)
        boundary = torch.LongTensor(np.array(boundary).astype('int32'))
        return img, mask, boundary, os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1

    @property
    def classes(self):
        """Category names."""
        return ('Background', 'fiber')
        # return ('Background', 'Shelf', 'Jar or Tank', 'Freezer', 'Window',
        #         'Glass Door', 'Eyeglass', 'Cup', 'Floor Glass', 'Glass Bow',
        #         'Water Bottle', 'Storage Box')


def _get_trans10k_pairs(root, folders, mode='train', useJacket=True):
    img_paths = []
    mask_paths = []
    for folder in folders:
        subfolder = os.path.join(root, folder)
        assert os.path.exists(subfolder), "Please put the data in {SEG_ROOT}/datasets/transparent"
        maskRoot = os.path.join(subfolder, 'masks')
        mask_subpaths = []
        img_subpaths = []
        for subdir, dirs, files in os.walk(maskRoot):
            for file in files:
                if not file == '.DS_Store' and (file[-4:] == '.png' or file[-4:] == '.jpg') and (('-no-jacket' in file) ^ useJacket): # xor operation for file filtering
                    mask_subpaths.append(os.path.join(subdir, file))
                    if useJacket:
                        image = os.path.join(subfolder, 'images', os.path.basename(subdir), file)
                    else:
                        image = os.path.join(subfolder, 'images', os.path.basename(subdir), file[:-14]+'.png')
                    img_subpaths.append(image)
        indexTrain = int(len(mask_subpaths) * 0.8)
        if mode == 'train':
            img_paths.extend(img_subpaths[:indexTrain])
            mask_paths.extend(mask_subpaths[:indexTrain])
        elif mode == "val":
            img_paths.extend(img_subpaths[indexTrain:])
            mask_paths.extend(mask_subpaths[indexTrain:])
        else:
            assert mode == "test"
            img_paths.extend(img_subpaths[indexTrain:])
            mask_paths.extend(mask_subpaths[indexTrain:])

    return img_paths, mask_paths


if __name__ == '__main__':
    train_dataset = TransparentUroDataSegmentationBoundary()