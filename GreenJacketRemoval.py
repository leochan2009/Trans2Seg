import argparse
import cv2, os
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(description='GreenJacketRemoval')
    # for visual
    parser.add_argument('--input-top-directory', type=str, default='.',
                        help='path to the top directory of input images')
    args = parser.parse_args()

    return args

def _get_trans10k_pairs(root, folders):
    img_paths = []
    mask_paths = []
    for folder in folders:
        subfolder = os.path.join(root, folder)
        assert os.path.exists(subfolder), "Please put the data in {SEG_ROOT}/datasets/transparent"
        maskRoot = os.path.join(subfolder, 'masks')
        for subdir, dirs, files in os.walk(maskRoot):
            for file in files:
                if not file == '.DS_Store' and (file[-4:] == '.png' or file[-4:] == '.jpg') and not '-no-jacket' in file:
                    mask_paths.append(os.path.join(subdir, file))
                    image = os.path.join(subfolder, 'images', os.path.basename(subdir), file)
                    img_paths.append(image)
    return img_paths, mask_paths

if __name__ == '__main__':
    args = parse_args()
    topDir = args.input_top_directory
    folders = ['GLDataset']#['Nov_ModifiedLVEdata', 'GLDataset']
    img_paths, mask_paths = _get_trans10k_pairs(topDir, folders)
    dilatation_size = 8
    element_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    erosion_size = 3
    element_ero = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
    for i in range(len(mask_paths)):
        mask = cv2.imread(mask_paths[i])
        img = cv2.imread(img_paths[i])
        #mask = cv2.imread('/Users/longquanchen/Desktop/Work/DeepLearning/Segment_Transparent_Objects/datasets/laserfiberBigData/SelectedVidsPostBlender/GLDataset/masks/lld2/lld2p_000081.png')
        #img =  cv2.imread('/Users/longquanchen/Desktop/Work/DeepLearning/Segment_Transparent_Objects/datasets/laserfiberBigData/SelectedVidsPostBlender/GLDataset/images/lld2/lld2p_000081.png')
        img_b = img[:, :, 0]
        img_g = img[:, :, 1]
        img_r = img[:, :, 2]
        img_jacket = (img_g>img_r) * (img_g>img_b)
        img_jacket = np.logical_and(img_jacket, mask[:,:,0])
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_jacket.astype('uint8'), 4, cv2.CV_32S)
        if stats.shape[0] >= 2:
            locSortIndex = np.argsort(centroids[:, 0])
            cX_jacket = centroids[locSortIndex[-1]][0]
            cY_jacket = centroids[locSortIndex[-1]][1]
            size_jacket = stats[locSortIndex[-1]][4] # perform some size filtering??
            label_jacket = (labels == locSortIndex[-1]).astype('uint8')
            label_jacket = cv2.dilate(label_jacket, element_dil) # these two line for filling the whole and enlarge the jacket mask a bit
            label_jacket = cv2.erode(label_jacket, element_ero)
            mask_noJacket = mask[:,:,0] - label_jacket*255
            cv2.imwrite(os.path.join(os.path.dirname(mask_paths[i]), os.path.basename(mask_paths[i])[:-4]+'-no-jacket.png') ,mask_noJacket)
        else: # when no jacket is detected, we still copy the mask to *-no-jacket.png file
            cv2.imwrite(os.path.join(os.path.dirname(mask_paths[i]), os.path.basename(mask_paths[i])[:-4] + '-no-jacket.png'),  mask)






