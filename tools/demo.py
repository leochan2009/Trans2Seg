import os, shutil
import sys
import torch
from subprocess import PIPE, Popen
import cv2
import numpy as np

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from torchvision import transforms
from PIL import Image
from segmentron.utils.visualize import get_color_pallete
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.config import cfg


class HighQualityVideoWriter:
    def __init__(self, out_fname):
        # ffmpeg setup
        self.pipe = Popen(
            [
                "ffmpeg",
                "-y",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "-r",
                "25",
                "-i",
                "-",
                "-vcodec",
                "h264",
                "-crf",
                "0",
                "-r",
                "25",
                out_fname,
            ],
            stdin=PIPE,
        )

    def write(self, frame):
        im = Image.fromarray(frame)
        im.save(self.pipe.stdin, "JPEG")

    def close(self):
        self.pipe.stdin.close()
        self.pipe.wait()
        self.pipe = None


def demo():
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()
    default_setup(args)
    outputVideo = False
    # output folder
    output_dir = os.path.join(cfg.VISUAL.OUTPUT_DIR, 'vis_result_{}_{}_{}_{}'.format(
        cfg.MODEL.MODEL_NAME, cfg.MODEL.BACKBONE, cfg.DATASET.NAME, cfg.TIME_STAMP))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    shutil.copy(args.config_file, output_dir)
    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
    ])

    model = get_segmentation_model().to(args.device)
    model.eval()
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.size())
    if args.input_img[-4:] == '.mp4':
        cap = cv2.VideoCapture(args.input_img)
        video_writer = HighQualityVideoWriter(output_dir+'.mp4')
        frame_num = -1

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, cfg.TEST.CROP_SIZE)
            image = image[cfg.TEST.ROI_START[0]:cfg.TEST.ROI_END[0], cfg.TEST.ROI_START[1]:cfg.TEST.ROI_END[1]]
            images = transform(image).unsqueeze(0).to(args.device)
            with torch.no_grad():
                output = model(images)

            pred = torch.argmax(output[0][0], 1).squeeze(0).cpu().data.numpy()
            pred = cv2.cvtColor(np.float32(pred), cv2.COLOR_GRAY2BGR)
            pred_filled = np.zeros(frame.shape)
            pred_filled[:,:,:] = np.zeros(frame.shape)
            #pred = cv2.resize(pred,(400,400)).astype('uint8')
            pred_filled[cfg.TEST.ROI_START[0]:cfg.TEST.ROI_END[0], cfg.TEST.ROI_START[1]:cfg.TEST.ROI_END[1],0] = pred[:,:,0]*255
            #mask = get_color_pallete(pred, 'trans10kv2')
            output_image = cv2.addWeighted(frame, 1, pred_filled.astype('uint8'), 0.3, 0)
            if outputVideo:
                video_writer.write(output_image)
            else:
                cv2.imwrite(os.path.join(output_dir, os.path.basename(args.input_img)[:-4] + '-'+ str(frame_num)+'.png'), output_image)
            print(frame_num)
        cap.release()
        if outputVideo:
            video_writer.close()
    else:
        img_paths = []
        if os.path.isdir(args.input_img):
            for x in os.listdir(args.input_img):
                if not '_mask' in x and (not x == '.DS_Store'):
                    img_paths.append(os.path.join(args.input_img, x))
        elif os.path.isfile(args.input_img):
            img_paths = [args.input_img]
        for img_path in img_paths:
            frame = cv2.imread(img_path)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, cfg.TEST.CROP_SIZE)
            image = image[cfg.TEST.ROI_START[0]:cfg.TEST.ROI_END[0], cfg.TEST.ROI_START[1]:cfg.TEST.ROI_END[1]]
            images = transform(image).unsqueeze(0).to(args.device)
            with torch.no_grad():
                output = model(images)

            pred = torch.argmax(output[0][0], 1).squeeze(0).cpu().data.numpy()
            pred = cv2.cvtColor(np.float32(pred), cv2.COLOR_GRAY2BGR)
            pred_filled = np.zeros(frame.shape)
            pred_filled[:, :, :] = np.zeros(frame.shape)
            # pred = cv2.resize(pred,(400,400)).astype('uint8')
            pred_filled[cfg.TEST.ROI_START[0]:cfg.TEST.ROI_END[0], cfg.TEST.ROI_START[1]:cfg.TEST.ROI_END[1], 0] = pred[:, :,0] * 255
            # mask = get_color_pallete(pred, 'trans10kv2')
            output_image = cv2.addWeighted(frame, 1, pred_filled.astype('uint8'), 0.3, 0)
            cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)[:-4] + '.png'), output_image)


if __name__ == '__main__':
    demo()
