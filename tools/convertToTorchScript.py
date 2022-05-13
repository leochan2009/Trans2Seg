import os
import sys
import torch

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from torch.autograd import Variable
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.config import cfg



def convertToTracedScriptModel(args):
    original_model = get_segmentation_model().to(args.device)
    original_model.eval()
    # run the tracing
    generated_input = Variable(
        torch.zeros(1, 3, cfg.TRAIN.ROI_END[0]-cfg.TRAIN.ROI_START[0] , cfg.TRAIN.ROI_END[1]-cfg.TRAIN.ROI_START[1])
    )
    traced_script_module = torch.jit.trace(original_model, generated_input, strict=False)
    # save the converted model
    traced_script_module.save("traced_best.pt")
    model = torch.jit.load('traced_best.pt')
    model.to(args.device)
    model.eval()

    for i in range(200):
        out = model(generated_input)
        ori_out = original_model(generated_input)
        print(out)
        print(ori_out)




if __name__ == '__main__':
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()
    default_setup(args)
    os.makedirs("models", exist_ok=True)
    convertToTracedScriptModel(args)
