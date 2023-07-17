from numpy.lib.npyio import load
from torch._C import device
import sys
sys.path.append('/scratch/shared/beegfs/szwu/projects/video3d/RAFT')
from core.raft import RAFT

from .utils import InputPadder
import torch


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class FlowModel():
    def __init__(self, model, device):
        args = AttrDict({'model': model, 'small': False, 'mixed_precision': False, 'alternate_corr': False})
        self.model = self.load_model(args, device)
        self.device = device

    @staticmethod
    def load_model(args, device):
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.model))

        model = model.module
        model.to(device)
        model.eval()
        return model

    def preprocess_image(self, image):
        # image = image[:, :, ::-1].copy()
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        image = image.to(self.device)
        image = image[None]
        # size = [540, 960]
        # image = torch.nn.functional.interpolate(image, size=size, mode='bilinear', align_corners=False)
        padder = InputPadder(image.shape)
        return padder.pad(image)[0], padder

    def compute_flow(self, frame, next_frame, iters=20):
        frame, padder = self.preprocess_image(frame)
        next_frame, padder = self.preprocess_image(next_frame)
        _, flow = self.model(frame, next_frame, iters=iters, test_mode=True)
        return padder.unpad(flow)[0].permute(1, 2, 0).cpu().numpy()
