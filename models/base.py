import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from auto_avsr.datamodule.transforms import AdaptiveTimeMask
from models.faceformer import Faceformer
from utils.renderer_pytorch3d import set_rasterizer, SRenderY
from utils.utils import *


class BaseModel(nn.Module):

    def __init__(self, args, mode="train"):
        super(BaseModel, self).__init__()

        self.args = args
        self.mode = mode

        # Build facial animator model
        self.facial_animator = Faceformer(args)
        self.facial_animator.to(args.device)

        if mode=="train":
            # Build lip reader model
            self.lipreader = self.build_lipreader(args.lipreader_path)
            self.lipreader.to(args.device)

            # Optimizer 
            parameters = list(
                filter(lambda p: p.requires_grad, self.facial_animator.parameters())
            ) + list(
                filter(lambda p: p.requires_grad, self.lipreader.parameters())
            )
            self.optimizer = torch.optim.Adam(parameters, lr=args.lr)

            # For Render
            set_rasterizer()
            self.render = SRenderY(
                image_size=224,
                obj_filename=args.obj_filename,
                uv_size=256,
            ).to(args.device)
            self.video_transform = torch.nn.Sequential(
                torchvision.transforms.CenterCrop(88),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.Normalize(0.421, 0.4),
            )

            # For lipreader
            self.adaptive_time_mask = AdaptiveTimeMask(6400, 16000)


    def train_mode(self):
        self.facial_animator.train()
        if self.mode=="train": self.lipreader.train()


    def eval_mode(self):
        self.facial_animator.eval()
        if self.mode=="train": self.lipreader.eval()


    def build_lipreader(self, model_path):
        # Load configurations
        from hydra import compose, initialize
        initialize(version_base="1.3", config_path="../auto_avsr/configs")
        cfg = compose(config_name="config")
        assert os.path.exists(model_path), "Lip reader model is not exist!"
        cfg.pretrained_model_path = model_path
        
        # Load lip reader model
        from auto_avsr.lightning_av import ModelModule
        lipread_model = ModelModule(cfg).model
        lipread_model.load_state_dict(
            torch.load(
                cfg.pretrained_model_path, map_location=lambda storage,loc:storage,
            )
        )
        print(f"Lip reader is loaded - {model_path}")

        return lipread_model
    

    def load_model(self):
        self.facial_animator.load_state_dict(
            torch.load(self.args.test_model_path)
        )


    def forward(
        self, audio, template, vertice, one_hot, waveform, text_token,
    ):

        # Loss from facial animator
        # -----------------------------------------------------------------------------
        geo_loss, gt_vertice, prediction = self.facial_animator(
            audio, template,  vertice, one_hot, teacher_forcing=False,
        )
        prediction = prediction.squeeze().reshape(
            -1, self.args.vertice_dim//3, 3,
        ) # (frame, num_verts, 3)

        # Lip vertex loss
        # -----------------------------------------------------------------------------
        mouth_map = get_lip_verts(self.args.dataset)
        lip_pred = prediction[:, mouth_map] # [frame, num_lip_verts, 3]
        lip_gt = gt_vertice.squeeze().reshape(
            -1, self.args.vertice_dim//3, 3,
        )[:, mouth_map] # [frame, num_lip_verts, 3]
        lip_vert_loss = nn.MSELoss()(lip_pred, lip_gt)
        
        # AV loss
        # -----------------------------------------------------------------------------
        
        # Transform vertices
        if self.args.dataset=="vocaset":
            proj_camera = torch.Tensor([8, 0, 0]).expand(len(prediction), -1).cuda()
        elif self.args.dataset=="BIWI":
            proj_camera = torch.Tensor([1.6, 0, 0]).expand(len(prediction), -1).cuda()
        trans_verts = batch_orth_proj(prediction, proj_camera) # [frame, num_verts, 3]
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
        
        # Render face 
        rendered_video = self.render.render_shape(prediction, trans_verts) # [frame, C=3, H=224, W=224]
        
        # Crop lip region
        if self.args.dataset=="vocaset":
            rendered_video = rendered_video[:, :, 107:203, 63:159] # [frame, 3, 96, 96]
        elif self.args.dataset=="BIWI":
            rendered_video = rendered_video[:, :, 107:203, 68:164] # [frame, 3, 96, 96]
        
        # Transform rendered video
        video = self.video_transform(rendered_video) # [frame, 1, 88, 88]
        video = video.unsqueeze(dim=0) # [B=1, frame, 1, 88, 88]

        # Compute AV loss
        waveform = cut_or_pad(waveform, video.shape[1] * 640)
        waveform = self.adaptive_time_mask(waveform)
        waveform = F.layer_norm(waveform, waveform.shape, eps=1e-8)
        av_loss, loss_ctc, loss_att, acc = self.lipreader(
            video=video, 
            audio=waveform.unsqueeze(0), 
            video_lengths=torch.Tensor([video.shape[1]]).cuda(),
            audio_lengths=torch.Tensor([waveform.shape[0]]).to(torch.int64).cuda(), 
            label=text_token,
        )

        # Combine all losses
        # -----------------------------------------------------------------------------
        all_loss = (
            geo_loss
            + lip_vert_loss * self.args.lip_vert_weight
            + av_loss * self.args.lipread_weight
        )

        return all_loss