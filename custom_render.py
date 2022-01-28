import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange

import mmcv
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils, dvgo
from lib.load_data import load_data, inward_nearfar_heuristic

from pathlib import Path
from pipelime.sequences.readers.filesystem import UnderfolderReader
from pipelime.sequences.samples import PlainSample, SamplesSequence, Sample
from pipelime.sequences.writers.filesystem import UnderfolderWriter
from typing import Tuple


@torch.no_grad()
def render_viewpoint(model, pose, HW, K, render_kwargs):
    rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
            HW[0], HW[1], K, pose, False, inverse_y=render_kwargs['inverse_y'],
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
    keys = ['rgb_marched', 'disp']
    render_result_chunks = [
        {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
        for ro, rd, vd in zip(rays_o.split(16, 0), rays_d.split(16, 0), viewdirs.split(16, 0))
    ]
    render_result = {
        k: torch.cat([ret[k] for ret in render_result_chunks])
        for k in render_result_chunks[0].keys()
    }
    rgb = render_result['rgb_marched'].cpu().numpy()
    disp = render_result['disp'].cpu().numpy()
    depth = 1 / disp
    return rgb, depth

@torch.no_grad()
def render_viewpoints(model, poses, HWs, Ks, render_kwargs):
    rgbs = []
    depths = []

    for pose, HW, K in tqdm(zip(poses, HWs, Ks), total=len(poses)):
        rgb, depth = render_viewpoint(model, pose, HW, K, render_kwargs)
        rgbs.append(rgb)
        depths.append(depth)

    rgbs = np.array(rgbs)
    depths = np.array(depths)

    return rgbs, depths


def load_data(folder, pose_key, camera_key):
    # how to get NEAR and FAR

    uf = UnderfolderReader(folder)
    poses = [x[pose_key] for x in uf]
    w, h = uf[0][camera_key]["intrinsics"]["image_size"]
    HWs = [[h, w] for _ in range(len(poses))]
    camera = uf[0][camera_key]["intrinsics"]["camera_matrix"]
    Ks = [camera for _ in range(len(poses))]
    near, far = inward_nearfar_heuristic(np.array(poses)[:, :3, 3], ratio=0)

    data = {}
    data["poses"] = torch.tensor(poses).float()
    data["HWs"] = HWs
    data["Ks"] = torch.tensor(Ks).float()
    data["near"] = near
    data["far"] = far

    return data


class Normalizer16Bit:
    @classmethod
    def normalization_key(cls, key: str) -> str:
        return f"info_{key}"

    @classmethod
    def normalize_16bits(cls, x: np.ndarray) -> Tuple[np.ndarray, int, int]:
        m, M = x.min(), x.max()
        x = (x - m) / (M - m)
        x = x * (2 ** 16 - 1)
        return x, m.item(), M.item()

    @classmethod
    def denormalize_16bits(cls, x: np.ndarray, m: int, M: int) -> np.ndarray:
        return x / (2 ** 16 - 1) * (M - m) + m

    @classmethod
    def get_float_item(cls, sample: Sample, key: str) -> np.ndarray:
        raw = sample[key]
        norm_key = cls.normalization_key(key)
        norm_dict = sample[norm_key]["normalization"]
        m, M = norm_dict["min"], norm_dict["max"]
        item = cls.denormalize_16bits(raw, m, M)
        return item

    @classmethod
    def set_float_item(cls, sample: Sample, key: str, array: np.ndarray) -> None:
        item, m, M = cls.normalize_16bits(array)
        item = item.astype(np.uint16)
        norm_dict = {
            "normalization": {
                "min": m,
                "max": M,
            }
        }
        norm_key = cls.normalization_key(key)
        sample[key] = item
        sample[norm_key] = norm_dict


if __name__=='__main__':

    # load setup
    cfg_file = "/home/federico/repos/DirectVoxGO/configs/tankstemple/Custom.py"
    ckpt_path = "/media/federico/b34ff4bd-2899-4632-a37f-2e3e062d5b7a/experiments/NERF/test3/OUT/exp1/fine_last.tar"
    uf = "/media/federico/b34ff4bd-2899-4632-a37f-2e3e062d5b7a/experiments/NERF/test3/train"
    pose_key = "pose"
    camera_key = "camera"
    near = 0
    far = 0.65
    output_folder = "/media/federico/b34ff4bd-2899-4632-a37f-2e3e062d5b7a/experiments/NERF/test3/myout_tr"

    cfg = mmcv.Config.fromfile(cfg_file)

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load images / poses / camera settings / data split
    # i just need pose and camera and resolution
    data_dict = load_data(uf, pose_key, camera_key)

    # load model for rendring
    ckpt_path = Path(ckpt_path)
    ckpt_name = ckpt_path.stem
    model = utils.load_model(dvgo.DirectVoxGO, ckpt_path).to(device)
    render_viewpoints_kwargs = {
        'model': model,
        # 'ndc': cfg.data.ndc,
        'render_kwargs': {
            # 'near': data_dict['near'],
            # 'far': data_dict['far'],
            'near': near,
            'far': far,
            'bg': 1 if cfg.data.white_bkgd else 0,
            'stepsize': cfg.fine_model_and_render.stepsize,
            'inverse_y': cfg.data.inverse_y,
            'flip_x': cfg.data.flip_x,
            'flip_y': cfg.data.flip_y,
        },
    }

    # render
    rgbs, depths = render_viewpoints(
        poses=data_dict['poses'],
        HWs=data_dict['HWs'],
        Ks=data_dict['Ks'],
        **render_viewpoints_kwargs,
    )
    samples = []
    for idx, (rgb, depth) in enumerate(zip(rgbs, depths)):
        sample = PlainSample(id=idx)
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        sample["image"] = rgb
        Normalizer16Bit.set_float_item(sample, "depth", depth)
        samples.append(sample)

    UnderfolderWriter(
        output_folder,
        extensions_map={"image": "png", "depth": "png"},
        num_workers=-1
    )(SamplesSequence(samples))