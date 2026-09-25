import os
import sys
sys.path.insert(0, os.path.dirname(__file__) + '/..')
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from pathlib import Path
from typing import Optional

import cv2
import tyro
import torch
import numpy as np
from torch.amp import autocast

from ultralytics import YOLO
from data_config import SMPLX_PATH
from prompt_hmr import load_model_from_folder
from prompt_hmr.smpl_family import SMPLX

from prompt_hmr.vis.traj import align_meshes_to_gravity
from prompt_hmr.models.inference import prepare_batch
from prompt_hmr.vis.viser import viser_vis_human

from pipeline.camcalib.model import CameraRegressorNetwork


def _check_assets(gravity_align: bool) -> None:
    required = [
        Path('data/pretrain/yolov8x.pt'),
        Path('data/pretrain/phmr/config.yaml'),
        Path('data/pretrain/phmr/checkpoint.ckpt'),
        Path('data/body_models/smpl/SMPL_NEUTRAL.pkl'),
        Path('data/body_models/smplx/SMPLX_NEUTRAL.npz'),
        Path('data/body_models/smplx2smpl.pkl'),
        Path('data/body_models/smplx2smpl_joints.npy'),
    ]
    if gravity_align:
        required.append(Path('data/pretrain/camcalib_sa_biased_l2.ckpt'))

    missing = [path for path in required if not path.is_file()]
    if missing:
        missing_text = '\n'.join(f'  - {path}' for path in missing)
        raise FileNotFoundError(
            'Missing PromptHMR assets; downloads are intentionally not attempted:\n'
            f'{missing_text}'
        )


def main(
    image: str = 'data/examples/example_1.jpg',
    gravity_align: bool = False,
    detect_conf: float = 0.3,
    render_overlap: bool = False,
    run_viser: bool = True,
    output_dir: Optional[str] = None,
):
    image_path = Path(image)
    if not image_path.is_file():
        raise FileNotFoundError(f'Input image does not exist: {image_path}')

    image_path = image_path.resolve()
    _check_assets(gravity_align)

    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f'Input image is not readable by OpenCV: {image_path}')
    img = img_bgr[:, :, ::-1]

    savedir = Path(output_dir) if output_dir is not None else Path(os.path.basename(image))
    savedir = savedir.resolve()
    savedir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO("data/pretrain/yolov8x.pt")
    detection = yolo(str(image_path), verbose=False, conf=detect_conf, classes=0)
    boxes = detection[0].boxes.data.cpu()
    if len(boxes) == 0:
        raise RuntimeError(f'No people detected in input image: {image_path}')

    smplx = SMPLX(SMPLX_PATH).cuda()
    phmr = load_model_from_folder('data/pretrain/phmr')

    # Prompt HMR
    inputs = [{'image_cv': img, 'boxes': boxes, 'text': None, 'masks': None}]

    # Inference
    with torch.no_grad(), autocast('cuda'):
        batch = prepare_batch(inputs, img_size=896, interaction=False)
        output = phmr(batch, use_mean_hands=True)[0]

    # Reconstruction
    keys = ['pose', 'betas', 'transl', 'rotmat', 'vertices', 'body_joints', 'cam_int']
    output = {k: output[k].detach().cpu() for k in keys}
    output_path = savedir / 'output.pt'
    torch.save(output, output_path)
    print(f'Saved tensor output: {output_path}')

    # Render in the original camera coordinates.
    verts = output['vertices']
    if render_overlap:
        try:
            from prompt_hmr.vis.renderer import Renderer
        except ImportError as exc:
            raise RuntimeError(
                'Overlay rendering requires a working PyTorch3D installation.'
            ) from exc

        focal = float(batch[0]['cam_int_original'][0, 0, 0])
        renderer = Renderer(img.shape[1], img.shape[0], focal, bin_size=0)
        img_rend = renderer.render_meshes(verts, smplx.faces, img)
        output_image_path = savedir / 'output.jpg'
        if not cv2.imwrite(str(output_image_path), img_rend[:, :, ::-1]):
            raise IOError(f'OpenCV could not write rendered image: {output_image_path}')
        print(f'Rendered image saved: {output_image_path}')

    if not run_viser:
        return

    # Align only the interactive display scene. Numerical output and the
    # photographic overlay above stay in the original camera coordinates.
    camera = np.eye(4)
    if gravity_align:
        spec = CameraRegressorNetwork()
        spec = spec.load_ckpt('data/pretrain/camcalib_sa_biased_l2.ckpt').to('cuda')
        with torch.no_grad():
            _, pred_pitch, pred_roll = spec(img, transform_data=True)
            gravity_cam = spec.to_gravity_cam(pred_pitch, pred_roll)

        verts, [gv, gf, _], R, T = align_meshes_to_gravity(
            verts,
            gravity_cam,
            floor_scale=2,
            floor_color=[[0.73, 0.78, 0.82], [0.61, 0.69, 0.72]],
        )
        cam_r = R.mT
        cam_t = -cam_r @ T
        camera[:3, :3] = cam_r.numpy()
        camera[:3, 3] = cam_t.numpy()
        floor = [gv.cpu().numpy(), gf.cpu().numpy()]
    else:
        rot_180 = np.eye(3)
        rot_180[1, 1] = -1
        rot_180[2, 2] = -1
        verts = verts @ torch.from_numpy(rot_180).to(verts)
        camera[:3, :3] = rot_180
        floor = None

    import viser
    server = viser.ViserServer(host='0.0.0.0', port=8080)
    print('Viser listening on http://0.0.0.0:8080 (Ctrl+C to stop).')
    viser_vis_human(
        verts,
        smplx.faces,
        cameras=[camera],
        floor=floor,
        image=img,
        server=server,
    )


if __name__ == '__main__':
    tyro.cli(main)