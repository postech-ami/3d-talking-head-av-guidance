'''
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.
You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and liable to prosecution.
Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.
More information about VOCA is available at http://voca.is.tue.mpg.de.
For comments or questions, please email us at voca@tue.mpg.de
'''

import os
import cv2
import tempfile
from tqdm import tqdm
import numpy as np
from subprocess import call
os.environ['PYOPENGL_PLATFORM'] = 'osmesa' #egl
import pyrender
import trimesh
from psbody.mesh import Mesh

from utils.utils import *


# The implementation of rendering is borrowed from VOCA: https://github.com/TimoBolkart/voca/blob/master/utils/rendering.py
def render_mesh_helper(args,mesh, t_center, rot=np.zeros(3), tex_img=None,  z_offset=0):
    if args.dataset == "BIWI":
        camera_params = {'c': np.array([400, 400]),
                         'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                         'f': np.array([4754.97941935 / 8, 4754.97941935 / 8])}
    elif args.dataset == "vocaset":
        camera_params = {'c': np.array([400, 400]),
                         'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                         'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}

    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    mesh_copy = Mesh(mesh.v, mesh.f)
    mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v-t_center).T).T+t_center
    
    intensity = 2.0
    rgb_per_v = None

    primitive_material = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor=[0.3, 0.3, 0.3, 1.0],
                metallicFactor=0.8, 
                roughnessFactor=0.8 
            )

    tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=rgb_per_v)
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material,smooth=True)

    if args.background_black:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])#[0, 0, 0] black,[255, 255, 255] white
    else:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])#[0, 0, 0] black,[255, 255, 255] white
    
    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                      fy=camera_params['f'][1],
                                      cx=camera_params['c'][0],
                                      cy=camera_params['c'][1],
                                      znear=frustum['near'],
                                      zfar=frustum['far'])

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3,3] = np.array([0, 0, 1.0-z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])

    angle = np.pi / 6.0
    pos = camera_pose[:3,3]
    light_color = np.array([1., 1., 1.])
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3,3] = pos
    scene.add(light, pose=light_pose.copy())
    
    light_pose[:3,3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] =  cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
        color, _ = r.render(scene, flags=flags)
    except:
        print('pyrender: Failed rendering frame')
        color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]

def render_sequence_meshes(
    args, sequence_vertices, template, out_path,
    predicted_vertices_path, vt, ft, tex_img,
):
    num_frames = sequence_vertices.shape[0]
    file_name_pred = predicted_vertices_path.split('/')[-1].split('.')[0]
    tmp_video_file_pred = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=out_path)
    writer_pred = cv2.VideoWriter(
        tmp_video_file_pred.name, cv2.VideoWriter_fourcc(*'mp4v'), 
        args.fps, (800, 800), True,
    )

    center = np.mean(sequence_vertices[0], axis=0)
    video_fname_pred = os.path.join(out_path, file_name_pred+'.mp4')
    for i_frame in range(num_frames):
        render_mesh = Mesh(sequence_vertices[i_frame], template.f)
        if vt is not None and ft is not None:
            render_mesh.vt, render_mesh.ft = vt, ft
        pred_img = render_mesh_helper(args,render_mesh, center, tex_img=tex_img)
        pred_img = pred_img.astype(np.uint8)
        img = pred_img
        writer_pred.write(img)

    writer_pred.release()

    audio_fname = os.path.join(
        args.dataset, args.wav_path, "_".join(file_name_pred.split("_")[:5])
    )
    cmd = [
        'ffmpeg', '-loglevel', 'panic', '-y',
        '-i', tmp_video_file_pred.name, # video
        '-i', audio_fname, # audio
        '-c', 'copy', '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p', '-qscale', '0', '-c:a', 'aac',
        video_fname_pred
    ]
    call(cmd)

def main():
    # Load configurations
    parser = argparse.ArgumentParser(
        description="Speech-Driven 3D Facial Animation with A-V Guidance"
    )
    parser.add_argument(
        "--dataset", type=str, default="vocaset", help="Dataset to train the model",
    )
    args = parser.parse_args()
    if args.dataset=="vocaset":
        args = load_config("config/vocaset.yaml")
    elif args.dataset == "BIWI":
        args = load_config("config/biwi.yaml")

    # Make directories to save results
    make_dirs(args.save_video_path)

    pbar = tqdm(os.listdir(args.save_pred_path))
    
    for file in pbar:
        if file.endswith("npy"):
            predicted_vertices_path = os.path.join(args.save_pred_path, file)
            if args.dataset == "BIWI":
                template_file = os.path.join(
                    args.dataset, args.render_template_path, "BIWI.ply"
                )
            elif args.dataset == "vocaset":
                template_file = os.path.join(
                    args.dataset, args.render_template_path, "FLAME_sample.ply"
                )
            pbar.set_description(file)
        
            template = Mesh(filename=template_file)
            vt, ft = None, None
            tex_img = None

            predicted_vertices = np.load(predicted_vertices_path)
            predicted_vertices = np.reshape(
                predicted_vertices, (-1, args.vertice_dim//3, 3)
            )

            render_sequence_meshes(
                args,predicted_vertices,
                template,
                args.save_video_path,
                predicted_vertices_path,
                vt, ft, tex_img,
            )


if __name__=="__main__":
    main()