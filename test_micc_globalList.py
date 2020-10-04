import argparse
import sys, os
import time

#
import ast
import trimesh

#
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage.io import imsave
import tools
from model import VggEncoder


# self
_curr_path = os.path.abspath(__file__) # /home/..../face
_cur_dir = os.path.dirname(_curr_path) # ./
_comp_dir = os.path.dirname(_cur_dir) # ./
_deep_learning_dir = os.path.dirname(_comp_dir) # ../
print(_deep_learning_dir)
sys.path.append(_deep_learning_dir) # /home/..../pytorch3d

#
from baselib_python.Render.Ortho.render_app import get_render


# mfs
from tf_face3d.base.common.io_helper import *

#
from tools_data.face_common.faceIO import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='/home/jshang/SHANG_Data/4_nips2020_testData/20_MICC_video_Howfar_50_MUL',
                    help='path to load images. It should include image name with: front|left|right')
parser.add_argument('--mode_global_list', type=str, default='test',
                    help='path to save 3D face shapes')
parser.add_argument('--generate_dir', type=str, default='/home/jshang/SHANG_Data/4_nips2020_evalResult/23_MICC_testMVF',
                    help='path to load images. It should include image name with: front|left|right')
parser.add_argument('--isVisual', type=ast.literal_eval, default=False,
                    help='path to load images. It should include image name with: front|left|right')

PARSER = parser.parse_args()

"""
python ./comparison/mvfnet/test_micc_globalList.py \
--dataset_dir /home/jshang/SHANG_Data/4_nips2020_testData/20_MICC_video_Howfar_50_MUL \
--generate_dir /home/jshang/SHANG_Data/4_nips2020_evalResult/23_MICC_testMVFCrop \
--isVisual False
"""

if __name__ == '__main__':

    if not os.path.exists(PARSER.generate_dir):
        os.makedirs(PARSER.generate_dir)

    image_file_list, cam_file_list, subfolders, frame_ids = format_file_list(PARSER.dataset_dir, PARSER.mode_global_list, fmt=None, sort=True)

    """ mvs input path list """
    # mvf start
    model = VggEncoder()
    model = torch.nn.DataParallel(model).cuda()
    ckpt = torch.load(os.path.join(_cur_dir, 'data/net.pth'))
    model.load_state_dict(ckpt)

    for i in range(len(image_file_list)):
        path_image = image_file_list[i]
        dic_subfolder, name_image = os.path.split(path_image)
        name_subfolder = subfolders[i]
        name_image_pure, _ = os.path.splitext(name_image)

        print("Sample: %d %s %d" % (i, name_image_pure, len(image_file_list)))

        # save
        dic_folder_save = os.path.join(PARSER.generate_dir, name_subfolder)
        if os.path.isdir(dic_folder_save) == True:
            pass
        else:
            os.makedirs(dic_folder_save)

        # mvf
        path_mesh_save = os.path.join(dic_folder_save, name_image_pure + '.ply')
        path_lm3d_save = os.path.join(dic_folder_save, name_image_pure + '_lm3d.txt')
        time_st = time.time()

        image = Image.open(path_image).convert('RGB')
        image = np.array(image)

        imgA, imgB, imgC, list_imgs = unpack_image_np(image, 224, 224, 2)
        imgA_np = Image.fromarray(imgA)
        imgB_np = Image.fromarray(imgB)
        imgC_np = Image.fromarray(imgC)

        crop_opt = True  # change to True if you want to crop the image
        if PARSER.isVisual:
            crop_opt = False
        # imgA = Image.open(path_image_ref).convert('RGB')
        # imgB = Image.open(path_image_left).convert('RGB')
        # imgC = Image.open(path_image_right).convert('RGB')

        if crop_opt:
            imgA_np = tools.crop_image(imgA_np)
            imgB_np = tools.crop_image(imgB_np)
            imgC_np = tools.crop_image(imgC_np)

        imgA = transforms.functional.to_tensor(imgA_np)
        imgB = transforms.functional.to_tensor(imgB_np)
        imgC = transforms.functional.to_tensor(imgC_np)


        # print model
        input_tensor = torch.cat([imgA, imgB, imgC], 0).view(1, 9, 224, 224).cuda()
        start = time.time()
        preds = model(input_tensor)
        print(time.time() - start)
        vertices, tri, kptA, kptB, kptC, lm3d, face_shapeA, face_shapeB, face_shapeC \
            = tools.preds_to_shape(preds[0].detach().cpu().numpy())

        # Save
        # corresponding colors
        mesh_tri = trimesh.Trimesh(
            (vertices).reshape(-1, 3),
            tri,
            process=False
        )
        #mesh_tri.export(os.path.join(dic_subfolder, name_image_pure + '.ply'))
        tools.write_ply(path_mesh_save, vertices/1000., tri)
        write_self_lm(path_lm3d_save, lm3d/1000.)

        if PARSER.isVisual:
            # Save image

            # Geo
            geo_color = np.array([0.629, 0.629, 0.629])
            geo_color = np.reshape(geo_color, [1, 3])
            geo_color = np.tile(geo_color, [np.shape(vertices)[0], 1])

            path_overlayGeo_save = os.path.join(dic_folder_save, name_image_pure + '_overlayGeo.jpg')

            renderGeo_A = get_render(face_shapeA, tri, geo_color, 224, 224, mesh_tri.vertex_normals)
            renderGeo_B = get_render(face_shapeB, tri, geo_color, 224, 224, mesh_tri.vertex_normals)
            renderGeo_C = get_render(face_shapeC, tri, geo_color, 224, 224, mesh_tri.vertex_normals)

            def overl(renderGeo_save, image):
                render_mask_save = renderGeo_save > 0
                render_mask_save = render_mask_save.astype(np.float32)
                overlayGeo_save = image * (1 - render_mask_save) + renderGeo_save * 255. * render_mask_save
                return overlayGeo_save

            overlayGeo_saveA = overl(renderGeo_A, imgA_np)
            overlayGeo_saveB = overl(renderGeo_B, imgB_np)
            overlayGeo_saveC = overl(renderGeo_C, imgC_np)
            overlayGeo_save = np.concatenate([overlayGeo_saveA, overlayGeo_saveB, overlayGeo_saveC], axis=1)
            overlayGeo_save = overlayGeo_save.astype(np.uint8)
            imsave(path_overlayGeo_save, overlayGeo_save)

            time_end = time.time()
            print(" Time used: %f s" % (time_end-time_st))
