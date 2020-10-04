import argparse
import sys, os
import time

#
import ast

#
import torch
import torchvision.transforms as transforms
from PIL import Image
import trimesh
from skimage.io import imsave
#
import tools
from model import VggEncoder

#self
_curr_path = os.path.abspath(__file__) # /home/..../face
_cur_dir = os.path.dirname(_curr_path) # ./
_tf_dir = os.path.dirname(_cur_dir) # ./
_deep_learning_dir = os.path.dirname(_tf_dir) # ../
print(_deep_learning_dir)
sys.path.append(_deep_learning_dir) # /home/..../pytorch3d

# mfs
from baselib_python.IO.GlobalList import *
from baselib_python.IO.PairList import *
from baselib_python.IO.Landmark import *

from baselib_python.Render.Ortho.render_app import get_render

parser = argparse.ArgumentParser()
parser.add_argument('--dic_image', type=str, default='/home/jshang/SHANG_Data/3DFace_Training/21_MICC_Render', help='')
parser.add_argument('--name_glt', type=str, default='test', help='path to save 3D face shapes')
parser.add_argument('--output_dir', type=str, default='/home/jshang/SHANG_Data/3DFace_Training/27_MICC_testMVF', help='')

parser.add_argument('--batch_size', type=int, default=1, help='')
parser.add_argument('--img_width', type=int, default=256, help='')
parser.add_argument('--img_height', type=int, default=256, help='')

parser.add_argument('--path_gpmm', type=str, default='/home/jshang/SHANG_Data/3DFace_Training/27_MICC_testMVF', help='')

parser.add_argument('--device', type=str, default='cpu', help='')

parser.add_argument('--flag_overlay_save', type=ast.literal_eval, default=False, help='')
parser.add_argument('--flag_overlayOrigin_save', type=ast.literal_eval, default=False, help='')
parser.add_argument('--flag_main_save', type=ast.literal_eval, default=False, help='')

parser.add_argument('--flag_visual', type=ast.literal_eval, default=True, help='')


# flags.DEFINE_boolean("flag_visual", True, "")
# flags.DEFINE_boolean("flag_fore", True, "")
#
# # visual
# flags.DEFINE_boolean("flag_overlay_save", False, "")
# flags.DEFINE_boolean("flag_overlayOrigin_save", False, "")
# flags.DEFINE_boolean("flag_main_save", False, "")
PARSER = parser.parse_args()

"""
1. Python 2.7 (Numpy, PIL, scipy)
2. Pytorch 0.4.0, torchvision
3. face-alignment package from [https://github.com/1adrianb/face-alignment](https://github.com/1adrianb/face-alignment). 
This code is used for face cropping and will be replaced by face detection algorithm in the future.

python comparison/MGCNet/local/test_image_glt.py \
--dic_image /data0/0_DATA/0_Face_3D/0_facescape/14_fs_glist_test_256 \
--output_dir /home/jiaxiangshang/SHANG_Exp/server_syn_test/2_mvfnet/0_facescape \
--path_gpmm /data0/2_Project/python/deeplearning_python/comparison/MGCNet/model/bfm09_trim_exp_uv_presplit.h5 \
--flag_overlay_save=False --flag_overlayOrigin_save=False --flag_main_save=False
"""

if __name__ == '__main__':
    device = torch.device(PARSER.device)

    if not os.path.exists(PARSER.output_dir):
        os.makedirs(PARSER.output_dir)

    # Read model
    model = VggEncoder()
    model = torch.nn.DataParallel(model).cuda()
    ckpt = torch.load(os.path.join(_cur_dir, 'data/net.pth'))
    model.load_state_dict(ckpt)
    #model.eval()

    # Read data
    # read global list
    emotion_list, dic_folderLeaf_list = parse_global_list(PARSER.dic_image, PARSER.name_glt)
    path_img_global_list = pairs_2_global(emotion_list, level_bl=0)

    # save global list
    path_train_list = os.path.join(PARSER.output_dir, "eval.txt")
    f_train_global = open(path_train_list, 'w')

    """ mvs input path list """
    for t in range(0, len(path_img_global_list), PARSER.batch_size):
        time_st = time.time()
        inputs = np.zeros(
            (PARSER.batch_size, 9, 224, 224), dtype=np.uint8
        )
        for b in range(PARSER.batch_size):
            idx = t + b
            if idx >= len(path_img_global_list):
                break
            print('Sample: ', idx)
            # if os.path.isfile(image_file_list[idx]) == False:
            #     continue

            num_view = 3
            list_images_multi = []
            for i_v in range(3):
                path_image = path_img_global_list[idx][i_v * num_view + 0]
                path_lmgt = path_img_global_list[idx][i_v * num_view + 2]
                lm_howfar = parse_self_lm(path_lmgt)

                imgA = Image.open(path_image).convert('RGB')
                if imgA is None:
                    print("Error: can not find ", path_image)
                imgA = tools.crop_image(imgA)
                imgA = transforms.functional.to_tensor(imgA)
                if 0:
                    # face image align by landmark
                    # we also provide a tools to generate 'std_224_bfm09'
                    lm_trans, img_warped, tform = crop_align_affine_transform(lm_howfar, image_rgb, FLAGS.img_height,
                                                                              std_224_bfm09)
                    # M_inv is used to back project the face reconstruction result to origin image
                    M_inv = np.linalg.inv(tform.params)
                    M = tform.params
                    inputs[b] = np.array(img_warped)

                list_images_multi.append(imgA)

        input_tensor = torch.cat(list_images_multi, 0).view(1, 9, 224, 224).cuda()

        #inputs = torch.from_numpy(inputs).float().to(device)
        start = time.time()
        preds = model(input_tensor)
        print("Time each batch: ", time.time() - time_st)

        with torch.no_grad():
            for b in range(PARSER.batch_size):
                # name
                idx = t + b
                if idx >= len(path_img_global_list):
                    break
                path_image = path_img_global_list[idx][0]
                dic_image, name_image = os.path.split(path_image)
                name_image_pure, _ = os.path.splitext(name_image)
                name_image_pure = '_'.join(name_image_pure.split('_')[:2])

                # Read gt
                path_lm3d = path_img_global_list[idx][15]
                dic_emotion, name_lm3d = os.path.split(path_lm3d)
                dic_sample, name_emotion = os.path.split(dic_emotion)
                _, name_subfolder = os.path.split(dic_sample)
                path_mesh_np = os.path.join(dic_emotion, name_emotion + '_modelRigt.ply')

                dic_image_save = os.path.join(PARSER.output_dir, name_subfolder, name_emotion)
                if os.path.isdir(dic_image_save) == False:
                    os.makedirs(dic_image_save)

                # result
                vertices, tri, kptA, kptB, kptC, lm3d, face_shapeA, face_shapeB, face_shapeC = tools.preds_to_shape(preds[b].detach().cpu().numpy())
                list_rst_lm2d = [kptA, kptB, kptC]
                list_rst_shape2d = [face_shapeA, face_shapeB, face_shapeC]
                # eval
                path_mesh_save = os.path.join(dic_image_save, name_image_pure + '.ply')
                mesh_tri = trimesh.Trimesh(
                    (vertices).reshape(-1, 3),
                    tri,
                    process=False
                )
                # mesh_tri.export(os.path.join(dic_subfolder, name_image_pure + '.ply'))
                tools.write_ply(path_mesh_save, vertices / 1000., tri)
                path_lm3d_save = os.path.join(dic_image_save, name_image_pure + '.txt')
                write_self_lm(path_lm3d_save, lm3d / 1000.)

                if PARSER.flag_visual:
                    # Render geo
                    geo_color = np.array([0.629, 0.629, 0.629])
                    geo_color = np.reshape(geo_color, [1, 3])
                    geo_color = np.tile(geo_color, [np.shape(vertices)[0], 1])

                    list_images_multi = []
                    list_images_overLay = []
                    for i_v in range(3):
                        path_image = path_img_global_list[idx][i_v * num_view + 0]
                        path_lmgt = path_img_global_list[idx][i_v * num_view + 2]
                        lm_howfar = parse_self_lm(path_lmgt)

                        imgA = Image.open(path_image).convert('RGB')
                        if imgA is None:
                            print("Error: can not find ", path_image)
                        imgA = tools.crop_image(imgA, pts_gt=lm_howfar)
                        imgA = np.asarray(imgA)
                        list_images_multi.append(imgA)

                        # visual lm2d
                        # if 0:
                        #     import cv2
                        #     for ind, pt2d in enumerate(list_rst_lm2d[i_v]):
                        #         u = int(pt2d[0])
                        #         v = int(pt2d[1])
                        #         cv2.circle(imgA, (u, v), 5, (10, 10, 10), -1)
                        #         cv2.putText(imgA, "%02d" % (ind), (u - 8, v + 4),
                        #                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.2, color=(255, 255, 255))
                        #     cv2.imshow("Image Debug", imgA)
                        #     k = cv2.waitKey(0) & 0xFF
                        #     if k == 27:
                        #         cv2.destroyAllWindows()

                        renderGeo_A = get_render(list_rst_shape2d[i_v], tri, geo_color, 224, 224, mesh_tri.vertex_normals)

                        def overl(renderGeo_save, image):
                            render_mask_save = renderGeo_save > 0
                            render_mask_save = render_mask_save.astype(np.float32)
                            overlayGeo_save = image * (1 - render_mask_save) + renderGeo_save * 255. * render_mask_save
                            return overlayGeo_save

                        overlayGeo_save = overl(renderGeo_A, imgA)
                        overlayGeo_save = overlayGeo_save.astype(np.uint8)
                        list_images_overLay.append(overlayGeo_save)
                    list_images_multi[1], list_images_multi[0] = list_images_multi[0], list_images_multi[1]
                    list_images_overLay[1], list_images_overLay[0] = list_images_overLay[0], list_images_overLay[1]
                    images_all_save = np.concatenate(list_images_multi, axis=1)
                    overlayGeo_all_save = np.concatenate(list_images_overLay, axis=1)
                    all_save = np.concatenate([images_all_save, overlayGeo_all_save], axis=0)
                    path_overlayGeo_save = os.path.join(dic_image_save, name_image_pure + '_overlayGeo.jpg')
                    imsave(path_overlayGeo_save, all_save)

        time_end = time.time()
        print(" Time used: %f s" % (time_end-time_st))



