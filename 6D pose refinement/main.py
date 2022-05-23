"""
Simple script to run a forward pass employing the Refiner on a SIXD dataset sample with a trained model.

Usage:
  test_refinement.py [options]
  test_refinement.py -h | --help

Options:
    -d --dataset=<string>        Path to SIXD dataset[default: E:\\lm_base\\lm]
    -o --object=<string>         Object to be evaluated [default: 02]
    -n --network=<string>        Path to trained network [default: models/refiner_linemod_obj_02.pb]
    -r --max_rot_pert=<float>    Max. Rotational Perturbation to be applied in Degrees [default: 20.0]
    -t --max_trans_pert=<float>  Max. Translational Perturbation to be applied in Meters [default: 0.10]
    -i --iterations=<int>        Max. number of iterations[default: 100]
    -h --help                    Show this message and exit
"""
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



#import tensorflow as tf
import yaml
import cv2
import numpy as np

from utils.sixd import load_sixd
from refiner.architecture import Architecture
from rendering.renderer import Renderer
from refiner.refiner import Refiner, Refinable
from rendering.utils import *
from timeit import default_timer as timer
from docopt import docopt
from utils import utils#自制模型预测pose导入


args = docopt(__doc__)

sixd_base = args["--dataset"]
network = args["--network"]
max_rot_pert = float(args["--max_rot_pert"]) / 180. * np.pi
max_trans_pert = float(args["--max_trans_pert"])
iterations = int(args["--iterations"])

def get_cam(path):
    cam = np.identity(3)
    cam_info = yaml.safe_load(open(path, 'rb'))
    cam[0, 0] = cam_info['fx']
    cam[0, 2] = cam_info['cx']
    cam[1, 1] = cam_info['fy']
    cam[1, 2] = cam_info['cy']
    scale_to_meters = 0.001 * cam_info['depth_scale']

    return cam, scale_to_meters


def train():

    seq_to_name = {'Ape': 1, 'B': 2}
    croppings = yaml.safe_load(open('config/croppings.yaml', 'rb'))  # 裁剪参数


    poses_gt = tf.placeholder(tf.float32, [4, 4])

    bench = utils.loadSIXDBench(dataset_path=sixd_base,metric_crop_shape=croppings,seq='Ape',seq_to_name=seq_to_name)
    cam_info = yaml.safe_load(open(sixd_base+'\\camera.yml', 'rb'))
    cam, scale = get_cam(sixd_base + '\\camera.yml')

    #loss = loss(poses_gt)
    #数据打包处理，每个物体加载的对象为mask,depth,rgb.根据训练数据的多少三合1组合

    learning_rate = 1e-4
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.1)  # 优化器设置，！！！！损失函数设置

    with tf.Session() as sess:
        architecture = Architecture(network_file=network, sess=sess)  # 此处为网络，应该有一个东西预测生成rt，然后才是Refinemnet
        #     init = tf.global_variables_initializer()#权值初始化
        #     sess.run(init)
        ren = Renderer((640, 480), cam)  # 生成渲染器,数据维度的转化
        refiner = Refiner(architecture=architecture, ren=ren, session=session)  # 数据渲染优化
        i = 0
        for fr in range(len(bench.frames)):
            #print("depth:", bench.frames[fr].depth)
            #transform_points(bench.model.vertices, bench.frames[fr].gt)
            #adi(bench.model.vertices,)
            #print("gt:",bench.frames[fr].gt)
            contour = get_viewpoint_cloud(bench.frames[fr].depth, cam_info,100)#每一帧获取gt轮廓信息
            #full_contour_point = get_full_viewpoint_cloud(bench.frames[fr].depth, cam_info,3)#投影生成的3d轮廓点云
            obj, scene = distance_transform(contour)#gt轮廓点distance_trans
            #A = transform_points(bench.model.vertices, distance)
            print("distance transform for 100点:\n", obj, "scene countour", scene)
            i += 1
            if i >= 0:
                break










    # with tf.Session() as sess:
    #     architecture = Architecture(network_file=network, sess=sess)  # 此处为网络，应该有一个东西预测生成rt，然后才是Refinemnet
    # #     init = tf.global_variables_initializer()#权值初始化
    # #     sess.run(init)
    #     loss = add_pose_loss(architecture, poses_gt)
    #
    #
    #
    #
    #
    #     for obj in objects[0]:
    #         bench = load_sixd(sixd_base, nr_frames=0, seq=obj)#数据打包处理，每个物体加载的对象为mask,depth,rgb.根据训练数据的多少三合1组合
    #         ren = Renderer((640, 480), bench.cam)  # 生成渲染器,数据维度的转化
    #         refiner = Refiner(architecture=architecture, ren=ren, session=session)  # 数据渲染优化
    #
    #         # optimizer.minimize(loss)
    #
    #
    #         i = 1
    #         for frame in bench.frames:
    #             col = frame.color.copy()
    #             # _, gt_pose, _ = frame.gt[0]
    #             _, gt_pose = frame.gt[0]  # corrected,获取原始GT数据
    #
    #             #optimizer.minimize(loss)
    #
    #             perturbed_pose = perturb_pose(gt_pose, max_rot_pert, max_trans_pert)
    #             #hypo_col, hypo_dep = hp.get_render_path(model=bench.models[str(int(obj))],hypo_pose=perturbed_pose)
    #             #feed_dict = hp.get_init_feedDict(architecture, ren, input_col=col,model=bench.models[str(int(obj))],hypo_pose=perturbed_pose)#喂养数据，导入扰乱的姿势
    #             #hypo_pose = hp.get_hypoPose(architecture,feed_dict,sess)#生成投影的pose
    #
    #
    #
    #             refinable = Refinable(model=bench.models[str(int(obj))], label=0, hypo_pose=perturbed_pose,
    #                                   metric_crop_shape=croppings[dataset_name]['obj_{:02d}'.format(int(obj))],
    #                                   input_col=col)
    #
    #
    #
    #
    #
    #             #start = timer() #可视化时，结果的时间的计算
    #             refiner.iterative_contour_alignment(refinable=refinable)
    #             #end = timer() #refinement的结果时间计算
    #
    #             #diff_t, diff_r = trans_rot_err(gt_pose, refinable.hypo_pose)
    #             # loss = diff_t+diff_r
    #             # print("rt差值",diff_r,diff_t)
    #             # print("epoch:", i,"loss:", loss)
    #             # optimizer.minimize(loss)
    #
    #
    #             i = i + 1
    #             if i > 50:
    #                 break;
                # sig, back_sig = distance_transform(refinable.hypo_dep)
                # print("epoch:", i, "loss:",sig+back_sig)
                # print("epoch:", i, "sig:", sig, "back_sig", back_sig)
                # print("gt:", gt_pose,"\n", "hypo_pose:", hypo_pose)
                # error = trans_rot_err(gt_pose,hypo_pose)
                # loss = trans_rot_err(gt_pose, refinable.hypo_pose)

                #以下为结果可视化
                # ren.clear()
                # ren.draw_background(col)
                # ren.draw_boundingbox(refinable.model, refinable.hypo_pose)
                # ren.draw_model(refinable.model, refinable.hypo_pose, ambient=0.5, specular=0, shininess=100,
                #                light_col=[1, 1, 1], light=[0, 0, -1])
                # render_col, _ = ren.finish()
                # render_col = render_col.copy()
                #
                # cv2.imshow("Input Image", col)
                #
                # # Draw FPS in top left corner
                # fps = "FPS: " + str(int(1 / (end - start)))
                #
                # cv2.rectangle(render_col, (0, 0), (133, 40), (1., 1., 1.), -1)
                # cv2.putText(render_col, fps, (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                #
                # cv2.imshow("Refined Output", render_col)
                # cv2.waitKey(300)






                #orig_trans_err, orig_angular_err = trans_rot_err(gt_pose, perturbed_pose)
               #optimizer.minimize(rotation_hy_to_gt,translation_hy_to_gt)#计算损失函数











if __name__ == '__main__':
    train()


