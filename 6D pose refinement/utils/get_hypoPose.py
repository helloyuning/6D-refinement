
from rendering import renderer
import numpy as np
from utils.quaternion import matrix2quaternion
from rendering.utils import verify_objects_in_scene
import cv2
from utils.sixd import load_sixd
from refiner.architecture import Architecture
from rendering.renderer import Renderer
from refiner.refiner import Refiner, Refinable
from rendering.utils import perturb_pose, trans_rot_err
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from utils import get_hypoPose as hp
from pyquaternion import Quaternion


def get_hypoPose(architecture, feed_dict, sess):#利用网络获取假设的pose
    rotation, translation = sess.run([architecture.rotation_hy_to_gt,
                                    architecture.translation_hy_to_gt],
                                    feed_dict=feed_dict)
    #return rotation, translation
    assert np.sum(np.isnan(translation[0])) == 0 and np.sum(np.isnan(rotation[0])) == 0

    hypo_pose = np.identity(4)
    hypo_pose[:3, :3] = Quaternion(rotation[0]).rotation_matrix
    hypo_pose[:3, 3] = translation[0]

    return hypo_pose


def get_init_feedDict(architecture, ren, input_col, model, hypo_pose=np.identity(4)):#生成网络需要的feed_dict
    input_shape = (architecture.input_shape[0], architecture.input_shape[1])
    scene_patch = cv2.resize(input_col, input_shape)
    #scene_path = input_col
    hypo_col, hypo_dep = get_render_path(ren, model,hypo_pose)#提取渲染的颜色
    render_path = cv2.resize(hypo_col, input_shape)#调整数据的形状为网络的大小
    #render_path = hypo_col
    hypo_trans = hypo_pose[:3, 3]  # 细化部分这么写，可以尝试不设置维度直接后期reshape
    hypo_rot = matrix2quaternion(hypo_pose[:3, :3])#来自原文的部分
    #crop_shift = [get_cropShift(hypo_dep)]

    centroid = verify_objects_in_scene(hypo_dep)  # 提取中心点，用于截取图片
    if centroid is None:
        print("Hypo outside of image plane")
        return centroid

    (x, y) = centroid
    x_normalized = x / 640.
    y_normalized = y / 480.

    feed_dict = {
        architecture.scene_patch: [scene_patch],
        architecture.render_patch: [render_path],
        architecture.hypo_rotation: hypo_rot.reshape(1, 4),
        architecture.hypo_translation: hypo_trans.reshape(1, 3),
        architecture.crop_shift: [[x_normalized, y_normalized]]
    }
    return feed_dict


def get_render_path(ren, model, hypo_pose):
    ren.clear()
    ren.draw_model(model, hypo_pose, ambient=0.5, specular=0, shininess=100,
                   light_col=[1, 1, 1], light=[0, 0, -1])
    hypo_col, hypo_dep = ren.finish()

    return hypo_col, hypo_dep
def get_cropShift(hypo_dep):
    centroid = verify_objects_in_scene(hypo_dep)#提取中心点，用于截取图片
    if centroid is None:
        print("Hypo outside of image plane")
        return centroid

    (x, y) = centroid
    x_normalized = x / 640.
    y_normalized = y / 480.
    return [x_normalized, y_normalized]

#if __name__ == '__main__':
    # sixd_base = "E:\\lm_base\\lm"
    # bench = load_sixd(sixd_base, nr_frames=0, seq=1)  # 数据打包处理，每个物体加载的对象为mask,depth,rgb.根据训练数据的多少三合1组合
    # ren = Renderer((640, 480), bench.cam)  # 生成渲染器,数据维度的转化
    # network = "models/refiner_linemod_obj_02.pb"
    #
    # max_rot_pert = float(20.0) / 180. * np.pi
    # max_trans_pert = float(0.10)
    # with tf.Session() as sess:
    #     architecture = Architecture(network_file=network, sess=sess)
    #     refiner = Refiner(architecture=architecture, ren=ren, session=sess)  # 数据渲染优化
    #     i = 1
    #     for frame in bench.frames:
    #         col = frame.color.copy()
    #         # _, gt_pose, _ = frame.gt[0]
    #         _, gt_pose = frame.gt[0]  # corrected,获取原始GT数据
    #         perturbed_pose = perturb_pose(gt_pose, max_rot_pert, max_trans_pert)
    #         feed_dict = hp.get_init_feedDict(architecture, ren, input_col=col, model=bench.models[str(int(1))])
    #         hypo_rotation, hypo_tans = hp.get_hypoPose(architecture, feed_dict, sess, perturbed_pose)  # 生成投影的pose