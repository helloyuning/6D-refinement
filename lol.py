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

import os

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)




import yaml
import cv2
import numpy as np
import time
from utils.sixd import load_sixd, load_yaml
from refiner.architecture import Architecture
from rendering.renderer import Renderer
#from refiner.refiner import Refiner, Refinable
from refiner.corrected_refiner import Refiner, Refinable
from rendering.utils import *
#from refiner.non_sess_network import Architecture
from timeit import default_timer as timer
from docopt import docopt
import graph_def_editor as ge
from utils import get_hypoPose as hp#自制模型预测pose导入
from tensorflow.python.training import training_util
from Network import man_net
from tqdm import tqdm
import random
import tfquaternion as tfq
import math
from tensorflow.python.framework import graph_util

output_checkpoint_dir = 'ckpt_model'
checkpoint_file = '6D_model.ckpt'

from Network.posenet import GoogLeNet as PoseNet


args = docopt(__doc__)

sixd_base = args["--dataset"]
network = args["--network"]
max_rot_pert = float(args["--max_rot_pert"]) / 180. * np.pi
max_trans_pert = float(args["--max_trans_pert"])
#iterations = int(args["--iterations"])
save_interval = 50
init_learning_rate = 0.01



def gen_data(bench,indices, max_loop):
    for i in range(indices, max_loop):
        col = bench.frames[i].color.copy()
        _, gt_pose = bench.frames[i].gt[0]  # corrected,获取原始GT数据
        hypo_pose = perturb_pose(gt_pose, max_rot_pert, max_trans_pert)
        yield gt_pose, hypo_pose, col

def gen_data_batch(bench, batch_size):

    gt_pose_batch = []
    hypo_pose_batch = []
    col_batch = []
    nrframs = bench.nrFrams
    random.seed()
    indices = random.randint(1, nrframs)
    print("random_number",indices)
    max_loop = indices + batch_size
    if max_loop >= nrframs:
        max_loop = nrframs
        indices = nrframs - batch_size
    for _ in range(batch_size):
        gt_pose, hypo_pose, col = next(gen_data(bench, indices, max_loop))#生成一批数据,随机生成
        #seq_batch.append(seq)
        gt_pose_batch.append(gt_pose)
        hypo_pose_batch.append(hypo_pose)
        col_batch.append(col)
    return gt_pose_batch, hypo_pose_batch, col_batch

def add_pose_loss(predict_r, predict_t, poses_r, poses_t):
    loss = None
    try:
        # predict_r, predict_t = net.full_Net(input)
        l1_r = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(predict_r, poses_r)))) * 0.3
        l1_t = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(predict_t, poses_t)))) * 150
        print("", l1_r)
        if loss is None:
            loss = l1_r + l1_t
        else:
            loss += l1_r + l1_t
    except:
        pass
    return loss




def Ds_loss(hypo_loss, gt_loss):
    loss = None

    try:
        loss1 = tf.reduce_sum(hypo_loss)
        loss2 = tf.reduce_sum(gt_loss)
        if loss is None:
            loss = loss1 + loss2
        else:
            loss += loss1 + loss2

    except:
        pass
    return loss
@tf.function
@tf.function
def get_min(at, bt, ct, v):
    min = None

    for i in range(100):
        v_0, v_1, v_2 = tf.split(tf.gather(v, i), num_or_size_splits=3, axis=-1)
        l1 = tf.abs(tf.subtract(v_0, at))
        l2 = tf.abs(tf.subtract(v_1, bt))
        l3 = tf.abs(tf.subtract(v_2, ct))
        d = tf.add(tf.add(l1,l2), l3)
        if min is None:
            min = d
        bool = tf.cond(d <= min, lambda: 1, lambda: 0)
        if bool == 1:
            min = d
    return min

def point_loss(v, q, t):
    loss = None


    w, x, y, z = tf.split(tf.gather(q, 0), num_or_size_splits=4, axis=-1)
    t_0, t_1, t_2 = tf.split(tf.gather(t, 0), num_or_size_splits=3, axis=-1)

    for i in range(100):
        v_0, v_1, v_2 = tf.split(tf.gather(v, i), num_or_size_splits=3, axis=-1)
        #d1, d2, d3 = tf.split(tf.gather(contour_points, i), num_or_size_splits=3, axis=-1)

        a1 = tf.multiply(tf.multiply(w, w), v_0)
        a2 = tf.multiply(tf.multiply(tf.multiply(2.0, y), w), v_2)

        a3 = tf.multiply(tf.multiply(tf.multiply(2.0, z), w), v_1)
        a4 = tf.multiply(tf.multiply(x, x), v_0)
        a5 = tf.multiply(tf.multiply(tf.multiply(2.0, y), x), v_1)
        a6 = tf.multiply(tf.multiply(tf.multiply(2.0, z), x), v_2)
        a7 = tf.multiply(tf.multiply(z, z), v_0)
        a8 = tf.multiply(tf.multiply(y, y), v_0)
        a = tf.subtract(tf.subtract(tf.add(tf.add(tf.add(tf.subtract(tf.add(a1, a2), a3), a4), a5), a6), a7), a8)
        # a = w*w*v[0] + 2*y*w*v[2] - 2*z*w*v[1] + x*x*v[0] + 2*y*x*v[1] + 2*z*x*v[2] - z*z*v[0] - y*y*v[0]

        b1 = tf.multiply(tf.multiply(tf.multiply(2.0, x), y), v_0)
        b2 = tf.multiply(tf.multiply(y, y), v_1)
        b3 = tf.multiply(tf.multiply(tf.multiply(2.0, x), y), v_2)
        b4 = tf.multiply(tf.multiply(tf.multiply(2.0, w), z), v_0)
        b5 = tf.multiply(tf.multiply(z, z), v_1)
        b6 = tf.multiply(tf.multiply(w, w), v_1)
        b7 = tf.multiply(tf.multiply(tf.multiply(2.0, x), w), v_2)
        b8 = tf.multiply(tf.multiply(x, x), v_1)
        # b = tf.add(b1,b2)
        b = tf.subtract(tf.subtract(tf.add(tf.subtract(tf.add(tf.add(tf.add(b1, b2), b3), b4), b5), b6), b7), b8)

        c1 = tf.multiply(tf.multiply(tf.multiply(2.0, x), z), v_0)
        c2 = tf.multiply(tf.multiply(tf.multiply(2.0, y), z), v_1)
        c3 = tf.multiply(tf.multiply(z, z), v_2)
        c4 = tf.multiply(tf.multiply(tf.multiply(2.0, w), y), v_0)
        c5 = tf.multiply(tf.multiply(y, y), v_2)
        c6 = tf.multiply(tf.multiply(tf.multiply(2.0, w), x), v_1)
        c7 = tf.multiply(tf.multiply(x, x), v_2)
        c8 = tf.multiply(tf.multiply(w, w), v_2)
        c = tf.add(tf.subtract(tf.add(tf.subtract(tf.subtract(tf.add(tf.add(c1, c2), c3), c4), c5), c6), c7), c8)

        at = tf.add(a, t_0)
        bt = tf.add(b, t_1)
        ct = tf.add(c, t_2)

        # l1 = tf.reduce_sum(tf.abs(tf.subtract(v_0, at)))
        # l2 = tf.reduce_sum(tf.abs(tf.subtract(v_1, bt)))
        # l3 = tf.reduce_sum(tf.abs(tf.subtract(v_2, ct)))

        min = get_min(at, bt, ct, v)
    try:
        if loss is None:
            #loss = l1 + l2 + l3
            loss = tf.reduce_sum(min)
        else:
            #loss += l1 + l2 + l3
            loss += tf.reduce_sum(min)
    except:
        pass


    return loss
def get_loss(predict_r, predict_t, dt_sign, dt_unsign):
    loss = None
    try:
        #q_r = tfq.vector3d_to_quaternion(predict_t)
        q_r = tfq.quaternion_conjugate(predict_r)#共轭四元数
        pi = tf.constant(math.pi)
        #tf.pad(a,[[1,0],[0,1]])
        dt_sign = tf.pad(dt_sign, [[0,0],[0,1]])
        dt_unsign = tf.pad(dt_unsign, [[0, 0], [0, 1]])

        predict_t = tf.pad(predict_t, [[0,0],[0,1]])
        value_l1 = (predict_r * dt_sign * q_r + predict_t) * pi
        value_l1 = value_l1 * dt_unsign
        sign_loss = tf.reduce_sum(value_l1)

        value_l2 = (q_r * dt_unsign * predict_r - predict_t) * pi
        value_l2 = value_l2 * dt_sign
        unsign_loss = tf.reduce_sum(value_l2)

        #loss = tf.reduce_sum(sign_loss + unsign_loss)
        #gray = tf.image.convert_image_dtype(pi * value, tf.uint8)

        if loss is None:
            loss = sign_loss + unsign_loss
        else:
            loss += sign_loss + unsign_loss

    except:
        pass
    return loss

def full_Net(input_shape):
    x = tf.concat([input_shape[0], input_shape[1]], axis=0)
    return x

def train(iterations=100, batch_size=16):

    croppings = yaml.safe_load(open('config/croppings.yaml', 'rb'))  # 裁剪参数
    dataset_name = 'linemod'#不知道啥用，先设置数据集名称



    scene_patches = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="input_patch")
    render_patches = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="hypo_patch")
    poses_r = tf.placeholder(tf.float32, [None, 4])
    poses_t = tf.placeholder(tf.float32, [None, 3])


    gt_contour = tf.placeholder(tf.float32, [100, 3])
    predict_r, predict_t = man_net.full_Net([scene_patches,render_patches])
    predict_r = tf.identity(predict_r, name="predict_r")  # 恒等函数映射，命名输出的节点
    predict_t = tf.identity(predict_t, name="predict_t")
    # #loss = get_loss(predict_r,predict_t,dt_sign, dt_unsign)
    #loss = add_pose_loss(predict_r, predict_t, poses_r, poses_t)
    loss = point_loss(gt_contour, predict_r, predict_t)

    crop = tf.placeholder(tf.float32, name="crop")
    # print('loss', loss)

    global_step = training_util.create_global_step()
    opt = tf.train.AdamOptimizer(learning_rate=0.0001,beta1=0.9,beta2=0.999,epsilon=0.00000001,use_locking=False,name='Adam').minimize(loss,global_step)
    print("神了")
    cam_info = load_yaml(os.path.join(sixd_base, 'camera.yml'))
    init = tf.global_variables_initializer()#权值初始化

    variables_to_save = tf.global_variables()
    saver = tf.train.Saver(variables_to_save)  # 设置保存变量的checkpoint存储模型
    bench = load_sixd(sixd_base, nr_frames=0, seq=1)#加载数据
    output_checkpoint = os.path.join(output_checkpoint_dir, checkpoint_file)
    print("nrframs",bench.nrFrams)
    with tf.Session() as sess:
        sess.run(init)
        cam = bench.cam
        iter = -1
        for sub in range(iterations):
            ren = Renderer((640, 480), cam)  # 生成渲染器,数据维度的转化

            gt_pose_batch, hypo_pose_batch, col_batch = gen_data_batch(bench, batch_size)
            #print("gt",len(gt_pose_batch),"hypo",len(hypo_pose_batch),"col",len(col_batch))
            batch_loss = 0
            index = 0
            iteration = 3
            for _ in tqdm(range(batch_size)):
                col = col_batch[index].copy()
                #col = col_batch[index]
                #print("index",index)
                # _, gt_pose, _ = frame.gt[0]
                # print(frame.gt[0])
                gt_pose = gt_pose_batch[index]  # corrected,获取原始GT数据

                perturbed_pose = perturb_pose(gt_pose, max_rot_pert, max_trans_pert)
                refinable = Refinable(model=bench.models[str(int(1))], label=0, hypo_pose=perturbed_pose,
                                      metric_crop_shape=croppings[dataset_name]['obj_{:02d}'.format(int(1))],
                                      # 这里的obj为物体的顺序
                                      input_col=col)

                # refiner.iterative_contour_alignment(refinable=refinable, opt = opt, loss=loss, hypo_r=poses_r, hypo_t=poses_t, input=input,
                #                                     crop=crop, predict_r=predict_r, predict_t=predict_t, i = i, max_iterations=3,display=1)

                '''以下来自corrected_refiner'''
                index = index + 1
                display = None  # 训练展示

                refinable.refined = False
                ren.clear()
                ren.draw_model(refinable.model, refinable.hypo_pose, ambient=0.5, specular=0, shininess=100,
                               light_col=[1, 1, 1], light=[0, 0, -1])
                refinable.hypo_col, refinable.hypo_dep = ren.finish()






                #rendering result of gt
                ren.draw_model(refinable.model, gt_pose, ambient=0.5, specular=0, shininess=100,
                               light_col=[1, 1, 1], light=[0, 0, -1])
                gt_col, gt_dep = ren.finish()
                contour_gt = get_viewpoint_cloud(gt_dep, cam_info, 100)  # 每一帧获取gt轮廓信息
                _, unsign_gt = distance_transform(gt_dep)  # gt轮廓点distance_trans






                # padding to prevent crash when object gets to close to border
                pad = int(refinable.metric_crop_shape[0] / 2)
                input_col = np.pad(refinable.input_col, ((pad, pad), (pad, pad), (0, 0)), 'wrap')
                hypo_col = np.pad(refinable.hypo_col, ((pad, pad), (pad, pad), (0, 0)), 'wrap')

                centroid = verify_objects_in_scene(refinable.hypo_dep)

                if centroid is None:
                    print("Hypo outside of image plane")
                    return refinable

                (x, y) = centroid
                x_normalized = x / 640.
                y_normalized = y / 480.
                crop_shift = [x_normalized, y_normalized]
                # crop to metric shape
                slice = (int(refinable.metric_crop_shape[0] / 2), int(refinable.metric_crop_shape[1] / 2))
                input_col = input_col[y: y + 2 * slice[1], x: x + 2 * slice[0]]
                hypo_col = hypo_col[y: y + 2 * slice[1], x: x + 2 * slice[0]]
                #input_shape = (140, 140)
                input_shape = (224, 224)

                # resize to input shape of architecture
                scene_patch = cv2.resize(input_col, input_shape)  # 原数据pose场景训练
                render_patch = cv2.resize(hypo_col, input_shape)  # 扰乱pose数据集

                #训练可视化

                # cv2.imshow("scene", scene_patch)
                # # cv2.imshow("render", render_patch)
                # cv2.waitKey(300)


                # write feed dict
                hypo_trans = refinable.hypo_pose[:3, 3]
                hypo_rot = matrix2quaternion(refinable.hypo_pose[:3, :3])
                if hypo_rot[0] < 0.:
                    hypo_rot *= -1
                # print("scene_patch", scene_patch,"render_patch", render_patch,"双人组形状",hypo_rot,hypo_trans)
                # image = scene_patch[np.newaxis, :]
                    # rendering result of gt



                feed_dict = {
                    render_patches: [render_patch],
                    scene_patches: [scene_patch],
                    poses_r: hypo_rot.reshape(1, 4),
                    poses_t: hypo_trans.reshape(1, 3),
                    gt_contour: contour_gt,
                    # gt_ds:l2,
                    crop: [[x_normalized, y_normalized]]}

                #refined_r, refined_t = sess.run([predict_r,predict_t], feed_dict=feed_dict)#这个测试的时候用


                sess.run(opt, feed_dict=feed_dict)
                loss_val = sess.run(loss, feed_dict=feed_dict)
                #if sub > 0 and sub % save_interval == 0:
                    # saver.save(sess, output_checkpoint, global_step=global_step)
                    # print('Intermediate file saved at: ' + output_checkpoint)
                iter = sub

                time.sleep(0.1)
                batch_loss += loss_val
                if index >= batch_size:
                    batch_loss = batch_loss / batch_size
                    print("epoch: " + str(sub + 1) + "\t" + "loss:" + str(batch_loss))
                if display:
                    concat = cv2.hconcat([refinable.input_col, refinable.hypo_col])
                    cv2.imshow('show_train', concat)
                    #concat2 = cv2.hconcat([sign, unsign])
                    cv2.imshow('show_train', concat)
                    #cv2.imshow('show_dt', sign)
                    cv2.waitKey(500)
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['input_patch','hypo_patch','predict_r','predict_t','crop'])
        #if iter > 0 and iter % save_interval != 0:
        with tf.gfile.FastGFile('ckpt_model/new_6D_model.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())
        # if iter > 0 and iter % save_interval != 0:
        #     saver.save(sess, output_checkpoint, global_step=global_step)
        #     print('Intermediate file saved at: ' + output_checkpoint)
        #saver.save(sess=sess, save_path='ckpt_model/6D_model.ckpt')  # 保存模型
if __name__ == '__main__':
   train(100, 64)