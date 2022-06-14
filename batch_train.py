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
#from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
#import tensorflow._api.v2.compat.v1 as tf
from tfquaternion import scope_wrapper

#tf.disable_v2_behavior()
# from tensorflow._api.v2.compat.v1 import ConfigProto
# from tensorflow._api.v2.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)



#import tensorflow as tf
import yaml
import cv2
import numpy as np
import time
from utils.sixd import load_sixd, load_yaml
#from refiner.architecture import Architecture
from rendering.renderer import Renderer
#from refiner.refiner import Refiner, Refinable
from refiner.corrected_refiner import Refiner, Refinable
from rendering.utils import *
#from refiner.non_sess_network import Architecture
from timeit import default_timer as timer
from docopt import docopt
# import graph_def_editor as ge
# from utils import get_hypoPose as hp#自制模型预测pose导入
from tensorflow.python.training import training_util

from Network import man_net
from tqdm import tqdm
import random
import tfquaternion as tfq
import math
from tensorflow.python.framework import graph_util
#from tensorflow_graphics.geometry.transformation import quaternion
#import tensorflow_addons as tfa
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
        print("l1_r_shape", tf.shape(l1_r))
        if loss is None:
            loss = l1_r + l1_t
        else:
            loss += l1_r + l1_t
    except:
        pass
    return loss



def get_loss_ds(contour_hypo, contour_gt, unsign_hypo, unsign_gt):
    loss2=0

    for pt in contour_hypo:
        loss1 += unsign_gt[pt.x][pt.y]#取出坐标值里的DS
    for pt in contour_gt:
        loss2 += unsign_hypo[pt.x][pt.y]

    return loss1+loss2

@tf.function
def Dsitancc_loss(v, q, t, cam, Ds_gt, pose_r, pose_t):#placeholder_get_pointsds
    # v: 3d point
    # q: quarternion (4d vector) = w + xi+ yj + zk
    # t: translation (3d vector)


    q = tf.convert_to_tensor(q)
    t = tf.convert_to_tensor(t)

    loss = None
    Ds_predict = None
    Ds_grondtrulth = None
    w, x, y, z = tf.split(tf.gather(q, 0), num_or_size_splits=4, axis=-1)
    t_0, t_1, t_2 = tf.split(tf.gather(t, 0), num_or_size_splits=3, axis=-1)

    w_g, x_g, y_g, z_g = tf.split(tf.gather(pose_r, 0), num_or_size_splits=4, axis=-1)
    gt_0, gt_1, gt_2 = tf.split(tf.gather(pose_t, 0), num_or_size_splits=3, axis=-1)

    for i in range(100):

        v_0, v_1, v_2 = tf.split(tf.gather(v, i), num_or_size_splits=3, axis=-1)


        a1 = tf.multiply(tf.multiply(w, w), v_0)
        a2 = tf.multiply(tf.multiply(tf.multiply(2.0,y),w),v_2)

        a3 = tf.multiply(tf.multiply(tf.multiply(2.0,z),w),v_1)
        a4 = tf.multiply(tf.multiply(x,x),v_0)
        a5 = tf.multiply(tf.multiply(tf.multiply(2.0,y),x),v_1)
        a6 = tf.multiply(tf.multiply(tf.multiply(2.0,z),x),v_2)
        a7 = tf.multiply(tf.multiply(z,z),v_0)
        a8 = tf.multiply(tf.multiply(y,y),v_0)
        a = tf.subtract(tf.subtract(tf.add(tf.add(tf.add(tf.subtract(tf.add(a1,a2),a3),a4),a5),a6),a7),a8)
        #a = w*w*v[0] + 2*y*w*v[2] - 2*z*w*v[1] + x*x*v[0] + 2*y*x*v[1] + 2*z*x*v[2] - z*z*v[0] - y*y*v[0]


        b1 = tf.multiply(tf.multiply(tf.multiply(2.0,x),y),v_0)
        b2 = tf.multiply(tf.multiply(y,y),v_1)
        b3 = tf.multiply(tf.multiply(tf.multiply(2.0,x),y),v_2)
        b4 = tf.multiply(tf.multiply(tf.multiply(2.0, w), z), v_0)
        b5 = tf.multiply(tf.multiply(z,z),v_1)
        b6 = tf.multiply(tf.multiply(w,w),v_1)
        b7 = tf.multiply(tf.multiply(tf.multiply(2.0, x), w), v_2)
        b8 = tf.multiply(tf.multiply(x,x),v_1)
        #b = tf.add(b1,b2)
        b = tf.subtract(tf.subtract(tf.add(tf.subtract(tf.add(tf.add(tf.add(b1,b2),b3),b4),b5),b6),b7),b8)

        c1 = tf.multiply(tf.multiply(tf.multiply(2.0,x),z),v_0)
        c2 = tf.multiply(tf.multiply(tf.multiply(2.0,y),z),v_1)
        c3 = tf.multiply(tf.multiply(z, z), v_2)
        c4 = tf.multiply(tf.multiply(tf.multiply(2.0,w),y),v_0)
        c5 = tf.multiply(tf.multiply(y,y),v_2)
        c6 = tf.multiply(tf.multiply(tf.multiply(2.0,w),x),v_1)
        c7 = tf.multiply(tf.multiply(x,x),v_2)
        c8 = tf.multiply(tf.multiply(w,w),v_2)
        c = tf.add(tf.subtract(tf.add(tf.subtract(tf.subtract(tf.add(tf.add(c1,c2),c3),c4),c5),c6),c7),c8)

        at = tf.add(a,t_0)
        bt = tf.add(b,t_1)
        ct = tf.add(c,t_2)

        k = tf.concat([at, bt, ct], axis=-1)
        l = tf.expand_dims(k,0)
        #print("现在的形状",tf.shape(a))

        res = project2(l, cam)
        out = tf.gather(res, 0)
        k = tf.gather(out, 0)
        m = tf.gather(out, 1)

        res = tf.gather(Ds_gt, tf.cast(k,dtype=tf.int32))
        res = tf.gather(res, tf.cast(m, dtype=tf.int32))
        #res = tf.reduce_sum(tf.gather(res, tf.cast(y, dtype=tf.int32)))
        if Ds_predict is None:
            Ds_predict = res

        else:
            Ds_predict += res



        #gt ds
        ga1 = tf.multiply(tf.multiply(w_g, w_g), v_0)
        ga2 = tf.multiply(tf.multiply(tf.multiply(2.0, y_g), w_g), v_2)

        ga3 = tf.multiply(tf.multiply(tf.multiply(2.0, z_g), w_g), v_1)
        ga4 = tf.multiply(tf.multiply(x_g, x_g), v_0)
        ga5 = tf.multiply(tf.multiply(tf.multiply(2.0, y_g), x_g), v_1)
        ga6 = tf.multiply(tf.multiply(tf.multiply(2.0, z_g), x_g), v_2)
        ga7 = tf.multiply(tf.multiply(z_g, z_g), v_0)
        ga8 = tf.multiply(tf.multiply(y_g, y_g), v_0)
        ga = tf.subtract(tf.subtract(tf.add(tf.add(tf.add(tf.subtract(tf.add(ga1, ga2), ga3), ga4), ga5), ga6), ga7), ga8)
        # a = w*w*v[0] + 2*y*w*v[2] - 2*z*w*v[1] + x*x*v[0] + 2*y*x*v[1] + 2*z*x*v[2] - z*z*v[0] - y*y*v[0]

        gb1 = tf.multiply(tf.multiply(tf.multiply(2.0, x_g), y_g), v_0)
        gb2 = tf.multiply(tf.multiply(y_g, y_g), v_1)
        gb3 = tf.multiply(tf.multiply(tf.multiply(2.0, x_g), y_g), v_2)
        gb4 = tf.multiply(tf.multiply(tf.multiply(2.0, w_g), z_g), v_0)
        gb5 = tf.multiply(tf.multiply(z_g, z_g), v_1)
        gb6 = tf.multiply(tf.multiply(w_g, w_g), v_1)
        gb7 = tf.multiply(tf.multiply(tf.multiply(2.0, x_g), w_g), v_2)
        gb8 = tf.multiply(tf.multiply(x_g, x_g), v_1)
        # b = tf.add(b1,b2)
        gb = tf.subtract(tf.subtract(tf.add(tf.subtract(tf.add(tf.add(tf.add(gb1, gb2), gb3), gb4), gb5), gb6), gb7), gb8)

        gc1 = tf.multiply(tf.multiply(tf.multiply(2.0, x_g), z_g), v_0)
        gc2 = tf.multiply(tf.multiply(tf.multiply(2.0, y_g), z_g), v_1)
        gc3 = tf.multiply(tf.multiply(z_g, z_g), v_2)
        gc4 = tf.multiply(tf.multiply(tf.multiply(2.0, w_g), y_g), v_0)
        gc5 = tf.multiply(tf.multiply(y_g, y_g), v_2)
        gc6 = tf.multiply(tf.multiply(tf.multiply(2.0, w_g), x_g), v_1)
        gc7 = tf.multiply(tf.multiply(x_g, x_g), v_2)
        gc8 = tf.multiply(tf.multiply(w_g, w_g), v_2)
        gc = tf.add(tf.subtract(tf.add(tf.subtract(tf.subtract(tf.add(tf.add(gc1, gc2), gc3), gc4), gc5), gc6), gc7), gc8)

        gat = tf.add(ga, gt_0)
        gbt = tf.add(gb, gt_1)
        gct = tf.add(gc, gt_2)

        gk = tf.concat([gat, gbt, gct], axis=-1)
        gl = tf.expand_dims(gk, 0)
        # print("现在的形状",tf.shape(a))

        gres = project2(gl, cam)
        gout = tf.gather(gres, 0)
        gk = tf.gather(gout, 0)
        gm = tf.gather(gout, 1)

        gres = tf.gather(Ds_gt, tf.cast(gk, dtype=tf.int32))
        gres = tf.gather(gres, tf.cast(gm, dtype=tf.int32))

        if Ds_grondtrulth is None:

            Ds_grondtrulth = gres

        else:
            Ds_grondtrulth += gres

    value = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(Ds_grondtrulth,Ds_predict))))
    if loss is None:
        loss = value
    else:
        loss += value
        #loss += Ds_gt[x, y]
        #print("过了")
        #print("guole,shape_res",tf.shape(res))

    return loss





@tf.function
def transform_pt(v, q, t, cam, Ds_gt):#placeholder_get_pointsds
    # v: 3d point
    # q: quarternion (4d vector) = w + xi+ yj + zk
    # t: translation (3d vector)

    # q = tf.convert_to_tensor(q)
    # t = tf.convert_to_tensor(t)

    loss = None
    Ds = None
    pad_t = tf.pad(t, [[0, 0], [0, 1]])
    w, x, y, z = tf.split(tf.gather(q, 0), num_or_size_splits=4, axis=-1)
    t_0, t_1, t_2 = tf.split(tf.gather(t, 0), num_or_size_splits=3, axis=-1)


    for i in range(100):
        v_0, v_1, v_2 = tf.split(tf.gather(v, i), num_or_size_splits=3, axis=-1)


        a1 = tf.multiply(tf.multiply(w, w), v_0)
        a2 = tf.multiply(tf.multiply(tf.multiply(2.0,y),w),v_2)

        a3 = tf.multiply(tf.multiply(tf.multiply(2.0,z),w),v_1)
        a4 = tf.multiply(tf.multiply(x,x),v_0)
        a5 = tf.multiply(tf.multiply(tf.multiply(2.0,y),x),v_1)
        a6 = tf.multiply(tf.multiply(tf.multiply(2.0,z),x),v_2)
        a7 = tf.multiply(tf.multiply(z,z),v_0)
        a8 = tf.multiply(tf.multiply(y,y),v_0)
        a = tf.subtract(tf.subtract(tf.add(tf.add(tf.add(tf.subtract(tf.add(a1,a2),a3),a4),a5),a6),a7),a8)
        #a = w*w*v[0] + 2*y*w*v[2] - 2*z*w*v[1] + x*x*v[0] + 2*y*x*v[1] + 2*z*x*v[2] - z*z*v[0] - y*y*v[0]


        b1 = tf.multiply(tf.multiply(tf.multiply(2.0,x),y),v_0)
        b2 = tf.multiply(tf.multiply(y,y),v_1)
        b3 = tf.multiply(tf.multiply(tf.multiply(2.0,x),y),v_2)
        b4 = tf.multiply(tf.multiply(tf.multiply(2.0, w), z), v_0)
        b5 = tf.multiply(tf.multiply(z,z),v_1)
        b6 = tf.multiply(tf.multiply(w,w),v_1)
        b7 = tf.multiply(tf.multiply(tf.multiply(2.0, x), w), v_2)
        b8 = tf.multiply(tf.multiply(x,x),v_1)
        #b = tf.add(b1,b2)
        b = tf.subtract(tf.subtract(tf.add(tf.subtract(tf.add(tf.add(tf.add(b1,b2),b3),b4),b5),b6),b7),b8)

        c1 = tf.multiply(tf.multiply(tf.multiply(2.0,x),z),v_0)
        c2 = tf.multiply(tf.multiply(tf.multiply(2.0,y),z),v_1)
        c3 = tf.multiply(tf.multiply(z, z), v_2)
        c4 = tf.multiply(tf.multiply(tf.multiply(2.0,w),y),v_0)
        c5 = tf.multiply(tf.multiply(y,y),v_2)
        c6 = tf.multiply(tf.multiply(tf.multiply(2.0,w),x),v_1)
        c7 = tf.multiply(tf.multiply(x,x),v_2)
        c8 = tf.multiply(tf.multiply(w,w),v_2)
        c = tf.add(tf.subtract(tf.add(tf.subtract(tf.subtract(tf.add(tf.add(c1,c2),c3),c4),c5),c6),c7),c8)

        at = tf.add(a,t_0)
        bt = tf.add(b,t_1)
        ct = tf.add(c,t_2)

        k = tf.concat([at, bt, ct], axis=-1)
        l = tf.expand_dims(k,0)
        #print("现在的形状",tf.shape(a))


        #res = project2(l, cam)
        res = proj(l, cam)
        out = tf.gather(res, 0)
        k = tf.gather(out, 0)
        m = tf.gather(out, 1)

        k= tf.cast(k, dtype=tf.int32)
        m = tf.cast(m, dtype = tf.int32)
        res_ds = tf.gather(Ds_gt, k)
        res_ds = tf.gather(res_ds, m)
        #res = tf.reduce_sum(tf.gather(res, tf.cast(y, dtype=tf.int32)))

        if Ds is None:
            Ds = res_ds #* tf.square(tf.subtract(q, pad_t))
            #print("guo")
        else:
            Ds += res_ds #* tf.square(tf.subtract(q, pad_t))
    Ds = tf.reduce_sum(Ds)
    print("Ds_shape",tf.shape(Ds))
    try:
        if loss is None:
            loss = Ds
        else:
            loss += Ds
            #loss += Ds_gt[x, y]
        #print("过了")

    except:
        pass#print("guole,shape_res",tf.shape(res))

    return loss
        #print(tf.shape(a))


        #print("tf.shape(res)", tf.shape(res))
        #r = project2(x, cam)
        #print("tf.shape(at)",tf.shape(at))
    #print(tf.shape(res))
    #return r
    #w, x, y, z = q
    #a = w*w*v[0] + 2*y*w*v[2] - 2*z*w*v[1] + x*x*v[0] + 2*y*x*v[1] + 2*z*x*v[2] - z*z*v[0] - y*y*v[0]
    # b = 2*x*y*v[0] + y*y*v[1] + 2*z*y*v[2] + 2*w*z*v[0] - z*z*v[1] + w*w*v[1] - 2*x*w*v[2] - x*x*v[1]
    # c = 2*x*z*v[0] + 2*y*z*v[1] + z*z*v[2] - 2*w*y*v[0] - y*y*v[2] + 2*w*x*v[1] - x*x*v[2] + w*w*v[2]
    # return a + t[0], b + t[1], c + t[2]

@tf.function
def get_v(x):#取值
    result = []
    for i in range(len(x)):
        t = tf.split(x[i], num_or_size_splits=3, axis=-1)
        result.append(t)

    return result

@tf.function
def fn(x):#取值
    w = []
    x = []
    y = []
    z = []

    for i in range(len(x)):
        w.append([x[i][0]])
        x.append([x[i][1]])
        y.append([x[i][2]])
        z.append([x[i][3]])
    return w,x,y,z
#square_if_positive(tf.range(10))

def proj(points_3d, cam):
    """ Project a numpy array of 3D points to the image plane
    :param points_3d: Input array of 3D points (N, 3)
    :param cam: Intrinsics of the camera
    :return: An array of projected 2D points
    """
    # x = cam[0, 2] + points_3d[:, 0] * cam[0, 0] / points_3d[:, 2]
    # y = cam[1, 2] + points_3d[:, 1] * cam[1, 1] / points_3d[:, 2]


    x1 = tf.gather(cam, 0)
    x1 = tf.gather(x1, 2)
    x2 = tf.gather(cam, 0)
    x2 = tf.gather(x2, 0)

    x3 = tf.multiply(points_3d[:, 0],x2)
    x3 = tf.divide(x3, points_3d[:, 2])

    x = tf.add(x1, x3)

    y1 = tf.gather(cam, 1)
    y1 = tf.gather(y1, 2)

    y2 = tf.gather(cam, 1)
    y2 = tf.gather(y2, 1)

    y3 = tf.multiply(points_3d[:, 1], y2)
    y3 = tf.divide(y3, points_3d[:, 2])

    y = tf.add(y1, y3)

    res = tf.stack((x, y), axis=1)
    res = tf.cast(res,dtype=tf.uint32)#转换数据类型

    return res

def project2(points_3d, cam):
    """ Project a numpy array of 3D points to the image plane
    :param points_3d: Input array of 3D points (N, 3)
    :param cam: Intrinsics of the camera
    :return: An array of projected 2D points
    """
    x = cam[0, 2] + points_3d[:, 0] * cam[0, 0] / points_3d[:, 2]
    y = cam[1, 2] + points_3d[:, 1] * cam[1, 1] / points_3d[:, 2]
    # x_n = x.numpy()
    # y_n = y.numpy()
    #x = tf.expand_dims(x, 0)
    #y = tf.expand_dims(y, 0)


    res = tf.stack((x, y), axis=1)
    res = tf.cast(res,dtype=tf.uint32)#转换数据类型
    #print("res_shape:", tf.shape(res))
    return res
    #return np.stack((x, y), axis=1).astype(np.uint32)



def Ds_loss(predict_r, predict_t, contour_3d, cam, Ds_gt):

    #q_r = tfq.quaternion_conjugate(predict_r)  # 共轭四元数
    #q = quaternion.conjugate(predict_r)
    #q_r =  quaternion.from_rotation_matrix(predict_r)#[none,4]
    #contour_3d = tf.pad(contour_3d, [[0, 0], [0, 1]])#[100,3]
    # predict_t_pad = tf.pad(predict_t, [[0, 0], [0, 1]])#[none,4]
    # print("神了")
    #
    #
    # #project_v = tf.add(tf.multiply(tf.multiply(predict_r,contour_3d),q_r),predict_t)不行
    # project_value = predict_r * contour_3d * q_r + predict_t_pad
    # print("pro_shape:",tf.shape(project_value))
    res = transform_pt(contour_3d, predict_r, predict_t,cam, Ds_gt)
    #print("计算好at...shape",tf.shape(res))

    #r = project2(res, cam)
    return res



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



def train(iterations=100, batch_size=16):

    croppings = yaml.safe_load(open('config/croppings.yaml', 'rb'))  # 裁剪参数
    dataset_name = 'linemod'#不知道啥用，先设置数据集名称



    scene_patches = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="input_patch")
    render_patches = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="hypo_patch")
    poses_r = tf.placeholder(tf.float32, [None, 4])
    poses_t = tf.placeholder(tf.float32, [None, 3])

    #predict_r, predict_t = man_net.full_Net(input)

    # hypo_ds = tf.placeholder(tf.float32)
    # gt_ds = tf.placeholder(tf.float32)

    # images = full_Net([scene_patches,render_patches])
    # net = PoseNet({'data': images})
    # predict_r = net.layers['cls3_fc_pose_wpqr']
    # predict_t = net.layers['cls3_fc_pose_xyz']
    #
    # predict_r = tf.identity(predict_r, name="predict_r")  # 恒等函数映射，命名输出的节点
    # predict_t = tf.identity(predict_t, name="predict_t")
    # l3_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(predict_r, poses_r)))) * 1
    # l3_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(predict_t, poses_t)))) * 500
    #
    # loss = l3_q + l3_x


    predict_r, predict_t = man_net.full_Net([scene_patches,render_patches])
    predict_r = tf.identity(predict_r, name="predict_r")  # 恒等函数映射，命名输出的节点
    predict_t = tf.identity(predict_t, name="predict_t")

    #loss = add_pose_loss(predict_r, predict_t, poses_r, poses_t)#单纯计算位姿损失


    contour_3d = tf.placeholder(tf.float32,shape=[100,3])#gt_3d_contour
    Ds_gt = tf.placeholder(tf.float32, shape=[480,640])  # ground truth的二值距离变换

    crop = tf.placeholder(tf.float32, name="crop")

    # print('loss', loss)

    global_step = training_util.create_global_step()


    # Set GPU options
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6833)

    cam_info = load_yaml(os.path.join(sixd_base, 'camera.yml'))
    init = tf.global_variables_initializer()#权值初始化

    variables_to_save = tf.global_variables()
    saver = tf.train.Saver(variables_to_save)  # 设置保存变量的checkpoint存储模型
    bench = load_sixd(sixd_base, nr_frames=0, seq=1)#加载数据
    #camera = tf.constant(bench.cam,tf.float32,shape=[3,3])
    camera = tf.placeholder(tf.float32, shape=[3,3])
    #output_checkpoint = os.path.join(output_checkpoint_dir, checkpoint_file)
    loss = Ds_loss(predict_r, predict_t, contour_3d, camera, Ds_gt)#ccalculate loss with Ds
    print("loss loaded")

    opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=0.00000001, use_locking=False,
                                 name='Adam').minimize(loss, global_step)
    #print("ds_shape",tf.shape(ds_loss))
    print("我是你的神")
    print("nrframs",bench.nrFrams)
    #config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
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

                #rendering result of perturbation
                # ren.draw_model(refinable.model, refinable.hypo_pose, ambient=0.5, specular=0, shininess=100,
                #                light_col=[1, 1, 1], light=[0, 0, -1])
                # refinable.hypo_col, refinable.hypo_dep = ren.finish()
                #
                # contour_hypo = get_viewpoint_cloud(refinable.hypo_dep, cam_info, 100)  # 每一帧获取gt轮廓信息
                # _, unsign_hypo = distance_transform(refinable.hypo_dep)  # gt轮廓点distance_trans
                #
                # for i, data in enumerate(contour_hypo):
                #     l2 = unsign_hypo[i][data], unsign_hypo[i][data], unsign_hypo[i][data]
                #
                #rendering result of gt
                ren.draw_model(refinable.model, gt_pose, ambient=0.5, specular=0, shininess=100,
                               light_col=[1, 1, 1], light=[0, 0, -1])
                gt_col, gt_dep = ren.finish()
                contour_gt = get_viewpoint_cloud(gt_dep, cam_info, 100)  # 每一帧获取gt轮廓信息
                unsign_gt, _ = distance_transform(gt_dep)  # gt轮廓点distance_trans



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



                feed_dict = {
                    render_patches: [render_patch],
                    scene_patches: [scene_patch],
                    poses_r: hypo_rot.reshape(1, 4),
                    poses_t: hypo_trans.reshape(1, 3),
                    contour_3d:contour_gt,#gt_contour
                    Ds_gt: unsign_gt,#gt_distance_transform
                    # gt_ds:l2,
                    crop: [[x_normalized, y_normalized]]}

                #predict_r, predict_t = sess.run([predict_r,predict_t], feed_dict=feed_dict)#这个测试的时候用
                #print("坐标点", Ds_loss(predict_r, predict_t, contour_gt))

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
    # x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # for i in range(len(x)):
    #     print("第",i,"次",x[i])

    # def condition(i, imgs_combined):
    #     return tf.less(i, 5)
    # def body(i, imgs_combined):
    #     c_image = tf.zeros(shape=(2, 3), dtype=tf.float32)
    #     imgs_combined = imgs_combined.write(i, c_image)
    #     return [tf.add(i, 1), imgs_combined]
    train(5, 64)
    # a1 = tf.multiply(tf.multiply(w, w), v_0)
    # a2 = tf.multiply(tf.multiply(tf.multiply(2.0, y), w), v_2)

    q = tf.constant([[1., 2., 3., 4.]])
    t = tf.constant([[5., 6., 7.]])

    v = tf.constant([[10., 11., 12.]])

    w, x, y, z = tf.split(tf.gather(q, 0), num_or_size_splits=4, axis=-1)
    t_0, t_1, t_2 = tf.split(tf.gather(t, 0), num_or_size_splits=3, axis=-1)
    v_0,v_1, v_2 = tf.split(tf.gather(v, 0), num_or_size_splits=3, axis=-1)

    a1 = tf.multiply(tf.multiply(w, w), v_0)
    a2 = tf.multiply(tf.multiply(tf.multiply(2.0, y), w), v_2)

    a3 = tf.multiply(tf.multiply(tf.multiply(2.0, z), w), v_1)
    a4 = tf.multiply(tf.multiply(x, x), v_0)
    a5 = tf.multiply(tf.multiply(tf.multiply(2.0, y), x), v_1)
    a6 = tf.multiply(tf.multiply(tf.multiply(2.0, z), x), v_2)
    a7 = tf.multiply(tf.multiply(z, z), v_0)
    a8 = tf.multiply(tf.multiply(y, y), v_0)
    a = tf.subtract(tf.subtract(tf.add(tf.add(tf.add(tf.subtract(tf.add(a1, a2), a3), a4), a5), a6), a7), a8)

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

    k = tf.concat([at, bt, ct], axis=-1)

    l = tf.expand_dims(k, 0)
    bench = load_sixd(sixd_base, nr_frames=0, seq=1)  # 加载数据
    #camera = tf.constant(bench.cam, tf.float32, shape=[3, 3])
    cam = tf.constant([[572.4114, 0., 325.2611], [0., 573.57043, 242.04899], [0., 0., 1.]])
    gres = project2(l, cam)
    gout = tf.gather(gres, 0)
    gk = tf.gather(gout, 0)
    gm = tf.gather(gout, 1)

    ds = tf.constant([[1., 2., 3., 4.],
                      [5., 6., 7., 8.],
                      [10., 12., 13., 14.]])

    gk2 = tf.cast(gk, dtype=tf.int32)
    gm2 = tf.cast(gm, dtype=tf.int32)

    cx2 = tf.subtract(gk2, tf.constant(463))
    cy2 = tf.subtract(gm2, tf.constant(478))
    gres = tf.gather(ds, cx2)
    gres = tf.gather(gres, cy2)


    sess = tf.Session()
    #print(type(gk))


    print(sess.run(cx2))
    print(sess.run(cy2))
    print(sess.run(gres))
    '''
    [[572.4114    0.      325.2611 ]
    [  0.      573.57043 242.04899]
    [  0.        0.        1.     ]]
       x = cam[0, 2] + points_3d[:, 0] * cam[0, 0] / points_3d[:, 2]
    y = cam[1, 2] + points_3d[:, 1] * cam[1, 1] / points_3d[:, 2]
    '''

    # rr = tf.constant([1])
    # ss = tf.constant([4])
    # v = tf.constant([[1, 2, 3]])
    # v_0, v_1, v_2, v_3 = tf.split(tf.gather(v, 0), num_or_size_splits=4, axis=-1)
    # d = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    #
    #
    # p = tf.constant([[2,1]])
    # k = tf.constant([2,1])
    # s = tf.gather(d, p, axis=0)
    # sess = tf.Session()
    # out = tf.gather(d,3)
    # r = tf.gather(out, 2)
    # print(sess.run(tf.multiply(v,d)))
    # for i in range(len(d)):
    #     for j in range(len(d[0])):
    #         if i == p[0][0] and j == p[0][1]:
    #             print(d[i][j])

    # i = tf.constant(0)
    # combined = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True, clear_after_read=False)
    # _, image_4d = tf.while_loop(condition, body, [i, combined])
    # image_4d = image_4d.stack()
    #a = tf.constant(0)
    #n = tf.constant(10)
    # d = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    # p = tf.constant([[2, 1]])
    #a, n = tf.while_loop(con,body,[a,n])
    #sess = tf.Session()
    # res = sess.run([a,n])
    # print(sess.run(d[p[0][0],p[0][1]]))
    # print(tf.size(d))

    # for i in range(len(d)):
    #     for j in range(len(d[0])):
    #         if i == p[0][0] and j == p[0][1]:
    #             print(d[i][j])

        #b = tf.constant([[7, 8, 9], [10, 11, 12]])
        #ab1 = tf.concat([a, b[i]], axis=0)

        #ab = tf.concat([s,a],axis = 0)
        #ab2 = tf.stack([a, b], axis=0)
        #sess = tf.Session()
    #s = tf.stack(s)
    #print(sess.run(ab1))
    #print(sess.run(s))

    #print(sess.run(ab2))
    # print(ab1)
    # print(ab2)






   # x = tf.placeholder(tf.int32, name='x')
   # y = tf.placeholder(tf.int32, name='y')
   # b = tf.Variable(1, name='b')
   # xy = tf.multiply(x, y)
   # op = tf.add(xy, b, name='op_to_store')
   # init_op = tf.global_variables_initializer()  # 初始化全部变量
   # saver = tf.train.Saver()  # 声明tf.train.Saver类用于保存模型
   # with tf.Session() as sess:
   #     sess.run(init_op)
   #
   #     constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store','x','y'])
   #     # convert_variables_to_constants()方法，可以固化模型结构，将计算图中的变量取值以常量的形式保存
   #     with tf.gfile.FastGFile('ckpt_model/test.pb', mode='wb') as f:
   #         f.write(constant_graph.SerializeToString())
