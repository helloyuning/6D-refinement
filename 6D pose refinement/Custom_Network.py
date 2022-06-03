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
from keras.preprocessing.image import ImageDataGenerator
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
import time
from tqdm import tqdm
from tqdm._tqdm import trange
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
from utils.utils import loadSIXDBench
from tensorflow.python.training import training_util
from Network import man_net
import random#导入随机数




args = docopt(__doc__)

sixd_base = args["--dataset"]
network = args["--network"]
max_rot_pert = float(args["--max_rot_pert"]) / 180. * np.pi
max_trans_pert = float(args["--max_trans_pert"])
iterations = int(args["--iterations"])

init_learning_rate = 0.01
batch_size = 128
image_size = (224, 224,3)


def add_pose_loss(predict_r, predict_t, poses_r, poses_t):
    loss = None
    try:
        # predict_r, predict_t = net.full_Net(input)
        l1_r = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(predict_r, poses_r)))) * 0.3
        l1_t = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(predict_t, poses_t)))) * 150
        #print("", l1_r)
        if loss is None:
            loss = l1_r + l1_t
        else:
            loss += l1_r + l1_t
    except:
        pass
    return loss



def gen_data(full_bench,batch_size):
    #indices = random.randint(64,full_bench.ran)
    random.seed()
    indices = random.randint(0, full_bench.ran)
    max_loop = indices + batch_size
    if max_loop >= full_bench.ran:
        max_loop = indices
        indices = indices - batch_size

    for i in range(indices, max_loop):

        #seq = full_bench.frames[i].gt[i][i]
        col = full_bench.frames[i].color
        samples = full_bench.frames[i].gt[0][1]
        gt_pose = samples[0][0]#[0][0]为gt_pose,[0][1]为perturbated_pose
        hypo_pose = samples[0][1]

        # print("gt_pose", gt_pose)
        # print("hypo_pose", hypo_pose)
        yield gt_pose, hypo_pose, col

def gen_data_batch(full_bench, batch_size):
    #data_gen = gen_data(full_bench)
    #seq_batch = []
    gt_pose_batch = []
    hypo_pose_batch = []
    col_batch = []
    for _ in range(batch_size):
        gt_pose, hypo_pose, col = next(gen_data(full_bench, batch_size))#生成一批数据,随机生成
        #seq_batch.append(seq)
        gt_pose_batch.append(gt_pose)
        hypo_pose_batch.append(hypo_pose)
        col_batch.append(col)
    return gt_pose_batch, hypo_pose_batch, col_batch

def train(iterations=iterations, batch_size=64):

    objects = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"]#总目标个数为15

    croppings = yaml.safe_load(open('config/croppings.yaml', 'rb'))  # 裁剪参数
    dataset_name = 'linemod'#不知道啥用，先设置数据集名称

    seq_to_name = {'Ape': 1}
    full_bench = loadSIXDBench(dataset_path=sixd_base, metric_crop_shape=croppings, seq='Ape', seq_to_name=seq_to_name)
    # samples = full_bench.frames[0].gt[0][1]
    # print("成功取出samples:",samples[0][0])#[0][0]为gt_pose,[0][1]为perturbated_pose

    scene_patches = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="input_patch")
    render_patches = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="hypo_patch")
    poses_r = tf.placeholder(tf.float32, [None, 4], name="poses_r")
    poses_t = tf.placeholder(tf.float32, [None, 3], name="poses_t")



    predict_r, predict_t = man_net.full_Net(scene_patches)
    predict_r = tf.identity(predict_r, name="predicted_r")#恒等函数映射，命名输出的节点
    predict_t = tf.identity(predict_t, name="predicted_t")
    loss = add_pose_loss(predict_r, predict_t, poses_r, poses_t)

    crop = tf.placeholder(tf.float32, name="crop")
    # print('loss', loss)

    global_step = training_util.create_global_step()
    opt = tf.train.AdamOptimizer(learning_rate=0.0001, name='Adam').minimize(loss,global_step)

    cam_info = load_yaml(os.path.join(sixd_base, 'camera.yml'))
    init = tf.global_variables_initializer()#权值初始化
    refinable_pose = []
    print("Total_epoch:", "\t", iterations, "btatch_size:", batch_size)
    saver = tf.train.Saver(tf.global_variables())  # 设置保存变量的checkpoint存储模型
    with tf.Session() as sess:
        sess.run(init)
        # summary_writer = tf.summary.FileWriter('./logs', sess.graph)
        #
        # _x = input[:, :, :, ::-1]
        # tf.summary.image('x', _x, 4)

        #summary_op = tf.summary.merge_all()
        cam = full_bench.cam

        for sub in range(iterations):
            #bench = load_sixd(sixd_base, nr_frames=0, seq=obj)#数据打包处理，每个物体加载的对象为mask,depth,rgb.根据训练数据的多少三合1组合
            #print(bench.cam)
            ren = Renderer((640, 480), cam)  # 生成渲染器,数据维度的转化
            #refiner = Refiner(ren=ren, session=session)  # 数据渲染优化

            #data_source = gen_data(full_bench)
            #seq_batch, gt_pose_batch, hypo_pose_batch = next(data_source)#生成batch_size个数据进行训练
            gt_pose_batch, hypo_pose_batch, col_batch = gen_data_batch(full_bench,batch_size)
            #print("第",sub+1,"批随机批次数据集")
            batch_loss = 0
            index = 0
            iteration = 3
            for j in tqdm(range(batch_size)):
                col = col_batch[index].copy()
                # _, gt_pose, _ = frame.gt[0]
                #print(frame.gt[0])
                gt_pose = gt_pose_batch[index]  # corrected,获取原始GT数据

                perturbed_pose = perturb_pose(gt_pose, max_rot_pert, max_trans_pert)
                refinable = Refinable(model=full_bench.model[str(int(1))], label=0, hypo_pose=perturbed_pose,
                                      metric_crop_shape=croppings[dataset_name]['obj_{:02d}'.format(int(1))],#这里的obj为物体的顺序
                                      input_col=col)

                # refiner.iterative_contour_alignment(refinable=refinable, opt = opt, loss=loss, hypo_r=poses_r, hypo_t=poses_t, input=input,
                #                                     crop=crop, predict_r=predict_r, predict_t=predict_t, i = i, max_iterations=3,display=1)

                '''以下来自corrected_refiner'''
                display = None#训练展示
                min_rotation_displacement = 0.5
                min_translation_displacement = 0.0025
                refinable.refined = False
                ren.clear()
                ren.draw_model(refinable.model, refinable.hypo_pose, ambient=0.5, specular=0, shininess=100,
                                    light_col=[1, 1, 1], light=[0, 0, -1])
                refinable.hypo_col, refinable.hypo_dep = ren.finish()

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
                input_shape = (224, 224)

                # resize to input shape of architecture
                scene_patch = cv2.resize(input_col, input_shape)#原数据pose场景训练
                render_patch = cv2.resize(hypo_col, input_shape)#扰乱pose数据集

                # write feed dict
                hypo_trans = refinable.hypo_pose[:3, 3]
                hypo_rot = matrix2quaternion(refinable.hypo_pose[:3, :3])
                if hypo_rot[0] < 0.:
                    hypo_rot *= -1
                #print("scene_patch", scene_patch,"render_patch", render_patch,"双人组形状",hypo_rot,hypo_trans)
                #image = scene_patch[np.newaxis, :]

                feed_dict = {
                    poses_r: hypo_rot.reshape(1, 4),
                    poses_t: hypo_trans.reshape(1, 3),
                    render_patches: [render_patch],
                    scene_patches: [scene_patch],
                    crop: [[x_normalized, y_normalized]]}

                # run network
                #print("开始feed")

                refined_rotation, refined_translation = sess.run([predict_r,
                                                                          predict_t], feed_dict=feed_dict)

                loss_val = sess.run(loss, feed_dict=feed_dict)
                sess.run(opt, feed_dict=feed_dict)
                index += 1
                time.sleep(0.1)
                batch_loss += loss_val
                if index >= batch_size:
                    batch_loss = batch_loss / batch_size
                    print("epoch: " + str(sub + 1) + "\t" + "train_loss:" + str(batch_loss))
                #print('Iteration: ' + str(index) + '\t' + 'Loss is: ' + str(loss_val))
                # assert np.sum(np.isnan(refined_translation[0])) == 0 and np.sum(np.isnan(refined_rotation[0])) == 0
                #
                # refined_pose = np.identity(4)
                # refined_pose[:3, :3] = Quaternion(refined_rotation[0]).rotation_matrix
                # refined_pose[:3, 3] = refined_translation[0]
                #
                # refinable.hypo_pose = refined_pose
                #
                # refinable.render_patch = render_patch.copy()
                # refinable.refined = True
                # assert refinable is not None
                #
                # last_pose = np.copy(refinable.hypo_pose)
                #
                # for max in range(iteration):
                #
                #     last_trans = last_pose[:3, 3]
                #     last_rot = Quaternion(matrix2quaternion(last_pose[:3, :3]))
                #
                #     cur_trans = refinable.hypo_pose[:3, 3]
                #     cur_rot = Quaternion(matrix2quaternion(refinable.hypo_pose[:3, :3]))
                #
                #     trans_diff = np.linalg.norm(cur_trans - last_trans)
                #     update_q = cur_rot * last_rot.inverse
                #     angular_diff = np.abs((update_q).degrees)
                #
                #     last_pose = np.copy(refinable.hypo_pose)
                #
                #     if display:
                #         concat = cv2.hconcat([refinable.input_col, refinable.hypo_col])
                #         cv2.imshow('test', concat)
                #         cv2.waitKey(50)
                #
                #     if angular_diff <= min_rotation_displacement and trans_diff <= min_translation_displacement:
                #         refinable.iterations = iteration + 1
                #         #return refinable
                #         refinable = refinable
                #         break;#!!!!
                # #print('Iteration: ' + str(index) + '\t' + 'Loss is: ' + str(loss_val))
                # refinable = refinable
                # batch_loss += loss_val
                # refinable.iterations = 3
                # #refinable_pose.append(refinable)
                # index += 1
                # time.sleep(0.1)
                # if index >= batch_size:
                #     batch_loss = batch_loss / batch_size
                #     print("epoch: " + str(sub+1) + "\t" + "train_loss:" + str(batch_loss))
        saver.save(sess=sess, save_path='ckpt_model/6D_model.ckpt')#保存模型





if __name__ == '__main__':
    train(150,32)

