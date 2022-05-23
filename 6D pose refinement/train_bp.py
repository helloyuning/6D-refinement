"""
Simple script to run a forward pass employing the Refiner on a SIXD dataset sample with a trained model.

Usage:
  test_refinement.py [options]
  test_refinement.py -h | --help

Options:
    -d --dataset=<string>        Path to SIXD dataset[default: E:\\lm_base\\lm]
    -o --object=<string>         Object to be evaluated [default: 01]
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

from utils.sixd import load_sixd, load_yaml
from refiner.architecture import Architecture
from rendering.renderer import Renderer
from refiner.refiner import Refiner, Refinable
from rendering.utils import *
#from refiner.non_sess_network import Architecture
from timeit import default_timer as timer
from docopt import docopt
import graph_def_editor as ge
from utils import get_hypoPose as hp#自制模型预测pose导入
from tensorflow.python.training import training_util




args = docopt(__doc__)

sixd_base = args["--dataset"]
network = args["--network"]
max_rot_pert = float(args["--max_rot_pert"]) / 180. * np.pi
max_trans_pert = float(args["--max_trans_pert"])
iterations = int(args["--iterations"])


def load_frozen_graph(network):  # 自己编写的graph导入
    with tf.gfile.GFile(network, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    with tf.Graph().as_default() as detection_graph:
        tf.import_graph_def(graph_def, name='')

    with detection_graph.as_default():  # 创建clone, 生成新的节点名称
        str = 'InceptionV4'
        const_var_name_pairs = {}
        probable_variables = [op for op in detection_graph.get_operations() if
                              op.type == "Const" and str not in op.name]  # 获取常量
        # probable_variables = [op for op in detection_graph.get_operations() if op.type == "Const"]  # 获取常量
        available_names = [op.name for op in detection_graph.get_operations()]  # 获取所有Operation名称

        for op in probable_variables:
            name = op.name
            if name + '/read' not in available_names:
                continue
            # print('{}:0'.format(name))
            tensor = detection_graph.get_tensor_by_name('{}:0'.format(name))
            with tf.compat.v1.Session() as s:
                tensor_as_numpy_array = s.run(tensor)
            var_shape = tensor.get_shape()
            # Give each variable a name that doesn't already exist in the graph
            # 生成对应节点的对应名称  原名字 + turned_var
            var_name = '{}_turned_var'.format(name)
            var = tf.get_variable(name=var_name, dtype='float32', shape=var_shape,
                                  initializer=tf.constant_initializer(tensor_as_numpy_array))  # 后期添加的初始化变量，使用原来的const
            # print(var_name)
            var = tf.Variable(name=var_name, dtype=op.outputs[0].dtype, initial_value=tensor_as_numpy_array,
                              trainable=True, shape=var_shape)
            const_var_name_pairs[name] = var_name  # 生成对应常量和对应可用变量名称的字典
    ge_graph = ge.Graph(detection_graph.as_graph_def())

    current_var = (tf.compat.v1.trainable_variables())
    name_to_op = dict([(n.name, n) for n in ge_graph.nodes])  # 获取原始图的节点，保存为字典

    for const_name, var_name in const_var_name_pairs.items():
        const_op = name_to_op[const_name]
        var_reader_op = name_to_op[var_name + '/read']
        ge.swap_outputs(ge.sgv(const_op), ge.sgv(var_reader_op))
    return detection_graph
def add_pose_loss(architecture):
    loss = None
    try:
        # predict_r, predict_t = net.full_Net(input)
        l1_r = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(architecture.rotation_hy_to_gt, architecture.hypo_rotation)))) * 0.3
        l1_t = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(architecture.translation_hy_to_gt, architecture.hypo_translation)))) * 150
        print("", l1_r)
        if loss is None:
            loss = l1_r + l1_t
        else:
            loss += l1_r + l1_t
    except:
        pass
    return loss
def get_loss(predict_r, predict_t, poses_r, poses_t):
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
def train():

    objects = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"]#总目标个数为15
    croppings = yaml.safe_load(open('config/croppings.yaml', 'rb'))  # 裁剪参数
    dataset_name = 'linemod'#不知道啥用，先设置数据集名称
    net = "models\\refiner_linemod_obj_02.pb"

    # rot_gt = tf.placeholder(tf.float32, [None, 4], "rot_gt")
    # trans_gt = tf.placeholder(tf.float32, [None, 3], "trans_gt")
    #graph = load_frozen_graph(net)#rot_gt是默认图

    #print(graph.get_operations())


    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.1, name="myNewAdam")  # 优化器设置，！！！！损失函数设置


    #print("trainable variables:", tf.trainable_variables())

    cam_info = load_yaml(os.path.join(sixd_base, 'camera.yml'))
    init = tf.global_variables_initializer()#权值初始化

    bench = load_sixd(sixd_base, nr_frames=0, seq=1)  # 数据打包处理，每个物体加载的对象为mask,depth,rgb.根据训练数据的多少三合1组合

    with tf.Session() as sess:
        sess.run(init)
        free_graph = load_frozen_graph(network)

        global_step = training_util.create_global_step()
        #architecture = Architecture(network_file=network, sess=sess)  # 此处为网络，应该有一个东西预测生成rt，然后才是Refinemnet
        with free_graph.as_default():
            scene_patch = free_graph.get_tensor_by_name('input_patches:0')
            render_patch = free_graph.get_tensor_by_name('hypo_patches:0')
            hypo_rotation = free_graph.get_tensor_by_name('hypo_rotations:0')
            hypo_translation = free_graph.get_tensor_by_name('hypo_translations:0')
            crop_shift = free_graph.get_tensor_by_name('cropshift:0')

            # in architecture saved information
            input_shape = [224, 224, 3]

            # output tensors
            rotation_hy_to_gt = free_graph.get_tensor_by_name('refined_rotation:0')
            translation_hy_to_gt = free_graph.get_tensor_by_name('refined_translation:0')
            #print("trainable variables:", tf.trainable_variables())
            loss = get_loss(rotation_hy_to_gt, translation_hy_to_gt, hypo_rotation, hypo_translation)
            opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=0.00000001,use_locking=False, name='Adam').minimize(loss, global_step)
            print("ok")



    #     for obj in objects[0]:
    #
    #         #print(bench.cam)
    #         ren = Renderer((640, 480), bench.cam)  # 生成渲染器,数据维度的转化
    #         refiner = Refiner(architecture=architecture, ren=ren, session=session)  # 数据渲染优化
    #         i = 1
    #         for frame in bench.frames:
    #             col = frame.color.copy()
    #             # _, gt_pose, _ = frame.gt[0]
    #             _, gt_pose = frame.gt[0]  # corrected,获取原始GT数据
    #             #unsigned, signed = distance_transform(frame.depth)
    #             # trans_p = transform_points(bench.models[str(int(obj))].vertices,gt_pose)
    #             # print(trans_p)
    #
    #             perturbed_pose = perturb_pose(gt_pose, max_rot_pert, max_trans_pert)
    #             #hypo_col, hypo_dep = hp.get_render_path(model=bench.models[str(int(obj))],hypo_pose=perturbed_pose)
    #             #feed_dict = hp.get_init_feedDict(architecture, ren, input_col=col,model=bench.models[str(int(obj))],hypo_pose=perturbed_pose)#喂养数据，导入扰乱的姿势
    #             #hypo_pose = hp.get_hypoPose(architecture,feed_dict,sess)#生成投影的pose
    #             refinable = Refinable(model=bench.models[str(int(obj))], label=0, hypo_pose=perturbed_pose,
    #                                   metric_crop_shape=croppings[dataset_name]['obj_{:02d}'.format(int(obj))],
    #                                   input_col=col)
    #             start = timer() #可视化时，结果的时间的计算
    #             refiner.iterative_contour_alignment(refinable=refinable,display=1)
    #             end = timer() #refinement的结果时间计算

                # i = i + 1
                # if i > 50:

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
                # cv2.rectangle(render_col, (0, 0), (133, 40), (1., 1., 1.), -1)
                # cv2.putText(render_col, fps, (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                # cv2.imshow("Refined Output", render_col)
                # cv2.waitKey(300)













if __name__ == '__main__':
    train()



