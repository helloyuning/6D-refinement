#import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import get_pb_graph
import pb_newTest
import graph_def_editor as ge
from rendering.utils import trans_rot_err
from utils.quaternion import matrix2quaternion
from pyquaternion import Quaternion
import numpy as np




class Architecture(object):

    def __init__(self, network_file, sess):

        assert sess is not None
        #恢复网络参数
        # self.load_frozen_graph(network=network_file)
        # self.get_newGraph(network=network_file)
        self.load_frozen_graph(network=network_file)#调用自己调整好的graph


        #input tensors
        self.scene_patch = sess.graph.get_tensor_by_name('input_patches:0')
        self.render_patch = sess.graph.get_tensor_by_name('hypo_patches:0')
        self.hypo_rotation = sess.graph.get_tensor_by_name('hypo_rotations:0')
        self.hypo_translation = sess.graph.get_tensor_by_name('hypo_translations:0')
        self.crop_shift = sess.graph.get_tensor_by_name('cropshift:0')

        # in architecture saved information
        self.input_shape = [224, 224, 3]


        # output tensors
        self.rotation_hy_to_gt = sess.graph.get_tensor_by_name('refined_rotation:0')
        self.translation_hy_to_gt = sess.graph.get_tensor_by_name('refined_translation:0')



        # tf.reshape(self.rotation_hy_to_gt,[1,4])
        # tf.reshape(self.translation_hy_to_gt,[1, 3])





    def load_frozen_graph(self, network):
        #载入训练好的模型
        """ Loads the provided network as the new default graph """
        with tf.gfile.FastGFile(network, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    # def load_frozen_graph(self,network):#自己编写的graph导入
    #     with tf.gfile.GFile(network, 'rb') as f:
    #         graph_def = tf.GraphDef()
    #         graph_def.ParseFromString(f.read())
    #         _ = tf.import_graph_def(graph_def, name='')
    #     with tf.Graph().as_default() as detection_graph:
    #         tf.import_graph_def(graph_def, name='')
    #
    #     with detection_graph.as_default():  # 创建clone, 生成新的节点名称
    #         str = 'InceptionV4'
    #         const_var_name_pairs = {}
    #         probable_variables = [op for op in detection_graph.get_operations() if
    #                               op.type == "Const" and str not in op.name]  # 获取常量
    #         # probable_variables = [op for op in detection_graph.get_operations() if op.type == "Const"]  # 获取常量
    #         available_names = [op.name for op in detection_graph.get_operations()]  # 获取所有Operation名称
    #
    #         for op in probable_variables:
    #             name = op.name
    #             if name + '/read' not in available_names:
    #                 continue
    #             # print('{}:0'.format(name))
    #             tensor = detection_graph.get_tensor_by_name('{}:0'.format(name))
    #             with tf.compat.v1.Session() as s:
    #                 tensor_as_numpy_array = s.run(tensor)
    #             var_shape = tensor.get_shape()
    #             # Give each variable a name that doesn't already exist in the graph
    #             # 生成对应节点的对应名称  原名字 + turned_var
    #             var_name = '{}_turned_var'.format(name)
    #             var = tf.get_variable(name=var_name, dtype='float32', shape=var_shape,
    #                                   initializer=tf.constant_initializer(tensor_as_numpy_array))  # 后期添加的初始化变量，使用原来的const
    #             # print(var_name)
    #             var = tf.Variable(name=var_name, dtype=op.outputs[0].dtype, initial_value=tensor_as_numpy_array, trainable=True, shape=var_shape)
    #             const_var_name_pairs[name] = var_name  # 生成对应常量和对应可用变量名称的字典
    #     ge_graph = ge.Graph(detection_graph.as_graph_def())
    #     print("trainable variables:", tf.trainable_variables())
    #     current_var = (tf.compat.v1.trainable_variables())
    #     name_to_op = dict([(n.name, n) for n in ge_graph.nodes])  # 获取原始图的节点，保存为字典
    #
    #     for const_name, var_name in const_var_name_pairs.items():
    #         const_op = name_to_op[const_name]
    #         var_reader_op = name_to_op[var_name + '/read']
    #         ge.swap_outputs(ge.sgv(const_op), ge.sgv(var_reader_op))
