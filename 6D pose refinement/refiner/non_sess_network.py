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




class Architecture_non(object):

    def __init__(self, network_file, sess):
        self.load_frozen_graph(network=network_file)
        self.scene_patch = sess.graph.get_tensor_by_name('input_patch:0')
        self.render_patch = sess.graph.get_tensor_by_name('hypo_patch:0')
        self.hypo_rotation = sess.graph.get_tensor_by_name('predict_r:0')
        self.hypo_translation = sess.graph.get_tensor_by_name('predict_t:0')
        self.crop_shift = sess.graph.get_tensor_by_name('crop:0')

        # in architecture saved information
        self.input_shape = [224, 224, 3]
        #self.input_shape = [140, 140, 3]#ape

        self.rotation_hy_to_gt = sess.graph.get_tensor_by_name('predict_r:0')
        self.translation_hy_to_gt = sess.graph.get_tensor_by_name('predict_t:0')



    def load_frozen_graph(self, network):
        #载入训练好的模型
        """ Loads the provided network as the new default graph """
        with tf.gfile.FastGFile(network, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')













