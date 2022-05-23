import tensorflow._api.v2.compat.v1 as tf
from tensorflow.python.platform import gfile

tf.disable_v2_behavior()
import numpy as np
import graph_def_editor as ge


def generator(pb_file):
    #with tf_graph.as_default() as detection_graph:
    # with detection_graph.as_default():#加载模型
    #     od_graph_def = tf.compat.v1.GraphDef()
    #     with tf.compat.v1.io.gfile.GFile(pb_file, 'rb') as fid:
    #         serialized_graph = fid.read()
    #         od_graph_def.ParseFromString(serialized_graph)
    #         tf.graph_util.import_graph_def(od_graph_def, name='')
    detection_graph = load_graph(pb_file)
    with detection_graph.as_default():#创建clone, 生成新的节点名称
        str = 'InceptionV4'
        const_var_name_pairs = {}
        probable_variables = [op for op in detection_graph.get_operations() if op.type == "Const" and str not in op.name]  # 获取常量
        #probable_variables = [op for op in detection_graph.get_operations() if op.type == "Const"]  # 获取常量
        available_names = [op.name for op in detection_graph.get_operations()]  # 获取所有Operation名称

        '''for op, name in zip(detection_graph.get_operations(), available_names):#获取冻结的非InceptionV4部分的节点
            if op.type == "Const" and str not in name:
                probable_variables.append(op)'''#zip测试用代码，只适用于两个数量完全相等的list

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
            var = tf.get_variable(name=var_name, dtype='float32', shape=var_shape, initializer=tf.constant_initializer(tensor_as_numpy_array))#后期添加的初始化变量，使用原来的const
            # print(var_name)
            #var = tf.Variable(name=var_name, dtype=op.outputs[0].dtype, initial_value=tensor_as_numpy_array, trainable=True, shape=var_shape)
            const_var_name_pairs[name] = var_name  # 生成对应常量和对应可用变量名称的字典
    ge_graph = ge.Graph(detection_graph.as_graph_def())
    name_to_op = dict([(n.name, n) for n in ge_graph.nodes])  # 获取原始图的节点，保存为字典

    for const_name, var_name in const_var_name_pairs.items():
        const_op = name_to_op[const_name]
        var_reader_op = name_to_op[var_name + '/read']
        ge.swap_outputs(ge.sgv(const_op), ge.sgv(var_reader_op))
    # detection_training_graph = ge_graph.to_tf_graph()  # 导出新的graph
    # with detection_training_graph.as_default():#
    #     writer = tf.compat.v1.summary.FileWriter('remap', detection_training_graph)
    #     writer.close
    # return detection_training_graph


def old_graph(pb_file,detection_graph):
    with detection_graph.as_default():  # 加载模型
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.io.gfile.GFile(pb_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.graph_util.import_graph_def(od_graph_def, name='')
    return detection_graph


def load_graph(pb_file):
    with tf.gfile.GFile(pb_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

if __name__ == '__main__':
    frozen_path = 'models/refiner_linemod_obj_02.pb'
    generator(frozen_path)