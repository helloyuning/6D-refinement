#import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from Network.InceptionV4 import Stem, InceptionBlockA, InceptionBlockB, \
    InceptionBlockC, ReductionA, ReductionB

from keras.layers import Input, merge, Dropout, Dense, Flatten, Activation

from keras.models import Model



def build_inception_block_a(n):
    block = tf.keras.Sequential()
    for _ in range(n):
        block.add(InceptionBlockA())
    return block


def build_inception_block_b(n):
    block = tf.keras.Sequential()
    for _ in range(n):
        block.add(InceptionBlockB())
    return block


def build_inception_block_c(n):
    block = tf.keras.Sequential()
    for _ in range(n):
        block.add(InceptionBlockC())
    return block


class InceptionV4(tf.keras.Model):
    def __init__(self):
        super(InceptionV4, self).__init__()
        self.stem = Stem()
        self.inception_a = build_inception_block_a(4)
        self.reduction_a = ReductionA(k=192, l=224, m=256, n=384)
        self.inception_b = build_inception_block_b(7)
        self.reduction_b = ReductionB()
        self.inception_c = build_inception_block_c(3)
        self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=(8, 8))
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        self.flat = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(units=1,
                                        activation=tf.keras.activations.softmax)

    def call(self, inputs, training=True, mask=None):
        x = self.stem(inputs, training=training)
        x = self.inception_a(x, training=training)
        # x = self.reduction_a(x, training=training)
        # x = self.inception_b(x, training=training)
        # x = self.reduction_b(x, training=training)
        # x = self.inception_c(x, training=training)

        #x = self.avgpool(x)#去除这一层，(None, 5, 5, 1536)，平均池化报错
        #x = self.dropout(x, training=training)
        # x = self.flat(x)
        # x = self.fc(x)

        return x

    def __repr__(self):
        return "InceptionV4"

def full_Net(input_shape):
    # xy = tf.multiply(x, y)
    # op = tf.add(xy, b, name='op_to_store')
    # s = tf.Variable(0.5, name='s')
    # m = tf.Variable(2.0, name='m')
    sub_input = tf.subtract(input_shape[0], 0.5)
    input_patch = tf.multiply(sub_input, 2)

    sub_hypo = tf.subtract(input_shape[1], 0.5)
    hypo_patch = tf.multiply(sub_hypo, 2)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)

        x = tf.concat([input_patch, hypo_patch], axis=0)

        x = InceptionV4().call(x)

        #x = tf.concat([rotation,translation], axis=-1)

        x1 = tf.keras.layers.Conv2D(192, 3, 2, 'same')(x)
        x2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),strides=2,padding="same")(x)
        x = tf.concat([x1, x2], axis=-1)

        x = InceptionBlockA().call(x)
        x = InceptionBlockA().call(x)

        r = tf.keras.layers.Conv2D(64, 3, 2, 'same')(x)
        #r = tf.keras.layers.Conv2D(4, 4, 2, 'valid')(r)
        r = tf.keras.layers.Conv2D(4, 6, 2, 'valid')(r)#(224,224,3)

        t = tf.keras.layers.Conv2D(64, 3, 2, 'same')(x)
        #t = tf.keras.layers.Conv2D(3, 4, 2, 'valid')(t)
        t = tf.keras.layers.Conv2D(3, 6, 2, 'valid')(t)#(224,224,3)



        # r = tf.pow(r_Reshape,2)
        # r = tf.reduce_sum(r,reduction_indices=1)
        # r_sqrt = tf.sqrt(r)
        # realdiv = tf.realdiv(r_Reshape,r_sqrt)
        #r = tf.keras.layers.Dense(4, activation=tf.keras.activations.relu)(r)
        r = tf.keras.layers.Flatten()(r)#(None,4)
        #r = tf.keras.layers.Reshape((4,1))(r)
        #r = tf.keras.layers.Flatten((1,4))(r)
        #
        #

        #t = tf.keras.layers.Dense(3, activation=tf.keras.activations.relu)(t)
        #t = tf.keras.layers.Reshape((-1, 3, 1))(t)
        t = tf.keras.layers.Flatten()(t)#(None,3)
        # h_t = tf.multiply(t,[25,25,25])
        # a = tf.add(h_t, t)
        # realdiv_t = tf.realdiv(a,[25,25,25])




        return r, t




if __name__ == '__main__':
    #model_input = Input((299, 299, 3))#正确为
    rotation = Input((224, 224, 3))
    translation = Input((224, 224, 3))
    model_input = [rotation,translation]
    #middel = InceptionV4().call(model_input)
    model = full_Net(model_input)
    Model = Model(model_input,model)
    Model.summary()
    #plot_model(model=Model, to_file='C:/Users/YuNing Ye/PycharmProjects/6D pose refinement/Network/full_model.png', show_shapes=True, show_dtype=True, show_layer_names=True,expand_nested=True, dpi=100)