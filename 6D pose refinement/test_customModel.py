"""
Simple script to run a forward pass employing the Refiner on a SIXD dataset sample with a trained model.

Usage:
  test_refinement.py [options]
  test_refinement.py -h | --help

Options:
    -d --dataset=<string>        Path to SIXD dataset[default: E:\\lm_base_T\\lm]
    -o --object=<string>         Object to be evaluated [default: 01]
    -n --network=<string>        Path to trained network [default: ckpt_model/new_6D_model.pb]
    -r --max_rot_pert=<float>    Max. Rotational Perturbation to be applied in Degrees [default: 0.0]
    -t --max_trans_pert=<float>  Max. Translational Perturbation to be applied in Meters [default: 0.0]
    -i --iterations=<int>        Max. number of iterations[default: 50]
    -h --help                    Show this message and exit
"""
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

#import tensorflow as tf
import yaml
import cv2
import numpy as np

from utils.sixd import load_sixd
from refiner.non_sess_network import Architecture_non
from rendering.renderer import Renderer
from refiner.refiner import Refiner, Refinable
from rendering.utils import perturb_pose, trans_rot_err

from timeit import default_timer as timer
from docopt import docopt

args = docopt(__doc__)

sixd_base = args["--dataset"]
obj = args["--object"]
network = args["--network"]
max_rot_pert = float(args["--max_rot_pert"]) / 180. * np.pi
max_trans_pert = float(args["--max_trans_pert"])
iterations = int(args["--iterations"])

print("max_rot_pert: ", max_rot_pert)

bench = load_sixd(sixd_base, nr_frames=1, seq=obj)#数据打包处理
croppings = yaml.safe_load(open('config/croppings.yaml', 'rb'))#裁剪参数


# print(bench.frames)

#saver = tf.train.Saver(max_to_keep=3)
with tf.Session() as session:
    #graph = tf.get_default_graph()
    # sess.run(init)
    # ckpt = tf.train.latest_checkpoint("ckpt_model/")
    # saver = tf.train.import_meta_graph("ckpt_model/6D_model.ckpt.meta")
    # saver.restore(session, ckpt)
    # print("读取成功")
    architecture = Architecture_non(network_file=network, sess=session)
    ren = Renderer((640, 480), bench.cam)  # 生成渲染器,数据维度的转化
    refiner = Refiner(architecture=architecture, ren=ren, session=session)  # 数据渲染优化

    for frame in bench.frames:
        col = frame.color.copy()
        # _, gt_pose, _ = frame.gt[0]
        _, gt_pose = frame.gt[0]  # corrected
        gt_t = gt_pose[:3, 3]
        gt_r = matrix2quaternion(gt_pose[:3, :3])

        print("origional_pose:", gt_r, gt_t)
        # print(frame.gt[0])

        perturbed_pose = perturb_pose(gt_pose, max_rot_pert, max_trans_pert)
        print("pertur_r:", matrix2quaternion(perturbed_pose[:3, :3]), "pertur_t:", perturbed_pose[:3, 3])
        refinable = Refinable(model=bench.models[str(int(obj))], label=0, hypo_pose=perturbed_pose,
                              metric_crop_shape=croppings['linemod']['obj_{:02d}'.format(int(obj))], input_col=col)
        print("iterations:", iterations)
        for i in range(iterations):
            # 完整的渲染过程
            refinable.input_col = col.copy()

            start = timer()
            refiner.iterative_contour_alignment(refinable=refinable, max_iterations=1)
            #refinable = refiner.refine(refinable)
            end = timer()

            # Rendering of results
            ren.clear()
            ren.draw_background(col)
            ren.draw_boundingbox(refinable.model, refinable.hypo_pose)
            ren.draw_model(refinable.model, refinable.hypo_pose, ambient=0.5, specular=0, shininess=100,
                           light_col=[1, 1, 1], light=[0, 0, -1])
            render_col, _ = ren.finish()
            render_col = render_col.copy()

            cv2.imshow("Input Image", col)

            # Draw FPS in top left corner
            fps = "FPS: " + str(int(1 / (end - start)))

            cv2.rectangle(render_col, (0, 0), (133, 40), (1., 1., 1.), -1)
            cv2.putText(render_col, fps, (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            cv2.imshow("Refined Output", render_col)
            cv2.waitKey(300)

        orig_trans_err, orig_angular_err = trans_rot_err(gt_pose, perturbed_pose)
        refined_trans_err, refined_angular_err = trans_rot_err(gt_pose, refinable.hypo_pose)

        print('Original Errors')
        print('Translation: {:.4f}m\tRotation: {:.4f}°\n'.format(orig_trans_err, orig_angular_err))

        print('Refined Errors')
        print('Translation: {:.4f}m\tRotation: {:.4f}°'.format(refined_trans_err, refined_angular_err))

















