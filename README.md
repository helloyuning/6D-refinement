# 6D-refinement
#The batch_train file in main folder uploaded(recent uploaded on 2022.06.03)
#oneBatch_train.py file uploaded on 2022.7.8

The following is the input of network:

    scene_patches = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="input_patch")
    render_patches = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="hypo_patch")
    poses_r = tf.placeholder(tf.float32, [None, 4])
    poses_t = tf.placeholder(tf.float32, [None, 3])
    
This part is the output of model and loss calculation:

    predict_r, predict_t = man_net.full_Net([scene_patches, render_patches])
    predict_r = tf.identity(predict_r, name="predict_r")  
    predict_t = tf.identity(predict_t, name="predict_t")
    loss = add_pose_loss(predict_r, predict_t, poses_r, poses_t)
    
Get the perturbation_pose in this section as the input to the network:

      for frame in bench.frames:
          col = frame.color.copy()
          # _, gt_pose, _ = frame.gt[0]
          #print(frame.gt[0])
          _, gt_pose = frame.gt[0]  # corrected,获取原始GT数据

      perturbed_pose = perturb_pose(gt_pose, max_rot_pert, max_trans_pert)
      refinable = Refinable(model=bench.models[str(int(obj))], label=0, hypo_pose=perturbed_pose,
                            metric_crop_shape=croppings[dataset_name]['obj_{:02d}'.format(int(obj))],
                            input_col=col)
                            
Render model to get hypo_col and hypo_depth:

refinable.refined = False
ren.clear()
ren.draw_model(refinable.model, refinable.hypo_pose, ambient=0.5, specular=0, shininess=100,
                light_col=[1, 1, 1], light=[0, 0, -1])
refinable.hypo_col, refinable.hypo_dep = ren.finish()

Padding to prevent crash when object gets to close to border:

pad = int(refinable.metric_crop_shape[0] / 2)
input_col = np.pad(refinable.input_col, ((pad, pad), (pad, pad), (0, 0)), 'wrap')
hypo_col = np.pad(refinable.hypo_col, ((pad, pad), (pad, pad), (0, 0)), 'wrap')

Crop image and resize the image to 224x224:

slice = (int(refinable.metric_crop_shape[0] / 2), int(refinable.metric_crop_shape[1] / 2))
                input_col = input_col[y: y + 2 * slice[1], x: x + 2 * slice[0]]
                hypo_col = hypo_col[y: y + 2 * slice[1], x: x + 2 * slice[0]]
                input_shape = (224, 224)

Resize to input shape of architecture:

scene_patch = cv2.resize(input_col, input_shape)#原数据pose场景训练
render_patch = cv2.resize(hypo_col, input_shape)#扰乱pose数据集

Get the input rotation and translation(from perturbed pose):

hypo_trans = refinable.hypo_pose[:3, 3]
hypo_rot = matrix2quaternion(refinable.hypo_pose[:3, :3])

Write feed dict to train network:

feed_dict = {
  poses_r: hypo_rot.reshape(1, 4),
  poses_t: hypo_trans.reshape(1, 3),
  render_patches: [render_patch],
  scene_patches: [scene_patch],
  crop: [[x_normalized, y_normalized]]}

##########################################################################################
Test part
Arbitrarily disturb the pose within this range(max_rot_pert, max_trans_pert) as test input pose:

perturbed_pose = perturb_pose(gt_pose, max_rot_pert, max_trans_pert)
refinable = Refinable(model=bench.models[str(int(obj))], label=0, hypo_pose=perturbed_pose,
    metric_crop_shape=croppings['linemod']['obj_{:02d}'.format(int(obj))], input_col=col)

Use trained model to predict pose,  output of network are refined_rotation and refined_translation separately:

feed_dict = {
      self.architecture.scene_patch: [scene_patch],
      self.architecture.render_patch: [render_path],
      self.architecture.hypo_rotation: hypo_rot.reshape(1, 4),
      self.architecture.hypo_translation: hypo_trans.reshape(1, 3),
      self.architecture.crop_shift: [[x_normalized, y_normalized]]
  }
# run network

refined_rotation, refined_translation = self.session.run([self.architecture.rotation_hy_to_gt,
                                                          self.architecture.translation_hy_to_gt],
                                                         feed_dict=feed_dict)


Refinement part, calculate the error from the previous pose each time, and generate a new pose iteratively. The current situation is that the predicted result and the input are basically unchanged, so the value of the diff is very small, so each iteration returns the result directly. Due to this condition:(angular_diff <= min_rotation_displacement and trans_diff <= min_translation_displacement):


def iterative_contour_alignment(self, refinable, max_iterations=3,
                              min_rotation_displacement=0.5,
                              min_translation_displacement=0.0025, display=False):
  assert refinable is not None

  last_pose = np.copy(refinable.hypo_pose)
  for i in range(max_iterations):

      refinable = self.refine(refinable=refinable) ---> the result of network above

      last_trans = last_pose[:3, 3]
      last_rot = Quaternion(matrix2quaternion(last_pose[:3, :3]))

      cur_trans = refinable.hypo_pose[:3, 3]
      cur_rot = Quaternion(matrix2quaternion(refinable.hypo_pose[:3, :3]))
      # print("last_pose:", last_pose)
      # print("current_trans:", cur_trans, "last_trans", last_trans)
      # print("cur_trans - last_trans",cur_trans - last_trans)
      trans_diff = np.linalg.norm(cur_trans - last_trans)
      update_q = cur_rot * last_rot.inverse
      angular_diff = np.abs((update_q).degrees)

      last_pose = np.copy(refinable.hypo_pose)
      #print("trans_diff:", round(trans_diff, 6), "angular_diff:", round(angular_diff, 6))
      refined_t = refinable.hypo_pose[:3, 3]
      refined_r = matrix2quaternion(refinable.hypo_pose[:3, :3])
      print('refined:', refined_r, refined_t)
      if display:
          concat = cv2.hconcat([refinable.input_col, refinable.hypo_col])
          cv2.imshow('test', concat)
          cv2.waitKey(500)

      if angular_diff <= min_rotation_displacement and trans_diff <= min_translation_displacement:
          #print("优化")
          refinable.iterations = i+1
          return refinable

  refinable.iterations = max_iterations


  return refinable
