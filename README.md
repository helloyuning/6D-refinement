print information

origional_pose - run test_customModel.py get gt_pose and print after converting rotation matrix to quaternion
pertur_r, pertur_t - run test_customModel.py get perturbedPose and print after converting rotation matrix to quaternion












# 6D-refinement
#The batch_train file in main folder uploaded(recent uploaded on 2022.06.03)
#oneBatch_train.py file uploaded on 2022.7.8
############################################
Data set reading part:

This function get camera infomation:
return: camera information

def load_info(path):
    with open(path, 'rb') as f:
        info = yaml.safe_load(f)
        for eid in info.keys():
            if 'cam_K' in info[eid].keys():
                info[eid]['cam_K'] = np.array(info[eid]['cam_K']).reshape((3, 3))
            if 'cam_R_w2c' in info[eid].keys():
                info[eid]['cam_R_w2c'] = np.array(info[eid]['cam_R_w2c']).reshape((3, 3))
            if 'cam_t_w2c' in info[eid].keys():
                info[eid]['cam_t_w2c'] = np.array(info[eid]['cam_t_w2c']).reshape((3, 1))
    return info

Load groundtrulth infomation:

load_gt return gt pose
def load_gt(path):
    with open(path, 'rb') as f:
        gts = yaml.safe_load(f)
        for im_id, gts_im in gts.items():
            for gt in gts_im:
                if 'cam_R_m2c' in gt.keys():
                    gt['cam_R_m2c'] = np.array(gt['cam_R_m2c']).reshape((3, 3))
                if 'cam_t_m2c' in gt.keys():
                    gt['cam_t_m2c'] = np.array(gt['cam_t_m2c']).reshape((3, 1))
    return gts


Class Frame store all rgb image, depth image, camera infomation from each object, ground pose
when train start class Benchmark store all data in frames(from Frame), models store 3D CAD model, cam store basical camera infomation  
class Frame:
    def __init__(self):
        self.nr = None
        self.color = None
        self.depth = None
        self.cam = np.identity(3)
        self.gt = []


class Benchmark:
    def __init__(self):
        self.frames = []#存放Frame，每一帧
        self.cam = np.identity(3)
        self.models = {}
       

load_sixd function return Benchmark class stored data
def load_sixd(base_path, seq, nr_frames=0, load_mesh=True, subset_models=[]):

    bench = Benchmark()
    bench.scale_to_meters = 0.001
    if os.path.exists(os.path.join(base_path, 'camera.yml')):
        cam_info = load_yaml(os.path.join(base_path, 'camera.yml'))
        bench.cam[0, 0] = cam_info['fx']
        bench.cam[0, 2] = cam_info['cx']
        bench.cam[1, 1] = cam_info['fy']
        bench.cam[1, 2] = cam_info['cy']
        bench.scale_to_meters = 0.001 * cam_info['depth_scale']
    else:
        raise FileNotFoundError

    #载入6D物体模型
    models_path = 'models'
    if not os.path.exists(os.path.join(base_path, models_path)):
        #models_path = 'models_reconst'
        models_path = 'lm_models\\models\\'#corrected

    model_info = load_yaml(os.path.join(base_path, models_path, 'models_info.yml'))
    for key, val in model_info.items():
        bench.models[str(key)] = Model3D()
        bench.models[str(key)].diameter = val['diameter']

    if seq is None:
        return bench

    #path = base_path + '/test/{:02d}/'.format(int(seq))
    #path = base_path + '\\lm_test_all\\test\\000002\\'.format(int(seq))#corrected
    path = base_path + '\\lm_test_all\\test\\{:06d}\\'.format(int(seq))  # corrected
    #path = base_path + '\\lm_test\\{:06d}\\'.format(int(seq))  # pbr_train_test
    info = load_info(path + 'info.yml')#camera_info
    #info = load_info(path + 'scene_camera.yml')  #
    gts = load_gt(path + 'gt.yml')#此处为scence_gt
    #gts = load_gt(path + 'scene_gt.yml')  #
    # Load frames

    nr_frames = nr_frames if nr_frames > 0 else len(info)
    print("Total number of dataset",nr_frames)
    bench.nrFames = nr_frames
    k = 0
    for i in range(1, nr_frames):
        fr = Frame()
        fr.nr = i
        #nr_string = '{:05d}'.format(i) if 'tudlight' in base_path else '{:04d}'.format(i)
        nr_string = '{:05d}'.format(i) if 'tudlight' in base_path else '{:06d}'.format(i)#corrected
        #print("path",path,"nr_string",nr_string)
        #fr.color = cv2.imread(os.path.join(path, "rgb", nr_string + ".png")).astype(np.float32) / 255.0
        #nr_string = "000006"
        fr.color = cv2.imread(os.path.join(path, "rgb", nr_string + ".png")).astype(np.float32) / 255.0
        fr.depth = cv2.imread(os.path.join(path, "depth", nr_string + ".png"), -1).astype(np.float32)\
                   * bench.scale_to_meters
        if 'tless' in base_path:  # TLESS depth is in micrometers... why not nano? :D
            fr.depth *= 10
        if os.path.exists(os.path.join(path, 'mask')):
            #fr.mask = cv2.imread(os.path.join(path, 'mask', nr_string + ".png"), -1)
            fr.mask = cv2.imread(os.path.join(path, 'mask', nr_string + '_000000' + ".png"), -1)#corrected

        #gts type is dicts

        for gt in gts[str(i)]:#corrected
            if subset_models and str(gt['obj_id']) not in subset_models:
                continue

            pose = np.identity(4)
            pose[:3, :3] = gt['cam_R_m2c']
            pose[:3, 3] = np.squeeze(gt['cam_t_m2c']) * bench.scale_to_meters

            if str(int(gt['obj_id'])) == str(int(seq)):
                #fr.gt.append((str(gt['obj_id']), pose, gt['obj_bb']))
                fr.gt.append((str(gt['obj_id']), pose))#corrected


        fr.cam = info[str(i)]['cam_K']
        bench.frames.append(fr)

    if load_mesh:
        # Build a set of all used model IDs for this sequence
        all_gts = list(itertools.chain(*[f.gt for f in bench.frames]))
        #print("all_gts:",all_gts)
        for ID in set([gt[0] for gt in all_gts]):
            #print("ID",ID)
            #bench.models[str(ID)].load(os.path.join(base_path, "models/obj_{:02d}.ply".format(int(ID))),
            bench.models[str(ID)].load(os.path.join(base_path, "lm_models\\models\\obj_{:06d}.ply".format(int(ID))),#corrected
                                       scale_to_meter=bench.scale_to_meters)

    return bench





The following is the input of network:
scene_patches:rgb image from dataset
render_patchs:rendering image from rendering CAD model
poses_r:hypo_rotation
poses_t:hypo_translation

    scene_patches = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="input_patch")
    render_patches = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="hypo_patch")
    poses_r = tf.placeholder(tf.float32, [None, 4])
    poses_t = tf.placeholder(tf.float32, [None, 3])
    
This part is the output of model and loss calculation:
predicr_r, predict_t: network output
    predict_r, predict_t = man_net.full_Net([scene_patches, render_patches])
    loss = add_pose_loss(predict_r, predict_t, poses_r, poses_t)
    
Get the perturbation_pose in this section as the input to the network:
Get data from dataset

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
