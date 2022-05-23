import yaml, cv2, glob
import numpy as np

# from ..rendering.model import Model3D
# from ..rendering.utils import perturb_pose
from rendering.model import Model3D
from rendering.utils import perturb_pose
import os
import itertools

def load_yaml(path):
    with open(path, 'rb') as f:
        #content = yaml.load(f)
        content = yaml.safe_load(f)#corrected
        return content
class Benchmark(object):
    def __init__(self):
        self.model = {}
        self.frames = []
        self.cam = np.identity(3)
        self.ran = None#数据的总个数


class Frame(object):
    def __init__(self, nr=-1, colorfile=None, depthfile=None, color=None, depth=None):
        self.nr = nr
        self.colorfile = colorfile
        self.detphfile = depthfile
        self.color = color
        self.depth = depth
        self.gt = []


def loadSIXDBench(dataset_path, seq, seq_to_name, nr_frames=-1, metric_crop_shape=None, rot_variation=0.174,
                  trans_variation=[0.2, 0.2, 0.2], load_mesh = True):
    assert seq in seq_to_name.keys()  # Make sure that there is no typo

    #model_path = dataset_path + "/models/obj_%02d" % seq_to_name[seq]
    model_path = dataset_path + "\\lm_models\\models\\obj_%06d" % seq_to_name[seq]#corrected

    bench = Benchmark()
    # bench.model = Model3D()
    # bench.model.load(model_path + ".ply", demean=False, scale_to_meter=0.001)
    # bench.metric_cropshape = metric_crop_shape

    models_path = 'models'
    if not os.path.exists(os.path.join(dataset_path, models_path)):
        # models_path = 'models_reconst'
        models_path = 'lm_models\\models\\'  # corrected

    model_info = load_yaml(os.path.join(dataset_path, models_path, 'models_info.yml'))
    #print(model_info)
    for key, val in model_info.items():
        bench.model[str(key)] = Model3D()
        bench.model[str(key)].diameter = val['diameter']

    if os.path.exists(os.path.join(dataset_path, 'camera.yml')):
        cam_info = load_yaml(os.path.join(dataset_path, 'camera.yml'))
        bench.cam[0, 0] = cam_info['fx']
        bench.cam[0, 2] = cam_info['cx']
        bench.cam[1, 1] = cam_info['fy']
        bench.cam[1, 2] = cam_info['cy']
        bench.scale_to_meters = 0.001 * cam_info['depth_scale']
    else:
        raise FileNotFoundError

    # Find min/max frame numbers
    # color_path = dataset_path + "/test/%02d/rgb/" % seq_to_name[seq]
    # depth_path = dataset_path + "/test/%02d/depth/" % seq_to_name[seq]
    color_path = dataset_path + "\\lm_test_all\\test\\%06d\\rgb\\" % seq_to_name[seq]#corrected
    depth_path = dataset_path + "\\lm_test_all\\test\\%06d\\depth\\" % seq_to_name[seq]
    #print(color_path)
    color_files = glob.glob(color_path + '/*')

    # Load frames
    max_frames = len(color_files) - 1

    if nr_frames == -1:
        ran = range(max_frames - 1)
    else:
        ran = np.random.randint(0, max_frames, nr_frames)

    poses = yaml.safe_load(open(dataset_path + "\\lm_test_all\\test\\%06d\\gt.yml" % seq_to_name[seq]))
    print("The number of train data:",max_frames)
    for i in ran:
        fr = Frame()
        fr.nr = i
        # fr.colorfile = color_path + "%04d.png" % i
        # fr.depthfile = depth_path + "%04d.png" % i
        fr.colorfile = color_path + "%06d.png" % i
        fr.depthfile = depth_path + "%06d.png" % i
        fr.color = cv2.imread(fr.colorfile).astype(np.float32) / 255.0
        fr.depth = cv2.imread(fr.depthfile, -1)
        fr.depth = 0.001 * fr.depth.astype(np.float32)

        pose = np.identity(4)
        samples = []

        for p in poses[str(i)]:
            if p["obj_id"] == seq_to_name[seq]:
                pose[:3, :3] = np.array(p["cam_R_m2c"]).reshape(3, 3)
                pose[:3, 3] = np.array(p["cam_t_m2c"]) / 1000.
                break

        perturbed_pose = [perturb_pose(pose, rot_variation=rot_variation, trans_variation=trans_variation)]
        samples.append((pose, perturbed_pose))

        fr.gt.append((seq, samples))
        bench.frames.append(fr)
        bench.ran = max_frames#数据个数
    if load_mesh:
        # Build a set of all used model IDs for this sequence
        all_gts = list(itertools.chain(*[f.gt for f in bench.frames]))
        #print("all_gts",all_gts)
        for ID in set([gt[0] for gt in all_gts]):
            #print("ID",ID)
            ID = 1
            #bench.models[str(ID)].load(os.path.join(base_path, "models/obj_{:02d}.ply".format(int(ID))),
            bench.model[str(ID)].load(os.path.join(dataset_path, "lm_models\\models\\obj_{:06d}.ply".format(int(ID))),#corrected
                                       scale_to_meter=bench.scale_to_meters)
    #print("处理完了")
    return bench
