import numpy as np
import matplotlib.gridspec as gridspec
import tensorflow as tf
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict_3dpose import create_model
import data_utils
import cameras

FLAGS = tf.app.flags.FLAGS

order = [15, 12, 25, 26, 27, 17, 18, 19, 1, 2, 3, 6, 7, 8]

def predict_3d(poses, use_gpu=False):
  enc_in = np.zeros((1, 64))
  enc_in[0] = [0 for i in range(64)]
  actions = data_utils.define_actions(FLAGS.action)


  SUBJECT_IDS = [1, 5, 6, 7, 8, 9, 11]
  rcams = cameras.load_cameras(FLAGS.cameras_path, SUBJECT_IDS)
  train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(
      actions, FLAGS.data_dir)
  train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(
      actions, FLAGS.data_dir, FLAGS.camera_frame, rcams, FLAGS.predict_14)

  device_count = {"GPU": 1} if use_gpu else {"GPU": 0}
  png_lib = []
  before_pose = None
  with tf.Session(config=tf.ConfigProto(device_count=device_count, allow_soft_placement=True)) as sess:
    batch_size = 128
    model = create_model(sess, actions, batch_size)
    _3d_predictions = []
    for n, pose in enumerate(poses):
      joints_array = np.zeros((1, 36))
      joints_array[0] = [float(item) for i in range(18) for item in pose[i]]
      twod_export_units = {}
      for __n in range(0, len(pose)):
        p = pose[__n]
        twod_export_units[__n] = {"translate": [p[0], p[1]]}
      _data = joints_array[0]
      # mapping all body parts or 3d-pose-baseline format
      for i in range(len(order)):
        for j in range(2):
          # create encoder input
          enc_in[0][order[i] * 2 + j] = _data[i * 2 + j]
      for j in range(2):
        # Hip
        enc_in[0][0 * 2 + j] = (enc_in[0][1 * 2 + j] + enc_in[0][6 * 2 + j]) / 2
        # Neck/Nose
        enc_in[0][14 * 2 + j] = (enc_in[0][15 * 2 + j] + enc_in[0][12 * 2 + j]) / 2
        # Thorax
        enc_in[0][13 * 2 + j] = 2 * enc_in[0][12 * 2 + j] - enc_in[0][14 * 2 + j]

      # set spine
      spine_x = enc_in[0][24]
      spine_y = enc_in[0][25]

      enc_in = enc_in[:, dim_to_use_2d]
      mu = data_mean_2d[dim_to_use_2d]
      stddev = data_std_2d[dim_to_use_2d]
      enc_in = np.divide((enc_in - mu), stddev)

      # ?
      dp = 1.0
      dec_out = np.zeros((1, 48))
      dec_out[0] = [0 for i in range(48)]
      _, _, poses3d = model.step(sess, enc_in, dec_out, dp, isTraining=False)
      all_poses_3d = []
      enc_in = data_utils.unNormalizeData(enc_in, data_mean_2d, data_std_2d, dim_to_ignore_2d)
      poses3d = data_utils.unNormalizeData(poses3d, data_mean_3d, data_std_3d, dim_to_ignore_3d)
      gs1 = gridspec.GridSpec(1, 1)
      gs1.update(wspace=-0.00, hspace=0.05)  # set the spacing between axes.
      all_poses_3d.append( poses3d )
      enc_in, poses3d = map( np.vstack, [enc_in, all_poses_3d] )
      subplot_idx, exidx = 1, 1
      _max = 0
      _min = 10000

      # ??????
      for i in range(poses3d.shape[0]):
       for j in range(32):
         tmp = poses3d[i][j * 3 + 2]
         poses3d[i][j * 3 + 2] = poses3d[i][j * 3 + 1]
         poses3d[i][j * 3 + 1] = tmp
         if poses3d[i][j * 3 + 2] > _max:
           _max = poses3d[i][j * 3 + 2]
         if poses3d[i][j * 3 + 2] < _min:
           _min = poses3d[i][j * 3 + 2]

      for i in range(poses3d.shape[0]):
        for j in range(32):
          poses3d[i][j * 3 + 2] = _max - poses3d[i][j * 3 + 2] + _min
          poses3d[i][j * 3] += (spine_x - 630)
          poses3d[i][j * 3 + 2] += (500 - spine_y)

      # np.min(poses3d) é o score do frame
      if False:# FLAGS.cache_on_fail ;; TODO: colocar regra pra não inserir keypoint
        if np.min(poses3d) < -1000:
          poses3d = before_pose

      p3d = poses3d
      x,y,z = [[] for _ in range(3)]
      if not poses3d is None:
        to_export = poses3d.tolist()[0]
      else:
        to_export = [0.0 for _ in range(96)]
      for o in range(0, len(to_export), 3):
        x.append(to_export[o])
        y.append(to_export[o+1])
        z.append(to_export[o+2])

      export_units = {}
      for jnt_index, (_x, _y, _z) in enumerate(zip(x,y,z)):
        export_units[jnt_index] = [_x, _y, _z]
      _3d_predictions.append(export_units)
  return _3d_predictions
