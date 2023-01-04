
import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp

from lietorch import SE3
from evo.tools import file_interface
from evo.core.trajectory import PosePath3D
from .base import RGBDDataset
from .stream import RGBDStream

cur_path = osp.dirname(osp.abspath(__file__))
test_split = osp.join(cur_path, 'tartan_test.txt')
test_split = open(test_split).read().split()


class VKITTI2(RGBDDataset):

    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, mode='training', **kwargs):
        self.mode = mode
        self.n_frames = 2
        super(VKITTI2, self).__init__(name='VKITTI2', **kwargs)

    @staticmethod 
    def is_test_scene(scene):
        # print(scene, any(x in scene for x in test_split))
        return any(x in scene for x in test_split)

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building VKITTI2 dataset")

        scene_info = {}
        scenes = glob.glob(osp.join(self.root, '*'))
        for scene in tqdm(sorted(scenes)):
            images = sorted(glob.glob(osp.join(scene, 'clone/frames/rgb/Camera_0/*.jpg')))
            depths = sorted(glob.glob(osp.join(scene, 'clone/frames/depth/Camera_0/*.png')))
            
            poses = self.read_vkitti2_poses_file(osp.join(scene, 'clone/extrinsic.txt'))
            poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]
            poses[:,:3] /= VKITTI2.DEPTH_SCALE
            intrinsics = [VKITTI2.calib_read()] * len(images)

            # graph of co-visible frames based on flow
            graph = self.build_frame_graph(poses, depths, intrinsics)

            scene = '/'.join(scene.split('/'))
            scene_info[scene] = {'images': images, 'depths': depths, 
                'poses': poses, 'intrinsics': intrinsics, 'graph': graph}

        return scene_info

    def read_vkitti2_poses_file(self, file_path):
        """
        parses pose file in Virtual KITTI 2 format (first 3 rows of SE(3) matrix per line)
        :param file_path: the trajectory file path (or file handle)
        :return: trajectory.PosePath3D
        """
        raw_mat = np.loadtxt(file_path, delimiter=' ', skiprows=1)[::2, 2:]
        error_msg = ("Virtual KITTI 2 pose files must have 16 entries per row "
                     "and no trailing delimiter at the end of the rows (space)")
        if raw_mat is None or (len(raw_mat) > 0 and len(raw_mat[0]) != 16):
            raise file_interface.FileInterfaceException(error_msg)
        try:
            mat = np.array(raw_mat).astype(float)
        except ValueError:
            raise file_interface.FileInterfaceException(error_msg)
        # yapf: disable
        poses = [np.linalg.inv(np.array([[r[0], r[1], r[2], r[3]],
                                         [r[4], r[5], r[6], r[7]],
                                         [r[8], r[9], r[10], r[11]],
                                         [r[12], r[13], r[14], r[15]]])) for r in mat]
        # yapf: enable
        if not hasattr(file_path, 'read'):  # if not file handle
            print("Loaded {} poses from: {}".format(len(poses), file_path))
        traj_ref = PosePath3D(poses_se3=poses)
        traj_ref_quat = np.hstack((traj_ref.positions_xyz, traj_ref.orientations_quat_wxyz))
        return traj_ref_quat

    @staticmethod
    def calib_read():
        return np.array([725.0087, 725.0087, 620.5, 187]) # 根据15-deg-left/intrinsic.txt

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        depth = cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE) / VKITTI2.DEPTH_SCALE
        depth[depth==np.nan] = 1.0
        depth[depth==np.inf] = 1.0
        return depth


class TartanAirStream(RGBDStream):
    def __init__(self, datapath, **kwargs):
        super(TartanAirStream, self).__init__(datapath=datapath, **kwargs)

    def _build_dataset_index(self):
        """ build list of images, poses, depths, and intrinsics """
        self.root = 'datasets/TartanAir'

        scene = osp.join(self.root, self.datapath)
        image_glob = osp.join(scene, 'image_left/*.png')
        images = sorted(glob.glob(image_glob))

        poses = np.loadtxt(osp.join(scene, 'pose_left.txt'), delimiter=' ')
        poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]

        poses = SE3(torch.as_tensor(poses))
        poses = poses[[0]].inv() * poses
        poses = poses.data.cpu().numpy()

        intrinsic = self.calib_read(self.datapath)
        intrinsics = np.tile(intrinsic[None], (len(images), 1))

        self.images = images[::int(self.frame_rate)]
        self.poses = poses[::int(self.frame_rate)]
        self.intrinsics = intrinsics[::int(self.frame_rate)]

    @staticmethod
    def calib_read(datapath):
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)


class TartanAirTestStream(RGBDStream):
    def __init__(self, datapath, **kwargs):
        super(TartanAirTestStream, self).__init__(datapath=datapath, **kwargs)

    def _build_dataset_index(self):
        """ build list of images, poses, depths, and intrinsics """
        self.root = 'datasets/mono'
        image_glob = osp.join(self.root, self.datapath, '*.png')
        images = sorted(glob.glob(image_glob))

        poses = np.loadtxt(osp.join(self.root, 'mono_gt', self.datapath + '.txt'), delimiter=' ')
        poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]

        poses = SE3(torch.as_tensor(poses))
        poses = poses[[0]].inv() * poses
        poses = poses.data.cpu().numpy()

        intrinsic = self.calib_read(self.datapath)
        intrinsics = np.tile(intrinsic[None], (len(images), 1))

        self.images = images[::int(self.frame_rate)]
        self.poses = poses[::int(self.frame_rate)]
        self.intrinsics = intrinsics[::int(self.frame_rate)]

    @staticmethod
    def calib_read(datapath):
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)