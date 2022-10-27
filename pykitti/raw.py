"""Provides 'raw', which loads and parses raw KITTI data."""

import datetime as dt
import glob
import os
from collections import namedtuple
from pathlib import Path

import cv2
import imageio
import numpy as np
from tqdm import tqdm

from pykitti.depth import fill_in_fast, velo2depth
from pykitti.tracklet import parseXML, TRUNC_IN_IMAGE, TRUNC_TRUNCATED

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# define class number
class_number = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person_sitting': 4,
    'Cyclist': 5,
    'Tram': 6,
    'Misc': 7,
    'DontCare': 8
}

# define Colors
class Colors:
    def __init__(self):
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()  # create instance for 'from pykitti.raw import colors'

# Per dataformat.txt
OxtsPacket = namedtuple('OxtsPacket',
                        'lat, lon, alt, ' +
                        'roll, pitch, yaw, ' +
                        'vn, ve, vf, vl, vu, ' +
                        'ax, ay, az, af, al, au, ' +
                        'wx, wy, wz, wf, wl, wu, ' +
                        'pos_accuracy, vel_accuracy, ' +
                        'navstat, numsats, ' +
                        'posmode, velmode, orimode')

# Bundle into an easy-to-access structure
OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')


def subselect_files(files, indices):
    try:
        files = [files[i] for i in indices]
    except:
        pass
    return files


def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def pose_from_oxts_packet(packet, scale):
    """Helper method to compute a SE(3) pose matrix from an OXTS packet.
    """
    er = 6378137.  # earth radius (approx.) in meters

    # Use a Mercator projection to get the translation vector
    tx = scale * packet.lon * np.pi * er / 180.
    ty = scale * er * np.log(np.tan((90. + packet.lat) * np.pi / 360.))
    tz = packet.alt
    t = np.array([tx, ty, tz])

    # Use the Euler angles to get the rotation matrix
    Rx = rotx(packet.roll)
    Ry = roty(packet.pitch)
    Rz = rotz(packet.yaw)
    R = Rz.dot(Ry.dot(Rx))

    # Combine the translation and rotation into a homogeneous transform
    return R, t


def load_oxts_packets_and_poses(oxts_files):
    """Generator to read OXTS ground truth data.

       Poses are given in an East-North-Up coordinate system
       whose origin is the first GPS position.
    """
    # Scale for Mercator projection (from first lat value)
    scale = None
    # Origin of the global coordinate system (first GPS position)
    origin = None

    oxts = []

    for filename in oxts_files:
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.split()
                # Last five entries are flags and counts
                line[:-5] = [float(x) for x in line[:-5]]
                line[-5:] = [int(float(x)) for x in line[-5:]]

                packet = OxtsPacket(*line)

                if scale is None:
                    scale = np.cos(packet.lat * np.pi / 180.)

                R, t = pose_from_oxts_packet(packet, scale)

                if origin is None:
                    origin = t

                T_w_imu = transform_from_rot_trans(R, t - origin)

                oxts.append(OxtsData(packet, T_w_imu))

    return oxts


def load_image(file, mode):
    """Load an image from file."""
    return cv2.imread(file, mode)


def yield_images(imfiles, mode):
    """Generator to read image files."""
    for file in imfiles:
        yield load_image(file, mode)


def load_velo_scan(file):
    """Load and parse a velodyne binary file."""
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 4))


def yield_velo_scans(velo_files):
    """Generator to parse velodyne binary files into arrays."""
    for file in velo_files:
        yield load_velo_scan(file)


def load_depth(file):
    return cv2.imread(file, cv2.CV_16UC1)


def yield_depth(files):
    for file in files:
        yield load_depth(file)


def load_filled_depth(file):
    return cv2.imread(file, cv2.CV_16UC1)


def yield_filled_depth(files):
    for file in files:
        yield load_filled_depth(file)


def load_label(file):
    result = []
    with open(file) as f:
        data = f.readlines()
    for line in data:
        ceil = list(map(float, line.strip("\n").split()))
        ceil[0] = int(ceil[0])
        result.append(ceil)
    return result


def yield_label(files):
    for file in files:
        yield load_label(file)


def tracklet2frame(track, totalnum, calib):
    frame = [[] for _ in range(totalnum)]
    for iTracklet, tracklet in enumerate(track):
        # print('tracklet {0: 3d}: {1}'.format(iTracklet, tracklet))
        name = tracklet.objectType
        h, w, l = tracklet.size
        corners = np.array([[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                            [0.0, 0.0, 0.0, 0.0, h, h, h, h]])

        # loop over all data in tracklet
        for t, rotation, state, occlusion, truncation, absoluteFrameNumber in tracklet:

            # determine if object is in the image; otherwise continue
            if truncation not in (TRUNC_IN_IMAGE, TRUNC_TRUNCATED):
                continue

            # re-create 3D bounding box in velodyne coordinate system
            rz = rotation[2]  # other rotations are 0 in all xml files I checked
            assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
            R = np.array([
                [np.cos(rz), -np.sin(rz), 0.0],
                [np.sin(rz), np.cos(rz), 0.0],
                [0.0, 0.0, 1.0]])
            corners_3D = np.dot(R, corners) + np.tile(t, (8, 1)).T
            corners_3D = np.dot(calib.T_cam0_velo, np.insert(corners_3D, 3, 1, axis=0))
            orientation_3D = np.dot(R, [[0.0, 0.7 * l], [0.0, 0.0], [0.0, 0.0]]) + np.tile(t, (2, 1)).T
            orientation_3D = np.dot(calib.T_velo2cam0, np.insert(orientation_3D, 3, 1, axis=0))
            if any(corners_3D[2, :] < 0.5) or any(orientation_3D[2, :] < 0.5):
                continue
            box2D = np.dot(calib.P_rect_00[:, 0:3], corners_3D[0:3, :])
            box2D[0, :] /= box2D[2, :]
            box2D[1, :] /= box2D[2, :]
            # solve xyxy
            x1 = np.round(np.min(box2D[0, :]), 3)
            x2 = np.round(np.max(box2D[0, :]), 3)
            y1 = np.round(np.min(box2D[1, :]), 3)
            y2 = np.round(np.max(box2D[1, :]), 3)
            pos = [class_number[name], x1, y1, np.round(x2 - x1 + 1, 3), np.round(y2 - y1 + 1, 3)]
            # print(f'tracklet: {iTracklet}, frame: {absoluteFrameNumber}, cls_xywh: {pos}')
            frame[absoluteFrameNumber].append(pos)
    return frame


class raw:
    """Load and parse raw data into a usable format."""

    def __init__(self, base_path, date, drive, **kwargs):
        """Set the path and preload calibration data and timestamps."""
        self.dataset = kwargs.get('dataset', 'sync')
        self.drive = date + '_drive_' + drive + '_' + self.dataset
        self.calib_path = os.path.join(base_path, date)
        self.data_path = os.path.join(base_path, date, self.drive)
        self.frames = kwargs.get('frames', None)

        # Default image file extension is '.png'
        self.imtype = kwargs.get('imtype', 'png')

        # Find all the data files
        self._get_file_lists()

        # Preload data that isn't returned as a generator
        self._load_calib()
        self._load_timestamps()
        self._load_oxts()

        # load tracklet_labels file and covert to label and save to file
        self._save_label()

        # load the image size
        self._load_imgsz()

        # save depth image and filled-depth image
        self._save_depth()
        self._save_filled_depth()

    def __len__(self):
        """Return the number of frames loaded."""
        return len(self.timestamps)

    @property
    def cam0(self):
        """Generator to read image files for cam0 (monochrome left)."""
        return yield_images(self.cam0_files, mode=0)

    def get_cam0(self, idx):
        """Read image file for cam0 (monochrome left) at the specified index."""
        return load_image(self.cam0_files[idx], mode=0)

    @property
    def cam1(self):
        """Generator to read image files for cam1 (monochrome right)."""
        return yield_images(self.cam1_files, mode=0)

    def get_cam1(self, idx):
        """Read image file for cam1 (monochrome right) at the specified index."""
        return load_image(self.cam1_files[idx], mode=0)

    @property
    def cam2(self):
        """Generator to read image files for cam2 (RGB left)."""
        return yield_images(self.cam2_files, mode=-1)

    def get_cam2(self, idx):
        """Read image file for cam2 (RGB left) at the specified index."""
        return load_image(self.cam2_files[idx], mode=-1)

    @property
    def cam3(self):
        """Generator to read image files for cam0 (RGB right)."""
        return yield_images(self.cam3_files, mode=-1)

    def get_cam3(self, idx):
        """Read image file for cam3 (RGB right) at the specified index."""
        return load_image(self.cam3_files[idx], mode=-1)

    @property
    def gray(self):
        """Generator to read monochrome stereo pairs from file.
        """
        return zip(self.cam0, self.cam1)

    def get_gray(self, idx):
        """Read monochrome stereo pair at the specified index."""
        return self.get_cam0(idx), self.get_cam1(idx)

    @property
    def rgb(self):
        """Generator to read RGB stereo pairs from file.
        """
        return zip(self.cam2, self.cam3)

    def get_rgb(self, idx):
        """Read RGB stereo pair at the specified index."""
        return self.get_cam2(idx), self.get_cam3(idx)

    @property
    def velo(self):
        """Generator to read velodyne [x,y,z,reflectance] scan data from binary files."""
        # Return a generator yielding Velodyne scans.
        # Each scan is a Nx4 array of [x,y,z,reflectance]
        return yield_velo_scans(self.velo_files)

    def get_velo(self, idx):
        """Read velodyne [x,y,z,reflectance] scan at the specified index."""
        return load_velo_scan(self.velo_files[idx])

    @property
    def depth(self):
        """Generator depth image from binary files.
        """
        return yield_depth(self.depth_files)

    def get_depth(self, idx):
        """Read depth image at the specified index."""
        return load_depth(self.depth_files[idx])

    @property
    def filled_depth(self):
        """Generator depth image from binary files.
        """
        return yield_filled_depth(self.filled_depth_files)

    def get_filled_depth(self, idx):
        """Read depth image at the specified index."""
        return load_filled_depth(self.filled_depth_files[idx])

    @property
    def label(self):
        """Generator depth image from binary files.
        """
        return yield_label(self.label_files)

    def get_label(self, idx):
        """Read depth image at the specified index."""
        return load_label(self.label_files[idx])

    def _get_file_lists(self):
        """Find and list data files for each sensor."""
        self.oxts_files = sorted(glob.glob(
            os.path.join(self.data_path, 'oxts', 'data', '*.txt')))
        self.cam0_files = sorted(glob.glob(
            os.path.join(self.data_path, 'image_00',
                         'data', '*.{}'.format(self.imtype))))
        self.cam1_files = sorted(glob.glob(
            os.path.join(self.data_path, 'image_01',
                         'data', '*.{}'.format(self.imtype))))
        self.cam2_files = sorted(glob.glob(
            os.path.join(self.data_path, 'image_02',
                         'data', '*.{}'.format(self.imtype))))
        self.cam3_files = sorted(glob.glob(
            os.path.join(self.data_path, 'image_03',
                         'data', '*.{}'.format(self.imtype))))
        self.velo_files = sorted(glob.glob(
            os.path.join(self.data_path, 'velodyne_points',
                         'data', '*.bin')))

        # the depth point image
        if os.path.exists(os.path.join(self.data_path, 'depth_00', 'data')):
            self.depth_files = sorted(glob.glob(
                os.path.join(self.data_path, 'depth_00', 'data', '*.{}'.format(self.imtype))))
            self.resave_depth = False
            if len(self.depth_files) != len(self.velo_files):
                self.depth_files = self.velo_files
                self.resave_depth = True
        else:
            os.makedirs(os.path.join(self.data_path, 'depth_00', 'data'))
            self.depth_files = self.velo_files
            self.resave_depth = True

        # the filled depth point image
        if os.path.exists(os.path.join(self.data_path, 'depth_01', 'data')):
            self.filled_depth_files = sorted(glob.glob(
                os.path.join(self.data_path, 'depth_01', 'data', '*.{}'.format(self.imtype))))
            self.resave_filled_depth = False
            if len(self.filled_depth_files) != len(self.velo_files):
                self.resave_filled_depth = True
        else:
            os.makedirs(os.path.join(self.data_path, 'depth_01', 'data'))
            self.resave_filled_depth = True

        # the label
        if os.path.exists(os.path.join(self.data_path, 'label', 'data')):
            self.label_files = sorted(glob.glob(os.path.join(self.data_path, 'label', 'data', '*.txt')))
            self.resave_label = False
            if len(self.filled_depth_files) != len(self.velo_files):
                self.resave_label = True
        else:
            os.makedirs(os.path.join(self.data_path, 'label', 'data'))
            self.resave_label = True

        # set total number
        self.totalnum = len(self.cam0_files)
        # Sub select the chosen range of frames, if any
        if self.frames is not None:
            self.oxts_files = subselect_files(
                self.oxts_files, self.frames)
            self.cam0_files = subselect_files(
                self.cam0_files, self.frames)
            self.cam1_files = subselect_files(
                self.cam1_files, self.frames)
            self.cam2_files = subselect_files(
                self.cam2_files, self.frames)
            self.cam3_files = subselect_files(
                self.cam3_files, self.frames)
            self.velo_files = subselect_files(
                self.velo_files, self.frames)

    def _load_calib_rigid(self, filename):
        """Read a rigid transform calibration file as a numpy.array."""
        filepath = os.path.join(self.calib_path, filename)
        data = read_calib_file(filepath)
        return transform_from_rot_trans(data['R'], data['T'])

    def _load_calib_cam_to_cam(self, velo_to_cam_file, cam_to_cam_file):
        # We'll return the camera calibration as a dictionary
        data = {}

        # Load the rigid transformation from velodyne coordinates
        # to un-rectified cam0 coordinates
        T_velo2cam0_unrect = self._load_calib_rigid(velo_to_cam_file)
        data['T_velo2cam0_unrect'] = T_velo2cam0_unrect

        # Load and parse the cam-to-cam calibration data
        cam_to_cam_filepath = os.path.join(self.calib_path, cam_to_cam_file)
        filedata = read_calib_file(cam_to_cam_filepath)

        # Create 3x4 projection matrices
        P_rect_00 = np.reshape(filedata['P_rect_00'], (3, 4))
        P_rect_10 = np.reshape(filedata['P_rect_01'], (3, 4))
        P_rect_20 = np.reshape(filedata['P_rect_02'], (3, 4))
        P_rect_30 = np.reshape(filedata['P_rect_03'], (3, 4))

        data['P_rect_00'] = P_rect_00
        data['P_rect_10'] = P_rect_10
        data['P_rect_20'] = P_rect_20
        data['P_rect_30'] = P_rect_30

        # Create 4x4 matrices from the rectifying rotation matrices
        R_rect_00 = np.eye(4)
        R_rect_00[0:3, 0:3] = np.reshape(filedata['R_rect_00'], (3, 3))
        R_rect_10 = np.eye(4)
        R_rect_10[0:3, 0:3] = np.reshape(filedata['R_rect_01'], (3, 3))
        R_rect_20 = np.eye(4)
        R_rect_20[0:3, 0:3] = np.reshape(filedata['R_rect_02'], (3, 3))
        R_rect_30 = np.eye(4)
        R_rect_30[0:3, 0:3] = np.reshape(filedata['R_rect_03'], (3, 3))

        data['R_rect_00'] = R_rect_00
        data['R_rect_10'] = R_rect_10
        data['R_rect_20'] = R_rect_20
        data['R_rect_30'] = R_rect_30

        # Compute the rectified extrinsic from velo to camN
        data['T_velo2cam0'] = P_rect_00.dot(R_rect_00.dot(T_velo2cam0_unrect))
        data['T_velo2cam1'] = P_rect_10.dot(R_rect_00.dot(T_velo2cam0_unrect))
        data['T_velo2cam2'] = P_rect_20.dot(R_rect_00.dot(T_velo2cam0_unrect))
        data['T_velo2cam3'] = P_rect_30.dot(R_rect_00.dot(T_velo2cam0_unrect))

        # Compute the rectified extrinsics from cam0 to camN
        T0, T1, T2, T3 = np.eye(4), np.eye(4), np.eye(4), np.eye(4)
        T0[0, 3] = P_rect_00[0, 3] / P_rect_00[0, 0]
        T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
        T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
        T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]
        # Compute the velodyne to rectified camera coordinate transforms
        data['T_cam0_velo'] = T0.dot(R_rect_00.dot(T_velo2cam0_unrect))
        data['T_cam1_velo'] = T1.dot(R_rect_00.dot(T_velo2cam0_unrect))
        data['T_cam2_velo'] = T2.dot(R_rect_00.dot(T_velo2cam0_unrect))
        data['T_cam3_velo'] = T3.dot(R_rect_00.dot(T_velo2cam0_unrect))

        # Compute the stereo baselines in meters by projecting the origin of
        # each camera frame into the velodyne frame and computing the distances between them
        p_cam = np.array([0, 0, 0, 1])
        p_velo0 = np.linalg.inv(data['T_cam0_velo']).dot(p_cam)
        p_velo1 = np.linalg.inv(data['T_cam1_velo']).dot(p_cam)
        p_velo2 = np.linalg.inv(data['T_cam2_velo']).dot(p_cam)
        p_velo3 = np.linalg.inv(data['T_cam3_velo']).dot(p_cam)
        data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
        data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)  # rgb baseline

        return data

    def _load_calib(self):
        """Load and compute intrinsic and extrinsic calibration parameters."""
        # We'll build the calibration parameters as a dictionary, then
        # convert it to a namedtuple to prevent it from being modified later
        data = {}

        # Load the rigid transformation from IMU to velodyne
        data['T_imu2velo'] = self._load_calib_rigid('calib_imu_to_velo.txt')

        # Load the camera intrinsics and extrinsics
        data.update(self._load_calib_cam_to_cam(
            'calib_velo_to_cam.txt', 'calib_cam_to_cam.txt'))

        # Pre-compute the IMU to rectified camera coordinate transforms
        data['T_imu2cam0'] = data['T_velo2cam0'].dot(data['T_imu2velo'])
        data['T_imu2cam1'] = data['T_velo2cam1'].dot(data['T_imu2velo'])
        data['T_imu2cam2'] = data['T_velo2cam2'].dot(data['T_imu2velo'])
        data['T_imu2cam3'] = data['T_velo2cam3'].dot(data['T_imu2velo'])

        self.calib = namedtuple('CalibData', data.keys())(*data.values())

    def _load_timestamps(self):
        """Load timestamps from file."""
        timestamp_file = os.path.join(
            self.data_path, 'oxts', 'timestamps.txt')

        # Read and parse the timestamps
        self.timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                # NB: datetime only supports microseconds, but KITTI timestamps
                # give nanoseconds, so need to truncate last 4 characters to
                # get rid of \n (counts as 1) and extra 3 digits
                t = dt.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                self.timestamps.append(t)

        # Sub-select the chosen range of frames, if any
        if self.frames is not None:
            self.timestamps = [self.timestamps[i] for i in self.frames]

    def _load_oxts(self):
        """Load OXTS data from file."""
        self.oxts = load_oxts_packets_and_poses(self.oxts_files)

    def _save_label(self):
        """Load tracklet_labels data from xml and save."""
        tracklet = parseXML(os.path.join(self.data_path, 'tracklet_labels.xml'))
        label_frame = tracklet2frame(tracklet, self.totalnum, self.calib)
        if self.resave_label:
            print(f'save label to file:')
            pbar = enumerate(label_frame)
            pbar = tqdm(pbar, total=len(label_frame), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            for i, data in pbar:
                path = os.path.join(self.data_path, 'label', 'data', f'{"%010d" % i}.txt')
                with open(path, 'w') as f:
                    for line in data:
                        for j in line:
                            f.write(str(j))
                            f.write(' ')
                        f.write('\n')
                    f.close()
            self.label_files = sorted(glob.glob(os.path.join(self.data_path, 'label', 'data', '*.txt')))
            print(f'save done!')
        if self.frames is not None:
            self.label_files = subselect_files(self.label_files, self.frames)

    def _load_imgsz(self):
        """Load img size from gray image."""
        self.H, self.W = self.get_cam0(0).shape

    def _save_depth(self):
        if self.resave_depth:
            depth_path = os.path.join(self.data_path, 'depth_00', 'data')
            print(f'convert point cloud to depth map :')
            pbar = enumerate(self.depth_files)
            pbar = tqdm(pbar, total=len(self.depth_files), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            for _, file in pbar:
                velo = velo2depth(load_velo_scan(file), self.calib, self.H, self.W)
                imageio.imwrite(os.path.join(depth_path, f'{Path(file).stem}.{self.imtype}'),
                                velo.astype(np.uint16))
            print(f'convert done!')

        # update the depth_files
        self.depth_files = sorted(glob.glob(
            os.path.join(self.data_path, 'depth_00',
                         'data', '*.{}'.format(self.imtype))))
        if self.resave_filled_depth:
            self.filled_depth_files = self.depth_files
        if self.frames is not None:
            self.depth_files = subselect_files(self.depth_files, self.frames)

    def _save_filled_depth(self):
        if self.resave_filled_depth:
            filled_depth_path = os.path.join(self.data_path, 'depth_01', 'data')
            print(f'filling depth map :')
            pbar = enumerate(self.filled_depth_files)
            pbar = tqdm(pbar, total=len(self.filled_depth_files), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            for _, file in pbar:
                depth_filed = fill_in_fast(np.float32(load_depth(file) / 256.0), extrapolate=False,
                                           blur_type='bilateral')
                imageio.imwrite(os.path.join(filled_depth_path, f'{Path(file).stem}.{self.imtype}'),
                                depth_filed.astype(np.uint16))
            print(f'filling done!')

        # update the depth_files
        self.filled_depth_files = sorted(glob.glob(
            os.path.join(self.data_path, 'depth_01',
                         'data', '*.{}'.format(self.imtype))))
        if self.frames is not None:
            self.filled_depth_files = subselect_files(self.filled_depth_files, self.frames)

# draw 2Dbox
def draw2Dbox(im, label):
    for line in label:
        color = colors(line[0], True)
        p1 = [int(line[1]), int(line[2])]
        p2 = [int(line[3]+line[1]-1), int(line[4]+line[2]-1)]
        cv2.rectangle(im, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)
        # add label
        class_name = list(class_number.keys())[line[0]]
        w, h = cv2.getTextSize(class_name, 0, fontScale=1/3, thickness=1)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im,class_name, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, 1 / 3, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    return im
