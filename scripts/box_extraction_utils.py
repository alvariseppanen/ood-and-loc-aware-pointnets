import numpy as np
from scipy.spatial import Delaunay
import open3d as o3d

from scripts.box_extraction_calib import Calibration


def load_point_clouds(path):
    '''
    Args:
        path: string, containing point clouds with extension .bin

    Returns:
        points: (N, 3)
    '''
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4) 
    return points[:, :3]


def load_3d_boxes(path, category):
    '''
    Load 3d bounding boxes from label

    Args:
        path: string, containing the label of 3d bounding boxes with extension .txt
        category: category to be loaded.
                {'Car','Van','Truck','Pedestrian','Person_sitting','Cyclist','Tram'}

    Returns:
        boxes: (M, 7), [x, y, z, dx, dy, dz, heading]
    '''
    def parse_line(line):
        label = line.strip().split(' ')
        lable = [float(item) for item in label[1:]]
        cat = label[0]
        h, w, l = [label[8]], [label[9]], [label[10]]
        loc = label[11:14]
        heading = [label[14]]
        boxes = loc + l + h + w + heading
        return np.array(boxes, dtype=np.float32), cat
        
    with open(path, 'r') as f:
        lines = f.readlines()

    boxes = []
    for line in lines:
        box, cat = parse_line(line)
        if cat.lower()==category.lower():
            boxes.append(box)

    print('{} {}s are labeled in the point cloud'.format(len(boxes), category))
    #assert len(boxes)!=0
        
    return np.array(boxes)


def label_rect_to_lidar(label):
    '''
    Transform the bbox from camera system to the lidar system.

    Args:
        label: [N, 7], [x, y, z, dx, dy, dz, heading]

    Returns:
        label_lidar: [N, 7]
    '''
    loc, dims, rots = label[:, :3], label[:, 3:6], label[:, 6]
    loc_lidar = Calibration.rect_to_lidar(loc)
    l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
    loc_lidar[:, 2] += h[:, 0] / 2
    label_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
    return label_lidar


def is_within_3d_box(points, corners3d):
    """
    Check whether a point is within bbox

    Args:
        points:  (N, 3)
        corners3d: (M, 8, 3), corners of M bboxes.

    Returns:
        flag: (M, N), bool
    """
    num_objects = len(corners3d)
    flag = []
    for i in range(num_objects):
        hull = Delaunay(corners3d[i])
        flag.append(hull.find_simplex(points) >= 0)
    flag = np.stack(flag, axis=0)
    return flag


def lidar_to_shapenet(points, box):
    '''
    Transform the coordinates from kitti(lidar system) to shapenet.
    kitti: x->forward, z->up
    shapenet: x->forward, y->up

    Args:
        points: (N, 3), points within bbox
        box: (7), box parameters

    Returns:
        points_new: (N, 3), points within bbox
        box_new: (7), box parameters
    '''
    rot = np.array(-90/180 * np.pi).reshape(1)
    points_new = rotate_points_along_x(points, rot)

    box_new = box.copy()
    box_new[4], box_new[5] = box_new[5], box_new[4]
    return points_new.squeeze(), box_new


def points_to_canonical(points, box):
    '''
    Transform points within bbox to a canonical pose and normalized scale

    Args:
        points: (N, 3), points within bbox
        box: (7), box parameters

    Returns:
        points_canonical: (N, 3)
    '''
    center = box[:3].reshape(1, 3)
    rot = -box[-1].reshape(1)
    points_centered = (points - center).reshape(1, -1, 3)
    points_centered_rot = rotate_points_along_z(points_centered, rot)
    scale = (1 / np.abs(box[3:6]).max()) * 0.999999
    points_canonical = points_centered_rot * scale

    box_canonical = box.copy()
    box_canonical[:3] = 0
    box_canonical[-1] = 0
    box_canonical = box_canonical * scale
    return points_canonical.squeeze(), box_canonical


def points_to_center(points, box):
    '''
    Transform points to origin

    Args:
        points: (N, 3), points within bbox
        box: (7), box parameters

    Returns:
        points_canonical: (N, 3)
    '''
    #center = box[:3].reshape(1, 3)
    #center = np.mean(points, axis=0).reshape(1, 3)
    #max_p = np.max(points, axis=0)
    #min_p = np.min(points, axis=0)
    #center = np.mean([max_p, min_p], axis=0).reshape(1, 3)
    center = np.expand_dims(np.mean(points, axis = 0), 0)
    #center = [np.mean(max(points[:,0]), min(points[:,1])), np.mean(max(points[:,1]), min(points[:,2])), np.mean(max(points[:,2]), min(points[:,2]))].reshape(1, 3)
    #rot = -box[-1].reshape(1)
    points_centered = (points - center).reshape(1, -1, 3)
    #points_centered_rot = rotate_points_along_z(points_centered, rot)
    #scale = (1 / np.abs(box[3:6]).max()) * 0.999999
    #points_canonical = points_centered_rot * scale
    points_canonical = points_centered

    box_canonical = box.copy()
    box_canonical[:3] = box_canonical[:3] - center
    #box_canonical[-1] = 0
    #box_canonical = box_canonical * scale
    return points_canonical.squeeze(), box_canonical


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
        corners3d: (N, 8, 3)
    """
    template = np.array([
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    ]) / 2

    corners3d = boxes3d[:, None, 3:6] * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d, boxes3d[:, 6]).reshape(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]
    return corners3d


def rotate_points_along_x(points, angle):
    """
    Args:
        points: (B, N, 3)
        angle: (B), angle along x-axis

    Returns:
    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    ones = np.ones_like(angle, dtype=np.float32)
    zeros = np.zeros_like(angle, dtype=np.float32)
    rot_matrix = np.stack((
        ones,  zeros, zeros,
        zeros, cosa, sina,
        zeros, -sina, cosa
    ), axis=1).reshape(-1, 3, 3)
    points_rot = np.matmul(points, rot_matrix)
    return points_rot


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3)
        angle: (B), angle along z-axis, angle increases x ==> y

    Returns:
    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    ones = np.ones_like(angle, dtype=np.float32)
    zeros = np.zeros_like(angle, dtype=np.float32)
    rot_matrix = np.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), axis=1).reshape(-1, 3, 3)
    points_rot = np.matmul(points, rot_matrix)
    return points_rot


def write_bboxes(bboxes, save_path):
    np.save(save_path, bboxes)
    return


def write_points(points, save_path):
    np.save(save_path, points)
    return


def write_points_pts(points, save_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(save_path, pcd)
    return

def write_bboxes_box(bboxes, save_path):
    f = open(save_path, "w")
    np.savetxt(f, bboxes)
    #f.write(bboxes)
    f.close()
    return