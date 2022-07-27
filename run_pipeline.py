#!/usr/bin/env python3
import torch
from torch.autograd import Variable
from PointNets.custom_models import PointNetCls
#from pointnet.model import PointNetCls
from PointNets.box_model import BoxNet
from PointNets.provider import write_detection_results_lidar
from PointNets.provider import write_detection_results_cam
from PointNets.model_utils import parse_output_to_tensors_cpu
from scripts.calibration import get_calib_from_file
import numpy as np
import cv2
import time
import glob, os
from scripts import custom_functions
from scripts import plane_fit
from scripts.laserscan import LaserScan
import argparse
from depth_cluster.build import Depth_Cluster
import scipy.special as sci

NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 3

parser = argparse.ArgumentParser()
parser.add_argument('--data_root',  dest= "data_root", default='/home/alvari/test_ws/',help="data root")
parser.add_argument('--cls_model',  dest= "cls_model", default='cls_29_6.pth',help="classifier weights")
parser.add_argument('--box_model',  dest= "box_model", default='box_shift_flip.pth',help="box estimator weights")
parser.add_argument('--split',  dest= "split", default='training',help="training or validation split")
parser.add_argument('--which_cluster', dest= "which_cluster", default=1, help="2: ScanLineRun clustering; 3: superVoxel clustering; 4: euclidean; 1: depth_cluster; ")
parser.add_argument('--sensor_type', dest= "sensor_type", default='KITTI', help="sensor type: KITTI or dbot")
args = parser.parse_args()

def key_func(x):
    return os.path.split(x)[-1]

# sensor parameter and data setup
if args.sensor_type == 'KITTI':
    dbot = False
    fov_up = 3.0
    fov_down = -25.0
    vertical_resolution = 64
    horizontal_resolution = 2048
    cropped_horizontal_resolution = 512
    vertical_angle_resolution = (abs(fov_up)+abs(fov_down))/vertical_resolution/180*np.pi
    horizontal_angle_resolution = np.pi*2/horizontal_resolution
    angle_threshold = 0.4 # 0.3 (angle th: smaller th less clusters)
    search_steps = 9
    e_th = 3.5

    # data
    split = 'training'
    #split = 'testing'
    lidar_data = sorted(glob.glob(args.data_root + '/kitti_dataset/' + split + '/velodyne/*.bin'), key=key_func)
    calib_data = sorted(glob.glob(args.data_root + '/kitti_dataset/' + split + '/calib/*.txt'), key=key_func)
    result_dir = args.data_root + '/kitti_dataset/' + split + '/'
    first_scan = len(lidar_data) - 1000
    last_scan = len(lidar_data)
    
if args.sensor_type == 'dbot':
    dbot = True
    fov_up = 15.0
    fov_down = -15.0
    vertical_resolution = 16
    horizontal_resolution = 2048
    cropped_horizontal_resolution = 2048
    vertical_angle_resolution = (abs(fov_up)+abs(fov_down))/vertical_resolution/180*np.pi
    horizontal_angle_resolution = np.pi*2/horizontal_resolution
    angle_threshold = 0.15
    search_steps = 9
    e_th = 3.0

    # data
    dataset = 'car'
    #dataset = 'pedestrian'
    lidar_data = sorted(glob.glob(args.data_root + '/dbot_dataset/' + dataset + '_dataset/velodyne/*.bin'), key=key_func)
    calib_data = sorted(glob.glob(args.data_root + '/kitti_dataset/training/calib/*.txt'), key=key_func)
    result_dir = args.data_root + '/dbot_dataset/' + dataset + '_dataset/'
    first_scan = 0
    last_scan = len(lidar_data)

if (len(lidar_data) < 1):
    print("No data")

Scan = LaserScan(project=True, flip_sign=False, H=vertical_resolution, W=horizontal_resolution, fov_up=fov_up, fov_down=fov_down)

# ground segmentation setup
plane = plane_fit.Plane()

# cluster setup
if args.which_cluster == 1:
	cluster = Depth_Cluster.Depth_Cluster(angle_threshold, search_steps)

# upper and lower bounds for points per proposal
n_points = 128
min_points = 10

# classifier setup
classifier = PointNetCls(k=3)
classifier.cpu()
#classifier.load_state_dict(torch.load('weights/cls_epoch_299.pth')) # 2
classifier.load_state_dict(torch.load('weights/cls_29_6.pth')) # 1
classifier.eval()

# box estimator setup
box_estimator = BoxNet(n_classes=3, n_channel=3)
box_estimator.cpu()
#box_estimator.load_state_dict(torch.load('weights/box_flip_only_new.pth')) # 1
#box_estimator.load_state_dict(torch.load('weights/box_shift_flip.pth')) # 2
box_estimator.load_state_dict(torch.load('weights/box_energy_249_2.pth'))
box_estimator.eval()

def appendCylindrical_np(xyz):
    ptsnew = np.dstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,:,0]**2 + xyz[:,:,1]**2
    ptsnew[:,:,3] = np.sqrt(xy) #radial
    ptsnew[:,:,4] = np.arctan2(xyz[:,:,1], xyz[:,:,0]) #azimuth, horizontal
    ptsnew[:,:,5] = xyz[:,:,2] #z, height
    return ptsnew

def full_scan():
    total_times = [0,0,0,0,0]
    for i in range(first_scan, last_scan):
        total_time = 0
        #i = 895
        Scan.open_scan(lidar_data[i])

        # organize points
        xyz_list = Scan.points
        range_img_pre = Scan.proj_range
        xyz = Scan.proj_xyz #runtime 0.001 ms
        orig_xyz = np.copy(xyz)

        # add cylindrical coordinates
        xyz_cyl = appendCylindrical_np(xyz) #runtime 9 ms

        # add label channel
        xyz_cyl = np.append(xyz_cyl, np.zeros((vertical_resolution, horizontal_resolution, 1)), axis=2) #runtime 3 ms

        # crop
        xyz_cyl = xyz_cyl[0:vertical_resolution, int(horizontal_resolution/2-cropped_horizontal_resolution/2):int(horizontal_resolution/2+cropped_horizontal_resolution/2)]
        xyz = xyz[0:vertical_resolution, int(horizontal_resolution/2-cropped_horizontal_resolution/2):int(horizontal_resolution/2+cropped_horizontal_resolution/2)]
        range_img_pre = range_img_pre[0:vertical_resolution, int(horizontal_resolution/2-cropped_horizontal_resolution/2):int(horizontal_resolution/2+cropped_horizontal_resolution/2)]
        orig_xyz = orig_xyz[0:vertical_resolution, int(horizontal_resolution/2-cropped_horizontal_resolution/2):int(horizontal_resolution/2+cropped_horizontal_resolution/2)]

        # sample ground points
        start = time.time()
        ground_i = custom_functions.groundRemoval(xyz_cyl, dbot=dbot)[:,:,6] #runtime 3 ms
        stop = time.time()
        total_time += (stop-start)
        ground_mask = ground_i > 0

        # fit plane with RANSAC
        ground_points = xyz[ground_mask]
        all_points = orig_xyz.reshape(-1, 3)
        start = time.time()
        best_eq, best_inliers = plane.fit(pts=ground_points, all_pts=all_points, thresh=0.15, minPoints=100, maxIteration=100)
        stop = time.time()
        total_time += (stop-start)
        ground_points_ransac = all_points[best_inliers]
        Scan.set_points(ground_points_ransac)
        ground_i_ransac = Scan.proj_mask
        ground_i_ransac = ground_i_ransac[0:vertical_resolution, int(horizontal_resolution/2-cropped_horizontal_resolution/2):int(horizontal_resolution/2+cropped_horizontal_resolution/2)]
        range_unprojected = range_img_pre.reshape(-1)
        range_unprojected[best_inliers] = 0
        range_projected = range_unprojected.reshape(vertical_resolution, cropped_horizontal_resolution)

        # clustering
        start = time.time()
        if args.sensor_type == 'KITTI':
            instance_label = cluster.Depth_cluster(range_projected.reshape(-1))
        if args.sensor_type == 'dbot':
            instance_label = cluster.dbot_Depth_cluster(range_projected.reshape(-1))
        stop = time.time()
        total_time += (stop-start)
        #print('clustering:', stop-start)
        instance_label = np.asarray(instance_label).reshape(vertical_resolution, cropped_horizontal_resolution)

        # build NN input tensor, note: this should be done parallel with clustering, since now it's pretty slow (10-30 ms)
        nn_input_points = np.zeros((n_points, 3, 0))
        rotation_angles = np.zeros((0))
        cluster_centers = np.zeros((1, 3, 0))
        nn_input_voxel = np.zeros((1, 3, 0))
        cluster_instance_ids = set(instance_label[instance_label != 0])
        for c in cluster_instance_ids:
            cluster_i = xyz[instance_label == c].reshape(-1, 3)
            if len(cluster_i) < min_points: continue
            original_len = len(cluster_i)
            
            # remove big cluster
            if abs(np.max(cluster_i[:, 0]) - np.min(cluster_i[:, 0])) > 6: continue
            if abs(np.max(cluster_i[:, 1]) - np.min(cluster_i[:, 1])) > 6: continue
            if abs(np.max(cluster_i[:, 2]) - np.min(cluster_i[:, 2])) > 6: continue

            # sample cluster
            choice = np.random.choice(len(cluster_i), n_points, replace=True)
            cluster_i = cluster_i[choice, :]
            
            # origin to cluster center
            center = np.expand_dims(np.mean(cluster_i, axis = 0), 0)
            if center[0][2] > 1: continue
            cluster_i = cluster_i - center

            # spherical voxel position
            center_in_spherical = np.zeros((1, 3))
            center_in_spherical[:,0] = np.sqrt(center[:,0]**2 + center[:,1]**2 + center[:,2]**2) # euclidean from origin 
            center_in_spherical[:,1] = np.arctan2(center[:,1], center[:,0]) # azimuth, horizontal angle
            center_in_spherical[:,2] = np.arctan2(np.sqrt(center[:,0]**2 + center[:,1]**2), center[:,2]) # for elevation angle defined from Z-axis down

            # define voxel grid coordinate
            r_bins = np.linspace(0, 75, 10)
            a_bins = np.linspace(-(np.pi/4+0.2), np.pi/4+0.2, 10)
            e_bins = np.linspace(1*np.pi/3, 2*np.pi/3, 10)
            voxel_coord = np.zeros((1, 3))
            voxel_coord[:,0] = np.digitize(center_in_spherical[:,0], r_bins).squeeze()
            voxel_coord[:,1] = np.digitize(center_in_spherical[:,1], a_bins).squeeze()
            voxel_coord[:,2] = np.digitize(center_in_spherical[:,2], e_bins).squeeze()
            
            # rotate
            theta = -center_in_spherical[:,1].squeeze()
            '''rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]]) # counter clockwise
            cluster_i[:,[0,1]] = cluster_i[:,[0,1]].dot(rotation_matrix)
            center[:,:2] = center[:,:2].dot(rotation_matrix)
            voxel_coord[:,1] = np.digitize(0, a_bins)

            cluster_i = cluster_i - center
            rotation_matrix = np.array([[np.cos(-theta), -np.sin(-theta)],[np.sin(-theta), np.cos(-theta)]])
            center[:,:2] = center[:,:2].dot(rotation_matrix)'''

            nn_input_points = np.dstack((nn_input_points, cluster_i))
            nn_input_voxel = np.dstack((nn_input_voxel, voxel_coord))
            cluster_centers = np.dstack((cluster_centers, center))
            rotation_angles = np.hstack((rotation_angles, theta))

            # visualize
            '''if i > 20:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(cluster_i)
                #o3d.visualization.draw_geometries([pcd])'''

        nn_input_points = torch.from_numpy(nn_input_points)
        nn_input_voxel = torch.from_numpy(nn_input_voxel)
        cluster_centers = torch.from_numpy(cluster_centers)
        nn_input_points = torch.swapaxes(nn_input_points, 0, 2)
        nn_input_voxel = torch.swapaxes(nn_input_voxel, 0, 2)
        cluster_centers = torch.squeeze(torch.swapaxes(cluster_centers, 1, 2))
        nn_input_points, nn_input_voxel = Variable(nn_input_points), Variable(nn_input_voxel)
        nn_input_points, nn_input_voxel, cluster_centers = nn_input_points.cpu().float(), nn_input_voxel.cpu().float(), cluster_centers.cpu().float()

        # classifier
        start = time.time()
        pred, global_feat, avg, out = classifier(nn_input_points, nn_input_voxel)
        stop = time.time()
        total_time += (stop-start)
        #print(stop-start)
        pred_choice = pred.data.max(1)[1]
        
        # energy score
        np_out = out.cpu().detach().numpy()
        energy = -sci.logsumexp(np_out, axis=1)

        # OODs dropout
        energy_threshold = -e_th
        nn_input_points = nn_input_points[energy < energy_threshold, :, :]
        nn_input_voxel = nn_input_voxel[energy < energy_threshold, :]
        cluster_centers = cluster_centers[energy < energy_threshold, :]
        rotation_angles = rotation_angles[energy < energy_threshold]
        pred_choice = pred_choice[energy < energy_threshold]
        energy = energy[energy < energy_threshold]

        # transform target scalar to 3x one hot vector
        hot1 = torch.zeros(np.count_nonzero(energy < energy_threshold))
        hot1[pred_choice == 0] = 1
        hot2 = torch.zeros(np.count_nonzero(energy < energy_threshold))
        hot2[pred_choice == 2] = 1
        hot3 = torch.zeros(np.count_nonzero(energy < energy_threshold))
        hot3[pred_choice == 1] = 1
        one_hot = torch.vstack((hot1, hot2, hot3))
        one_hot = one_hot.transpose(1, 0)

        write_empty_label = True
        if (nn_input_points.shape[0] > 0):
            # boxnet (bug: if batch size = 0 -> boxnet crashes)
            start = time.time()
            box_pred, center_delta = box_estimator(nn_input_points, one_hot, nn_input_voxel)
            stop = time.time()
            total_time += (stop-start)

            # parse output
            center_boxnet, \
            heading_scores, heading_residual_normalized, heading_residual, \
            size_scores, size_residual_normalized, size_residual = \
                    parse_output_to_tensors_cpu(box_pred)

            # flipped box uncertainty is allowed
            offset = int(heading_scores.shape[1]/2)*torch.ones(heading_scores.shape[0]).cpu()
            offset = torch.argmax(heading_scores, dim=1)-offset
            offset[offset < 0] = int(heading_scores.shape[1])+offset[offset < 0]
            heading_scores[torch.arange(heading_scores.shape[0]), offset.long()] = -torch.inf

            # energy score
            np_heading_scores = heading_scores.cpu().detach().numpy()
            np_size_scores = size_scores.cpu().detach().numpy()
            h_energy = -sci.logsumexp(np_heading_scores, axis=1)
            s_energy = -sci.logsumexp(np_size_scores, axis=1)

            size_th = np.array([-7.8, -7.5, -6.0])
            heading_th = np.array([-2.5, -2.5, -1.5])
            #size_th = np.array([-8.0, -7.7, -6.5])
            #heading_th = np.array([-3.0, -2.3, -1.6])
            
            pass_through_mask = np.zeros((nn_input_points.shape[0]), dtype=bool)
            for kk in range(nn_input_points.shape[0]):
                if s_energy[kk] < size_th[pred_choice[kk]] and h_energy[kk] < heading_th[pred_choice[kk]]:
                    pass_through_mask[kk] = True

            nn_input_points = nn_input_points[pass_through_mask, :, :]
            nn_input_voxel = nn_input_voxel[pass_through_mask, :]
            cluster_centers = cluster_centers[pass_through_mask, :]
            center_delta = center_delta[pass_through_mask, :]
            center_boxnet = center_boxnet[pass_through_mask, :]
            rotation_angles = rotation_angles[pass_through_mask]
            pred_choice = pred_choice[pass_through_mask]
            energy = energy[pass_through_mask]
            s_energy = s_energy[pass_through_mask]
            heading_scores = heading_scores[pass_through_mask, :]
            heading_residual = heading_residual[pass_through_mask, :]
            size_scores = size_scores[pass_through_mask, :]
            size_residual = size_residual[pass_through_mask, :]

            if (nn_input_points.shape[0] > 0):
                write_empty_label = False

                # move boxes to real positions using cluster center 
                stage1_center = cluster_centers + center_delta
                box3d_center = center_boxnet + stage1_center

                # to correct format
                class_to_string = {0:'Car', 1:'Cyclist', 2:'Pedestrian'}
                type_list = [class_to_string[class_i] for class_i in pred_choice.detach().numpy()]
                id_list = i * np.ones((len(type_list))) # id of the scan
                box2d_list = np.zeros((len(type_list), 4)) # leave to zero
                center_list = box3d_center.detach().numpy()
                heading_cls_list = heading_scores.data.max(1)[1].detach().numpy()
                hcls_onehot = np.eye(NUM_HEADING_BIN)[heading_cls_list]
                heading_res_list = np.sum(heading_residual.detach().numpy() * hcls_onehot, axis=1)
                size_cls_list = size_scores.data.max(1)[1].detach().numpy()
                scls_onehot = np.eye(NUM_SIZE_CLUSTER)[size_cls_list]
                scls_onehot_repeat = scls_onehot.reshape(-1, NUM_SIZE_CLUSTER, 1).repeat(1, 2)
                size_res_list = np.sum(size_residual.detach().numpy() * scls_onehot_repeat, axis=1)

                limiter = 3.0 # 3 is good
                score_list1 = np.clip(-energy+energy_threshold, 0.01, limiter) 
                score_list = (score_list1-0.01)/limiter 

                # in lidar coordinates
                write_predictions = write_detection_results_lidar(result_dir, id_list, type_list, box2d_list, center_list, \
                                    heading_cls_list, heading_res_list, size_cls_list, size_res_list, score_list)

                # in camera coordinates
                calib_file = get_calib_from_file(calib_data[i])
                V2C = calib_file['Tr_velo2cam']
                R0 = calib_file['R0']
                write_predictions = write_detection_results_cam(result_dir, id_list, type_list, box2d_list, center_list, \
                                    heading_cls_list, heading_res_list, size_cls_list, size_res_list, score_list, V2C, R0, calib_data[i], rotation_angles)
        if write_empty_label:
            if result_dir is None: return
            results = {} # map from idx to list of strings, each string is a line (without \n)
            idx = i # idx is the number of the scan
            output_str = "DontCare -1.00 -1 -10.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00"
            #output_str = " "
            if idx not in results: results[idx] = []
            results[idx].append(output_str)

            # Write TXT files
            if not os.path.exists(result_dir): os.mkdir(result_dir)
            output_dir = os.path.join(result_dir, 'pred_lidar')
            if not os.path.exists(output_dir): os.mkdir(output_dir)
            for idx in results:
                pred_filename = os.path.join(output_dir, '%06d.txt'%(idx))
                fout = open(pred_filename, 'w')
                for line in results[idx]:
                    fout.write(line+'\n')
                fout.close() 

            # Write TXT files
            if not os.path.exists(result_dir): os.mkdir(result_dir)
            output_dir = os.path.join(result_dir, 'pred')
            if not os.path.exists(output_dir): os.mkdir(output_dir)
            for idx in results:
                pred_filename = os.path.join(output_dir, '%06d.txt'%(idx))
                fout = open(pred_filename, 'w')
                for line in results[idx]:
                    fout.write(line+'\n')
                fout.close()
        
        #print(int(i/len(lidar_data)*100), '%')
        total_times.append(total_time)
        total_times.pop(0)

        print(' ')
        print('FPS: {:.1f}'.format(1/np.mean(total_times)))
        print('amount of IDs: ', np.count_nonzero(energy < energy_threshold))
        print('index: ', i)

        # visualize
        '''normed2 = cv2.normalize(instance_label, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color2 = cv2.applyColorMap(normed2, cv2.COLORMAP_JET)
        scale_percent = 300 # percent of original size
        width = int(color2.shape[1] * scale_percent / 100)
        height = int(color2.shape[0] * scale_percent / 100)
        dim = (width, height)
        color2 = cv2.resize(color2, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("test", color2)
        cv2.waitKey(1000)'''

def main():
    full_scan()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()