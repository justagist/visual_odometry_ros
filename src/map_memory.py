import cv2
import sys
import numpy as np
class Map_Memory():

    def __init__(self):
        self.patch_2d_pts = []
        self.patch_3d_pts = []
        self.proj_mats = []
        self.image_mats = []
        self.stored_mats = []
        self.pose_mats = []
        self.cam_poses = []
        self.proj_mats_out = []
        self.rot_mats = []
        self.trans_mats = []
    def store_proj_mat(self,P,flag = None,ind = None):
        if flag == None:
            self.proj_mats.append(P)

            if self.proj_mats_out==[]:
                self.proj_mats_out = P
            else:
                self.proj_mats_out = np.vstack((self.proj_mats_out,P))
        elif flag == 'replace':
            if ind == len(self.proj_mats)-1:
                self.proj_mats[ind] = P
            else:
                print 'Error in Projection matrix storing index'
    # def update_2d_pts(self):
    def clear_memory(self):
        self.patch_2d_pts = []
        self.patch_3d_pts = []
        self.proj_mats = []
        self.image_mats = []
        self.stored_mats = []
        self.pose_mats = []
        self.cam_poses = []
        self.proj_mats_out = []
        self.rot_mats = []
        self.trans_mats = []

    def store_image_mats(self,I):
        self.image_mats.append((I))
        # if self.image_mats[len(self.image_mats)-1] == self.image_mats[len(self.image_mats)-2]:
        #     print 'yes'
    def update_patches(self,p2,p3,flag = None,flag_indices = None): #if flag is 'remove_lost', flag_indices must be a list of rows to be removed. If it is 'add_modify', flag_indices should contain the corresponding index of patch_%dd_points. If 'starting_trouble', the None values are appended
        if flag == 'remove_lost' or flag == None:
            if len(p2) == len(p3):
                self.patch_3d_pts.append(p3)
            elif len(p2)>len(p3):
                sys.exit('Patch coordinates ERROR!')
            else:
                newp3 = np.delete(p3,flag_indices,axis=0)
                self.patch_3d_pts.append(newp3)
            self.patch_2d_pts.append(p2)
        elif flag == 'add_modify':
            # print 'updating map with new features'
            self.patch_2d_pts[flag_indices] = np.vstack((self.patch_2d_pts[flag_indices],p2))
            self.patch_3d_pts[flag_indices] = np.vstack((self.patch_3d_pts[flag_indices],p3))
        elif flag == 'starting_trouble':
            self.patch_2d_pts.append(None)
            self.patch_3d_pts.append(None)
        elif flag == 'remove_from_last':
            # print self.patch_3d_pts[len(self.patch_3d_pts)-1] == p3
            # print self.patch_2d_pts[len(self.patch_2d_pts)-1] == p2
            if len(flag_indices)==0:
                self.patch_3d_pts[len(self.patch_3d_pts)-1] = p3
                self.patch_2d_pts[len(self.patch_2d_pts)-1] = p2
            elif len(p2)>len(p3):
                sys.exit('Patch coordinates ERROR!')
            else:
                newp3 = np.delete(p3,flag_indices,axis=0)
                newp2 = np.delete(p2,flag_indices,axis=0)
                self.patch_3d_pts[len(self.patch_3d_pts)-1] = newp3
                self.patch_2d_pts[len(self.patch_2d_pts)-1] = newp2

        else:
            print 'Patch update Flag ERROR!'

    def store_patches(self,ptch2,ptch3,index_check):
            if len(self.stored_mats) != index_check:
                print 'Patch storing index mismatch error'
            else:
                self.stored_mats.append((ptch2,ptch3))

    def store_cam_pose(self,Mat,flag = None,ind = None):
        
        R = Mat[:,0:3]
        Tr = np.array([[Mat[0,3]],[Mat[1,3]],[Mat[2,3]]])
        pose = np.dot(-(R.T),Tr)
        J = np.array([[0,0,0,1]])
        Mat = np.concatenate((R,Tr),axis = 1)
        Mat = np.concatenate((Mat,J),axis = 0)
        if flag == None:
            self.cam_poses.append(pose)
            self.pose_mats.append(Mat)
            self.rot_mats.append(R)
            self.trans_mats.append(Tr)
        elif flag == 'replace':
            if ind == len(self.cam_poses)-1 and ind == len(self.pose_mats)-1:
                self.cam_poses[ind] = pose
                self.pose_mats[ind] = Mat
                self.rot_mats[ind] = R
                self.trans_mats[ind] = Tr
            else:
                print 'Error in storing cam poses'
        # self.cam_poses = np.asarray(self.cam_poses).reshape(-1,3)

    def replace_cam_poses_sba(self,pose,idx):
        self.cam_poses[idx] = pose