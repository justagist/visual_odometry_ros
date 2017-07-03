import numpy as np
import cv2
import sys
import csv
from map_memory import Map_Memory
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sba
from pyrr import Quaternion, Matrix33
import pyrr
import math
import yaml
from visual_odometry_stam.msg import TransmatMsg
import roslib
import rospy
# from std_msgs.msg import String
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge, CvBridgeError
# from cv_cam import image_from_ardrone

class STAM():

    # baseline_threshold = (175.0,50.0,35.0)

    def __init__(self,scene,baseline_threshold = (100.0,80.0,35.0)):
        self.scene_no = scene
        self.camera_matrix, self.distortion_coeff = self.extract_intrinsics()
        self.scene_prefix = "/home/saif/Desktop/ismar/S0%d_INPUT/S0%dL03_VGA/S0%dL03_VGA_"%(self.scene_no,self.scene_no,self.scene_no)
        self.patch_prefix = "/home/saif/Desktop/ismar/S0%d_INPUT/S0%dL03_patch/S0%dL03_VGA_"%(self.scene_no,self.scene_no,self.scene_no)
        self.base_thresh = baseline_threshold[self.scene_no-1]
        self.patch_3d_file = "/home/saif/Desktop/ismar/S0%d_INPUT/S0%d_3Ddata.csv"%(self.scene_no,self.scene_no)
        self.detect_alg = cv2.xfeatures2d.SIFT_create() #cv.ORB()
        self.matcher = cv2.BFMatcher()#cv2.BFMatcher()
        self.tot_repr_error = 0
        ## ------------- alternate matcher - FLANN ------
        # self.orb_flann_index_params = dict(algorithm = FLANN_INDEX_LSH,table_number = 6{#12},key_size = 12{#20}, multi_probe_level = 1) {#2}
        # FLANN_INDEX_KDTREE = 0
        # self.sift_flann_index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        # self.flann_search_params = dict(checks=50)   # or pass empty dictionary
        # self.matcher = cv2.FlannBasedMatcher(self.sift_flann_index_params,self.flann_search_params)

    def match_features(self,img_gray, path_prefix,visualize_patch_flag=False):
        ## cv2.namedWindow("Initial Patches# uses the matchTemplate function to find the most likely match of the feature in the main image. The minMaxLoc function is used to find the most likely location of the feature in the image. 
        ## Returns an array of the most likey location for each of the features given in the first image.
        ## if visualize_patch_flag is set True, the patches will be marked and displayed in the first image. (Used for testing the function when there was only one image, and not a series of sequential images.)
        patch_no = 0
        patch_2d = np.array([],np.float64)
        present = True
        while present == True:
            patch = cv2.imread(path_prefix+"patch_%04d.png"%patch_no)
            if patch is None:
                if patch_no == 0:
                    sys.exit('ERROR! NO PATCHES FOUND!')
                else:
                    print "Total of %d patches obtained"%(patch_no)
                    present = False
            else:
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                w, h = patch.shape[::-1]
                # print w,h
                result = cv2.matchTemplate(img_gray,patch,cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                # print max_val,min_val
                patch_centre = ((max_loc[0]+max_loc[0]+w)/2,(max_loc[1]+max_loc[1]+h)/2)
                
                if patch_no == 0:
                    patch_2d = np.array([patch_centre],np.float64)
                else:
                    patch_2d = np.vstack((patch_2d,[patch_centre]))

                if visualize_patch_flag == True:
                    rect_start = max_loc
                    rect_end = (rect_start[0] + w, rect_start[1] + h)
                    cv2.rectangle(img_gray,rect_start, rect_end, 255, 2)
                    cv2.circle(img_gray, patch_centre, 2, 255, 3)
            
            patch_no += 1
        
        if visualize_patch_flag == True:
            cv2.imshow("Initial Patches",img_gray)
            cv2.waitKey(50) 
        
        return patch_2d
        
    def extract_intrinsics(self):
        stream = file("/home/saif/Desktop/ismar/ismar/intrinsics.yaml")
        calib_data = yaml.load(stream)
        cam_info_width = calib_data['image_width']
        cam_info_height = calib_data['image_height']
        cam_info_K = calib_data['camera_matrix']['data']
        cam_mat = np.asarray(cam_info_K).reshape(3,3)
        cam_info_D = calib_data['distortion_coefficients']['data']
        dist_mat = np.asarray(cam_info_D)
        cam_info_R = calib_data['rectification_matrix']['data']
        cam_info_P = calib_data['projection_matrix']['data']
        cam_info_distortion_model = calib_data['distortion_model']
        return cam_mat,dist_mat
            

    def test_results(self,pts_3d,pts_2d,P_mat):
        tot_error = 0
        for i in range(len(pts_2d)):

            pts_world_hom = np.array([[pts_3d[i][0]],[pts_3d[i][1]],[pts_3d[i][2]],[1]])
            pts_2d_projected_hom = np.dot(P_mat,pts_world_hom)
            pts_2d_projected = (pts_2d_projected_hom/pts_2d_projected_hom[2])[:2,0]
            # if len(pts_2d)==15:
            #     print np.linalg.norm(abs(pts_2d[i]-pts_2d_projected))
            #     print pts_2d[i],pts_2d_projected
            # print i,pts_2d[i], pts_2d_projected
            # tot_error+=np.sqrt(np.sum(pts_2d[i]-pts_2d_projected)**2)
            tot_error+=np.linalg.norm(abs(pts_2d[i]-pts_2d_projected))
            # print np.sqrt(np.sum(pts_2d[i]-pts_2d_projected)**2)

        self.tot_repr_error+=tot_error/len(pts_2d)
        print 'Mean Reprojection Error: %f'%(tot_error/len(pts_2d)),'\tNo. of Tracked Features:',len(pts_2d)
        self.mean_repr_error = tot_error/len(pts_2d)

    def test_projection_and_update(self):
        pt2d = self.mapmem.patch_2d_pts[len(self.mapmem.patch_2d_pts)-1]
        pt3d = self.mapmem.patch_3d_pts[len(self.mapmem.patch_3d_pts)-1]
        proj_mat = self.mapmem.proj_mats[len(self.mapmem.proj_mats)-1]
        del_idx = []
        for i in range(len(pt2d)):
            hom_wrld = np.array([[pt3d[i][0]],[pt3d[i][1]],[pt3d[i][2]],[1]])
            hom_prj = np.dot(proj_mat,hom_wrld)
            prj = (hom_prj/hom_prj[2])[:2,0]
            err = np.linalg.norm(abs(pt2d[i]-prj))
            if err > 25: #100 #25 for scene 1!,35 for scene 2
                del_idx.append(i)
        # print len(self.mapmem.patch_2d_pts)
        self.mapmem.update_patches(pt2d,pt3d,'remove_from_last',del_idx)


    def get_projection_matrix(self,objectPoints, imagePoints):
        _, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints, imagePoints, self.camera_matrix,np.array([0,0,0,0,0],np.float64),reprojectionError=3)
        K = self.camera_matrix
        R = cv2.Rodrigues(rvec)[0]
        T = tvec
        # print (T)
        trans_mat = np.concatenate((R,T),axis=1)
        # print R
        # print R.T
        # print np.matmul(K,trans_mat)
        # print np.dot(-(R.T),T)
        # print T
        # print trans_mat
        # print (-np.dot(R,T))
        proj_mat = np.dot(K,trans_mat)
        # print proj_mat
        return proj_mat,trans_mat
        
    def get3Ddata(self,csv_file):
        with open(csv_file, 'rb') as f:
            reader = csv.reader(f)
            x = []
            for row in reader:
                if len(x) == 0:
                    x = np.array([float(row[0]),float(row[1]),float(row[2])],np.float64)
                else:
                    x = np.vstack((x,[float(row[0]),float(row[1]),float(row[2])]))
        return x
           
    def check_baseline_and_update(self,ind):
        if len(self.mapmem.cam_poses)>1 and len(self.mapmem.cam_poses) == ind+1:
            baseline_status = cv2.norm(self.mapmem.cam_poses[ind-1]-self.mapmem.cam_poses[ind]) >= self.base_thresh
            if baseline_status:
                # self.run_status = False
                # if self.bundlestate:
                #     self.bundle_adjust(self.bundle_size)
                #     self.bundlestate = False
                # self.bundle_size = 0
                new2d,new3d = self.find_new_patches(ind)
                self.mapmem.update_patches(new2d,new3d,'add_modify',ind)
                new_proj,new_pose = self.get_projection_matrix(self.mapmem.patch_3d_pts[ind],self.mapmem.patch_2d_pts[ind])
                self.mapmem.store_proj_mat(new_proj,'replace',ind)
                self.mapmem.store_cam_pose(new_pose,'replace',ind)
                # self.test_projection_and_update()
                
        elif len(self.mapmem.cam_poses)>1 or ind>1:
            print 'Error: len(self.mapmem.pose_mats) = %d, index = %d'%(len(self.mapmem.pose_mats),ind)

    def find_new_patches(self,idx):
        kp_curr,des_curr = self.detect_alg.detectAndCompute(self.mapmem.image_mats[idx],None)
        kp_prev,des_prev = self.detect_alg.detectAndCompute(self.mapmem.image_mats[idx-1],None)
        match_data_curr2prev = self.matcher.knnMatch(des_curr,des_prev, k=2)
        match_data_prev2curr = self.matcher.knnMatch(des_prev,des_curr, k=2)
        good_match_data_curr = self.ratio_test(match_data_curr2prev)
        good_match_data_prev = self.ratio_test(match_data_prev2curr)
        mutual_match_pts_curr,mutual_match_pts_prev = self.find_mutual_matches(good_match_data_curr,good_match_data_prev,kp_curr,kp_prev)
        new_2D_curr,new_2D_prev = self.find_Fmat_n_optimise_matches(mutual_match_pts_curr,mutual_match_pts_prev)
        und_2D_curr,und_2D_prev = self.undistort_patches(new_2D_curr.reshape(-1,1,2),new_2D_prev.reshape(-1,1,2))
        new_3D_curr = self.triangulate_pts(und_2D_curr,und_2D_prev,idx)
        return und_2D_curr[:,0,:],new_3D_curr
    def ratio_test(self,matches):# ratio test as per Lowe's paper (for FLANN)
        good = []
        i = 0
        for m,n in matches:
            if m.distance < 0.5*n.distance:#0.5 seems to be good for scene 1, 0.75 for scene 2
                good.append(matches[i])
            i += 1
        return good    

    def find_mutual_matches(self,mtch1,mtch2,kp1,kp2):
        mutual_match = []
        k = 0
        for i in range(len(mtch1)):
            for j in range(len(mtch2)):
                if mtch1[i][0].queryIdx == mtch2[j][0].trainIdx and mtch2[j][0].queryIdx == mtch1[i][0].trainIdx:
                    # mutual_match.append(mtch1[i][0])
                    x2 = kp1[mtch1[i][0].queryIdx].pt[0]
                    y2 = kp1[mtch1[i][0].queryIdx].pt[1]
                    x1 = kp2[mtch1[i][0].trainIdx].pt[0]
                    y1 = kp2[mtch1[i][0].trainIdx].pt[1]
                    if k == 0:
                        pts1 = np.array([[x1,y1]],np.float64)
                        pts2 = np.array([[x2,y2]],np.float64)
                        k=1
                    else:
                        pts1 = np.vstack((pts1,[[x1,y1]]))
                        pts2 = np.vstack((pts2,[[x2,y2]]))
                        # print pts1[len(pts1)-1],pts2[len(pts2)-1]
        return pts1,pts2

    def find_Fmat_n_optimise_matches(self,pts1,pts2):
        F_mat, inliers = cv2.findFundamentalMat(np.float32(pts2),np.float32(pts1),cv2.FM_RANSAC,param1 = 1.0,param2=0.99)
        j = 0
        for ins in range(len(inliers)):
            if inliers[ins] == 1:
                if j == 0:
                    # print match[ins]
                    # print pts1[ins]
                    # print pts2[ins]
                    new2d_1 = np.array([pts1[ins]],np.float64)
                    new2d_2 = np.array([pts2[ins]],np.float64)
                    j = 1
                else:
                    new2d_1 = np.vstack((new2d_1,[pts1[ins]]))
                    new2d_2 = np.vstack((new2d_2,[pts2[ins]]))
        return new2d_1,new2d_2

    def undistort_patches(self,p1,p2):
        und1 = cv2.undistortPoints(p1, self.camera_matrix, self.distortion_coeff,P=self.camera_matrix)
        und2 = cv2.undistortPoints(p2, self.camera_matrix, self.distortion_coeff,P=self.camera_matrix)
        return und1,und2

    def triangulate_pts(self,p_curr,p_prev,indx):
        self.mapmem.proj_mats[indx]
        hom_curr_4D = cv2.triangulatePoints(self.mapmem.proj_mats[indx-1], self.mapmem.proj_mats[indx], p_prev, p_curr)
        curr_3D = hom_curr_4D/hom_curr_4D[3] #recovering 3d pots from homogeneneous
        curr_3D =  curr_3D.T
        return curr_3D[:,:3]
        # print (hom_curr_4D[0][0],hom_curr_4D[0][1],hom_curr_4D[0][2],hom_curr_4D[0][3]),len(hom_curr_4D)



    def updatepatch_klt(self,prev_img,curr_img,prev_2d):
        patch, patch_status, err = cv2.calcOpticalFlowPyrLK(prev_img,curr_img,np.float32(prev_2d),np.float32(prev_2d).copy())
        j = 0
        lost_point_indices = []
        for i in range(len(patch)):
            if patch_status[i] == 1 and err[i] < 10: #best result: 10 for scenes 1 & 2
                if j == 0:
                    new_patch = np.array([patch[i]],np.float64)
                    j = 1
                else:
                    new_patch = np.vstack((new_patch,[patch[i]]))
            else:
                lost_point_indices.append(i)
                # if len(prev_2d)<20:
                #     print i
        
            # print prev_2d,'prev'
            # print new_patch,'new'
            # print patch_status
        return new_patch,lost_point_indices

    def process(self,index):
        imgs = cv2.imread(self.scene_prefix+"%04d.png"%index)
        gray_img = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
        self.mapmem.store_image_mats(gray_img)
        if index == 0:
            patch_3D = self.get3Ddata(self.patch_3d_file)
            patch_2D = self.match_features(gray_img,self.patch_prefix)
            self.mapmem.update_patches(patch_2D,patch_3D)
        else:
            patch_2D,lost_pt_index = self.updatepatch_klt(self.mapmem.image_mats[index-1],gray_img,self.mapmem.patch_2d_pts[index-1])
            self.mapmem.update_patches(patch_2D,self.mapmem.patch_3d_pts[index-1],'remove_lost',lost_pt_index)
            patch_3D = self.mapmem.patch_3d_pts[index]
        projection_matrix,cam_pose = self.get_projection_matrix(patch_3D, patch_2D)
        self.mapmem.store_proj_mat(projection_matrix)
        self.mapmem.store_cam_pose(cam_pose)
        return imgs
        # print patch_2D
        # return imgs,patch_3D,patch_2D,projection_matrix,cam_pose

    def visualize_cam_pose(self,pose_mat):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        poses = np.asarray(self.mapmem.cam_poses).reshape(-1,3)
        z = poses[:,1]
        y = poses[:,0]
        x = poses[:,2]
        # print 'z',x
        # print 'x',y
        # print 'y',z
        # print poses
        # r = z**2 + 1
        # x = float(pose[0])
        # y = float(pose[1])
        ax.scatter(x, y, z, c='r', marker='o')
        ax.scatter(x[0],y[0],z[0],c='g',marker='x',s=500)
        # this = ax.plot(x, y, z, label='parametric curve')
        ax.legend()
        ax.set_xlabel('z')
        ax.set_ylabel('x')
        ax.set_zlabel('y')
        ax.set_xlim(-1000, 1000)
        ax.set_ylim(-200, 2000)
        ax.set_zlim(-1000, 0)
        ax.invert_zaxis()
        ax.invert_xaxis()
        # ax.invert_yaxis()

    def bundle_adjust(self,n):
        i = len(self.mapmem.patch_3d_pts)-n
        count = 0
        cameras = []
        list3d = []
        for feats in range(len(self.mapmem.patch_3d_pts[i])):
            featpt = self.mapmem.patch_3d_pts[i][feats]
            list3d.append([featpt[0],featpt[1],featpt[2]])
        fx = self.camera_matrix[0,0]
        fy = self.camera_matrix[1,1]
        cx = self.camera_matrix[0,2]
        cy = self.camera_matrix[1,2]
        ar = fy/fx
        s = 0
        ds1 = self.distortion_coeff[0]
        ds2 = self.distortion_coeff[1]
        ds3 = self.distortion_coeff[2]
        ds4 = self.distortion_coeff[3]
        ds5 = self.distortion_coeff[4]
        x = np.zeros((len(list3d),n,2),dtype=np.double)
        vmask = np.zeros((len(list3d),n),dtype=np.byte)
        for j in range(i,len(self.mapmem.patch_3d_pts)):
            # print 'mat',self.mapmem.rot_mats[j]
            quat = Quaternion(self.mapmem.rot_mats[j])
            # a = Matrix33(quat)
            # print type(a)
            # print a
            # print a[0]
            # print a[2][1]            
            # print 'quat',quat
            trans = self.mapmem.trans_mats[j]
            tx = trans[0]
            ty = trans[1]
            tz = trans[2]
            # trans = self.mapmem.cam_poses[j]
            # tx = trans[0]
            # ty = trans[1]
            # tz = trans[2]
            q1 = quat[0]
            q2 = quat[1]
            q3 = quat[2]
            # fx cx cy AR s r2 r4 t1 t2 r6 qi qj qk tx ty tz
            cameras.append((fx,cx,cy,ar,s,ds1,ds2,ds3,ds4,ds5,q1,q2,q3,tx,ty,tz))
            for val in range(len(self.mapmem.patch_3d_pts[j])):
                obj = self.mapmem.patch_3d_pts[j][val]
                imgpt = self.mapmem.patch_2d_pts[j][val]
                # ind = np.where(np.all(vals==list3d,axis=1))[0][0]
                ind = list3d.index([obj[0],obj[1],obj[2]])
                x[ind][count][0] = imgpt[0]
                x[ind][count][1] = imgpt[1]
                vmask[ind][count] = 1
            count+=1
        # print sba.Cameras(cameras)
        # print len(list3d)
        # print x
        # print cameras
        if count !=n:
            print 'count error in bundle adjustment!!'
        # print vmask.shape
        sba_points = sba.Points(list3d,x,vmask)
        sba_cameras = sba.Cameras(cameras)
        # print sba_cameras.camarray
        # print 'pts',sba_points.X
        # print 'cams',sba_cameras
        # print sba.Options.optsToC
        # options = sba.Options
        # print options.optsToC
       

        newcams, newpts, info = sba.SparseBundleAdjust(sba_cameras,sba_points)
        for count,real_idx in enumerate(range(i,len(self.mapmem.patch_3d_pts))):
            new_cam_tr = np.array([[newcams.camarray[count,13]],[newcams.camarray[count,14]],[newcams.camarray[count,15]]])
            quatr = Quaternion()
            # print 'newcam_',new_cam_tr
            quatr.x = newcams.camarray[count,10]
            quatr.y = newcams.camarray[count,11]
            quatr.z = newcams.camarray[count,12]
            quatr.w = math.sqrt(1-float(quatr.x)**2-float(quatr.y)**2-float(quatr.z)**2)
            rot = Matrix33(quatr)
            rot_mat = np.array([[rot[0][0],rot[0][1],rot[0][2]],[rot[1][0],rot[1][1],rot[1][2]],[rot[2][0],rot[2][1],rot[2][2]]])
            pose = np.dot(-(self.mapmem.rot_mats[real_idx].T),new_cam_tr)
            self.mapmem.replace_cam_poses_sba(pose[:,0],real_idx)
            print pose
            # print 'qss',quatr
            # print newcams.camarray[count]

            # print math.sqrt(1-0^2-0^2)
            # new_cam_quat = 
            # print quat.normalised
            # new_cam_rot = Matrix33(Quaternion())
            # print newcams.camarray[count]
            # print Matrix33(quat)
            # print new_cam_tr
        # print newcams.camarray
        # print sba.Options.optsToC
        # for camera_array,new_pts in zip(newcams.camarray,newpts.X:
        # for a in range(n):
        #     new2dpts = newpts.X[:,a,:]

        # print newpts.X
        # print info
        print 'done'

        
    def start(self):
        self.mapmem = Map_Memory()
        i = 0
        self.run_status = True
        self.bundle_size = 1
        self.bundlestate = True
        transmatPublisher = rospy.Publisher('/trans_mat',TransmatMsg,queue_size=10)
        # options = sba
        # print options
        # print type(options.Cameras)
        # print options.Options.optsarray
        while not cv2.imread(self.scene_prefix+"%04d.png"%i) is None and self.run_status == True:
        # while i<50:
            curr_img = self.process(i)
            if len(self.mapmem.patch_2d_pts[i])<200:
                self.check_baseline_and_update(i)

            # if i%5 == 0:
            #### -------- Use for scenes 1 & 2 --------------
            self.test_projection_and_update()
            #### -------------------------------------------
            # self.mapmem.store_patches(pts_2d,pts_3d,i)#########?????????????
            # print len(self.mapmem.patch_2d_pts[i]),len(self.mapmem.patch_3d_pts[i])#,len(self.mapmem.proj_mats),len(self.mapmem.image_mats),len(self.mapmem.stored_mats),len(self.mapmem.pose_mats),len(self.mapmem.cam_poses)
            
            for point in self.mapmem.patch_2d_pts[i]:
                cv2.circle(curr_img, (int(point[0]),int(point[1])), 1, (0,0,255), 3)

            cv2.imshow("Tracked Points",curr_img)
            cv2.moveWindow("Tracked Points", 1, 1)
            cv2.waitKey(1)
            print i,
            self.test_results(self.mapmem.patch_3d_pts[i],self.mapmem.patch_2d_pts[i],self.mapmem.proj_mats[i])
            poseMat = self.mapmem.pose_mats[i]
            poseMatmsg = TransmatMsg()
            poseMatmsg.vals = [poseMat[0,0],poseMat[0,1],poseMat[0,2],poseMat[0,3],poseMat[1,0],poseMat[1,1],poseMat[1,2],poseMat[1,3],poseMat[2,0],poseMat[2,1],poseMat[2,2],poseMat[2,3],poseMat[3,0],poseMat[3,1],poseMat[3,2],poseMat[3,3]]
            # = TransmatMsg(matArray)
            # print matArray
            transmatPublisher.publish(poseMatmsg)
            # if len(self.mapmem.patch_2d_pts[i])<20:
            #     # print self.mapmem.patch_2d_pts[i],len(self.mapmem.patch_2d_pts[i])
            #     print self.mapmem.patch_3d_pts[i],len(self.mapmem.patch_3d_pts[i])
            # print self.mapmem.cam_poses[i]
            # if i > 1:
            #     f = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
            #     for j in range(i):
            #         print f
            #         f = np.dot(f,self.mapmem.pose_mats[j])
            #     R = f[:,0:3]
            #     Tr = f[:,3]
            #     print np.dot(-(R.T),Tr)
            # if i == 40:
                # a = Quaternion(self.mapmem.rot_mats[i])
                # print a.x,a.y,a.z,a.w,a
                # q = Quaternion(Matrix33.identity())
                # p = Quaternion()
                # p.y = -0.210806446999
                # p.z = -0.0479394688719
                # p.w = 0.953074674275
                # p.x = math.sqrt(1-p.y**2-p.z**2-p.w**2)
                # print 'q',q
                # print type(p)
                # print a
                # print type(a)
                # p = pyrr.quaternion.create(1,2,1,2)
                # print p,type(p)
                # p = Matrix33(Quaternion(1,1,2,1))
                # print type(p), p
                # print math.sqrt(1-Quaternion(self.mapmem.rot_mats[i])[1]**2-Quaternion(self.mapmem.rot_mats[i])[2]**2-Quaternion(self.mapmem.rot_mats[i])[3]**2)
                # print 'trrr',self.mapmem.trans_mats[i][1]
                # self.run_status = False
            # print i+1,len(self.mapmem.patch_2d_pts),len(self.mapmem.patch_3d_pts),len(self.mapmem.proj_mats)
            # print self.mapmem.cam_poses[len(self.mapmem.cam_poses)-1]
            i+=1
            # self.bundle_size+=1
        cv2.destroyAllWindows()
        # self.bundle_adjust(self.bundle_size)
        # print self.mapmem.patch_3d_pts[i-1][1][1]
        
        if i != 0:
            print i,'total mean error:',self.tot_repr_error/i
        # f = open('a.txt', 'w')
        # f.write(np.asarray(self.mapmem.proj_mats))
        # np.savetxt('scene%02d_projmat.txt'%self.scene_no,self.mapmem.proj_mats_out,delimiter=' ')#,fmt='%f')
            self.visualize_cam_pose(self.mapmem.cam_poses[len(self.mapmem.cam_poses)-1]) #=====================
            plt.show() #==============
        # csd = curr_img[:300,:150]
        # cv2.imshow('this',csd)
        # cv2.waitKey(0)
        # print curr_img.shape[::-1]