#!/usr/bin/env python
import rclpy
from rclpy.node import Node

import os
import argparse

import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray
from nav_msgs.msg import Path
from std_msgs.msg import Float32
import time
import numpy as np
import itertools
import matplotlib.pyplot as plt

from camera_model import Camera
from feature_model import FeatureModel
from feature_extraction import featureAssociation, AdaptiveThreshold2, AdaptiveThresholdPeak
from pose_estimation import DSPoseEstimator
from perception_utils import plotPoseImageInfo
from perception_ros_utils import vectorToPose, vectorToTransform, poseToVector, lightSourcesToMsg, featurePointsToMsg
from perception import Perception


class PerceptionNode(Node):
    def __init__(self, featureModel, hz, cvShow=False, hatsMode="valley"):
        super().__init__('perception_node')

        self.cameraTopic = "lolo_camera"
        self.cameraInfoSub = self.create_subscription(
                CameraInfo, "lolo_camera/camera_info", self._getCameraCallback, 1)
        self.camera = None
        self.start_rate = self.create_rate(1.0)
        while rclpy.ok() and self.camera is None:
            print("Waiting for camera info to be published")
            self.start_rate.sleep()

        self.hz = hz
        self.rate = self.create_rate(self.hz)
        self.cvShow = cvShow

        self.perception = Perception(self.camera, featureModel, hatsMode=hatsMode)

        self.imageMsg = None
        self.bridge = CvBridge()
        self.imgSubsciber = self.create_subscription(
                Image, 'lolo_camera/image_rect_color', self._imgCallback, 1)

        # Publish some images for visualization.
        self.imgProcPublisher = self.create_publisher(
                Image, 'lolo/perception/debug/image_masked', 1)
        self.imgProcDrawPublisher = self.create_publisher(
                Image, 'lolo/preception/debug/image_peaks', 1)
        self.imgPosePublisher = self.create_publisher(
                Image, 'lolo/perception/debug/image_pose_overlay', 1)

        # Publish associated light source image points as a PoseArray.
        self.associatedImagePointsPublisher = self.create_publisher(
                PoseArray, 'lolo_camera/associated_image_points', 1)

        # Publish estimated pose.
        self.posePublisher = self.create_publisher(
                PoseWithCovarianceStamped, 'lolo/perception/optical_pose', 1)
        self.camPosePublisher = self.create_publisher(
                PoseWithCovarianceStamped, 'lolo_camera/estimated_pose', 10)
        self.mahalanobisDistPub = self.create_publisher(
                Float32, 'lolo/perception/optical_pose/uncertainty', 1)

        # Publish transform of estimated pose.
        self.transformPublisher = self.create_publisher(
                tf.msg.tfMessage, "/tf", 1)

        # Publish placement of the light sources as a PoseArray (published in the docking_station frame).
        self.featurePosesPublisher = self.create_publisher(
                PoseArray, 'lolo/perception/optical_poses', 1)

        # Sed in perception.py to update the estimated pose (estDSPose) for better prediction of the ROI.
        self._cameraPoseMsg = None
        self.cameraPoseSub = self.create_subscription(
                PoseWithCovarianceStamped, "lolo/camera/pose", self._cameraPoseSub, 1)

    def _getCameraCallback(self, msg):
        """
        Use either K and D or just P
        https://answers.ros.org/question/119506/what-does-projection-matrix-provided-by-the-calibration-represent/
        https://github.com/dimatura/ros_vimdoc/blob/master/doc/ros-camera-info.txt
        """
        # Using only P (D=0), we should subscribe to the rectified image topic.
        camera = Camera(cameraMatrix=np.array(msg.P, dtype=np.float32).reshape((3,4))[:, :3],
                        distCoeffs=np.zeros((1,4), dtype=np.float32),
                        resolution=(msg.height, msg.width))
        self.camera = camera

        # We only want one message.
        self.cameraInfoSub.destroy()

    def _imgCallback(self, msg):
        self.imageMsg = msg

    def _cameraPoseSub(self, msg):
        self._cameraPoseMsg = msg

    def update(self,
               imgColor,
               estDSPose=None,
               publishPose=True,
               publishCamPose=False,
               publishImages=True):

        # TODO: move this to run()?
        cameraPoseVector = None
        if self._cameraPoseMsg:
            t, r = poseToVector(self._cameraPoseMsg)
            cameraPoseVector = np.array(list(t) + list(r))
            self._cameraPoseMsg = None


        start = time.time()

        (dsPose,
         poseAquired,
         candidates,
         processedImg,
         poseImg) = self.perception.estimatePose(imgColor,
                                                 estDSPose,
                                                 estCameraPoseVector=cameraPoseVector)

        if dsPose and dsPose.covariance is None:
            dsPose.calcCovariance()

        elapsed = time.time() - start
        virtualHZ = 1./elapsed
        hz = min(self.hz, virtualHZ)

        cv.putText(poseImg,
                   "FPS {}".format(round(hz, 1)),
                   (int(poseImg.shape[1]*4/5), 25),
                   cv.FONT_HERSHEY_SIMPLEX,
                   0.7,
                   color=(0,255,0),
                   thickness=2,
                   lineType=cv.LINE_AA)

        cv.putText(poseImg,
                   "Virtual FPS {}".format(round(virtualHZ, 1)),
                   (int(poseImg.shape[1]*4/5), 45),
                   cv.FONT_HERSHEY_SIMPLEX,
                   0.7,
                   color=(0,255,0),
                   thickness=2,
                   lineType=cv.LINE_AA)

        timeStamp = self.get_clock().now()
        # Publish pose if pose has been aquired.
        if publishPose and poseAquired and dsPose.detectionCount >= 10: # TODO: set to 10
            # Publish transform.
            dsTransform = vectorToTransform("lolo/camera_link",
                                            "service_boat/estimated/fiducials_link",
                                            dsPose.translationVector,
                                            dsPose.rotationVector,
                                            timeStamp=timeStamp)
            self.transformPublisher.publish(tf.msg.tfMessage([dsTransform]))

            # Publish placement of the light sources as a PoseArray (published in the docking_station frame).
            pArray = featurePointsToMsg("service_boat/estimated/fiducials_link", self.perception.featureModel.features, timeStamp=timeStamp)
            self.featurePosesPublisher.publish(pArray)

            # Publish estimated pose.
            self.posePublisher.publish(
                vectorToPose("lolo/camera_link",
                dsPose.translationVector,
                dsPose.rotationVector,
                dsPose.covariance,
                timeStamp=timeStamp)
                )
            # Publish mahalanobis distance.
            if not dsPose.mahaDist and estDSPose:
                dsPose.calcMahalanobisDist(estDSPose)
                self.mahalanobisDistPub.publish(Float32(dsPose.mahaDist))

            if publishCamPose:
                print("!!!Publishing cam pose with covariance!!!")
                self.camPosePublisher.publish(
                    vectorToPose("estimated/fiducials_link",
                    dsPose.camTranslationVector,
                    dsPose.camRotationVector,
                    dsPose.calcCamPoseCovariance(),
                    timeStamp=timeStamp)
                    )

        if publishImages:
            self.imgProcDrawPublisher.publish(self.bridge.cv2_to_imgmsg(processedImg))
            self.imgProcPublisher.publish(self.bridge.cv2_to_imgmsg(self.perception.featureExtractor.img))
            self.imgPosePublisher.publish(self.bridge.cv2_to_imgmsg(poseImg))

        if dsPose:
            # If the light source candidates have been associated, we pusblish the associated candidates.
            self.associatedImagePointsPublisher.publish(lightSourcesToMsg(dsPose.associatedLightSources, timeStamp=timeStamp))
        else:
            # Otherwise we publish all candidates.
            self.associatedImagePointsPublisher.publish(lightSourcesToMsg(candidates, timeStamp=timeStamp))

        return dsPose, poseAquired, candidates

    def run(self, poseFeedback=True, publishPose=True, publishCamPose=False, publishImages=True):
        # currently estimated docking station pose
        # send the pose as an argument in update
        # for the feature extraction to consider only a region of interest
        # near the estimated pose
        estDSPose = None

        while rclpy.ok():

            if self.imageMsg:
                try:
                    imgColor = self.bridge.imgmsg_to_cv2(self.imageMsg, 'bgr8')
                except CvBridgeError as e:
                    print(e)
                else:
                    if not poseFeedback:
                        estDSPose = None

                    self.imageMsg = None
                    (dsPose,
                     poseAquired,
                     candidates) = self.update(imgColor,
                                                estDSPose=estDSPose,
                                                publishPose=publishPose,
                                                publishCamPose=publishCamPose,
                                                publishImages=publishImages)

                    if not poseAquired:
                        estDSPose = None
                    else:
                        estDSPose = dsPose

            if self.cvShow:
                show = False
                if self.perception.poseImg is not None:
                    show = True

                    img = self.perception.poseImg
                    if img.shape[0] > 720:
                        cv.resize(img, (1280,720))
                    cv.imshow("pose image", img)

                if self.perception.processedImg is not None:
                    show = True

                    img = self.perception.processedImg
                    if img.shape[0] > 720:
                        cv.resize(img, (1280,720))
                    cv.imshow("processed image", img)

                if show:
                    cv.waitKey(1)

            self.rate.sleep()


if __name__ == '__main__':
    rclpy.init(args=None)  # TODO(aldoteran) pass in args correctly.

    # TODO(aldoteran): figure out how to read in these parameters!
    # TODO(aldoteran) featureModelYaml = rospy.get_param("~feature_model_yaml")
    hz = 30  # TODO(aldoteran): rospy.get_param("~hz")
    cvShow = False  # TODO(aldoteran) rospy.get_param("~cv_show")
    publishCamPose = True  # TODO(aldoteran) rospy.get_param("~publish_cam_pose")
    hatsMode = "valley"  # TODO(aldoteran) rospy.get_param("~hats_mode")
    poseFeedBack = True  # TODO(aldoteran) rospy.get_param("~pose_feedback")

    # TODO(aldoteran) yolo with the yaml... gotta figure it out.
    # featureModelYamlPath = os.path.join(
    #         rospkg.RosPack().get_path("lolo_perception"),
    #         "feature_models/{}".format(featureModelYaml))
    # featureModel = FeatureModel.fromYaml(featureModelYamlPath)
    featureModel = FeatureModel(
            "asko_240611_station",
            np.array([[0., 0., -0.20], # Center.
                      [-0.21, -0.22, 0.], # Top left.
                      [0.21, -0.22,  0.], # Top right.
                      [0.21, 0.16, 0.], # Botttom right.
                      [-0.21, 0.16, 0.]]), # Bottom left.
            placementUncertainty=0.01,
            detectionTolerance=0.1)

    perception = PerceptionNode(featureModel, hz, cvShow=cvShow, hatsMode=hatsMode)

    # TODO(aldoteran) this should be migrated to `rclpy.spin(perception)`.
    perception.run(poseFeedback=poseFeedBack, publishPose=True, publishCamPose=publishCamPose, publishImages=True)
