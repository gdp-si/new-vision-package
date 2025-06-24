#!/usr/bin/env python3
"""ROS2 node for ArUco detection with pose estimation and image saving for debugging."""
import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np

class ArucoDetectionNode(Node):
    def __init__(self):
        super().__init__('aruco_detection_node')

        # Initialize CvBridge to convert between ROS and OpenCV images
        self.bridge = CvBridge()

        # Subscribe directly to the camera image topic
        self.subscription = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.listener_callback, 10)

        # Load the predefined ArUco dictionary (6x6 markers with 250 markers)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

        # Initialize ArUco detector parameters (using direct class instantiation)
        self.parameters = cv2.aruco.DetectorParameters()

        # Camera calibration parameters (replace with your actual calibration values)
        self.camera_matrix = np.array([[617.0, 0.0, 320.0],
                                       [0.0, 617.0, 240.0],
                                       [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.zeros((5, 1))  # Assuming no distortion for simplicity

        # Log that the node is active and subscribed to the image topic
        self.get_logger().info("ArucoDetectionNode is active and subscribed to /camera/color/image_raw")

    def listener_callback(self, msg):
        """Callback function to process the camera image and detect ArUco markers."""
        try:
            # Convert the ROS Image message to an OpenCV image
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Convert the image to grayscale for ArUco detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect ArUco markers in the grayscale image
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

            # If markers are detected, log their IDs and draw them on the image
            if ids is not None:
                self.get_logger().info(f"Detected ArUco markers with IDs: {ids.flatten()}")

                # Draw detected markers on the image
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                # Estimate the pose of each detected marker
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, 0.05, self.camera_matrix, self.dist_coeffs)

                for i, marker_id in enumerate(ids):
                    # Log the translation and rotation vectors for the detected markers
                    tvec = tvecs[i][0]  # Translation vector
                    rvec = rvecs[i][0]  # Rotation vector
                    self.get_logger().info(f"Marker {marker_id}: Translation {tvec}, Rotation {rvec}")

                # Save the image with detected markers to disk for debugging
                cv2.imwrite(f"/tmp/aruco_detected_{self.get_clock().now().to_msg().sec}.jpg", frame)
            else:
                # No markers detected in this frame
                self.get_logger().info("No ArUco markers detected in this frame.")

                # Save the image even if no markers were detected (for debugging)
                cv2.imwrite(f"/tmp/aruco_no_marker_{self.get_clock().now().to_msg().sec}.jpg", frame)

            # Optionally display the image (requires a graphical environment)
            cv2.imshow('Aruco Detection', frame)
            cv2.waitKey(3)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def destroy_node(self):
        """Ensure OpenCV windows are closed when the node is destroyed."""
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    """Main function to start the ArUco detection node."""
    rclpy.init(args=args)
    node = ArucoDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

