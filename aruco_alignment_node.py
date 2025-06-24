#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from roast_interfaces.srv import DetectAruco  # Import the service type

class AlignmentNode(Node):
    def __init__(self):
        super().__init__('alignment_node')

        # Create a client for the 'detect_aruco' service
        self.client = self.create_client(DetectAruco, 'detect_aruco')

        # Wait until the service is available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for detect_aruco service...')

        self.get_logger().info('detect_aruco service is available!')

        # Send a request to the service
        self.send_aruco_detection_request()

    def send_aruco_detection_request(self):
        """Send a request to the detect_aruco service and handle the response."""
        request = DetectAruco.Request()

        # Set parameters for the service request
        request.timeout = 5  # Timeout for detection (5 seconds)
        request.image_topic = '/camera/color/image_raw'  # Image topic from camera

        # Call the service and wait for the response
        future = self.client.call_async(request)
        future.add_done_callback(self.handle_response)

    def handle_response(self, future):
        """Handle the response from the detect_aruco service."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('Aruco marker detected!')
                # Perform alignment logic here (e.g., move the robot, etc.)
            else:
                self.get_logger().info('Aruco marker not detected.')
        except Exception as e:
            self.get_logger().error(f'Failed to call detect_aruco service: {e}')

def main(args=None):
    """Main function for the Alignment Node."""
    rclpy.init(args=args)
    node = AlignmentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()

