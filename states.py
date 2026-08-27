import time
import rclpy
import numpy

from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist
import numpy as np
import math

class GetToGoal(Node):
    def __init__(self):
        super().__init__('get_to_goal')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.front_distance = 0.0
        self.sign = 6.0
        self.decided_sign = 6.0
        self.flag = 0.0
        self.state = "DRIVING"
        self.current_yaw = 0.0

        # Scanning state
        self.scan_phase = None        # "SCAN_RIGHT", "SCAN_LEFT", "SCAN_CENTER"
        self.scan_base_yaw = None
        self.scan_target_yaw = None
        self.sign_buffer = []         # collected signs during scan

        self.obj_sub = self.create_subscription(
            Point, '/object_position', self.position_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.timer = self.create_timer(0.1, self.control_loop)

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def start_turn(self, degrees):
        radians = math.radians(degrees)
        self.target_yaw = self.normalize_angle(self.current_yaw + radians)
        self.state = "TURNING"
        if hasattr(self, 'drive_heading'):
            del self.drive_heading
        self.get_logger().info(f"Starting turn: {degrees}° | target yaw: {math.degrees(self.target_yaw):.1f}°")

    def start_scan(self):
        """Stop, record base heading, begin scanning right 20°"""
        self.state = "SCANNING"
        self.scan_base_yaw = self.current_yaw
        self.scan_target_yaw = self.normalize_angle(self.scan_base_yaw - math.radians(5))
        self.scan_phase = "SCAN_RIGHT"
        self.sign_buffer = []
        self.get_logger().info("SCANNING: starting right scan")

    def position_callback(self, msg):
        self.front_distance = msg.x
        self.sign = msg.y
        self.flag = msg.z

    def control_loop(self):
        if self.flag == 1.0 and self.state != "TURNING":
            self.get_logger().info(
                f"STATE={self.state} | dist={self.front_distance:.2f} | sign={self.sign}"
            )
        command = Twist()

        if self.state == "TURNING":
            error = self.normalize_angle(self.target_yaw - self.current_yaw)
            if abs(error) > 0.02:
                command.angular.z = 0.1 * np.sign(error)
                command.linear.x = 0.0
            else:
                self.get_logger().info("Turn complete, driving forward")
                self.state = "DRIVING"

        # flag 1.0 front 
        #flag 0.0 none 
        # flag 2.0 right
        #flag 3.0 left
        #flag 4.0 is front + right
        # flagh 5.0 front + left
        # flag 7.0 front + right + left
        elif self.state == "DRIVING":
            if self.flag == 1.0:
                command.linear.x = 0.0
                command.angular.z = 0.0
                self.start_scan()
            elif self.flag == 2.0:
                # command.linear.x = 0.0
                # command.angular.z = 0.0
                
                wall_dist = self.front_distance  # now carries right_dist
                desired = 0.45
                error = desired - wall_dist      # negative = too close, positive = too far
                command.linear.x = 0.07
                command.angular.z = -1.5 * error
                self.get_logger().info("WALL FOLLOWING RIGHT")
                
                # right wall following
            elif self.flag == 3.0:
                # command.linear.x = 0.0
                # command.angular.z = 0.0
                wall_dist = self.front_distance  # now carries left_dist
                desired = 0.43
                error = desired - wall_dist
                command.linear.x = 0.07
                command.angular.z = 1.5 * error 
                self.get_logger().info("WALL FOLLOWING RIGHT")
                
                # left wall following
            # elif self.flag == 4.0:
            #     command.linear.x = 0.0
            #     command.angular.z = 0.0
            #     #self.start_scan()
            #     # SCAN + right wall following
            # elif self.flag == 5.0:
            #     command.linear.x = 0.0
            #     command.angular.z = 0.0
            #     self.start_scan()
            # elif self.flag == 7.0:
            #     command.linear.x = 0.0
            #     command.angular.z = 0.0
            #     self.start_scan()
            else:
                if not hasattr(self, 'drive_heading'):
                    self.drive_heading = self.current_yaw
                heading_error = self.normalize_angle(self.drive_heading - self.current_yaw)
                command.linear.x = 0.07
               
                if abs(heading_error) > 0.01:                         
                    command.angular.z = min(2.5 * heading_error, 0.3) 
                else:                                                  
                    command.angular.z = 0.0    

        elif self.state == "SCANNING":
            command.linear.x = 0.0
            error = self.normalize_angle(self.scan_target_yaw - self.current_yaw)

            # Collect sign every tick (ignore unknown 6.0)
            if self.sign != 6.0:
                self.sign_buffer.append(self.sign)

            if abs(error) > 0.02:
                command.angular.z = 0.1 * np.sign(error)

            else:
                # Current scan target reached
                if self.scan_phase == "SCAN_RIGHT":
                    # Reached right, go back to center
                    self.scan_target_yaw = self.scan_base_yaw
                    self.scan_phase = "SCAN_CENTER_1"
                    self.get_logger().info("SCANNING: returning to center")

                elif self.scan_phase == "SCAN_CENTER_1":
                    # Reached center, now go left
                    self.scan_target_yaw = self.normalize_angle(self.scan_base_yaw + math.radians(15))
                    self.scan_phase = "SCAN_LEFT"
                    self.get_logger().info("SCANNING: starting left scan")

                elif self.scan_phase == "SCAN_LEFT":
                    # Reached left, go back to center
                    self.scan_target_yaw = self.scan_base_yaw
                    self.scan_phase = "SCAN_CENTER_2"
                    self.get_logger().info("SCANNING: returning to center")

                elif self.scan_phase == "SCAN_CENTER_2":
                    # Scan complete — pick most common sign
                    if self.sign_buffer:
                        self.decided_sign = max(self.sign_buffer, key=self.sign_buffer.count)
                        self.get_logger().info(f"Scan complete. Buffer: {self.sign_buffer} → decided={self.decided_sign}")
                    else:
                        self.get_logger().warning("Scan complete but no signs collected, defaulting to 6.0")
                        self.decided_sign = 6.0
                    self.state = "READING_SIGN"

        elif self.state == "READING_SIGN":
            command.linear.x = 0.0
            command.angular.z = 0.0

            if self.decided_sign == 1.0:
                self.start_turn(90) #turn left
            elif self.decided_sign == 2.0 or self.decided_sign == 0.0: #turn right
                self.start_turn(-90)
            elif self.decided_sign == 4.0 or self.decided_sign == 5.0 or self.decided_sign == 3.0: #
                self.start_turn(180)
            else:
                self.get_logger().warning("Unknown sign, waiting...")

        self.cmd_pub.publish(command)


def main(args=None):
    rclpy.init(args=args)
    node = GetToGoal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()