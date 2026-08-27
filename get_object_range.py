import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, CompressedImage
from geometry_msgs.msg import Pose2D,Point, Twist
import numpy as np
from rclpy.qos import qos_profile_sensor_data
import math
import os
import cv2
import pickle
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy


class GetObjectRange(Node):

    def __init__(self):
        super().__init__('get_object_range')
        self.flag = 0.0
        MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'model.pkl')
        self.model = self.initialize_model(MODEL_PATH)
        image_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.BEST_EFFORT,
		    history=QoSHistoryPolicy.KEEP_LAST,
		    durability=QoSDurabilityPolicy.VOLATILE,
		    depth=1
		)
        self.label = 6.0
        self.img = None
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)


        # Subscribe to LiDAR
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            qos_profile_sensor_data
        )

        # Publish object position
        self.cmd_pub = self.create_publisher(
            Point,
            '/object_position',
            10
        )
        self.img_subscriber = self.create_subscription(
				CompressedImage,
				'/image_raw/compressed',
				self.image_callback,
				image_qos_profile)
		#self._video_subscriber # Prevents unused variable warning.


    def initialize_model(self, model_path=None):
        if model_path is None:
            raise ValueError("model_path must be provided.")
        with open(model_path, "rb") as f:
            meta = pickle.load(f)
        knn = cv2.ml.KNearest_load(meta["knn_path"])
        return {"knn": knn, "k": meta["k"], "img_size": meta["img_size"]}

    def predict(self, model, image):
        img_size = model["img_size"]
        knn = model["knn"]
        k = model["k"]

        # Must exactly match extract_features() in train_knn.py
        resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        hog = cv2.HOGDescriptor(
            _winSize=(img_size, img_size),
            _blockSize=(24, 24),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9
        )

        features = hog.compute(gray).flatten()              # shape (2916,)
        features = np.array(features, dtype=np.float32)     # ensure float32
        features = np.ascontiguousarray(features.reshape(1, -1))  # shape (1, 2916), contiguous

        #self.get_logger().info(f"Features shape: {features.shape}, dtype: {features.dtype}")

        _, results, _, _ = knn.findNearest(features, k)
        return float(results.flatten()[0])
    def image_callback(self, msg):
        #self.label = self.predict(self.model, msg)
        np_arr = np.frombuffer(msg.data, np.uint8)
        img2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img2 is None:
            self.get_logger().warning("no image")
            return
        
        self.img = img2
        return img2
        #return label
    def lidar_callback(self, msg):

        front_ranges = []
        right_ranges = []
        left_ranges = []
        self.front_dist = float("inf")
        self.right_dist = float("inf")
        self.left_dist = float("inf")
        self.flag = 0.0

        for i, r in enumerate(msg.ranges):

            if math.isinf(r) or math.isnan(r):
                continue

            angle = msg.angle_min + i * msg.angle_increment
            #self.get_logger().info( f"ANGLE={angle}")

            # front ±45°
            if (-0.785/4) < angle < (0.785/4):
                front_ranges.append(r)

            # right (250° to 290)
            if (1.48) < angle < (1.66):
                right_ranges.append(r)
            if (-1.66) < angle < (-1.48):
                left_ranges.append(r)

        
        
        valid_front = [r for r in front_ranges if r <= 0.57] # keep 60cm in front as threashold
        valid_left = [r for r in left_ranges if r <= 0.4]
        valid_right = [r for r in right_ranges if r <= 0.4]
        


        #self.front_dist = min(front_ranges) if len(front_ranges) > 0 else float("inf")

        # self.get_logger().info( f"VALID_front={(len(valid_front) > 5)}")
        # flag 1.0 front 
        #flag 0.0 none 
        # flag 2.0 right
        #flag 3.0 left
        #flag 4.0 is front + right
        # flagh 5.0 front + left
        # flag 7.0 front + right + left
        if len(valid_front) > 2 and len(valid_right) < 1 and len(valid_left) < 1:
            


            # command = Twist()
            # command.linear.x = 0.0
            # command.angular.z = 0.0
            # self.cmd_pub.publish(command)


            self.flag = 1.0
            
            if self.img is not None:
                
                #self.get_logger().warning("wall detected, see an image")
                self.label = self.predict(self.model, self.img)
                #self.get_logger().info(f"Wall detected! Distance: {self.front_dist:.2f}m | Sign: {self.label}")
            else:
                self.label = 6.0
                #self.get_logger().warning("Wall detected but no image yet")

        elif len(valid_right) > 2 and len(valid_front) < 1 and len(valid_left) < 1:
            self.flag = 2.0
        elif len(valid_left) > 2 and len(valid_front) < 1 and len(valid_right) < 1:
            self.flag = 3.0

        # elif len(valid_right) > 0 and len(valid_front) > 0 and len(valid_left)==0:
        #     self.flag = 4.0
        #     if self.img is not None:
                
        #         #self.get_logger().warning("wall detected, see an image")
        #         self.label = self.predict(self.model, self.img)
        #         #self.get_logger().info(f"Wall detected! Distance: {self.front_dist:.2f}m | Sign: {self.label}")
        #     else:
        #         self.label = 6.0
        # elif len(valid_left) > 0 and len(valid_front) > 0 and len(valid_left)==0:
        #     self.flag = 5.0
        #     if self.img is not None:
                
        #         #self.get_logger().warning("wall detected, see an image")
        #         self.label = self.predict(self.model, self.img)
        #         #self.get_logger().info(f"Wall detected! Distance: {self.front_dist:.2f}m | Sign: {self.label}")
        #     else:
        #         self.label = 6.0
        # elif len(valid_right) > 0 and len(valid_front) > 0 and len(valid_left) > 0:
        #     self.flag = 7.0
        #     if self.img is not None:
                
        #         #self.get_logger().warning("wall detected, see an image")
        #         self.label = self.predict(self.model, self.img)
                #self.get_logger().info(f"Wall detected! Distance: {self.front_dist:.2f}m | Sign: {self.label}")
        
        
        else:
            self.flag = 0.0
            
            



        

        publish = Point()
        #publish.x = self.front_dist
        # publish.y = self.right_dist
        publish.y = float(self.label) #sign
        publish.z = self.flag #flag use to detect whether we want to follow the wall
        if self.flag == 2.0:
            publish.x = min(right_ranges) if right_ranges else float("inf")
        elif self.flag == 3.0:
            publish.x = min(left_ranges) if left_ranges else float("inf")
        else:
            publish.x = self.front_dist  # keep front dist for other states
        self.cmd_pub.publish(publish)


def main():

    rclpy.init()

    node = GetObjectRange()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()