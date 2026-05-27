#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import math
import time
import subprocess
from datetime import datetime


class ExploreNode(Node):
    def __init__(self):
        super().__init__('explore_node')

        # Publisher & Subscribers
        self.cmd_vel_pub = self.create_publisher(msg_type=TwistStamped, topic='/cmd_vel', qos_profile=10)
        self.scan_sub = self.create_subscription(msg_type=LaserScan, topic='/scan', callback=self.scan_callback, qos_profile=10)
        self.odom_sub = self.create_subscription(msg_type=Odometry, topic='/odom', callback=self.odom_callback, qos_profile=10)

        # Timer for control loop aka. timaer_callback executes every 0.1 seconds or publish_rate = 10Hz ... --> not needed timer period set up directly
        self.timer = self.create_timer(timer_period_sec=0.1, callback=self.control_loop)  # 10 Hz

        # FSM
        self.goal = 'ESCAPE_CENTER'
        self.state = 'GO_TO_TARGET'
        self.prev_goal = None
        self.prev_state = None

        # Motion params
        self.forward_speed = 0.26
        self.slow_speed = 0.18
        self.turn_speed = 1.2

        self.heading_kp = 1.2
        self.max_heading_turn = 0.7
        self.rotate_in_place_yaw = 0.55

        # thresholds
        self.front_threshold = 0.3
        self.corner_threshold = 0.3
        self.side_threshold = 0.2
        self.target_clear_threshold = 0.5
        self.target_sector_half_width_deg = 10.0
        self.origin_tolerance = 0.20
        self.zone_margin = 0.3

        # Sensor data
        self.scan_ready = False
        self.odom_ready = False
        self.ranges = []

        self.front_dist = float('inf')
        self.front_left_dist = float('inf')
        self.front_right_dist = float('inf')

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # Avoidance state
        self.avoid_direction = None

        # Zones
        self.visited_zones = set()
        self.current_zone = None
        self.zones = {
            1: (1.0, 2.0, 1.0, 2.0),
            2: (1.0, 2.0, 0.0, 1.0),
            3: (1.0, 2.0, -1.0, 0.0),
            4: (1.0, 2.0, -2.0, -1.0),
            5: (0.0, 1.0, -2.0, -1.0),
            6: (-1.0, 0.0, -2.0, -1.0),
            7: (-2.0, -1.0, -2.0, -1.0),
            8: (-2.0, -1.0, -1.0, 0.0),
            9: (-2.0, -1.0, 0.0, 1.0),
            10: (-2.0, -1.0, 1.0, 2.0),
            11: (-1.0, 0.0, 1.0, 2.0),
            12: (0.0, 1.0, 1.0, 2.0),
        }

        self.zone_order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] # [1,2,3,8,9,7,12,11,10,4,5,6] # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        # Zone skip — if stuck on a zone for too long, skip it
        self.zone_skip_timeout  = 20.0  # seconds before skipping
        self.zone_skip_timer    = 0.0   # accumulated time on current target zone
        self.current_target_zone = None  # zone currently being attempted
        self.skipped_zones      = set() # zones skipped, retried at end if time allows

        # Autosave map every 10s while SLAM is still alive
        self.timestamp      = datetime.now().strftime('%H-%M-%S')
        self.map_save_path  = f'/home/student/ros2_ws/src/ele434_team05_2026/maps/explore_map'
        self.map_save_timer = self.create_timer(5.0, self._save_map)

        self.get_logger().info(f"The '{self.get_name()}' Zone explore FSM node is initialised.")

    # Utils
    def log_fsm_changes(self):
        if self.goal != self.prev_goal:
            self.get_logger().info(f'CURRENT GOAL: {self.goal}')
            self.prev_goal = self.goal

        if self.state != self.prev_state:
            self.get_logger().info(f'CURRENT STATE: {self.state}')
            self.prev_state = self.state

    def clamp(self, value, min_value, max_value):
        return max(min(value, max_value), min_value)

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def build_twist(self, linear_x=0.0, angular_z=0.0):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = linear_x
        msg.twist.angular.z = angular_z
        return msg

    # LiDAR
    def clean_ranges(self, ranges, range_min, range_max):
        cleaned = []
        for r in ranges:
            if math.isinf(r) or math.isnan(r):
                cleaned.append(float('inf'))
            elif r < range_min or r > range_max:
                cleaned.append(float('inf'))
            else:
                cleaned.append(r)
        return cleaned

    def get_sector_min(self, ranges, start_idx, end_idx):
        if start_idx <= end_idx:
            sector = ranges[start_idx:end_idx + 1]
        else:
            sector = ranges[start_idx:] + ranges[:end_idx + 1]
        if not sector:
            return float('inf')
        return min(sector)

    def get_target_sector_distance(self, yaw_error):
        """
        Revisa con el LiDAR si la dirección relativa al target ya está libre.
        Asume scan ~360 samples.
        """
        if not self.ranges:
            return float('inf')

        center_deg = math.degrees(yaw_error)
        half_width = self.target_sector_half_width_deg

        start_deg = int(round(center_deg - half_width)) % 360
        end_deg = int(round(center_deg + half_width)) % 360

        return self.get_sector_min(self.ranges, start_deg, end_deg)

    def scan_callback(self, msg: LaserScan):
        self.ranges = self.clean_ranges(msg.ranges, msg.range_min, msg.range_max)

        self.front_dist = self.get_sector_min(self.ranges, 350, 10)
        self.front_left_dist = self.get_sector_min(self.ranges, 20, 60)
        self.front_right_dist = self.get_sector_min(self.ranges, 300, 340)

        self.scan_ready = True

    # Odom
    def odom_callback(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        self.odom_ready = True

    # Zones
    def get_current_zone(self, x, y):
        for zone_id, (xmin, xmax, ymin, ymax) in self.zones.items():
            xmin_adj = xmin + self.zone_margin
            xmax_adj = xmax - self.zone_margin
            ymin_adj = ymin + self.zone_margin
            ymax_adj = ymax - self.zone_margin

            if xmin_adj <= x <= xmax_adj and ymin_adj <= y <= ymax_adj:
                return zone_id

        return None

    def get_zone_center(self, zone_id):
        xmin, xmax, ymin, ymax = self.zones[zone_id]
        return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)

    def update_zone_tracking(self):
        self.current_zone = self.get_current_zone(self.x, self.y)
        if self.current_zone is not None and self.current_zone not in self.visited_zones:
            self.visited_zones.add(self.current_zone)
            self.get_logger().info(f'Visited zone: {self.current_zone} \n' f'---> Visited: {sorted(self.visited_zones)} of ' f'Total: {len(self.visited_zones)}/12')

    def get_next_target_zone(self):
        for zone_id in self.zone_order:
            if zone_id not in self.visited_zones and zone_id not in self.skipped_zones:
                return zone_id
        # All non-skipped zones done — retry skipped ones
        for zone_id in self.zone_order:
            if zone_id not in self.visited_zones:
                return zone_id
        return None

    # Target helpers
    def compute_target_yaw_error(self, target_x, target_y):
        dx = target_x - self.x
        dy = target_y - self.y
        desired_yaw = math.atan2(dy, dx)
        yaw_error = self.normalize_angle(desired_yaw - self.yaw)
        return yaw_error

    def target_direction_clear(self, target_x, target_y):
        yaw_error = self.compute_target_yaw_error(target_x, target_y)
        dist = self.get_target_sector_distance(yaw_error)
        return dist > self.target_clear_threshold

    # Avoidance
    def obstacle(self):
        return (self.front_dist < self.front_threshold or
            self.front_left_dist < self.corner_threshold or
            self.front_right_dist < self.corner_threshold)

    def avoid_control(self, target_x, target_y):
        angular = 0.0
        if self.avoid_direction is None:
            if self.front_left_dist > self.front_right_dist:
                self.avoid_direction = 'left'
            else:
                self.avoid_direction = 'right'
        if self.obstacle():
            if self.avoid_direction == 'left':
                return self.build_twist(0.0, self.turn_speed)
            else:
                return self.build_twist(0.0, -self.turn_speed)
        if self.target_direction_clear(target_x, target_y):
            self.state = 'GO_TO_TARGET'
            self.avoid_direction = None
            return self.build_twist(0.0, 0.0)
        # Si el frente está libre pero el target aún no, avanzar ajustando
        if self.avoid_direction == 'left':
            if self.front_left_dist < self.side_threshold:
                angular = -0.5
        else:
            if self.front_right_dist < self.side_threshold:
                angular = 0.5
        return self.build_twist(self.slow_speed, angular)

    # State logic
    def go_to_target_control(self, target_x, target_y):
        yaw_error = self.compute_target_yaw_error(target_x, target_y)
        angular = self.heading_kp * yaw_error
        angular = self.clamp(angular, -self.max_heading_turn, self.max_heading_turn)
        if abs(yaw_error) > self.rotate_in_place_yaw:
            linear = 0.03
        else:
            linear = self.forward_speed
        if self.front_left_dist < self.side_threshold:
            angular -= 0.80
            linear = min(linear, self.slow_speed)
        if self.front_right_dist < self.side_threshold:
            angular += 0.80
            linear = min(linear, self.slow_speed)
        angular = self.clamp(angular, -self.max_heading_turn, self.max_heading_turn)
        return self.build_twist(linear, angular)

    def go_to_target(self, target_x, target_y):
        if self.state == 'GO_TO_TARGET':
            if self.obstacle():
                print("OBSTACLE FOR DEBUG ONLY")
                self.state = 'AVOID_OBSTACLE'
                self.avoid_direction = None
                return self.build_twist(0.0, 0.0)
            return self.go_to_target_control(target_x, target_y)
        elif self.state == 'AVOID_OBSTACLE':
            return self.avoid_control(target_x, target_y)
        return self.build_twist(0.0, 0.0)

    def state_logic(self):
        if self.goal == 'ESCAPE_CENTER':
            target_x, target_y = self.get_zone_center(self.zone_order[0])
            cmd = self.go_to_target(target_x, target_y)
        elif self.goal == 'OUTER_ZONE_EXPLORATION':
            target_zone = self.get_next_target_zone()
            if target_zone is None:
                return self.build_twist(0.0, 0.0)
            target_x, target_y = self.get_zone_center(target_zone)
            cmd = self.go_to_target(target_x, target_y)
        elif self.goal == 'RETURN_TO_ORIGIN':
            target_x = 0.0
            target_y = 0.0
            cmd = self.go_to_target(target_x, target_y)
        else:
            cmd = self.build_twist(0.0, 0.0)
        return cmd

    # FSM update
    def update_state(self):
        self.update_zone_tracking()
        if self.goal == 'ESCAPE_CENTER':
            if self.current_zone == self.zone_order[0]:
                self.goal = 'OUTER_ZONE_EXPLORATION'
                self.state = 'GO_TO_TARGET'
        elif self.goal == 'OUTER_ZONE_EXPLORATION':
            target_zone = self.get_next_target_zone()

            # Track time spent on current target zone
            if target_zone != self.current_target_zone:
                self.current_target_zone = target_zone
                self.zone_skip_timer     = 0.0
            else:
                self.zone_skip_timer += 0.1  # control loop is 10 Hz

            # Skip zone if stuck for too long
            if (self.zone_skip_timer >= self.zone_skip_timeout
                    and target_zone is not None
                    and target_zone not in self.visited_zones):
                self.skipped_zones.add(target_zone)
                self.zone_skip_timer     = 0.0
                self.current_target_zone = None
                self.state               = 'GO_TO_TARGET'
                self.get_logger().warn(
                    f'[SKIP] Zone {target_zone} unreachable after '
                    f'{self.zone_skip_timeout}s → skipping. '
                    f'Skipped: {sorted(self.skipped_zones)}')

            if len(self.visited_zones) == 12:
                self.goal = 'RETURN_TO_ORIGIN'
                self.state = 'GO_TO_TARGET'
                self._save_map()  # save immediately while SLAM is still alive
        elif self.goal == 'RETURN_TO_ORIGIN':
            if math.hypot(self.x, self.y) < self.origin_tolerance:
                self.goal = 'DONE'
                self.state = 'STOP'

    def control_loop(self):
        if not self.scan_ready or not self.odom_ready:
            return

        self.update_state()
        self.log_fsm_changes()
        cmd = self.state_logic()
        self.cmd_vel_pub.publish(cmd)

    # SHUTDOWN — stop robot first, then save map
    def _save_map(self):
        self.get_logger().info(f'[MAP] Saving map to {self.timestamp}...')
        try:
            result = subprocess.run(
                ['ros2', 'run', 'nav2_map_server', 'map_saver_cli',
                 '-f', self.map_save_path, '--fmt', 'png', 
                 '--ros-args', '-p', 'save_map_timeout:=10.0'],
                timeout=20,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                self.get_logger().info(f'[MAP] Map saved: map_{self.timestamp}.pgm / .yaml')
            else:
                self.get_logger().error(f'[MAP] Map save failed: {result.stderr}')
        except Exception as e:
            self.get_logger().error(f'[MAP] Map save error: {e}')

    def on_shutdown(self):
        # 1. Stop the robot immediately
        self.get_logger().info("Stopping the robot...")
        self.cmd_vel_pub.publish(TwistStamped())
        time.sleep(0.5)  # give cmd_vel time to reach robot before ROS tears down

        # 2. Save SLAM map
        self.get_logger().info(f'[SHUTDOWN] Saving map to {self.map_save_path}...')
        try:
            result = subprocess.run(
                ['ros2', 'run', 'nav2_map_server', 'map_saver_cli',
                 '-f', self.map_save_path,
                 '--ros-args', '-p', 'save_map_timeout:=10.0'],
                timeout=20,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                self.get_logger().info(f'[SHUTDOWN] Map saved: map_{self.timestamp}.pgm / .yaml')
            else:
                self.get_logger().error(f'[SHUTDOWN] Map save failed: {result.stderr}')
        except Exception as e:
            self.get_logger().error(f'[SHUTDOWN] Map save error: {e}')

def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = ExploreNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print(f"{node.get_name()} received a shutdown request (Ctrl+C).")
    finally:
        node.on_shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()