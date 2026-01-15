#!/usr/bin/env python3
import math
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


def quat_to_yaw(q):
    # yaw from quaternion (x,y,z,w)
    # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def norm_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class HuskyTerminalCommander(Node):
    def __init__(self):
        super().__init__('husky_terminal_commander')

        # --- Params ---
        self.declare_parameter('cmd_vel_topic', '/husky/platform/cmd_vel_unstamped')
        self.declare_parameter('odom_topic', '/husky/platform/odom/filtered')

        # Motion tuning (simple open-loop-ish controller using odom feedback)
        self.declare_parameter('base_speed', 0.25)          # m/s
        self.declare_parameter('turn_speed', 0.8)           # rad/s (max)
        self.declare_parameter('goal_tolerance', 0.35)      # meters
        self.declare_parameter('yaw_tolerance', 0.25)       # radians
        self.declare_parameter('control_rate_hz', 20.0)

        # If you spawned at origin, keep (0,0). If not, change these.
        self.declare_parameter('world_origin_x', 0.0)
        self.declare_parameter('world_origin_y', 0.0)

        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value

        self.pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        # --- Pose state (from odom) ---
        self.pose_lock = threading.Lock()
        self.have_odom = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # Use initial odom pose as reference for "world" (so coordinates align with spawn frame)
        self.odom0_set = False
        self.odom0_x = 0.0
        self.odom0_y = 0.0
        self.odom0_yaw = 0.0
        self.world_origin_x = float(self.get_parameter('world_origin_x').value)
        self.world_origin_y = float(self.get_parameter('world_origin_y').value)

        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 20)

        # --- Goals database (EDIT THESE to match your world coordinates) ---
        self.zones = {
            "red":    (10.0, 10.0),
            "blue":   (-10.0, 10.0),
            "yellow": (10.0, -10.0),
            "green":  (-10.0, -10.0),
        }

        self.objects = {
            "object1": (8.0, -4.0),
            "object2": (6.0,  6.0),
            "object3": (-6.0,  6.0),
            "object4": (-8.0, -3.0),
            "object5": (2.0,  -8.0),
            "object6": (-2.0, -8.0),
        }

        # --- Runtime navigation state ---
        self.active_goal_name = None
        self.active_goal_xy = None
        self.paused = True

        self.base_speed = float(self.get_parameter('base_speed').value)
        self.turn_speed = float(self.get_parameter('turn_speed').value)
        self.goal_tol = float(self.get_parameter('goal_tolerance').value)
        self.yaw_tol = float(self.get_parameter('yaw_tolerance').value)

        rate = float(self.get_parameter('control_rate_hz').value)
        self.timer = self.create_timer(1.0 / rate, self.control_loop)

        self.get_logger().info(f"cmd_vel: {self.cmd_vel_topic}")
        self.get_logger().info(f"odom:    {self.odom_topic}")
        self.get_logger().info("Commands: go_to_red/blue/yellow/green, go_to_object1..6, pick_object1..6, place_object1..6, stop, move, slow, fast, status, help, quit")

    def odom_cb(self, msg: Odometry):
        with self.pose_lock:
            ox = msg.pose.pose.position.x
            oy = msg.pose.pose.position.y
            oyaw = quat_to_yaw(msg.pose.pose.orientation)

            if not self.odom0_set:
                # first odom reading becomes reference (spawn pose)
                self.odom0_x = ox
                self.odom0_y = oy
                self.odom0_yaw = oyaw
                self.odom0_set = True

            # Convert odom pose to "world-ish" pose:
            # world = (odom - odom0) + world_origin
            self.x = (ox - self.odom0_x) + self.world_origin_x
            self.y = (oy - self.odom0_y) + self.world_origin_y

            # yaw relative to start
            self.yaw = norm_angle(oyaw - self.odom0_yaw)

            self.have_odom = True

    def publish_stop(self):
        t = Twist()
        self.pub.publish(t)

    def set_goal(self, name: str, xy):
        self.active_goal_name = name
        self.active_goal_xy = xy
        self.paused = False
        self.get_logger().info(f"GOAL set: {name} -> ({xy[0]:.2f}, {xy[1]:.2f})")

    def control_loop(self):
        if self.paused or self.active_goal_xy is None:
            return

        with self.pose_lock:
            if not self.have_odom:
                return
            rx, ry, ryaw = self.x, self.y, self.yaw

        gx, gy = self.active_goal_xy
        dx = gx - rx
        dy = gy - ry
        dist = math.hypot(dx, dy)

        # If reached goal, stop
        if dist < self.goal_tol:
            self.publish_stop()
            self.get_logger().info(f"Reached goal: {self.active_goal_name} (dist={dist:.2f})")
            self.active_goal_xy = None
            self.active_goal_name = None
            self.paused = True
            return

        desired_yaw = math.atan2(dy, dx)
        yaw_err = norm_angle(desired_yaw - ryaw)

        # Simple “turn then go” controller
        cmd = Twist()

        # Turn strongly if facing away
        if abs(yaw_err) > self.yaw_tol:
            cmd.angular.z = max(-self.turn_speed, min(self.turn_speed, 1.5 * yaw_err))
            cmd.linear.x = 0.0
        else:
            # go forward and small steering
            cmd.linear.x = self.base_speed
            cmd.angular.z = max(-self.turn_speed, min(self.turn_speed, 1.0 * yaw_err))

        self.pub.publish(cmd)

    def print_status(self):
        with self.pose_lock:
            if not self.have_odom:
                print("STATUS: no odom received yet.")
                return
            rx, ry, ryaw = self.x, self.y, self.yaw

        print(f"\nRobot pose (world-ish from odom): x={rx:.2f}, y={ry:.2f}, yaw={math.degrees(ryaw):.1f} deg")
        if self.active_goal_xy:
            gx, gy = self.active_goal_xy
            print(f"Active goal: {self.active_goal_name} -> ({gx:.2f},{gy:.2f}) dist={math.hypot(gx-rx, gy-ry):.2f}")
        else:
            print("Active goal: none")

        print("\nDistances to zones:")
        for k, (zx, zy) in self.zones.items():
            print(f"  {k:6s}: {math.hypot(zx-rx, zy-ry):.2f} m")

        print("\nDistances to objects:")
        for k, (ox, oy) in self.objects.items():
            print(f"  {k:7s}: {math.hypot(ox-rx, oy-ry):.2f} m")
        print("")

    def repl(self):
        help_text = """
Commands:
  go_to_red | go_to_blue | go_to_yellow | go_to_green
  go_to_object1 .. go_to_object6
  pick_object1 .. pick_object6        (placeholder)
  place_object1 .. place_object6      (placeholder)
  stop        pause immediately
  move        resume toward last goal
  slow        reduce speed by 1/8
  fast        increase speed by 1/8
  status      print pose + distances (updates as robot moves)
  help        show this
  quit        exit
"""
        print(help_text.strip() + "\n")

        while rclpy.ok():
            try:
                cmd = input().strip()
            except EOFError:
                break

            if cmd in ("help", "?"):
                print(help_text.strip() + "\n")
                continue

            if cmd in ("quit", "exit"):
                self.publish_stop()
                break

            if cmd == "status":
                self.print_status()
                continue

            if cmd == "stop":
                self.paused = True
                self.publish_stop()
                self.get_logger().info("Paused (stop).")
                continue

            if cmd == "move":
                if self.active_goal_xy is None:
                    self.get_logger().info("No previous goal to resume.")
                else:
                    self.paused = False
                    self.get_logger().info("Resumed (move).")
                continue

            if cmd == "slow":
                self.base_speed *= (7.0/8.0)
                self.get_logger().info(f"Speed -> {self.base_speed:.3f} m/s")
                continue

            if cmd == "fast":
                self.base_speed *= (9.0/8.0)
                self.get_logger().info(f"Speed -> {self.base_speed:.3f} m/s")
                continue

            # go_to_zone
            if cmd.startswith("go_to_"):
                key = cmd[len("go_to_"):]
                if key in self.zones:
                    self.set_goal(key, self.zones[key])
                    continue
                if key in self.objects:
                    self.set_goal(key, self.objects[key])
                    continue
                self.get_logger().info(f"Unknown target: {key}")
                continue

            # placeholders for pick/place
            if cmd.startswith("pick_object"):
                self.get_logger().info(f"{cmd}: placeholder (base-only for now).")
                continue

            if cmd.startswith("place_object"):
                self.get_logger().info(f"{cmd}: placeholder (base-only for now).")
                continue

            self.get_logger().info("Unknown command. Type 'help'.")

        self.publish_stop()


def main():
    rclpy.init()
    node = HuskyTerminalCommander()

    t = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t.start()

    node.repl()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
