# TurtleBot3 Autonomous Zone Exploration

ROS 2 package for the ELE434 autonomous robotics assessment at the University of Sheffield. The robot must autonomously visit 12 outer zones of a 4×4 m arena within 90 seconds without colliding with any obstacles.

<img width="720" height="auto" alt="ROS_TurtleBot" src="https://github.com/user-attachments/assets/cb7689f5-525c-4aea-ae12-5393a279a219" />


## Task Overview

The arena is divided into a 4×4 grid of 1×1 m zones. The 12 outer zones each score 1 mark when the robot's full body enters them. Four coloured cylinders and four wooden wall assemblies are placed randomly each run.

- **Robot:** TurtleBot3 WafflePi (ROS 2 Jazzy, Ubuntu 24.04)
- **Max linear speed:** 0.26 m/s
- **Max angular speed:** 1.82 rad/s

## Package Structure

```
scripts/
  zone_exploration.py        # Approach 1: FSM + reactive obstacle avoidance
  zone_exploration_dstar.py  # Approach 2: D* Lite path planning + pure pursuit
launch/
  explore.launch.py          # Launches SLAM + exploration node
```

## Approaches

### Approach 1 — FSM + Reactive Avoidance (`zone_exploration.py`)

A finite state machine drives the robot around the 12 outer zones in order. Obstacle avoidance is purely reactive: when a LiDAR hit is detected in the forward cone, the robot spins in the direction of greater clearance until the path to the target is clear again.

### Approach 2 — D* Lite Path Planning (`zone_exploration_dstar.py`)

Builds a live 80×80 occupancy grid (0.05 m/cell) from LiDAR via Bresenham ray-tracing, inflated by robot radius for collision-free planning. D* Lite (Koenig & Likhachev, 2002) runs incrementally on the inflated grid — only affected edges are updated when new obstacles are observed. A pure-pursuit follower tracks the planned path, with front-clearance speed scaling as the only reactive safety layer.

## Dependencies

- ROS 2 Jazzy
- Standard ROS 2 packages: `rclpy`, `geometry_msgs`, `nav_msgs`, `sensor_msgs`
