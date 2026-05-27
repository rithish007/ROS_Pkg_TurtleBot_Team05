#!/usr/bin/env python3
"""
ELE434 zone exploration using D* Lite path planning + reactive safety.

Architecture
------------
- OccupancyGrid: 40x40 cells over [-2, 2]^2, 0.10 m resolution, built
  incrementally from LiDAR via Bresenham ray-trace. Inflated by robot
  radius for planning.
- DStarLite: Koenig & Likhachev (2002) incremental search on the
  inflated grid. Replans on edge-cost changes only - no from-scratch
  rebuilds while the goal stays fixed.
- PathFollower: pure-pursuit on the D* Lite path. Linear speed scaled
  by front clearance (the only reactive layer; planner owns steering).
- ExploreNode: ROS plumbing, picks closest unvisited zone, triggers
  replans on goal change / path invalidation / stuck timeout.
"""
import heapq
import math

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


# ---------------------------------------------------------------------------
# Occupancy grid
# ---------------------------------------------------------------------------

class OccupancyGrid:
    UNKNOWN = -1
    FREE = 0
    OCCUPIED = 1

    def __init__(self, size=80, res=0.05, origin=(-2.0, -2.0),
                 inflation_radius=3, wall_inflation_radius=2,
                 shoulder_pad_radius=1):
        # 0.05 m cells let the grid represent ~0.30 m gaps between
        # obstacles. Inflation 3 cells = 0.15 m disc (circular kernel
        # below), roughly matching the chassis footprint with antennas.
        # The diagonals are reclaimed vs a square kernel, so the planner
        # can still squeeze through narrow gaps at 45 degrees.
        #
        # `shoulder_pad_radius` adds a directional pad at the 4 corner
        # diagonals only - is_blocked treats a cell as blocked if any
        # of (i+-k, j+-k) is in the inflated set. The Waffle's
        # half-diagonal corner reach (~0.205 m) exceeds the 0.15 m disc
        # by ~0.055 m, which is what makes pivots near obstacles graze
        # the rear corners. k=1 cell adds sqrt(2)*0.05 ~ 0.071 m at the
        # diagonals without padding the cardinal directions (so the
        # tight cylinder/wall gap stays open).
        self.size = size
        self.res = res
        self.origin = origin
        self.inflation_radius = inflation_radius
        self.wall_inflation_radius = wall_inflation_radius
        self.shoulder_pad_radius = shoulder_pad_radius
        self.grid = [[self.UNKNOWN] * size for _ in range(size)]
        self.occupied = set()
        self.inflated = set()

    def world_to_cell(self, x, y):
        i = int((x - self.origin[0]) / self.res)
        j = int((y - self.origin[1]) / self.res)
        i = max(0, min(self.size - 1, i))
        j = max(0, min(self.size - 1, j))
        return (i, j)

    def cell_to_world(self, i, j):
        return (self.origin[0] + (i + 0.5) * self.res,
                self.origin[1] + (j + 0.5) * self.res)

    def is_valid(self, i, j):
        return 0 <= i < self.size and 0 <= j < self.size

    def _set_cell(self, i, j, value):
        if not self.is_valid(i, j):
            return False
        prev = self.grid[i][j]
        if prev == value:
            return False
        self.grid[i][j] = value
        if value == self.OCCUPIED:
            self.occupied.add((i, j))
        elif prev == self.OCCUPIED:
            self.occupied.discard((i, j))
        return True

    def _bresenham(self, x0, y0, x1, y1):
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            cells.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return cells

    def update_from_scan(self, ranges, angle_min, angle_increment,
                         rx, ry, ryaw, range_max, beam_step=4):
        """Ray-trace each LiDAR beam, mark FREE along + OCCUPIED at hit."""
        n = len(ranges)
        if n == 0:
            return
        rc = self.world_to_cell(rx, ry)
        endpoints_occupied = set()
        for i in range(0, n, beam_step):
            r = ranges[i]
            ang = angle_min + i * angle_increment + ryaw
            if math.isinf(r) or math.isnan(r) or r >= range_max:
                r_trace = range_max
                hit = False
            else:
                r_trace = r
                hit = True
            ex = rx + r_trace * math.cos(ang)
            ey = ry + r_trace * math.sin(ang)
            ei = int((ex - self.origin[0]) / self.res)
            ej = int((ey - self.origin[1]) / self.res)
            ei = max(0, min(self.size - 1, ei))
            ej = max(0, min(self.size - 1, ej))
            cells = self._bresenham(rc[0], rc[1], ei, ej)
            for c in cells[:-1]:
                if c in endpoints_occupied:
                    continue
                if not self.is_valid(*c):
                    continue
                if self.grid[c[0]][c[1]] != self.OCCUPIED:
                    self._set_cell(c[0], c[1], self.FREE)
            if cells:
                last = cells[-1]
                if hit and self.is_valid(*last):
                    self._set_cell(last[0], last[1], self.OCCUPIED)
                    endpoints_occupied.add(last)
                elif self.is_valid(*last) and last not in endpoints_occupied:
                    if self.grid[last[0]][last[1]] != self.OCCUPIED:
                        self._set_cell(last[0], last[1], self.FREE)

    def recompute_inflation(self):
        """Rebuild the inflated set; return (added, removed) cells."""
        new_inflated = set()
        for (i, j) in self.occupied:
            is_wall = (i == 0 or i == self.size - 1 or
                       j == 0 or j == self.size - 1)
            r = self.wall_inflation_radius if is_wall else self.inflation_radius
            r_sq = r * r
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if di * di + dj * dj > r_sq:
                        continue
                    ni, nj = i + di, j + dj
                    if self.is_valid(ni, nj):
                        new_inflated.add((ni, nj))
        added = new_inflated - self.inflated
        removed = self.inflated - new_inflated
        self.inflated = new_inflated
        return added, removed

    def is_blocked(self, i, j):
        if (i, j) in self.inflated:
            return True
        k = self.shoulder_pad_radius
        if k > 0:
            if (i + k, j + k) in self.inflated: return True
            if (i + k, j - k) in self.inflated: return True
            if (i - k, j + k) in self.inflated: return True
            if (i - k, j - k) in self.inflated: return True
        return False

    def clear_inflation_around(self, cell, radius=1):
        """Temporarily un-block cells near the robot so the planner never
        sees the robot's own cell as blocked."""
        ci, cj = cell
        removed = set()
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                c = (ci + di, cj + dj)
                if c in self.inflated:
                    self.inflated.discard(c)
                    removed.add(c)
        return removed


# ---------------------------------------------------------------------------
# D* Lite
# ---------------------------------------------------------------------------

class DStarLite:
    """D* Lite (Koenig & Likhachev, 2002), 8-connected, lazy-deletion heap."""

    SQRT2 = math.sqrt(2)

    def __init__(self, grid: OccupancyGrid):
        self.grid = grid
        self.start = None
        self.goal = None
        self.k_m = 0.0
        self.g = {}
        self.rhs = {}
        self.U = []
        self.entry_finder = {}
        self.counter = 0

    def _get_g(self, s):
        return self.g.get(s, math.inf)

    def _get_rhs(self, s):
        return self.rhs.get(s, math.inf)

    def _h(self, a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return max(dx, dy) + (self.SQRT2 - 1.0) * min(dx, dy)

    def _cost(self, u, v):
        if self.grid.is_blocked(*u) or self.grid.is_blocked(*v):
            return math.inf
        dx = abs(u[0] - v[0])
        dy = abs(u[1] - v[1])
        if dx + dy == 1:
            return 1.0
        if dx == 1 and dy == 1:
            return self.SQRT2
        return math.inf

    def _neighbors(self, u):
        i, j = u
        out = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if self.grid.is_valid(ni, nj):
                    out.append((ni, nj))
        return out

    def _calculate_key(self, s):
        m = min(self._get_g(s), self._get_rhs(s))
        return (m + self._h(self.start, s) + self.k_m, m)

    def _push(self, s, key):
        self.counter += 1
        entry = [key, self.counter, s]
        self.entry_finder[s] = entry
        heapq.heappush(self.U, entry)

    def _remove(self, s):
        entry = self.entry_finder.pop(s, None)
        if entry is not None:
            entry[-1] = None

    def _top(self):
        while self.U:
            key, _, s = self.U[0]
            if s is None:
                heapq.heappop(self.U)
                continue
            return key, s
        return None, None

    def initialize(self, start, goal):
        self.start = start
        self.goal = goal
        self.k_m = 0.0
        self.g = {}
        self.rhs = {}
        self.U = []
        self.entry_finder = {}
        self.counter = 0
        self.rhs[goal] = 0.0
        self._push(goal, self._calculate_key(goal))

    def update_vertex(self, u):
        if u != self.goal:
            best = math.inf
            for s in self._neighbors(u):
                c = self._cost(u, s) + self._get_g(s)
                if c < best:
                    best = c
            self.rhs[u] = best
        if u in self.entry_finder:
            self._remove(u)
        if self._get_g(u) != self._get_rhs(u):
            self._push(u, self._calculate_key(u))

    def compute_shortest_path(self, max_iters=5000):
        iters = 0
        while iters < max_iters:
            iters += 1
            top_key, u = self._top()
            if u is None:
                break
            start_key = self._calculate_key(self.start)
            start_consistent = (
                self._get_rhs(self.start) == self._get_g(self.start)
            )
            if not (top_key < start_key or not start_consistent):
                break
            k_new = self._calculate_key(u)
            if top_key < k_new:
                self._remove(u)
                self._push(u, k_new)
            elif self._get_g(u) > self._get_rhs(u):
                self.g[u] = self._get_rhs(u)
                self._remove(u)
                for s in self._neighbors(u):
                    self.update_vertex(s)
            else:
                self.g[u] = math.inf
                self.update_vertex(u)
                for s in self._neighbors(u):
                    self.update_vertex(s)
        return iters

    def update_start(self, new_start):
        if new_start != self.start:
            self.k_m += self._h(self.start, new_start)
            self.start = new_start

    def apply_edge_changes(self, changed_cells):
        touched = set()
        for c in changed_cells:
            touched.add(c)
            for n in self._neighbors(c):
                touched.add(n)
        for u in touched:
            self.update_vertex(u)

    def extract_path(self, max_steps=400):
        """Greedy walk from start to goal via min(cost + g) successor."""
        path = [self.start]
        s = self.start
        seen = {s}
        for _ in range(max_steps):
            if s == self.goal:
                break
            best, best_s = math.inf, None
            for ns in self._neighbors(s):
                c = self._cost(s, ns) + self._get_g(ns)
                if c < best:
                    best, best_s = c, ns
            if best_s is None or math.isinf(best) or best_s in seen:
                break
            path.append(best_s)
            seen.add(best_s)
            s = best_s
        return path

    def path_blocked(self, path):
        if not path:
            return True
        for c in path:
            if self.grid.is_blocked(*c):
                return True
        return False


# ---------------------------------------------------------------------------
# ROS Node
# ---------------------------------------------------------------------------

class ExploreNode(Node):
    ROBOT_RADIUS = 0.176
    ZONE_MARGIN = 0.176

    def __init__(self):
        super().__init__('explore_dstar_node')

        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.timer = self.create_timer(0.1, self.control_loop)

        # Motion params
        self.forward_speed = 0.24
        self.slow_speed = 0.18
        self.max_angular = 1.6
        self.heading_kp = 1.6
        self.lookahead_dist = 0.30
        self.stop_clearance = 0.22
        self.slow_clearance = 0.35

        # Sensor state
        self.scan_ready = False
        self.odom_ready = False
        self.ranges = []
        self.scan_angle_min = 0.0
        self.scan_angle_inc = 0.0
        self.scan_range_max = 3.5
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # Zones
        self.zones = {
            1:  (1.0,  2.0,  1.0,  2.0),
            2:  (1.0,  2.0,  0.0,  1.0),
            3:  (1.0,  2.0, -1.0,  0.0),
            4:  (1.0,  2.0, -2.0, -1.0),
            5:  (0.0,  1.0, -2.0, -1.0),
            6:  (-1.0, 0.0, -2.0, -1.0),
            7:  (-2.0,-1.0, -2.0, -1.0),
            8:  (-2.0,-1.0, -1.0,  0.0),
            9:  (-2.0,-1.0,  0.0,  1.0),
            10: (-2.0,-1.0,  1.0,  2.0),
            11: (-1.0, 0.0,  1.0,  2.0),
            12: (0.0,  1.0,  1.0,  2.0),
        }
        self.visited = set()
        self.current_zone = None
        self.target_zone = None
        self.target_entry = None
        self.target_arrival_time = None
        self.arrival_radius = 0.30
        self.zone_timeout = 10.0

        # Planner
        self.grid = OccupancyGrid(size=80, res=0.05,
                                  origin=(-2.0, -2.0),
                                  inflation_radius=3,
                                  wall_inflation_radius=2)
        self.planner = DStarLite(self.grid)
        self.planner_ready = False
        self.path = []
        self.path_world = []
        self.goal_cell = None

        # Stuck detection
        self.stuck_x = None
        self.stuck_y = None
        self.stuck_t0 = None
        self.stuck_timeout = 4.0
        self.stuck_progress_eps = 0.08

        # Logging
        self.prev_target = None
        self.tick = 0
        self._was_degenerate = False
        self._diag_last_tick = 0

        self.get_logger().info("D* Lite explore node initialised.")

    # ---- ROS callbacks ------------------------------------------------
    def _scan_cb(self, msg: LaserScan):
        cleaned = []
        for r in msg.ranges:
            if math.isinf(r) or math.isnan(r):
                cleaned.append(math.inf)
            elif r < msg.range_min or r > msg.range_max:
                cleaned.append(math.inf)
            else:
                cleaned.append(r)
        self.ranges = cleaned
        self.scan_angle_min = msg.angle_min
        self.scan_angle_inc = msg.angle_increment
        self.scan_range_max = msg.range_max
        self.scan_ready = True

    def _odom_cb(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)
        self.odom_ready = True

    # ---- Utilities ----------------------------------------------------
    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    @staticmethod
    def _norm(a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def _twist(self, lin=0.0, ang=0.0):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = float(lin)
        msg.twist.angular.z = float(ang)
        return msg

    # ---- Zone tracking ------------------------------------------------
    def _zone_at(self, x, y):
        for zid, (xmn, xmx, ymn, ymx) in self.zones.items():
            if (xmn + self.ZONE_MARGIN <= x <= xmx - self.ZONE_MARGIN and
                    ymn + self.ZONE_MARGIN <= y <= ymx - self.ZONE_MARGIN):
                return zid
        return None

    def _zone_centre(self, zid):
        xmn, xmx, ymn, ymx = self.zones[zid]
        return ((xmn + xmx) / 2.0, (ymn + ymx) / 2.0)

    def _zone_entry_point(self, zid):
        """Closest unblocked cell in the zone's 'full body inside' sub-rectangle."""
        xmn, xmx, ymn, ymx = self.zones[zid]
        fxmn = xmn + self.ZONE_MARGIN
        fxmx = xmx - self.ZONE_MARGIN
        fymn = ymn + self.ZONE_MARGIN
        fymx = ymx - self.ZONE_MARGIN
        i0, j0 = self.grid.world_to_cell(fxmn, fymn)
        i1, j1 = self.grid.world_to_cell(fxmx, fymx)
        best, best_d = None, math.inf
        for i in range(min(i0, i1), max(i0, i1) + 1):
            for j in range(min(j0, j1), max(j0, j1) + 1):
                if self.grid.is_blocked(i, j):
                    continue
                wx, wy = self.grid.cell_to_world(i, j)
                d = math.hypot(wx - self.x, wy - self.y)
                if d < best_d:
                    best, best_d = (wx, wy), d
        if best is not None:
            return best
        cx = max(fxmn, min(fxmx, self.x))
        cy = max(fymn, min(fymx, self.y))
        return (cx, cy)

    def _update_visited(self):
        z = self._zone_at(self.x, self.y)
        self.current_zone = z
        if z is not None and z not in self.visited:
            self.visited.add(z)
            self.get_logger().info(
                f'Visited zone {z} | total {len(self.visited)}/12 | '
                f'{sorted(self.visited)}'
            )

    def _pick_next_zone(self):
        best, best_d = None, math.inf
        for zid in self.zones:
            if zid in self.visited:
                continue
            cx, cy = self._zone_centre(zid)
            d = math.hypot(cx - self.x, cy - self.y)
            if d < best_d:
                best, best_d = zid, d
        return best

    # ---- Replan logic -------------------------------------------------
    def _replan(self, reason: str):
        if self.target_zone is not None:
            self.target_entry = self._zone_entry_point(self.target_zone)
        start = self.grid.world_to_cell(self.x, self.y)
        self.grid.clear_inflation_around(start, radius=3)
        gx, gy = self.target_entry if self.target_entry is not None \
            else self._zone_centre(self.target_zone)
        goal = self.grid.world_to_cell(gx, gy)
        if goal == start:
            self.path = [start]
            self.path_world = [(gx, gy)]
            return
        if self.grid.is_blocked(*goal):
            goal = self._nudge_free(goal, start)
        self.planner.initialize(start, goal)
        self.planner.compute_shortest_path()
        self.path = self.planner.extract_path()
        self.path_world = [self.grid.cell_to_world(*c) for c in self.path]
        self.goal_cell = goal
        self.planner_ready = True
        self.get_logger().info(
            f'Replan ({reason}) -> zone {self.target_zone}: '
            f'{len(self.path)} cells, g(start)={self.planner._get_g(start):.2f}'
        )

    def _nudge_free(self, goal, start):
        gi, gj = goal
        si, sj = start
        for _ in range(20):
            if not self.grid.is_blocked(gi, gj):
                return (gi, gj)
            di = (si - gi)
            dj = (sj - gj)
            sgn_i = 0 if di == 0 else (1 if di > 0 else -1)
            sgn_j = 0 if dj == 0 else (1 if dj > 0 else -1)
            gi += sgn_i
            gj += sgn_j
            if (gi, gj) == start:
                return start
        return (gi, gj)

    def _maintain_planner(self):
        self.grid.update_from_scan(
            self.ranges, self.scan_angle_min, self.scan_angle_inc,
            self.x, self.y, self.yaw, self.scan_range_max, beam_step=4,
        )
        added, removed = self.grid.recompute_inflation()

        start = self.grid.world_to_cell(self.x, self.y)
        peeled = self.grid.clear_inflation_around(start, radius=3)
        if peeled:
            added = added | peeled
            removed = removed - peeled

        if self.target_zone is None or self.target_zone in self.visited:
            nz = self._pick_next_zone()
            target_changed = (nz != self.target_zone)
            self.target_zone = nz
            if nz is not None:
                self.target_arrival_time = None
            else:
                self.target_entry = None
                self.target_arrival_time = None
            if nz is None:
                return
        else:
            target_changed = False

        if target_changed or not self.planner_ready:
            self._replan('new target')
            return

        self.planner.update_start(start)
        if added or removed:
            self.planner.apply_edge_changes(added | removed)
            self.planner.compute_shortest_path()
            self.path = self.planner.extract_path()
            self.path_world = [self.grid.cell_to_world(*c) for c in self.path]
            if self.planner.path_blocked(self.path) or not self.path \
                    or self.path[-1] != self.planner.goal:
                self._replan('path stale')

    # ---- Path following ----------------------------------------------
    def _cone_clearance(self, centre_idx):
        """Min range within a +-30 degree cone centred on `centre_idx`."""
        if not self.ranges:
            return math.inf
        n = len(self.ranges)
        half = max(1, int(round((math.radians(30.0)) / self.scan_angle_inc))) \
            if self.scan_angle_inc > 0 else 15
        best = math.inf
        for k in range(-half, half + 1):
            r = self.ranges[(centre_idx + k) % n]
            if not math.isinf(r) and r < best:
                best = r
        return best

    def _front_clearance(self):
        return self._cone_clearance(0)

    def _rear_clearance(self):
        n = len(self.ranges) if self.ranges else 360
        return self._cone_clearance(n // 2)

    def _lookahead_world(self):
        if not self.path_world:
            return None
        for px, py in self.path_world:
            if math.hypot(px - self.x, py - self.y) >= self.lookahead_dist:
                return (px, py)
        return self.path_world[-1]

    def _follow_path(self):
        start = self.grid.world_to_cell(self.x, self.y)
        g_start = self.planner._get_g(start) if self.planner.start else math.inf
        degenerate = (len(self.path_world) <= 1) or math.isinf(g_start)

        if degenerate and self.target_zone is not None:
            target = self.target_entry if self.target_entry is not None \
                else self._zone_centre(self.target_zone)
            target_source = 'zone_entry'
        else:
            target = self._lookahead_world()
            target_source = 'lookahead'
        if target is None:
            self._diag_log('drift', degenerate, None, None, math.inf,
                           self.slow_speed, 0.0)
            return self._twist(self.slow_speed, 0.0)

        tx, ty = target
        bearing = math.atan2(ty - self.y, tx - self.x)
        herr = self._norm(bearing - self.yaw)
        ang = self._clamp(self.heading_kp * herr,
                          -self.max_angular, self.max_angular)

        clearance = self._front_clearance()
        if clearance < self.stop_clearance:
            if degenerate:
                rear = self._rear_clearance()
                if rear > 0.35:
                    self._diag_log('backup', degenerate, target_source,
                                   herr, clearance, -self.slow_speed, 0.0)
                    return self._twist(-self.slow_speed, 0.0)
            self._diag_log('rot_only', degenerate, target_source, herr,
                           clearance, 0.0, ang)
            return self._twist(0.0, ang)

        if degenerate:
            lin = self.slow_speed if abs(herr) < 0.6 else 0.0
            self._diag_log('degen_creep', degenerate, target_source, herr,
                           clearance, lin, ang)
            return self._twist(lin, ang)

        if clearance < self.slow_clearance or abs(herr) > 0.6:
            lin = self.slow_speed
        else:
            lin = self.forward_speed
        self._diag_log('follow', degenerate, target_source, herr,
                       clearance, lin, ang)
        return self._twist(lin, ang)

    def _diag_log(self, mode, degenerate, target_source, herr, clearance,
                  lin, ang):
        transition = (degenerate != self._was_degenerate)
        self._was_degenerate = degenerate
        if not transition and (self.tick - self._diag_last_tick) < 10:
            return
        self._diag_last_tick = self.tick
        herr_str = '   --' if herr is None else f'{math.degrees(herr):+5.0f}d'
        clr_str = ' inf' if math.isinf(clearance) else f'{clearance:.2f}'
        tag = '*DEGEN*' if degenerate else '       '
        path_len = len(self.path_world)
        self.get_logger().info(
            f'[DIAG] {tag} mode={mode:11s} tgt={self.target_zone} '
            f'pos=({self.x:+.2f},{self.y:+.2f}) yaw={math.degrees(self.yaw):+4.0f}d '
            f'src={target_source} herr={herr_str} clr={clr_str} '
            f'cmd=({lin:+.2f},{ang:+.2f}) path={path_len}'
        )

    # ---- Stuck handling ----------------------------------------------
    def _check_stuck(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.stuck_x is None:
            self.stuck_x, self.stuck_y, self.stuck_t0 = self.x, self.y, now
            return False
        moved = math.hypot(self.x - self.stuck_x, self.y - self.stuck_y)
        if moved > self.stuck_progress_eps:
            self.stuck_x, self.stuck_y, self.stuck_t0 = self.x, self.y, now
            return False
        if now - self.stuck_t0 > self.stuck_timeout:
            return True
        return False

    # ---- Main control loop -------------------------------------------
    def control_loop(self):
        if not self.scan_ready or not self.odom_ready:
            return
        self.tick += 1

        self._update_visited()

        if len(self.visited) >= 12:
            self.cmd_vel_pub.publish(self._twist(0.0, 0.0))
            return

        self._maintain_planner()

        if self.target_zone is None:
            self.cmd_vel_pub.publish(self._twist(0.0, 0.0))
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        if (self.target_arrival_time is None
                and self.target_entry is not None):
            ex, ey = self.target_entry
            if math.hypot(self.x - ex, self.y - ey) <= self.arrival_radius:
                self.target_arrival_time = now
                self.get_logger().info(
                    f'Arrived near zone {self.target_zone}; '
                    f'{self.zone_timeout:.0f}s budget started.'
                )

        if (self.target_arrival_time is not None
                and now - self.target_arrival_time > self.zone_timeout):
            self.get_logger().warn(
                f'Zone {self.target_zone} post-arrival timeout '
                f'({self.zone_timeout:.0f}s); skipping.'
            )
            self.visited.add(self.target_zone)
            self.target_zone = None
            self.target_entry = None
            self.target_arrival_time = None
            self.stuck_x = self.stuck_y = self.stuck_t0 = None
            return

        if self._check_stuck():
            self.get_logger().warn(
                f'Stuck at zone {self.target_zone}; marking visited & skipping.'
            )
            self.visited.add(self.target_zone)
            self.target_zone = None
            self.target_entry = None
            self.target_arrival_time = None
            self.stuck_x = self.stuck_y = self.stuck_t0 = None
            return

        if self.target_zone != self.prev_target:
            self.get_logger().info(f'Target zone -> {self.target_zone}')
            self.prev_target = self.target_zone

        cmd = self._follow_path()
        self.cmd_vel_pub.publish(cmd)

    def on_shutdown(self):
        self.get_logger().info('Stopping robot.')
        self.cmd_vel_pub.publish(TwistStamped())


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = ExploreNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print(f'{node.get_name()} received a shutdown request (Ctrl+C).')
    finally:
        node.on_shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
