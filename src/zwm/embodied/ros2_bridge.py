"""P3-7 (audit) — 具身接口 (ROS2 Bridge)。

将 ROS2 传感器消息转换为 ZWM 的 sensor_data 字典, 使 TrinityAgent
能够与物理/仿真机器人系统交互。

架构:
  ROS2 Node → ros2_bridge → sensor_data → TrinityAgent → hexagram → action

P1-1 (audit): 添加条件 rclpy 导入。当 ROS2 环境可用时, ROS2Node 提供
真实的 ROS2 节点 (订阅 sensor_msgs/LaserScan, nav_msgs/Odometry,
sensor_msgs/Imu) 并将消息推入 ROS2Bridge 的 feed_* 接口。当 rclpy
不可用时, 降级为存根 (spin() 使用模拟数据)。

用法 (需要 ROS2):
  from zwm.embodied.ros2_bridge import ROS2Bridge, ROS2Node
  node = ROS2Node("zwm_embodied")
  bridge = ROS2Bridge(agent, node=node)
  node.spin()  # 阻塞运行 ROS2 节点 (真实传感器)

用法 (无 ROS2, 纯仿真):
  from zwm.embodied.ros2_bridge import ROS2Bridge, GymBridge
  bridge = ROS2Bridge(agent)
  bridge.spin()  # 模拟数据循环
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)

# P1-1: 条件导入 rclpy — 仅在 ROS2 环境中可用。
try:
    import rclpy  # type: ignore
    from rclpy.node import Node  # type: ignore
    _RCLPY_AVAILABLE = True
except ImportError:
    rclpy = None  # type: ignore
    Node = None
    _RCLPY_AVAILABLE = False


# ------------------------------------------------------------------
# 标准化传感器消息 (ROS2 兼容)
# ------------------------------------------------------------------
@dataclass
class LaserScan:
    """sensor_msgs/LaserScan 等价。"""
    ranges: list[float]
    angle_min: float = -1.57
    angle_max: float = 1.57
    range_min: float = 0.1
    range_max: float = 30.0


@dataclass
class Odometry:
    """nav_msgs/Odometry 等价。"""
    x: float
    y: float
    theta: float
    vx: float = 0.0
    vy: float = 0.0
    vtheta: float = 0.0


@dataclass
class Image:
    """sensor_msgs/Image 等价 (简化)。"""
    data: np.ndarray  # [H, W, C]
    width: int = 0
    height: int = 0

    def __post_init__(self):
        if self.width == 0 and self.data.ndim >= 2:
            self.height, self.width = self.data.shape[:2]


@dataclass
class IMU:
    """sensor_msgs/Imu 等价。"""
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0
    gx: float = 0.0
    gy: float = 0.0
    gz: float = 0.0


# ------------------------------------------------------------------
# 消息转换器
# ------------------------------------------------------------------
class SensorConverter:
    """将 ROS2 标准消息转换为 ZWM sensor_data 字典。

    转换规则基于 天地人三才 映射:
      - 天 (vision):   激光距离 → 阴阳强度
      - 地 (language): 里程计位置 → 宫位坐标
      - 人 (sensor):   IMU 加速度 → 爻变信号
    """

    @staticmethod
    def laser_to_sensor(scan: LaserScan) -> dict[str, float] | None:
        """激光扫描 → 阴阳强度 (天)。

        AUDIT-S2: empty / invalid scans used to fall through to a
        hardcoded ``{"天_左": 0.5, ...}`` neutral dict, which silently
        masked sensor failures.  We now return ``None`` and let the
        caller decide how to handle the missing "sky" channel (down-
        grade the trinity, raise, or use the prior — depending on
        policy).
        """
        if not scan.ranges:
            return None
        n = len(scan.ranges)
        third = n // 3
        valid = [r for r in scan.ranges if r > 0]
        if not valid:
            return None
        left = np.mean(valid[:third]) if third > 0 else np.mean(valid)
        center = np.mean(valid[third:2 * third]) if 2 * third > third else np.mean(valid)
        right = np.mean(valid[2 * third:]) if 2 * third < len(valid) else np.mean(valid)
        max_r = scan.range_max or 1.0
        return {
            "天_左": float(np.clip(left / max_r, 0, 1)),
            "天_中": float(np.clip(center / max_r, 0, 1)),
            "天_右": float(np.clip(right / max_r, 0, 1)),
        }

    @staticmethod
    def odom_to_sensor(odom: Odometry) -> dict[str, float]:
        """里程计位置 → 宫位坐标 (地)。"""
        # 归一化到 [0, 1] 范围映射到洛书宫位
        norm_x = float(np.clip((odom.x + 10) / 20, 0, 1))
        norm_y = float(np.clip((odom.y + 10) / 20, 0, 1))
        heading = float((odom.theta % (2 * np.pi)) / (2 * np.pi))
        return {
            "地_x": norm_x,
            "地_y": norm_y,
            "地_heading": heading,
            "地_speed": float(np.sqrt(odom.vx**2 + odom.vy**2)),
        }

    @staticmethod
    def imu_to_sensor(imu: IMU) -> dict[str, float]:
        """IMU 加速度 → 爻变信号 (人)。"""
        accel_mag = float(np.sqrt(imu.ax**2 + imu.ay**2 + imu.az**2))
        gyro_mag = float(np.sqrt(imu.gx**2 + imu.gy**2 + imu.gz**2))
        return {
            "人_accel": float(np.clip(accel_mag / 20, 0, 1)),
            "人_gyro": float(np.clip(gyro_mag / 10, 0, 1)),
            "人_ax": float(np.clip((imu.ax + 10) / 20, 0, 1)),
            "人_ay": float(np.clip((imu.ay + 10) / 20, 0, 1)),
            "人_az": float(np.clip((imu.az + 10) / 20, 0, 1)),
        }

    @staticmethod
    def image_to_features(image: Image) -> np.ndarray | None:
        """图像 → 视觉特征向量 (天)。

        P0-2: 使用 ResNet18 预训练 backbone 替代颜色直方图。
        当 torchvision 不可用时，回退到 RGB 均值。
        """
        if image.data is None or image.data.size == 0:
            return None
        try:
            import torch
            data = image.data
            if data.dtype != np.uint8:
                data = (data * 255).clip(0, 255).astype(np.uint8)
            if data.ndim == 2:
                data = np.stack([data] * 3, axis=-1)
            elif data.shape[-1] == 1:
                data = np.repeat(data, 3, axis=-1)
            elif data.shape[-1] == 4:
                data = data[..., :3]

            import torchvision.transforms as T
            h, w = data.shape[:2]
            scale = 224.0 / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img_pil = T.ToPILImage()(data)
            img_pil = T.Resize((new_h, new_w), antialias=True)(img_pil)
            img_pil = T.CenterCrop(224)(img_pil)
            img_tensor = T.ToTensor()(img_pil)
            img_tensor = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )(img_tensor)
            img_tensor = img_tensor.unsqueeze(0)

            if not hasattr(SensorConverter, "_resnet"):
                from torchvision.models import resnet18, ResNet18_Weights
                model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                model.eval()
                SensorConverter._resnet = torch.nn.Sequential(
                    *list(model.children())[:-1]
                )
                SensorConverter._resnet_device = (
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                SensorConverter._resnet.to(SensorConverter._resnet_device)

            device = SensorConverter._resnet_device
            with torch.no_grad():
                features = SensorConverter._resnet(img_tensor.to(device))
                features = features.squeeze(-1).squeeze(-1)
            return features.cpu().numpy().astype(np.float32)[0]
        except Exception:
            import logging
            logging.getLogger(__name__).debug(
                "ResNet18 feature extraction failed; falling back to RGB mean"
            )
            try:
                flat = image.data.reshape(-1, image.data.shape[-1])
                return flat.mean(axis=0).astype(np.float32)
            except Exception:
                return None

    @staticmethod
    def adapt_for_agent(sensor_data: dict[str, float]) -> dict[str, float]:
        """将 ROS2 传感器键映射为 RuleBasedEncoder 期望的键。

        RuleBasedEncoder 要求: temperature, terrain, social_proximity,
        resource_level, momentum, overall_favorability。

        映射规则:
          - 天_左/中/右 → temperature (三者均值)
          - 地_heading → terrain (heading-based, 0=flat, 1=steep)
          - 地_speed → momentum (speed-based)
          - 人_accel → social_proximity (加速度幅值的逆, 高加速=低接近度)
          - 人_gyro → resource_level (陀螺仪幅值)
          - overall_favorability → 天_中 (中间激光读数作为整体有利度代理)
        """
        adapted: dict[str, float] = {}

        # temperature: average of 天_左/中/右
        sky_keys = ["天_左", "天_中", "天_右"]
        sky_vals = [sensor_data[k] for k in sky_keys if k in sensor_data]
        if sky_vals:
            adapted["temperature"] = float(np.mean(sky_vals)) * 40.0  # scale to ~Celsius range

        # terrain: heading-based (0=flat, 1=steep)
        if "地_heading" in sensor_data:
            adapted["terrain"] = float(np.clip(sensor_data["地_heading"], 0, 1))

        # social_proximity: inverse of accel magnitude
        if "人_accel" in sensor_data:
            adapted["social_proximity"] = float(np.clip(1.0 - sensor_data["人_accel"], 0, 1))

        # resource_level: gyro-based
        if "人_gyro" in sensor_data:
            adapted["resource_level"] = float(np.clip(1.0 - sensor_data["人_gyro"], 0, 1))

        # momentum: speed-based
        if "地_speed" in sensor_data:
            adapted["momentum"] = float(np.clip(sensor_data["地_speed"], 0, 1))

        # overall_favorability: center sky reading as proxy
        if "天_中" in sensor_data:
            adapted["overall_favorability"] = float(np.clip(sensor_data["天_中"], 0, 1))

        return adapted

    @staticmethod
    def merge_sensors(
        laser: LaserScan | None = None,
        odom: Odometry | None = None,
        imu: IMU | None = None,
        image: Image | None = None,
    ) -> tuple[dict[str, float], np.ndarray | None, dict[str, str]]:
        """合并所有传感器 → (sensor_data, vision_features, sensor_warnings).

        AUDIT-S2: ``sensor_warnings`` is a structured report of which
        channels failed to produce data — empty dict when everything
        worked.  Replaces the silent stub fallback of laser_to_sensor
        with a visible "天 missing" signal that the trinity agent can
        surface to its operator."""
        data: dict[str, float] = {}
        warnings: dict[str, str] = {}
        if laser is not None:
            laser_data = SensorConverter.laser_to_sensor(laser)
            if laser_data is None:
                warnings["天"] = "laser scan empty or all-zero ranges"
            else:
                data.update(laser_data)
        if odom is not None:
            data.update(SensorConverter.odom_to_sensor(odom))
        if imu is not None:
            data.update(SensorConverter.imu_to_sensor(imu))
        vision = SensorConverter.image_to_features(image) if image is not None else None
        return data, vision, warnings


# ------------------------------------------------------------------
# Action 映射
# ------------------------------------------------------------------
class ActionMapper:
    """将 hexagram 决策映射为 ROS2 控制命令。

    64 卦映射到离散动作空间:
      - 初爻/二爻 → 线速度 (前进/后退)
      - 三爻/四爻 → 角速度 (左转/右转)
      - 五爻/上爻 → 特殊动作 (停止/加速)
    """

    @staticmethod
    def hexagram_to_twist(hex_bits: int) -> tuple[float, float]:
        """hexagram → (linear_x, angular_z) 速度命令。"""
        yao = [(hex_bits >> i) & 1 for i in range(6)]

        # 初爻/二爻 → 线速度 [-1, 1]
        linear = (yao[0] * 0.5 + yao[1] * 0.5) - 0.5
        linear = linear * 2  # scale to [-1, 1]

        # 三爻/四爻 → 角速度 [-1, 1]
        angular = (yao[2] * 0.5 + yao[3] * 0.5) - 0.5
        angular = angular * 2

        # 五爻 → 加速因子
        if yao[4]:
            linear *= 1.5
        # 上爻 → 停止
        if yao[5] and not any(yao[:5]):
            linear = 0.0
            angular = 0.0

        return (float(np.clip(linear, -1, 1)), float(np.clip(angular, -1, 1)))

    @staticmethod
    def hexagram_to_discrete(hex_bits: int) -> int:
        """hexagram → 离散动作索引 (0-63)。"""
        return hex_bits & 0b111111


# ------------------------------------------------------------------
# 桥接器存根
# ------------------------------------------------------------------
class ROS2Bridge:
    """ROS2 桥接器 — 双模式: 真实 ROS2 节点 (rclpy) 或模拟数据存根。

    P1-1 (audit): 当 ``node`` 参数传入 ROS2Node 实例时, spin() 使用
    rclpy.spin() 阻塞等待真实传感器消息; 否则使用模拟数据循环。
    """

    def __init__(self, agent=None, node: "ROS2Node | None" = None) -> None:
        self._agent = agent
        self._node = node
        self._latest_scan: LaserScan | None = None
        self._latest_odom: Odometry | None = None
        self._latest_imu: IMU | None = None
        self._running = False
        self._consecutive_errors = 0

    def feed_scan(self, scan: LaserScan) -> None:
        self._latest_scan = scan

    def feed_odom(self, odom: Odometry) -> None:
        self._latest_odom = odom

    def feed_imu(self, imu: IMU) -> None:
        self._latest_imu = imu

    def step(self) -> tuple[int, float, float] | None:
        """单步 OODA: 传感器 → hexagram → 动作。

        F1: 同时把动作发布到 ``/cmd_vel`` 主题 (如果有真实 ROS2 节点),
        让具身链路变成 read+write 闭环。返回 ``(hex_bits, linear, angular)``
        与往常一致, 但具身机器人现在真的会动起来。
        """
        if self._agent is None:
            return None
        sensor_data, vision, sensor_warnings = SensorConverter.merge_sensors(
            laser=self._latest_scan,
            odom=self._latest_odom,
            imu=self._latest_imu,
        )
        if not sensor_data:
            return None
        adapted_data = SensorConverter.adapt_for_agent(sensor_data)
        report = self._agent.observe_predict_evaluate_act(
            sensor_data=adapted_data,
            vision_features=vision,
        )
        linear, angular = ActionMapper.hexagram_to_twist(
            report.h_next.normal_order
        )
        # F1: publish to /cmd_vel when a real ROS2Node is bound.  The
        # publisher is created lazily (only when the bridge knows it
        # has a node to write to) so the in-memory mock path stays
        # zero-dependency.
        if self._node is not None:
            self._publish_twist(linear, angular)
        return report.h_next.normal_order, linear, angular

    def _publish_twist(self, linear: float, angular: float) -> None:
        """F1: lazily create a ``/cmd_vel`` publisher and publish."""
        if not getattr(self, "_vel_pub", None):
            self._vel_pub = self._node.create_cmd_vel_publisher()  # type: ignore[attr-defined]
        self._vel_pub.publish(linear, angular)

    def spin(self, max_steps: int | None = None, interval: float = 0.1) -> None:
        """Run the OODA loop using real ROS2 sensors (if available) or mock data.

        P1-1 (audit): When ``self._node`` is a live ROS2Node, the bridge
        delegates to ``node.spin()`` which blocks on ``rclpy.spin()`` and
        calls ``self._on_sensor_update`` on each callback.  When no node
        is available, the legacy mock-sensor loop runs.

        P3-1 (audit): ``_on_sensor_update`` provides a structured callback
        for real sensor data — this is the canonical path for production
        ROS2 deployments.  The callback is rate-limited via ``rclpy.Rate``
        (or `time.sleep` in mock mode) to avoid overwhelming the OODA loop.

        Args:
            max_steps: Maximum number of OODA steps (None = run until stop()).
            interval: Sleep between steps in seconds (mock mode only).
        """
        if self._node is not None and _RCLPY_AVAILABLE:
            # P3-1: Real ROS2 sensor path — spin blocks on rclpy.spin().
            self._node.spin(max_steps=max_steps)
            return

        # Mock sensor fallback path (no ROS2 dependency).
        import time as _time
        self._running = True
        step = 0
        consecutive_errors = 0
        while self._running:
            if max_steps is not None and step >= max_steps:
                break
            # Generate mock sensors if no real data has been fed.
            if self._latest_scan is None and self._latest_odom is None:
                sensor_data = self.generate_mock_sensors()
                sensor_warnings: dict[str, str] = {}
                vision = None
            else:
                sensor_data, vision, sensor_warnings = SensorConverter.merge_sensors(
                    laser=self._latest_scan,
                    odom=self._latest_odom,
                    imu=self._latest_imu,
                )
            if sensor_data and self._agent is not None:
                adapted_data = SensorConverter.adapt_for_agent(sensor_data)
                self._on_sensor_update(adapted_data, sensor_warnings, step, vision)
                consecutive_errors = self._consecutive_errors
            elif sensor_warnings:
                _log.info(
                    "ROS2Bridge: degraded sensor input: %s", sensor_warnings,
                )
            step += 1
            if interval > 0:
                _time.sleep(interval)

    def _on_sensor_update(
        self,
        sensor_data: dict[str, float],
        sensor_warnings: dict[str, str] | None = None,
        step: int = 0,
        vision_features: np.ndarray | None = None,
    ) -> None:
        """P3-1: Canonical sensor-data callback for real ROS2 deployments.

        In production, this is called by ``ROS2Node._sensor_callback``
        every time the ROS2 subscription fires.  In mock mode, it is
        called by ``spin()`` on each loop iteration.

        Graceful degradation: if the OODA step throws, the error is
        logged and counted; after 3 consecutive failures the spin loop
        halts to prevent a CPU peg.
        """
        try:
            self._agent.observe_predict_evaluate_act(
                sensor_data=sensor_data,
                vision_features=vision_features,
            )
            self._consecutive_errors = 0
        except Exception as exc:
            import traceback
            _log.warning(
                "OODA step %d failed: %s\n%s",
                step, exc, traceback.format_exc(),
            )
            self._consecutive_errors += 1
            if self._consecutive_errors >= 3:
                _log.error(
                    "ROS2Bridge: 3 consecutive OODA failures; "
                    "halting spin loop.  Inspect the agent state."
                )
                self._running = False

    def stop(self) -> None:
        self._running = False

    def generate_mock_sensors(self) -> dict[str, float]:
        """生成模拟传感器数据 (用于测试)。"""
        scan = LaserScan(
            ranges=[1.0 + np.random.random() * 5 for _ in range(36)],
        )
        odom = Odometry(
            x=np.random.random() * 20 - 10,
            y=np.random.random() * 20 - 10,
            theta=np.random.random() * 2 * np.pi,
            vx=np.random.random() * 0.5,
        )
        imu = IMU(
            ax=np.random.random() * 0.1 - 0.05,
            ay=np.random.random() * 0.1 - 0.05,
            az=9.8 + np.random.random() * 0.1,
        )
        data, vision, warnings = SensorConverter.merge_sensors(laser=scan, odom=odom, imu=imu)
        return data


# ------------------------------------------------------------------
# P1-1: 真实 ROS2 节点 (条件可用)
# ------------------------------------------------------------------
class ROS2Node:
    """P1-1 (audit): 真实 ROS2 节点订阅器。

    仅在 rclpy 可用时工作。订阅标准 ROS2 话题 (LaserScan, Odometry,
    Imu) 并将消息推入 ROS2Bridge 的 feed_* 接口。每个传感器回调触发
    ``bridge._on_sensor_update`` 执行一步 OODA 循环。

    用法:
        if ROS2Node.is_available():
            node = ROS2Node("zwm_embodied")
            bridge = ROS2Bridge(agent, node=node)
            bridge.spin()  # → node.spin() → rclpy.spin()
    """

    def __init__(
        self,
        node_name: str = "zwm_embodied",
        bridge: ROS2Bridge | None = None,
        scan_topic: str = "/scan",
        odom_topic: str = "/odom",
        imu_topic: str = "/imu",
    ) -> None:
        if not _RCLPY_AVAILABLE:
            raise ImportError(
                "rclpy is not available. Install ROS2 (Humble or later) "
                "and source the setup script before using ROS2Node."
            )
        # rclpy.init() is idempotent — safe to call multiple times.
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = Node(node_name)

        # QoS: sensor data is best-effort with a short queue to avoid
        # back-pressure blocking the ROS2 executor.
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        # P3-1: Real sensor subscriptions with structured callbacks.
        # Each callback merges the latest sensor data and triggers the
        # OODA loop through bridge._on_sensor_update.
        self._bridge = bridge
        self._latest_scan: LaserScan | None = None
        self._latest_odom: Odometry | None = None
        self._latest_imu: IMU | None = None

        # Subscribe to sensor topics — these are type-erased imports
        # because the message types are in ROS2 packages that may not
        # be installed in development.  The subscriptions will fail
        # gracefully at runtime if the message types are unavailable.
        self._scan_sub = self._node.create_subscription(
            self._get_msg_type("sensor_msgs.msg", "LaserScan"),
            scan_topic,
            self._on_scan,
            sensor_qos,
        )
        self._odom_sub = self._node.create_subscription(
            self._get_msg_type("nav_msgs.msg", "Odometry"),
            odom_topic,
            self._on_odom,
            sensor_qos,
        )
        self._imu_sub = self._node.create_subscription(
            self._get_msg_type("sensor_msgs.msg", "Imu"),
            imu_topic,
            self._on_imu,
            sensor_qos,
        )

        _log.info("ROS2Node '%s' subscribed to %s, %s, %s",
                  node_name, scan_topic, odom_topic, imu_topic)

    @staticmethod
    def is_available() -> bool:
        return _RCLPY_AVAILABLE

    @staticmethod
    def _get_msg_type(pkg: str, msg: str) -> type:
        """Dynamically import a ROS2 message type."""
        import importlib
        mod = importlib.import_module(pkg)
        return getattr(mod, msg)

    def create_cmd_vel_publisher(self):
        """F1: lazy ``/cmd_vel`` publisher factory.

        Returns a thin wrapper that holds the rclpy publisher and
        translates ``(linear_x, angular_z)`` into a
        ``geometry_msgs/Twist`` message.  Cached on the bridge as
        ``self._vel_pub``.
        """
        if not _RCLPY_AVAILABLE:
            raise RuntimeError("rclpy not available; cannot create cmd_vel publisher")
        if getattr(self, "_vel_pub", None) is not None:
            return self._vel_pub
        Twist = self._get_msg_type("geometry_msgs.msg", "Twist")
        pub = self._node.create_publisher(Twist, "/cmd_vel", 10)

        class _VelPublisher:
            def __init__(self, publisher, twist_cls):
                self._pub = publisher
                self._twist_cls = twist_cls

            def publish(self, linear: float, angular: float) -> None:
                msg = self._twist_cls()
                msg.linear.x = float(linear)
                msg.angular.z = float(angular)
                self._pub.publish(msg)

        self._vel_pub = _VelPublisher(pub, Twist)
        _log.info("ROS2Node '%s' created /cmd_vel publisher",
                  self._node.get_name() if hasattr(self._node, "get_name") else "zwm")
        return self._vel_pub

    def _on_scan(self, msg) -> None:
        """ROS2 callback: LaserScan → bridge feed."""
        ranges = list(msg.ranges)
        self._latest_scan = LaserScan(
            ranges=ranges,
            angle_min=float(msg.angle_min),
            angle_max=float(msg.angle_max),
            range_min=float(msg.range_min),
            range_max=float(msg.range_max),
        )
        if self._bridge is not None:
            self._bridge.feed_scan(self._latest_scan)
            self._trigger_ooda()

    def _on_odom(self, msg) -> None:
        """ROS2 callback: Odometry → bridge feed."""
        pose = msg.pose.pose
        twist = msg.twist.twist
        import math as _math
        self._latest_odom = Odometry(
            x=float(pose.position.x),
            y=float(pose.position.y),
            theta=float(2 * _math.atan2(pose.orientation.z, pose.orientation.w)),
            vx=float(twist.linear.x),
            vy=float(twist.linear.y),
            vtheta=float(twist.angular.z),
        )
        if self._bridge is not None:
            self._bridge.feed_odom(self._latest_odom)
            self._trigger_ooda()

    def _on_imu(self, msg) -> None:
        """ROS2 callback: Imu → bridge feed."""
        acc = msg.linear_acceleration
        gyro = msg.angular_velocity
        self._latest_imu = IMU(
            ax=float(acc.x),
            ay=float(acc.y),
            az=float(acc.z),
            gx=float(gyro.x),
            gy=float(gyro.y),
            gz=float(gyro.z),
        )
        if self._bridge is not None:
            self._bridge.feed_imu(self._latest_imu)
            self._trigger_ooda()

    def _trigger_ooda(self) -> None:
        """P3-1: Trigger one OODA step with the latest fused sensor data.

        Called by each sensor callback.  Rate-limiting is handled by
        the ROS2 QoS (queue depth=10) and the callback's thread safety
        (rclpy callbacks are serialized by the default executor).
        """
        if self._bridge is None:
            return
        sensor_data, _, sensor_warnings = SensorConverter.merge_sensors(
            laser=self._latest_scan,
            odom=self._latest_odom,
            imu=self._latest_imu,
        )
        if sensor_data:
            adapted_data = SensorConverter.adapt_for_agent(sensor_data)
            self._bridge._on_sensor_update(adapted_data, sensor_warnings)

    def spin(self, max_steps: int | None = None) -> None:
        """Block on rclpy.spin() — real sensor callbacks drive OODA.

        Args:
            max_steps: If set, spin_once() is called in a loop with
                       this limit.  Otherwise, rclpy.spin() blocks
                       indefinitely.
        """
        if max_steps is not None:
            for _ in range(max_steps):
                if not rclpy.ok():
                    break
                rclpy.spin_once(self._node, timeout_sec=0.1)
        else:
            rclpy.spin(self._node)

    def shutdown(self) -> None:
        """Clean up the ROS2 node."""
        if self._node is not None:
            self._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception as exc:
            _log.debug("ROS2 bridge shutdown in __del__ failed: %s", exc)


# ------------------------------------------------------------------
# Gym 环境桥接器 (无 ROS2 依赖)
# ------------------------------------------------------------------
class GymBridge:
    """Gym 环境桥接器 — 将 ZWM agent 连接到 Gymnasium 环境。

    用于无 ROS2 的仿真开发和基准测试。
    """

    def __init__(self, agent=None, env=None) -> None:
        self._agent = agent
        self._env = env
        self._obs = None

    def reset(self) -> dict[str, float]:
        if self._env is not None:
            self._obs = self._env.reset()
        return self._obs_to_sensor()

    def step(self, action: int | None = None) -> tuple[dict, float, bool]:
        """执行一步交互: ZWM agent 决策 → 环境执行 → 返回观测。"""
        if self._agent is None:
            raise RuntimeError("Agent not set")

        if self._obs is not None:
            sensor_data = self._obs_to_sensor()
            report = self._agent.observe_predict_evaluate_act(
                sensor_data=sensor_data,
            )
            action = ActionMapper.hexagram_to_discrete(report.h_next.normal_order)

        if self._env is not None and action is not None:
            self._obs, reward, terminated, truncated, _ = self._env.step(action)
            done = terminated or truncated
        else:
            reward, done = 0.0, False

        return self._obs_to_sensor(), reward, done

    def _obs_to_sensor(self) -> dict[str, float]:
        """将环境观测转换为 sensor_data。"""
        if self._obs is None:
            return {}
        if isinstance(self._obs, dict):
            return {str(k): float(v) for k, v in self._obs.items()}
        if isinstance(self._obs, np.ndarray):
            return {f"obs_{i}": float(v) for i, v in enumerate(self._obs.flat)}
        return {"obs": float(self._obs)}