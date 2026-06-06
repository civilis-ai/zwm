"""ZWM 具身接口 (P3-7 audit)。

提供 ROS2 Bridge + Gym 环境桥接器, 将 TrinityAgent 连接到
物理/仿真机器人系统。

⚠️ AUDIT-A1 — opt-in only
==========================
This subpackage is **not** wired into ``zwm.planner`` by default.  The
``ROS2Bridge`` / ``GymBridge`` / ``ActionMapper`` are real, but they are
not part of the OODA critical path and the imports below were silently
succeeding without ever being exercised.  Importing this module
*no longer* pulls ``rclpy`` / ``gym`` into the runtime — those
dependencies are imported lazily on first instantiation of the bridge.

If you want to use the embodied layer, you must explicitly import it::

    from zwm.embodied import ROS2Bridge, GymBridge  # noqa: F401
    bridge = ROS2Bridge(agent=my_agent, ros2_node=my_node)

This keeps the canonical ``import zwm`` lightweight and prevents the
silent stub surface from masquerading as a finished feature.
"""
from __future__ import annotations

__all__ = [
    "ActionMapper",
    "GymBridge",
    "IMU",
    "Image",
    "LaserScan",
    "Odometry",
    "ROS2Bridge",
    "SensorConverter",
]


def __getattr__(name: str):
    """PEP 562 lazy import — avoid pulling rclpy / gym at import time.

    AUDIT-A1: previously the module eagerly imported rclpy / gym
    stubs at the top level, which (a) made the standard ``import zwm``
    payload heavier, and (b) implied the embodied layer was fully
    integrated with the agent when in fact nothing in ``zwm.planner``
    instantiates ``ROS2Bridge`` or ``GymBridge``.
    """
    if name in __all__:
        from . import ros2_bridge
        return getattr(ros2_bridge, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")