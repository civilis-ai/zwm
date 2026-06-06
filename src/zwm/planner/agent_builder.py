"""P2-arch — AgentBuilder: 将 TrinityAgent 的子系统初始化提取为独立构造器。

解决 audit 报告中"God object"问题: TrinityAgent.__init__ 导入了 16 个
子模块并直接在构造函数中组装所有子系统。AgentBuilder 将每个子系统
的构造拆分为独立方法, 支持:
  * 依赖注入 (测试时替换子系统)
  * 延迟构造 (仅在使用时创建)
  * 构造顺序保证 (内存/文件句柄在构造失败时不会泄漏)

用法::

    builder = AgentBuilder(config)
    builder.build_planning()
    builder.build_perception()
    ...
    agent = builder.agent  # 完全组装的 TrinityAgent
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zwm.planner.agent import TrinityAgent
    from zwm.planner.agent_config import TrinityConfig


@dataclass
class AgentBuilder:
    """延迟构造 TrinityAgent 子系统。

    每个 ``build_*`` 方法初始化一个子系统并附加到 ``self._agent``。
    ``build_all()`` 按依赖顺序调用所有 build_* 方法, 返回完全组装的 agent。

    构造顺序保证: planner → perception → learning → vsa → world_model
    → particle_filter → topology → memory/checkpoint。内存/文件句柄最后
    打开, 确保前面的组件构造失败时不会泄漏 OS 资源。
    """

    config: "TrinityConfig"
    _agent: "TrinityAgent | None" = None

    # ------------------------------------------------------------------
    def with_agent(self, agent: "TrinityAgent") -> "AgentBuilder":
        """绑定一个已存在 (可能是部分构造) 的 agent 实例。"""
        self._agent = agent
        return self

    @property
    def agent(self) -> "TrinityAgent":
        if self._agent is None:
            raise RuntimeError("Agent not built — call build_all() or set with_agent() first")
        return self._agent

    # ------------------------------------------------------------------
    def build_all(self) -> "TrinityAgent":
        """按依赖顺序构建所有子系统, 返回完全组装的 agent。"""
        from zwm.planner.agent import TrinityAgent
        if self._agent is None:
            raise RuntimeError("call with_agent() before build_all()")
        agent = self._agent
        self.build_planning()
        self.build_perception()
        self.build_learning_systems()
        self.build_vsa()
        self.build_world_model()
        self.build_particle_filter()
        self.build_topology()
        self.build_constitution()
        self.build_memory_and_checkpoint()
        self.build_react()
        self.build_observability()
        return agent

    # ------------------------------------------------------------------
    def build_planning(self) -> None:
        """Stateless MCTS + MoE evaluator."""
        from zwm.planner.loop import TrinityPlanner
        self._agent.planner = TrinityPlanner(
            mcts_iterations=self.config.mcts_iterations,
            use_diffusion=self.config.use_diffusion,
        )

    def build_perception(self) -> None:
        """Sensor / time / multimodal encoders."""
        from zwm.encoder.base import RuleBasedEncoder
        from zwm.scene_field.calendar import GanzhiTime, MultiScaleCalendar
        self._agent.encoder = RuleBasedEncoder()
        self._agent.calendar = MultiScaleCalendar()
        self._agent.ganzhi = GanzhiTime()
        self._agent._multimodal = None  # lazy-created on first multimodal input

    def build_learning_systems(self) -> None:
        """Persistent learners: online DPO, curiosity, growth, Hebbian."""
        from zwm.learning.hebbian import HebbianAssociator
        from zwm.learning.online import CuriosityScheduler, GrowthManager, OnlineLearner
        self._agent.learner = OnlineLearner()
        self._agent.curiosity = CuriosityScheduler()
        self._agent.growth = GrowthManager()
        self._agent.hebbian = HebbianAssociator()

    def build_vsa(self) -> None:
        """In-memory VSA codebook + persistent long-term memory buffer."""
        from zwm.hexaembed.vsa import TrainableVSACodebook, VSACodebook, VSAMemoryBuffer
        self._agent.vsa = (
            TrainableVSACodebook() if self.config.use_trainable_vsa else VSACodebook()
        )
        db_dir = os.path.dirname(self.config.db_path) or "."
        if db_dir == "." and self.config.db_path == ":memory:":
            persist_dir = os.environ.get(
                "ZWM_DATA_DIR",
                os.path.join(os.environ.get("TEMP", os.getcwd()), "zwm_vsa"),
            )
            os.makedirs(persist_dir, exist_ok=True)
        else:
            persist_dir = os.environ.get("ZWM_DATA_DIR", db_dir)
        self._agent.vsa_buffer = VSAMemoryBuffer(
            persist_path=os.path.join(persist_dir, "vsa_consolidated.npz"),
        )
        self._agent.vsa_buffer.load_persisted()

    def build_world_model(self) -> None:
        """JEPA predictor + SquareCircularJoint + VQ-VAE + value head + FSDP2 + quant."""
        from zwm.core.constants import Z_WORLD_DIM
        from zwm.jepa.predictor import HierarchicalJEPAPredictor, JEPAPredictor
        from zwm.jepa.square_encoder import FixedWeightSquareGNN, LearnableSquareGNN, SquareCircularJoint

        learnable = self.config.learnable_encoder
        hierarchical = self.config.hierarchical

        if learnable:
            self._agent._square_learnable = LearnableSquareGNN()
            self._agent._square_fixed = FixedWeightSquareGNN()
            self._agent.square = self._agent._square_learnable
        else:
            self._agent._square_learnable = None
            self._agent._square_fixed = FixedWeightSquareGNN()
            self._agent.square = self._agent._square_fixed

        self._agent.joint = SquareCircularJoint(self._agent.square)

        if hierarchical:
            self._agent.jepa = HierarchicalJEPAPredictor(input_dim=Z_WORLD_DIM)
        else:
            self._agent.jepa = JEPAPredictor(input_dim=Z_WORLD_DIM)
        if self._agent._square_learnable is not None:
            self._agent.jepa.attach_square_encoder(self._agent._square_learnable)

        # MuZero-style latent V(s) head.
        try:
            self._agent.jepa.init_value_head()
        except Exception as exc:
            _log.warning("init_value_head failed: %s", exc)

        # VQ-VAE discretisation.
        try:
            self._agent.jepa.init_vq(num_codes=64, beta=0.25)
        except Exception as exc:
            _log.warning("init_vq failed: %s", exc)

        # FSDP2 multi-GPU.
        if self.config.use_fsdp2:
            try:
                from zwm.jepa.distributed import wrap_fsdp2, wrap_fsdp2_hierarchical
                self._agent.jepa = (
                    wrap_fsdp2_hierarchical(self._agent.jepa)
                    if hierarchical
                    else wrap_fsdp2(self._agent.jepa)
                )
            except Exception as exc:
                _log.warning("FSDP2 wrap failed (single-GPU is fine): %s", exc)

        # Quantization.
        q = self.config.quantize
        if q in ("4bit", "qlora"):
            result = self._agent.jepa.quantize_4bit()
            if "error" in result:
                _log.warning("4-bit quantization skipped: %s", result["error"])
            else:
                _log.info("4-bit quantized %d layers", len(result))
        if q in ("lora", "qlora"):
            adapted = self._agent.jepa.apply_lora(rank=8, alpha=16.0)
            _log.info("LoRA adapted %d layers", len(adapted))

    def build_particle_filter(self) -> None:
        """Particle filter for ensemble EFE in JEPA latent space."""
        from zwm.core.constants import LATENT_DIM
        from zwm.self_field.particle_filter import ParticleFilter
        n_particles = self.config.n_particles
        self._agent._n_particles = n_particles
        if n_particles > 0:
            self._agent._particle_filter = ParticleFilter(
                n_particles=n_particles, dim=LATENT_DIM, noise_std=0.05, obs_std=0.1,
            )
        else:
            self._agent._particle_filter = None

    def build_topology(self) -> None:
        """Multi-scale palace scaffold + per-palace visit counter."""
        from zwm.topology.recursive import RecursiveTopology, expand_topology
        depth = self.config.topology_max_depth
        self._agent.topology = expand_topology(max_depth=depth)
        self._agent._palace_visits = {}

    def build_constitution(self) -> None:
        """Install the constitutional safety guardrail before any data-emitting subsystem."""
        from zwm.safety.constitution import ConstitutionalGuard
        self._agent.constitution = ConstitutionalGuard(
            enabled=self.config.enable_constitution,
        )

    def build_memory_and_checkpoint(self) -> None:
        """Episodic + semantic stores — opened LAST to prevent handle leaks."""
        from zwm.self_field.palace_graph import LuoshuGrid
        from zwm.storage.episodic_db import EpisodicStore, SemanticStore
        from zwm.learning.checkpoint import load_checkpoint

        self._agent.grid = (
            self.config.grid if self.config.grid is not None else LuoshuGrid()
        )
        self._agent.store = EpisodicStore(db_path=self.config.db_path)
        try:
            self._agent.semantic = (
                SemanticStore(file_path=self.config.semantic_path)
                if self.config.semantic_path
                else None
            )
        except Exception as exc:
            _log.error("SemanticStore init failed: %s", exc)
            self._agent.store.close()
            raise

        if self.config.checkpoint_path is not None:
            try:
                load_checkpoint(self._agent, self.config.checkpoint_path)
            except Exception as exc:
                _log.warning("Checkpoint restore failed (starting fresh): %s", exc)

    def build_react(self) -> None:
        """ReAct / Tool-Use loop (2026 SOTA agent architecture)."""
        self._agent._react_loop = None
        if self.config.use_react:
            try:
                from zwm.planner.react import ReActLoop
                self._agent._react_loop = ReActLoop(self._agent, max_steps=3)
            except Exception as exc:
                _log.warning("ReAct loop init failed: %s", exc)

    def build_observability(self) -> None:
        """Auto-init OTLP / OTel tracing."""
        try:
            from zwm.tracing import configure_otlp_from_env, get_tracer
            configure_otlp_from_env()
            get_tracer()
        except Exception as exc:
            _log.debug("auto-init observability skipped: %s", exc)
