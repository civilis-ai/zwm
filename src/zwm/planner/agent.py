"""TrinityAgent — the persistent OODA orchestrator.

This is the object the WIRING_PLAN calls for: a single owner of cross-tick
learning state that closes the Observe → Predict → Evaluate → Act → Learn loop
and feeds every previously-orphaned subsystem real data.

  Observe   sensor_data -> RuleBasedEncoder -> hexagram; calendar -> time_phase
  Predict   UnifiedField -> SquareCircularJoint -> z_world -> JEPAPredictor
  Evaluate  TrinityPlanner.plan (MCTS + live EFE + preference-biased MoE),
            warm-started by Hebbian + episodic memory priors.
            P3: ParticleFilter provides ensemble EFE for robust uncertainty.
            P3: Learned JEPA value head V(z) replaces EMA table bootstrap.
  Act       apply the top mutation -> next hexagram
  Learn     EpisodicStore + VSA memory, OnlineLearner preference feedback,
            MoE router gradient step, Hebbian association update,
            JEPA latent-prediction training (real backprop).
            P3: Codon-amino-acid features enrich episode context.
            P3: Checkpoint save/load persists all learning state.

The planner stays a stateless evaluator; all mutable state lives here.

P2-1 (audit): this file is intentionally slim — it only owns:
  * __init__ (subsystem wiring)
  * __enter__/__exit__/close (lifecycle)
  * tick / observe_predict_evaluate_act (OODA dispatcher)
P4-6 (audit): static configuration now lives in
  :class:`zwm.planner.agent_config.TrinityConfig` — a frozen dataclass
  that the CLI / API / MCP layers can introspect.
Heavy lifting lives in:
  * agent_data.py     — TickReport, TickPrediction, GOOD_OUTCOME
  * agent_config.py   — TrinityConfig (NEW)
  * agent_priors.py   — memory_priors, _combined_priors, _world_vector, _calendar_context
  * agent_phases.py   — _observe / _predict / _evaluate / _act / _learn
  * agent_train.py    — _joint_train_step, _train_jepa, _reinforce_router, …
"""
from __future__ import annotations

import math
import logging
import os
import time

import numpy as np

from zwm.core.hexagram import Hexagram
from zwm.core.constants import Z_WORLD_DIM, LATENT_DIM
from zwm.encoder.base import RuleBasedEncoder
from zwm.encoder.multimodal import MultimodalEncoder
from zwm.encoder.multimodal import LanguageBackbone
from zwm.hexaembed.vsa import TrainableVSACodebook, VSACodebook, VSAMemoryBuffer
from zwm.jepa.predictor import HierarchicalJEPAPredictor, JEPAPredictor
from zwm.jepa.square_encoder import FixedWeightSquareGNN, SquareCircularJoint
from zwm.learning.checkpoint import load_checkpoint, save_checkpoint
from zwm.learning.hebbian import HebbianAssociator
from zwm.learning.online import CuriosityScheduler, GrowthManager, OnlineLearner
from zwm.planner.agent_config import TrinityConfig
from zwm.planner.agent_priors import memory_priors
from zwm.planner.agent_data import GOOD_OUTCOME, TickPrediction, TickReport
from zwm.planner.agent_phases import _act, _evaluate, _learn, _observe, _predict
from zwm.planner.loop import PlanResult, TrinityPlanner
from zwm.scene_field.calendar import GanzhiTime, MultiScaleCalendar
from zwm.self_field.palace_graph import LuoshuGrid
from zwm.self_field.particle_filter import ParticleFilter
from zwm.storage.episodic_db import EpisodicStore, SemanticStore
from zwm.topology.recursive import RecursiveTopology, expand_topology

_log = logging.getLogger(__name__)

# Re-export the data contract for backward compat.
__all__ = ["TrinityAgent", "TrinityConfig", "TickReport", "TickPrediction", "GOOD_OUTCOME"]


class TrinityAgent:
    """Owns persistent learning state and runs the closed OODA loop.

    P4-6 (audit): static configuration is held in ``self.config`` (a
    :class:`TrinityConfig` dataclass).  ``self._cfg`` is gone.

    P2-arch: for dependency-injected construction (testable, swappable
    subsystems), use :class:`zwm.planner.agent_builder.AgentBuilder`::

        from zwm.planner.agent_builder import AgentBuilder
        builder = AgentBuilder(config).with_agent(TrinityAgent.__new__(TrinityAgent))
        agent = builder.build_all()

    The direct ``TrinityAgent(config=...)`` constructor remains the
    primary path for CLI/API/MCP usage.
    """

    def __init__(
        self,
        config: TrinityConfig | None = None,
        # Backwards-compat kwargs — kept so legacy callers continue to work.
        # Each one is a shadow of a ``TrinityConfig`` field; if ``config`` is
        # provided they are *ignored*.
        db_path: str = "zwm_episodes.db",
        semantic_path: str | None = None,
        checkpoint_path: str | None = None,
        mcts_iterations: int = 200,
        grid: LuoshuGrid | None = None,
        learnable_encoder: bool = True,
        use_trainable_vsa: bool = True,
        hierarchical: bool = False,
        n_particles: int = 16,
        use_diffusion: bool = True,
        use_fsdp2: bool = False,
        use_react: bool = True,
        quantize: str | None = None,
    ) -> None:
        # Build the config dataclass — either the caller's or ours.
        if config is not None:
            self.config = config
        else:
            self.config = TrinityConfig(
                db_path=db_path,
                semantic_path=semantic_path,
                checkpoint_path=checkpoint_path,
                mcts_iterations=mcts_iterations,
                grid=grid,
                learnable_encoder=learnable_encoder,
                use_trainable_vsa=use_trainable_vsa,
                hierarchical=hierarchical,
                n_particles=n_particles,
                use_diffusion=use_diffusion,
                use_fsdp2=use_fsdp2,
                use_react=use_react,
                quantize=quantize,
            )

        # P4-8 (audit): install the constitutional safety guardrail
        # *before* any subsystem that might emit data (planner, encoder,
        # …).  Every input and output of the OODA loop will pass
        # through ``self.constitution.check_input`` / ``check_output``.
        from zwm.safety.constitution import ConstitutionalGuard
        self.constitution: ConstitutionalGuard = ConstitutionalGuard(
            enabled=self.config.enable_constitution,
        )

        # LLM-as-Judge safety layer — additional check alongside the
        # constitutional guard.  Only created when API keys are
        # available; otherwise ``None`` and the check is skipped.
        self._llm_judge = None
        self._init_safety()

        # ── "我" — 第一人称定位锚点 (必须在一切子系统之前) ──
        # 部署时设定日干, agent 的所有感知/预测/行动以此为中心
        self._init_self()

        # Build subsystems in dependency order:
        #   planner → perception → learners → vsa → world model
        #   → particle filter → metrics → topology → memory/checkpoint.
        # Memory (SQLite) is opened LAST so a failure in the heavy
        # components above does not leak an open handle.
        self._init_planning()
        self._init_perception()
        self._init_learning_systems()
        self._init_vsa()
        self._init_world_model()
        self._init_particle_filter()
        self._init_metrics()
        self._init_topology()
        self._init_memory_and_checkpoint()

        # F8: auto-init observability so the agent emits OTLP / OTel
        # spans on first use without requiring the host to call
        # ``init_observability()`` explicitly.  ``configure_otlp_from_env``
        # auto-detects ``OTEL_EXPORTER_OTLP_ENDPOINT`` and binds the
        # batch span processor; ``get_tracer()`` is a no-op when the
        # SDK is absent and returns the in-process tracer.  Silently
        # downgrades when neither is available (e.g. unit tests
        # that monkey-patch tracing out).
        try:
            from zwm.tracing import configure_otlp, configure_otlp_from_env, get_tracer
            if self.config.enable_otlp:
                configure_otlp(
                    endpoint=self.config.otlp_endpoint,
                    service_name=self.config.otlp_service_name,
                )
            else:
                configure_otlp_from_env()
            get_tracer()  # ensure the singleton is built
        except Exception as exc:  # pragma: no cover — defensive
            _log.debug("auto-init observability skipped: %s", exc)

    # ------------------------------------------------------------------
    # Component initialisers — split from __init__ for maintainability
    # ------------------------------------------------------------------
    def _init_self(self) -> None:
        """P0 — "我"的初始化: 永远在中宫的第一人称锚点.

        日干由部署时环境变量 ZWM_DAY_GAN 设定 (默认 "甲"=木).
        "我"永远在中宫(5). 八方六亲由日干五行固定.
        """
        import os as _os
        day_gan = _os.environ.get("ZWM_DAY_GAN", "甲")
        from zwm.self_field.self_state import SelfState
        self._self_state = SelfState(day_gan=day_gan)
        _log.info("SelfState: 日%s·%s @中宫, 六亲=%s",
                  day_gan, self._self_state.self_element,
                  {k: v for k, v in self._self_state.six_relations.items()
                   if v != "兄弟"})

    @property
    def self_state(self) -> "SelfState":
        """P0 — agent 的"我"."""
        return self._self_state

    def _init_safety(self) -> None:
        """Initialise the LLM-as-Judge safety layer.

        Creates an :class:`LLMJudgeRule` via :func:`make_auto_judge`
        when at least one LLM API key is available.  The judge is
        stored as ``self._llm_judge`` and used as an additional
        safety check alongside the constitutional guard in
        ``check_output``.
        """
        try:
            from zwm.safety.llm_judge import LLMJudgeRule, make_auto_judge
            from zwm.safety.constitution import Severity
            judge_fn = make_auto_judge()
            # make_auto_judge returns _noop_judge when no API key is found;
            # only wire the rule when we got a real judge.
            if judge_fn.__name__ != "_noop_judge":
                self._llm_judge = LLMJudgeRule(
                    name="llm-output-judge",
                    judge_fn=judge_fn,
                    severity=Severity.WARN,
                )
                _log.info("LLM judge safety layer initialised")
            else:
                _log.debug("No LLM API key found; LLM judge safety layer disabled")
        except Exception as exc:
            _log.debug("LLM judge init skipped: %s", exc)

    def _init_planning(self) -> None:
        """Stateless MCTS + MoE evaluator."""
        self.planner = TrinityPlanner(
            mcts_iterations=self.config.mcts_iterations,
            use_diffusion=self.config.use_diffusion,
        )

    def _init_perception(self) -> None:
        """Sensor / time / multimodal encoders (天 — visual, 地 — temporal).

        Field encoder: 传感器→64卦场 (非单卦), 384-bit 状态场.
        当 use_field_encoder=True 时, OODA 使用卦象场而非单卦。
        """
        self.encoder = RuleBasedEncoder()
        # HexagramFieldEncoder — 将传感器编码为 64 卦 × 6 爻场
        self.field_encoder = None
        if getattr(self.config, "use_field_encoder", True):
            try:
                from zwm.encoder.field_encoder import HexagramFieldEncoder
                self.field_encoder = HexagramFieldEncoder(strategy="adaptive")
                _log.info("Field encoder: HexagramFieldEncoder (strategy=%s)",
                          self.field_encoder.strategy)
            except Exception as exc:
                _log.debug("Field encoder init skipped: %s", exc)

        self.calendar = MultiScaleCalendar()
        # GanzhiTime: 60-ganzhi cycle for richer time signal than
        # the raw MultiScaleCalendar phases alone.
        self.ganzhi = GanzhiTime()
        # MultimodalEncoder — lazily created on first multimodal input.
        # 天(vision)=0.3, 地(language)=0.3, 人(sensor)=0.4
        self._multimodal: MultimodalEncoder | None = None

    def _init_learning_systems(self) -> None:
        """Persistent learners: online DPO, curiosity, growth, Hebbian, EWC."""
        self.learner = OnlineLearner()
        self.curiosity = CuriosityScheduler()
        self.growth = GrowthManager()
        self.hebbian = HebbianAssociator()
        # P3a: EWC 防灾难遗忘 — 延迟初始化, 等待 world model 构建完成
        self._ewc = None  # type: ignore

    def _init_vsa(self) -> None:
        """In-memory VSA codebook + persistent long-term memory buffer."""
        self.vsa: TrainableVSACodebook | VSACodebook = (
            TrainableVSACodebook() if self.config.use_trainable_vsa else VSACodebook()
        )
        # Persist consolidated episodes into ZWM_DATA_DIR (or, when
        # the DB is :memory:, into the OS temp dir — the VSA buffer
        # persistence is best-effort and should never write into CWD).
        db_dir = os.path.dirname(self.config.db_path) or "."
        if db_dir == "." and self.config.db_path == ":memory:":
            persist_dir = os.environ.get(
                "ZWM_DATA_DIR",
                os.path.join(os.environ.get("TEMP", os.getcwd()), "zwm_vsa"),
            )
            os.makedirs(persist_dir, exist_ok=True)
        else:
            persist_dir = os.environ.get("ZWM_DATA_DIR", db_dir)
        self.vsa_buffer = VSAMemoryBuffer(
            persist_path=os.path.join(persist_dir, "vsa_consolidated.npz"),
        )
        # Eagerly restore any previously consolidated episodes.
        self.vsa_buffer.load_persisted()

    def _init_world_model(self) -> None:
        """JEPA predictor + SquareCircularJoint + VQ-VAE + value head + FSDP2.

        支持两种方图编码器:
          - 单卦 (旧): SquareGNN — 单个 hexagram 填入 Fuxi 方图
          - 场编码 (新): FieldSquareGNN — 64 个独立卦象的图神经网络

        当 field_encoder 可用时, 默认使用 FieldSquareGNN。
        """
        from zwm.jepa.square_encoder import LearnableSquareGNN, FixedWeightSquareGNN

        learnable = self.config.learnable_encoder
        hierarchical = self.config.hierarchical

        # ─── 场 GNN (新 — 64 卦独立) ───
        self._field_gnn = None
        if self.field_encoder is not None:
            try:
                from zwm.jepa.field_gnn import FieldSquareGNN, FieldSquareCircularJoint
                self._field_gnn = FieldSquareGNN(
                    hidden_dim=64, num_layers=3,
                )
                self.joint = FieldSquareCircularJoint(self._field_gnn)
                _log.info("World model: FieldSquareGNN (64卦独立场)")
            except Exception as exc:
                _log.warning("FieldSquareGNN init failed: %s; falling back to SquareGNN", exc)
                self._field_gnn = None

        # ─── 单卦 GNN (旧 — 回退) ───
        if self._field_gnn is None:
            self._square_learnable: LearnableSquareGNN | None
            if learnable:
                self._square_learnable = LearnableSquareGNN()
                self._square_fixed = FixedWeightSquareGNN()
                self.square = self._square_learnable
            else:
                self._square_learnable = None
                self._square_fixed = FixedWeightSquareGNN()
                self.square = self._square_fixed
            from zwm.jepa.square_encoder import SquareCircularJoint
            self.joint = SquareCircularJoint(self.square)
        else:
            # 场模式: 单卦 GNN 不需要, 但保留占位以防引用错误
            self._square_learnable = None
            self._square_fixed = None
            self.square = None  # type: ignore
        # 确定 JEPA 输入维度: 多场用 256, 单卦用 106
        jepa_input_dim = Z_WORLD_DIM
        if self._field_gnn is not None:
            jepa_input_dim = max(Z_WORLD_DIM, 256)  # 多场融合输出 256 dim
        if hierarchical:
            self.jepa = HierarchicalJEPAPredictor(input_dim=jepa_input_dim)
        else:
            self.jepa = JEPAPredictor(input_dim=jepa_input_dim)
        # Sync the learnable encoder's params into the JEPA optimiser scope.
        # 场模式: FieldSquareGNN 的梯度由 _predict 路径中的 forward 管理,
        # 不需要 attach 到 JEPA (JEPA 直接消费 MultiFieldJoint 的 256-dim z_world)
        if self._square_learnable is not None:
            self.jepa.attach_square_encoder(self._square_learnable)
        # MuZero-style latent V(s) head.
        try:
            self.jepa.init_value_head()
        except Exception as exc:
            _log.warning("init_value_head failed: %s", exc)
        # VQ-VAE discretisation.
        try:
            self.jepa.init_vq(num_codes=64, beta=0.25)
        except Exception as exc:
            _log.warning("init_vq failed: %s", exc)
        # FSDP2 (torch.distributed.fsdp.fully_shard).
        if self.config.use_fsdp2:
            try:
                from zwm.jepa.predictor import wrap_fsdp2, wrap_fsdp2_hierarchical
                self.jepa = (
                    wrap_fsdp2_hierarchical(self.jepa)
                    if hierarchical
                    else wrap_fsdp2(self.jepa)
                )
            except Exception as exc:
                _log.warning("FSDP2 wrap failed (single-GPU is fine): %s", exc)

        # Quantization — 2026 SOTA for inference efficiency.
        # ``quantize="4bit"`` → NF4 quantization (bitsandbytes)
        # ``quantize="lora"`` → LoRA adapters (rank-8)
        # ``quantize="qlora"`` → 4-bit + LoRA (Q-LoRA)
        q = self.config.quantize
        if q in ("4bit", "qlora"):
            result = self.jepa.quantize_4bit()
            if "error" in result:
                _log.warning("4-bit quantization skipped: %s", result["error"])
            else:
                _log.info("4-bit quantized %d layers", len(result))
        if q in ("lora", "qlora"):
            adapted = self.jepa.apply_lora(rank=8, alpha=16.0)
            _log.info("LoRA adapted %d layers", len(adapted))

        # P3a: EWC 防灾难遗忘 — 在 world model 构建完成后初始化
        try:
            from zwm.learning.ewc import EWCRegularizer
            self._ewc = EWCRegularizer(
                self.jepa,
                importance=float(os.environ.get("ZWM_EWC_IMPORTANCE", "100.0")),
                max_tasks=int(os.environ.get("ZWM_EWC_MAX_TASKS", "64")),
            )
            _log.info("EWC regularizer initialised (importance=%.1f)", self._ewc._importance)
        except Exception as exc:
            _log.debug("EWC init skipped: %s", exc)
            self._ewc = None

    def _init_particle_filter(self) -> None:
        """Particle filter for ensemble EFE in JEPA latent space (64-dim)."""
        n_particles = self.config.n_particles
        self._n_particles = n_particles
        if n_particles > 0:
            # 64-dim matches the JEPA latent space.  The particle filter
            # operates entirely in latent space: predict uses
            # ``jepa.predict_latent`` (64→64) and update uses
            # ``jepa.context_encode`` (106→64) to map observations.
            self._particle_filter = ParticleFilter(
                n_particles=n_particles, dim=LATENT_DIM, noise_std=0.05, obs_std=0.1,
            )
        else:
            self._particle_filter = None

    def _init_metrics(self) -> None:
        """Cross-tick counters + optional metrics logger (env override)."""
        self._step_count: int = 0
        # 卦象场缓存: 由 OODA observe 阶段填充, predict 阶段消费
        self._last_sensor_data: dict | None = None
        self._last_hex_field: np.ndarray | None = None
        # P1 FIX: Removed unnecessary MetricsLogger instantiation here.
        # _log_telemetry() uses get_logger() singleton which lazily creates
        # its own instance; self._metrics was never read, causing useless
        # file-handle and directory allocation.

        # ReAct / Tool-Use loop (2026 SOTA agent architecture).
        # When enabled, the agent runs a reasoning-acting-observing
        # cycle *before* the OODA planner, which enriches the prior
        # distribution with tool-derived insights (memory, harmony,
        # risk, topology, time).
        # P0: LLM 路由器 — 自动检测环境变量中的 API key 并注入 ReActLoop
        self._react_loop = None
        self._llm_router = None
        if self.config.use_react:
            try:
                from zwm.planner.react import ReActLoop
                llm_router = self._init_llm_router()
                self._react_loop = ReActLoop(
                    self, max_steps=3, llm_router=llm_router,
                )
                self._llm_router = llm_router
            except Exception as exc:
                _log.warning("ReAct loop init failed: %s", exc)

    def _init_topology(self) -> None:
        """Multi-scale palace scaffold + per-palace visit counter.

        P4-6 (audit): the topology is built once here and is part of
        the agent's persistent state.  ``expand_topology`` used to be
        imported at module level; inlining it removes a
        cross-module runtime dependency for a function that was
        only called here.
        """
        depth = self.config.topology_max_depth
        self.topology: RecursiveTopology = self._build_topology(depth)
        self._palace_visits: dict[int, int] = {}

    @staticmethod
    def _build_topology(max_depth: int) -> RecursiveTopology:
        """P4-6 — inlined topology builder, delegates to ``expand_topology``.

        Kept as a method (not a lambda) so subclasses can override the
        topology shape (e.g. a hex-diamond layout) without touching
        ``__init__``.
        """
        return expand_topology(max_depth=max_depth)

    def _init_memory_and_checkpoint(self) -> None:
        """Episodic + semantic stores — opened LAST.

        If anything in ``_init_world_model`` or below threw earlier, the
        SQLite handle was never opened, so nothing leaks.
        """
        # Per-construction grid: caller can pass one in, otherwise build.
        self.grid = (
            self.config.grid if self.config.grid is not None else LuoshuGrid()
        )
        # grid 的 self_position 永远是 5 — "我"在中宫
        self.grid.self_position = 5

        self.store = EpisodicStore(db_path=self.config.db_path)
        try:
            self.semantic = (
                SemanticStore(file_path=self.config.semantic_path)
                if self.config.semantic_path
                else None
            )
        except Exception as exc:
            _log.error("SemanticStore init failed: %s", exc)
            self.store.close()
            raise

        # P3: Checkpoint restore.
        if self.config.checkpoint_path is not None:
            try:
                load_checkpoint(self, self.config.checkpoint_path)
            except Exception as exc:
                _log.warning("Checkpoint restore failed (starting fresh): %s", exc)


    # ------------------------------------------------------------------
    # P0: LLM 路由器初始化
    # ------------------------------------------------------------------
    @staticmethod
    def _init_llm_router():
        """P0 — 自动检测并初始化 LLM 推理后端.

        检测顺序: DEEPSEEK_API_KEY > ANTHROPIC_API_KEY > OPENAI_API_KEY.
        当没有任何 API key 时, 返回 None (ReAct 使用启发式回退).
        """
        try:
            from zwm.llm.backends import auto_detect_backend
            backend = auto_detect_backend()
            from zwm.llm.router import LLMRouter
            _log.info("LLM router initialised: %s / %s", backend.name, backend.model_name)
            return LLMRouter(backend)
        except RuntimeError as exc:
            _log.info("No LLM backend available (%s); ReAct will use heuristic fallback", exc)
            return None
        except Exception as exc:
            _log.info("LLM router init skipped: %s", exc)
            return None

    @property
    def llm_router(self):
        """P0: 返回注入的 LLM 路由器 (可能为 None)."""
        return self._llm_router

    @property
    def ewc(self):
        """P3a: 返回 EWC 正则化器 (可能为 None)."""
        return self._ewc

    def register_ewc_task(self, task_id: str | None = None) -> None:
        """P3a: 将当前模型状态注册为 EWC 任务.

        在新 hexagram 的学习阶段调用, 防止遗忘旧知识。
        """
        if self._ewc is None:
            return
        tid = task_id or f"hex_{self._step_count}"
        try:
            self._ewc.register_task(tid)
        except Exception as exc:
            _log.debug("EWC register_task failed: %s", exc)

    # Context-manager support so the SQLite handle is always released, even on
    # an exception mid-loop.
    def __enter__(self) -> "TrinityAgent":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # OODA 入口
    # ------------------------------------------------------------------
    def observe_predict_evaluate_act(
        self,
        sensor_data: dict | None = None,
        h_current: Hexagram | None = None,
        grid: LuoshuGrid | None = None,
        year: int = 2026,
        month: int = 1,
        day: int = 1,
        hour: int = 0,
        time_phase: float | None = None,
        target_palace: int | None = None,
        day_gan: str | None = None,
        reward: float = 0.0,
        vision_features: np.ndarray | None = None,
        language_features: np.ndarray | None = None,
        language_text: str | None = None,
    ) -> TickReport:
        # R2: stash the multimodal inputs on the agent so downstream
        # consumers (F6's _act call to ``_multimodal.encode_multimodal``)
        # can actually read them.  Before this fix the
        # ``_last_vision_features`` / ``_last_language_text`` attributes
        # were ghost fields — read by _act but never written.
        self._last_vision_features = vision_features
        self._last_language_text = language_text
        self._last_language_features = language_features

        # P0-2: Convert raw text to embeddings via LanguageBackbone.
        if language_text is not None and language_features is None:
            if self._multimodal is None or self._multimodal.language_backbone is None:
                # Lazy-create the backbone.
                lb = LanguageBackbone()
                if self._multimodal is None:
                    self._multimodal = MultimodalEncoder(
                        sensor_weight=0.4, vision_weight=0.3, language_weight=0.3,
                        language_backbone=lb,
                    )
                else:
                    self._multimodal.language_backbone = lb
            language_features = self._multimodal.encode_text(language_text)

        # OBSERVE — perception turns sensors into a hexagram.
        if h_current is None:
            if sensor_data is None:
                raise ValueError("Provide either sensor_data or h_current")
            if vision_features is not None or language_features is not None:
                if self._multimodal is None:
                    self._multimodal = MultimodalEncoder(
                        sensor_weight=0.4, vision_weight=0.3, language_weight=0.3,
                    )
                h_current = self._multimodal.encode_multimodal(
                    sensor_data=sensor_data,
                    visual_features=vision_features,
                    text_embedding=language_features,
                )
            else:
                h_current = self.encoder.encode(sensor_data)

        if time_phase is None:
            time_phase = self.calendar.time_layers(year, month, day, hour)["年"]

        return self.tick(
            h_current=h_current,
            grid=grid,
            time_phase=time_phase,
            target_palace=target_palace,
            day_gan=day_gan,
            reward=reward,
            year=year,
            month=month,
            day=day,
            hour=hour,
            vision_features=vision_features,
            language_features=language_features,
            language_text=language_text,
            sensor_data=sensor_data,
        )

    def tick(
        self,
        h_current: Hexagram,
        grid: LuoshuGrid | None = None,
        time_phase: float = 0.0,
        target_palace: int | None = None,
        day_gan: str | None = None,
        reward: float = 0.0,
        year: int = 2026,
        month: int = 1,
        day: int = 1,
        hour: int = 0,
        vision_features: np.ndarray | None = None,
        language_features: np.ndarray | None = None,
        language_text: str | None = None,
        sensor_data: dict | None = None,
    ) -> TickReport:
        """OODA loop orchestrator — dispatches to focused phase modules.

        P4-8: input is gated through ``self.constitution.check_input``
        before any phase runs; the resulting ``TickReport`` is gated
        through ``check_output`` before being returned.  This makes
        every tick auditable against the safety rules in
        :data:`zwm.safety.constitution.DEFAULT_CONSTITUTION`.
        """
        from zwm.observability import PhaseStopwatch, metrics
        from zwm.tracing import get_tracer

        grid = grid if grid is not None else self.grid
        # 缓存传感器数据 — 供 field encoder 在 _predict 中使用。直接
        # tick(h_current=...) 没有感知载荷时才生成回退传感器；REST /
        # runtime / multimodal 入口会通过 observe_predict_evaluate_act()
        # 把真实 sensor_data 传进来。
        if sensor_data is None:
            sensor_data = {
                "temperature": 0.5 + 0.4 * math.sin(self._step_count / 5.0),
                "terrain": 0.5 + 0.3 * math.cos(self._step_count / 7.0),
                "social_proximity": abs(math.sin(self._step_count / 10.0)),
                "resource_level": 0.5 + 0.2 * math.sin(self._step_count / 3.0 + 1.0),
                "momentum": 0.5 * math.cos(self._step_count / 4.0),
                "overall_favorability": 0.5 + 0.3 * math.sin(self._step_count / 6.0),
            }
        self._last_sensor_data = sensor_data

        # Reward validation runs *first* so NaN/Inf surface as
        # ``ValueError`` (the long-standing contract — the existing
        # ``test_nan_reward_rejected`` test asserts that).  The
        # constitution is the second line of defence for out-of-range
        # but finite values, malformed hexagram indices, etc.
        reward = self._validate_reward(reward)

        # P4-8 — INPUT GATE.  Reject the tick *before* any expensive
        # work if the payload is malformed.
        self.constitution.check_input({
            "h_current": h_current.normal_order,
            "target_palace": target_palace,
            "reward": reward,
            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
            "time_phase": time_phase,
        })

        t_tick0 = time.perf_counter()
        metrics.set_reward(reward)
        metrics.set_hex_bits(int(h_current.normal_order))
        metrics.set_mcts_iterations(int(self.planner._mcts_iterations))
        if self._particle_filter is not None:
            metrics.set_particles(int(self._particle_filter.belief.n))

        # P4-9 — per-phase child spans (ooda.observe / .predict /
        # .evaluate / .act / .learn) are added inside each phase
        # below.  The child spans are sufficient for offline analysis
        # — they self-record start, end, status, and phase-specific
        # attributes (EFE, surprise, episode id, …).  No parent
        # OODA span is opened here so we don't have to indent the
        # whole function body under a ``with`` block.
        tracer = get_tracer()

        # 1) OBSERVE
        try:
            with PhaseStopwatch("observe"):
                with tracer.start_as_current_span("ooda.observe") as span:
                    target_palace = _observe(
                        self, grid, year, month, day, hour, target_palace,
                    )
                    span.set_attribute("zwm.target_palace", int(target_palace) if target_palace is not None else -1)
        except Exception:
            metrics.inc_errors()
            raise

        # 2) PREDICT
        try:
            with PhaseStopwatch("predict"):
                with tracer.start_as_current_span("ooda.predict") as span:
                    prediction = _predict(
                        self, h_current, grid, time_phase, day_gan,
                        year, month, day, hour,
                    )
                    self._last_prediction = prediction
                    span.set_attribute("zwm.z_world_dim", int(getattr(prediction, "z_world", np.zeros(0)).shape[0]) if hasattr(prediction, "z_world") else 0)
        except Exception:
            metrics.inc_errors()
            raise

        # 3) EVALUATE
        try:
            with PhaseStopwatch("evaluate"):
                with tracer.start_as_current_span("ooda.evaluate") as span:
                    result = _evaluate(
                        self, h_current, grid, time_phase, target_palace, day_gan,
                        prediction, vision_features, language_features,
                    )
                    self._last_plan = result
                    self._last_target_palace = target_palace
                    if result is not None and getattr(result, "hexagram_scores", None):
                        efe = float(result.hexagram_scores[0][1])
                        metrics.set_efe_value(efe)
                        span.set_attribute("zwm.efe_value", efe)
                    if result is not None and getattr(result, "moe_active_experts", None):
                        n_experts = len(result.moe_active_experts)
                        metrics.set_active_experts(n_experts)
                        span.set_attribute("zwm.active_experts", n_experts)
        except Exception:
            metrics.inc_errors()
            raise

        # 4) ACT
        try:
            with PhaseStopwatch("act"):
                with tracer.start_as_current_span("ooda.act") as span:
                    h_next, world_next, surprise, codon, codon_aa, mutation_class = _act(
                        self, h_current, grid, time_phase, day_gan,
                        prediction, result,
                    )
                    span.set_attribute("zwm.hex_bits_out", int(h_next.normal_order))
                    span.set_attribute("zwm.surprise", float(surprise))
        except Exception:
            metrics.inc_errors()
            raise

        # 5) LEARN
        try:
            with PhaseStopwatch("learn"):
                with tracer.start_as_current_span("ooda.learn") as span:
                    report = _learn(
                        self, h_current, h_next, grid, time_phase, result, reward,
                        world_next, surprise, vision_features, language_features,
                        codon, codon_aa, mutation_class, year, month, day, hour,
                        target_palace=target_palace,
                    )
                    self._last_report = report
                    span.set_attribute("zwm.episode_id", int(report.episode_id))
                    span.set_attribute("zwm.mutation_class", str(report.mutation_class))
        except Exception:
            metrics.inc_errors()
            raise

        # P2-1: publish tick-level metrics.
        elapsed = time.perf_counter() - t_tick0
        metrics.observe_tick_duration(elapsed)
        metrics.inc_ticks()
        if report.jepa_loss is not None:
            metrics.set_jepa_loss(float(report.jepa_loss))
        if report.router_loss is not None:
            metrics.set_router_loss(float(report.router_loss))
        metrics.set_surprise(float(report.surprise))
        try:
            metrics.set_episodes_stored(self.store.count())
        except Exception as exc:
            _log.debug("Prometheus gauge episodes_stored failed: %s", exc)
        # ReAct reflections gauge — cheap COUNT(*).
        try:
            if hasattr(self.store, "count_react_reflections"):
                metrics.set_react_reflections(self.store.count_react_reflections())
        except Exception as exc:
            _log.debug("Prometheus gauge react_reflections failed: %s", exc)
        metrics.set_hex_bits(int(h_next.normal_order))

        # P2-3: 复频谱 metrics — surface the resonance / phase coherence
        # of the chosen action so observability can spot "bad-vibe" ticks
        # (low phase coherence, destructive interference) at a glance.
        try:
            interference = getattr(self, "_last_interference", None)
            if interference is not None:
                metrics.set_interference_resonance(float(interference.resonance))
                metrics.set_interference_phase_coherence(float(interference.phase_coherence))
                metrics.set_dominant_harmonic(int(interference.dominant_harmonic))
        except Exception as exc:
            _log.debug("Prometheus gauge interference metrics failed: %s", exc)

        # P4-8 — OUTPUT GATE.  Validate the produced TickReport against
        # the constitution.  Block-severity failures raise here and
        # the caller (CLI / API / MCP) is expected to surface them.
        output_payload = {
            "h_current": report.h_current.normal_order,
            "h_next": report.h_next.normal_order,
            "top_mutation": report.top_mutation,
            "top_score": report.top_score,
            "reward": report.reward,
            "surprise": report.surprise,
        }
        self.constitution.check_output(output_payload)
        # Additional LLM-judge safety check (when available).
        if self._llm_judge is not None:
            try:
                verdict = self._llm_judge.check(output_payload)
                if not verdict.passed:
                    _log.warning("LLM judge rejected output: %s", verdict.reason)
            except Exception as exc:
                _log.warning("LLM judge check failed (fail-open): %s", exc)
        return report

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_reward(reward: float) -> float:
        """Validate the reward at the loop boundary; clamp to [-1, 1]."""
        try:
            r = float(reward)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"reward must be a real number, got {reward!r}") from exc
        if not math.isfinite(r):
            raise ValueError(f"reward must be finite, got {reward!r}")
        return max(-1.0, min(1.0, r))

    def memory_priors(self, h_current: "Hexagram", k: int = 5) -> dict[int, float]:
        """Public thin wrapper around :func:`memory_priors` for tests / tools."""
        return memory_priors(self, h_current, k=k)

    def close(self) -> None:
        # P3: Checkpoint save
        if self.config.checkpoint_path is not None:
            try:
                save_checkpoint(self, self.config.checkpoint_path)
            except Exception as exc:
                _log.debug("Checkpoint save skipped: %s", exc)
        self.store.close()
        if self.semantic is not None:
            self.semantic.close()
