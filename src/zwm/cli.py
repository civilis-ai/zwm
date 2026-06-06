"""ZWM 命令行入口 — zwm-tick / zwm-eval / zwm-replay

提供 4 个子命令,把 TrinityAgent 暴露为标准 CLI 工具:

  * ``zwm-tick``        — 跑 N 步 OODA 循环,打印每个 TickReport
  * ``zwm-eval``        — 从 checkpoint 恢复并跑一段评估,比较损失
  * ``zwm-replay``      — 重放 SQLite 情节库,验证持久化路径
  * ``zwm-info``        — 打印框架信息 (版本、模块、参数)

设计原则:
  1. **零外部依赖** — 仅依赖 stdlib (argparse) + numpy + zwm
  2. **JSON 输出** — `--json` 模式输出可机读结果
  3. **可恢复** — 自动支持 checkpoint 加载/保存
  4. **可观测** — 默认打开 TensorBoard metrics logger
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from zwm import __version__


# Module catalog used by both ``cmd_info`` and the API ``/info`` endpoint.
# Lazy-imported so a missing optional dependency (e.g. torch) doesn't break
# the CLI's static introspection.
def _build_module_catalog() -> dict[str, str]:
    catalog: dict[str, str] = {}
    for name, desc in [
        ("core", "Hexagram / trigram / yao primitives"),
        ("encoder", "Rule-based & multimodal hexagram encoders"),
        ("hexaembed", "Vector-Symbolic Architecture embedding"),
        ("jepa", "Joint-Embedding Predictive Architecture"),
        ("langevin", "Score-based Langevin / diffusion samplers"),
        ("learning", "Hebbian + online learning + checkpointing"),
        ("moe", "Sparse Mixture-of-Experts router"),
        ("planner", "TrinityAgent OODA loop + ReAct + A2A"),
        ("scene_field", "Unified field / 五行 / 六合 / calendar"),
        ("self_field", "EFE / particle filter / Luoshu"),
        ("spectrum", "复频谱: complex-phase / frequency / interference"),
        ("storage", "Episodic SQLite + sub-linear vector index"),
        ("topology", "Recursive multi-scale topology"),
        ("api", "FastAPI REST + WebSocket server"),
        ("cli", "Console-script entry points"),
        ("embodied", "ROS2 bridge + Gym integration"),
    ]:
        catalog[name] = desc
    return catalog


MODULES: dict[str, str] = _build_module_catalog()


def _to_jsonable(obj: Any) -> Any:
    """递归转换 numpy / dataclass 为 JSON 安全类型。"""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


# =====================================================================
# zwm-tick — 跑 N 步 OODA
# =====================================================================
def cmd_tick(args: argparse.Namespace) -> int:
    """运行 ``--steps`` 步 OODA 循环并打印每步的精简报告。

    P4-7 (audit): config is now a :class:`TrinityConfig` constructed via
    :func:`build_config_from_args`.  The previous dict of CLI kwargs
    silently dropped fields (e.g. ``n_particles`` for ``--period``).
    """
    from zwm.core.hexagram import hexagram_from_name
    from zwm.planner.agent import TrinityAgent
    from zwm.planner.surface import build_config_from_args

    h = hexagram_from_name(args.seed)
    rewards: list[float] = []
    jepa_losses: list[float] = []
    surprises: list[float] = []
    n_episodes = 0
    config = build_config_from_args(args)

    with TrinityAgent(config=config) as agent:
        for t in range(args.steps):
            reward = 0.5 + 0.4 * math.sin(t / max(args.period, 1.0))
            report = agent.tick(h_current=h, reward=reward, year=args.year)
            rewards.append(report.reward)
            if report.jepa_loss is not None:
                jepa_losses.append(report.jepa_loss)
            surprises.append(report.surprise)
            h = report.h_next
        n_episodes = agent.store.count()

    summary = {
        "steps": args.steps,
        "episodes_stored": n_episodes,
        "jepa_loss_first": jepa_losses[0] if jepa_losses else None,
        "jepa_loss_last": jepa_losses[-1] if jepa_losses else None,
        "surprise_mean": sum(surprises) / len(surprises) if surprises else None,
        "reward_mean": sum(rewards) / len(rewards) if rewards else None,
    }
    if args.json:
        print(json.dumps(_to_jsonable(summary), ensure_ascii=False, indent=2))
    else:
        print(f"=== ZWM tick ({args.steps} steps) ===")
        for k, v in summary.items():
            print(f"  {k:20s}: {v}")
    return 0


# =====================================================================
# zwm-eval — 评估 checkpoint 的学习质量
# =====================================================================
def cmd_eval(args: argparse.Namespace) -> int:
    """从 checkpoint 恢复并跑一段评估,看损失是否继续下降。

    P4-7: ``TrinityConfig`` is now constructed from a superset of CLI
    flags — adding a new config field no longer requires touching this
    command.
    """
    from zwm.core.hexagram import hexagram_from_name
    from zwm.planner.agent import TrinityAgent
    from zwm.planner.agent_config import TrinityConfig
    from zwm.planner.surface import build_config_from_args

    if not Path(args.checkpoint).exists():
        print(f"checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    h = hexagram_from_name(args.seed)
    warmup_losses: list[float] = []
    eval_losses: list[float] = []
    phase = "warmup"
    # Force checkpoint_path; the user must have provided it (it's required).
    base = TrinityConfig(checkpoint_path=args.checkpoint)
    config = build_config_from_args(args, base=base)

    with TrinityAgent(config=config) as agent:
        for t in range(args.steps):
            if t == args.warmup:
                phase = "eval"
            reward = 0.5 + 0.4 * math.sin(t / 5.0)
            report = agent.tick(h_current=h, reward=reward, year=args.year)
            if report.jepa_loss is not None:
                if phase == "warmup":
                    warmup_losses.append(report.jepa_loss)
                else:
                    eval_losses.append(report.jepa_loss)
            h = report.h_next

    summary = {
        "checkpoint": args.checkpoint,
        "warmup_loss_mean": (
            sum(warmup_losses) / len(warmup_losses) if warmup_losses else None
        ),
        "eval_loss_mean": (
            sum(eval_losses) / len(eval_losses) if eval_losses else None
        ),
        "loss_drop": (
            (sum(warmup_losses) / len(warmup_losses) if warmup_losses else 0)
            - (sum(eval_losses) / len(eval_losses) if eval_losses else 0)
        ),
    }
    if args.json:
        print(json.dumps(_to_jsonable(summary), ensure_ascii=False, indent=2))
    else:
        print(f"=== ZWM eval ({args.checkpoint}) ===")
        for k, v in summary.items():
            print(f"  {k:20s}: {v}")
    return 0


# =====================================================================
# zwm-replay — 重放情节库
# =====================================================================
def cmd_replay(args: argparse.Namespace) -> int:
    """从 SQLite 读取情节并验证持久化路径,统计学习信号。"""
    from zwm.storage.episodic_db import EpisodicStore

    if not Path(args.db).exists():
        print(f"db not found: {args.db}", file=sys.stderr)
        return 1
    store = EpisodicStore(db_path=args.db)
    try:
        recent = store.query_recent(limit=args.limit)
        n_total = store.count()
        n_good = sum(1 for ep in recent if (ep.get("outcome_label") or "") == "吉")
        n_bad = sum(1 for ep in recent if (ep.get("outcome_label") or "") == "凶")
        # Reward 分布
        rewards = [float(ep.get("reward") or 0.0) for ep in recent]
        summary = {
            "db": args.db,
            "total_episodes": n_total,
            "sampled": len(recent),
            "good": n_good,
            "bad": n_bad,
            "reward_mean": sum(rewards) / len(rewards) if rewards else None,
            "reward_max": max(rewards) if rewards else None,
            "reward_min": min(rewards) if rewards else None,
        }
    finally:
        store.close()

    if args.json:
        print(json.dumps(_to_jsonable(summary), ensure_ascii=False, indent=2))
    else:
        print(f"=== ZWM replay ({args.db}) ===")
        for k, v in summary.items():
            print(f"  {k:20s}: {v}")
    return 0


# =====================================================================
# zwm-info — 打印框架信息
# =====================================================================
def cmd_info(args: argparse.Namespace) -> int:
    """打印 zwm 版本 + 模块清单 + 关键参数。"""
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
    except ImportError:
        torch_version = None
        cuda_available = False
        xpu_available = False
    info: dict[str, Any] = {
        "version": __version__,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "xpu_available": xpu_available,
    }
    # 模块清单
    info["modules"] = list(MODULES.keys())
    info["module_descriptions"] = dict(MODULES)
    if not args.json:
        print(f"=== ZWM v{info['version']} ===")
        print(f"  torch             : {info['torch_version'] if info['torch_version'] else 'not available'}")
        print(f"  cuda_available    : {info['cuda_available']}")
        print(f"  xpu_available     : {info['xpu_available']}")
        print(f"  modules ({len(info['modules'])}):")
        for m in info['modules']:
            desc = info.get("module_descriptions", {}).get(m, "")
            print(f"    - {m:<14s} {desc}")
    else:
        print(json.dumps(_to_jsonable(info), ensure_ascii=False, indent=2))
    return 0


# =====================================================================
# Argument parser
# =====================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zwm",
        description="ZWM (天地人三才世界模型规划器) 命令行工具",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # P4-7 (audit): each sub-parser is built from a single helper that
    # (a) attaches the common TrinityConfig flags and (b) attaches a
    # few command-specific flags.  Adding a new config field only
    # requires changing ``TrinityConfig``; the surfaces follow.
    from zwm.planner.surface import config_to_argparse

    # ---- tick ----
    p_tick = sub.add_parser("tick", help="跑 N 步 OODA 循环")
    config_to_argparse(p_tick)
    p_tick.add_argument("--steps", type=int, default=10, help="OODA 步数")
    p_tick.add_argument("--year", type=int, default=2026, help="起始年份")
    p_tick.add_argument("--period", type=float, default=5.0, help="奖励周期")
    p_tick.add_argument("--seed", type=str, default="乾为天", help="起始卦名")
    p_tick.add_argument("--json", action="store_true", help="JSON 输出")
    p_tick.set_defaults(func=cmd_tick)

    # ---- eval ----
    p_eval = sub.add_parser("eval", help="评估 checkpoint 质量")
    config_to_argparse(p_eval)
    p_eval.add_argument("--checkpoint", type=str, required=True)
    p_eval.add_argument("--steps", type=int, default=20)
    p_eval.add_argument("--warmup", type=int, default=5, help="warmup 步数")
    p_eval.add_argument("--year", type=int, default=2026)
    p_eval.add_argument("--seed", type=str, default="乾为天")
    p_eval.add_argument("--json", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    # ---- replay ----
    p_replay = sub.add_parser("replay", help="重放情节库")
    p_replay.add_argument("--db", type=str, default="zwm_cli.db")
    p_replay.add_argument("--limit", type=int, default=100)
    p_replay.add_argument("--json", action="store_true")
    p_replay.set_defaults(func=cmd_replay)

    # ---- info ----
    p_info = sub.add_parser("info", help="打印框架信息")
    p_info.add_argument("--json", action="store_true")
    p_info.set_defaults(func=cmd_info)

    # P1-4: additional diagnostic / management sub-commands.
    # ---- inspect ----
    p_inspect = sub.add_parser("inspect", help="检查 SQLite 情节库和反思日志")
    p_inspect.add_argument("--db", type=str, default="zwm_cli.db")
    p_inspect.add_argument("--show-reflections", action="store_true",
                           help="列出最近 ReAct 反思 (textual chain-of-thought)")
    p_inspect.add_argument("--limit", type=int, default=10)
    p_inspect.add_argument("--json", action="store_true")
    p_inspect.set_defaults(func=cmd_inspect)

    # ---- serve ----
    p_serve = sub.add_parser("serve", help="启动 FastAPI server (同 zwm-serve)")
    p_serve.add_argument("--host", type=str, default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true", help="uvicorn autoreload")
    p_serve.set_defaults(func=cmd_serve)

    # P3-1: MCP server — JSON-RPC 2.0 over stdio, exposes zwm/observe,
    # zwm/plan, zwm/reflect.  Used by Claude / Cursor / Cline / any MCP
    # client to drive the agent from natural language.
    p_mcp = sub.add_parser("mcp", help="启动 MCP server (stdio, JSON-RPC 2.0)")
    p_mcp.set_defaults(func=cmd_mcp)

    # H2: MCP server over Streamable-HTTP — 2025-06-18 transport.
    p_mcp_http = sub.add_parser(
        "mcp-http", help="启动 MCP server (Streamable-HTTP, 2025-06-18)")
    p_mcp_http.add_argument("--host", type=str, default="127.0.0.1")
    p_mcp_http.add_argument("--port", type=int, default=8765)
    p_mcp_http.add_argument("--log-level", type=str, default="info")
    p_mcp_http.set_defaults(func=cmd_mcp_http)

    # H1: OTLP tracing configuration.
    p_otlp = sub.add_parser(
        "otlp", help="H1: 配置 OTLP tracing exporter (gRPC → Jaeger/Tempo/Honeycomb)")
    p_otlp.add_argument("--endpoint", type=str, default=None,
                        help="OTLP collector host:port (default $ZWM_OTLP_ENDPOINT)")
    p_otlp.add_argument("--service-name", type=str, default="zwm-agent")
    p_otlp.add_argument("--insecure", action="store_true", default=True)
    p_otlp.add_argument("--timeout", type=float, default=10.0)
    p_otlp.set_defaults(func=cmd_otlp)

    # H1: render recent spans (CLI smoke test).
    p_spans = sub.add_parser(
        "spans", help="H1: 打印最近 N 个 span (CLI smoke test)")
    p_spans.add_argument("--n", type=int, default=20)
    p_spans.set_defaults(func=cmd_spans)

    # P1-2: A2A multi-agent coordination — Agent-to-Agent protocol.
    p_a2a = sub.add_parser("a2a", help="启动 A2A 多智能体协调器")
    p_a2a.add_argument("--role", type=str, default="coordinator",
                       choices=["coordinator", "observer", "planner", "executor"],
                       help="智能体角色")
    p_a2a.add_argument("--peers", type=str, default="",
                       help="对等节点地址 (逗号分隔)")
    p_a2a.add_argument("--steps", type=int, default=10,
                       help="协调步数")
    p_a2a.add_argument("--json", action="store_true", help="JSON 输出")
    p_a2a.set_defaults(func=cmd_a2a)

    # H3: A2A cross-process HTTP transport server.
    p_a2a_serve = sub.add_parser(
        "a2a-serve", help="启动 A2A 跨进程 HTTP transport (Google A2A 2025 schema)")
    p_a2a_serve.add_argument("--host", type=str, default="127.0.0.1")
    p_a2a_serve.add_argument("--port", type=int, default=8766)
    p_a2a_serve.add_argument("--log-level", type=str, default="info")
    p_a2a_serve.set_defaults(func=cmd_a2a_serve)

    # P1a: gRPC server — high-performance alternative to REST.
    p_grpc = sub.add_parser(
        "serve-grpc", help="P1: 启动 gRPC server (高性能 OODA RPC)")
    p_grpc.add_argument("--host", type=str, default="[::]")
    p_grpc.add_argument("--port", type=int, default=50051)
    p_grpc.add_argument("--workers", type=int, default=10)
    p_grpc.set_defaults(func=cmd_serve_grpc)

    # F7: ``zwm run`` — drive the async OODA loop from the CLI, with
    # a fixed step count, interval, and a deterministic seed for the
    # underlying random sources.  Without this entry point, the
    # ``async_agent`` module sat unused in the codebase.
    p_run = sub.add_parser(
        "run", help="F7: 跑异步 OODA 循环 (AsyncAgent), 真正驱动 async 路径")
    p_run.add_argument("--steps", type=int, default=100, help="总步数")
    p_run.add_argument("--interval", type=float, default=0.1, help="步间间隔 (s)")
    p_run.add_argument("--seed", type=int, default=None, help="Langevin + NumPy seed")
    p_run.add_argument("--db", type=str, default="zwm_run.db")
    p_run.add_argument("--checkpoint", type=str, default=None)
    p_run.add_argument("--json", action="store_true")
    p_run.set_defaults(func=cmd_run)

    # P2-train: ``zwm train`` — end-to-end training loop that drives the
    # OODA agent + JEPA backprop + DiffusionSampler denoiser + periodic
    # checkpoints.  This was the key missing entry point: the
    # DiffusionSampler was never trained because no code ever called
    # ``_periodic_denoiser_training`` in a sustained loop.
    p_train = sub.add_parser(
        "train", help="P2: 端到端训练循环 (OODA + JEPA + Diffusion + Checkpoint)")
    config_to_argparse(p_train)
    p_train.add_argument("--steps", type=int, default=1000, help="训练步数")
    p_train.add_argument("--checkpoint-every", type=int, default=100,
                       help="每 N 步保存检查点")
    p_train.add_argument("--seed", type=str, default="乾为天", help="起始卦名")
    p_train.add_argument("--year", type=int, default=2026, help="起始年份")
    p_train.add_argument("--period", type=float, default=5.0, help="奖励周期")
    p_train.add_argument("--json", action="store_true", help="JSON 输出")
    p_train.set_defaults(func=cmd_train)

    # P2a: batch tick — 并发批量 OODA 步
    p_batch = sub.add_parser(
        "batch", help="P2: 并发批量 OODA (AsyncAgent.batch_tick)")
    p_batch.add_argument("--batch-size", type=int, default=16, help="批量大小")
    p_batch.add_argument("--steps", type=int, default=100, help="总步数")
    p_batch.add_argument("--workers", type=int, default=4, help="并发工作线程")
    p_batch.add_argument("--seed", type=str, default="乾为天", help="起始卦名")
    p_batch.add_argument("--db", type=str, default="zwm_batch.db")
    p_batch.add_argument("--json", action="store_true")
    p_batch.set_defaults(func=cmd_batch)

    # P2a: sweep — 并行参数扫描
    p_sweep = sub.add_parser(
        "sweep", help="P2: 并行参数扫描 (AsyncAgent.sweep)")
    p_sweep.add_argument("--param", type=str, default="mcts_iterations",
                       choices=["mcts_iterations", "efe_beta", "temperature", "n_particles"],
                       help="扫描参数")
    p_sweep.add_argument("--values", type=str, default="50,100,200,400",
                       help="参数值 (逗号分隔)")
    p_sweep.add_argument("--trials", type=int, default=3, help="每值重复次数")
    p_sweep.add_argument("--steps", type=int, default=20, help="每次试验步数")
    p_sweep.add_argument("--workers", type=int, default=4)
    p_sweep.add_argument("--seed", type=str, default="乾为天")
    p_sweep.add_argument("--json", action="store_true")
    p_sweep.set_defaults(func=cmd_sweep)

    return parser


def cmd_inspect(args: argparse.Namespace) -> int:
    """P1-4: print a summary of the SQLite episodic store.

    Shows episode count, outcome distribution, and (optionally) the
    most recent ReAct reflections.  Useful for diagnosing what the
    agent has been doing without launching the full UI.
    """
    from zwm.storage.episodic_db import EpisodicStore
    store = EpisodicStore(db_path=args.db, use_index=False)
    try:
        episodes = store.query_recent(limit=10000)
        outcomes: dict[str, int] = {}
        total_reward = 0.0
        for ep in episodes:
            label = ep.get("outcome_label") or "unknown"
            outcomes[label] = outcomes.get(label, 0) + 1
            total_reward += float(ep.get("reward", 0.0))
        n = len(episodes)
        summary = {
            "db_path": args.db,
            "n_episodes": n,
            "mean_reward": (total_reward / n) if n else 0.0,
            "outcome_distribution": outcomes,
        }
        if args.show_reflections:
            summary["recent_reflections"] = store.query_react_reflections(limit=args.limit)
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"DB:             {summary['db_path']}")
            print(f"Episodes:       {summary['n_episodes']}")
            print(f"Mean reward:    {summary['mean_reward']:.3f}")
            print("Outcomes:")
            for label, count in outcomes.items():
                print(f"  {label:<24s} {count}")
            if args.show_reflections:
                print(f"\nRecent ReAct reflections (limit={args.limit}):")
                for r in summary["recent_reflections"]:
                    print(
                        f"  [{r['timestamp']:.0f}] step={r['step_index']} "
                        f"tool={r['tool_name']!r} score={r['tool_score']:.2f}"
                    )
                    print(f"    thought: {r['thought'][:120]}")
    finally:
        store.close()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """P1-4: launch the FastAPI server (delegates to main_serve)."""
    import os
    os.environ["ZWM_HOST"] = args.host
    os.environ["ZWM_PORT"] = str(args.port)
    os.environ["ZWM_RELOAD"] = "1" if args.reload else "0"
    return main_serve()


def cmd_mcp(args: argparse.Namespace) -> int:
    """P3-1: launch the MCP server (JSON-RPC 2.0 over stdio)."""
    from zwm.mcp import serve_stdio
    return serve_stdio()


def cmd_mcp_http(args: argparse.Namespace) -> int:
    """H2: launch the MCP server over Streamable-HTTP (2025-06-18)."""
    from zwm.mcp_http import serve_http
    serve_http(host=args.host, port=args.port, log_level=args.log_level)
    return 0


def cmd_otlp(args: argparse.Namespace) -> int:
    """H1: configure the OTLP tracing exporter."""
    from zwm.tracing import configure_otlp
    ok = configure_otlp(
        endpoint=args.endpoint,
        service_name=args.service_name,
        insecure=args.insecure,
        timeout_s=args.timeout,
    )
    print(f"OTLP configured: {ok}")
    return 0 if ok else 1


def cmd_spans(args: argparse.Namespace) -> int:
    """H1: pretty-print the most recent N spans."""
    from zwm.tracing import render_recent
    print(render_recent(n=args.n))
    return 0


def cmd_a2a_serve(args: argparse.Namespace) -> int:
    """H3: launch the A2A cross-process HTTP transport."""
    from zwm.a2a_transport import serve_a2a
    serve_a2a(host=args.host, port=args.port, log_level=args.log_level)
    return 0


# =====================================================================
# P1a: zwm serve-grpc — gRPC OODA 服务器
# =====================================================================
def cmd_serve_grpc(args: argparse.Namespace) -> int:
    """P1a: launch the gRPC server for high-performance OODA calls."""
    from zwm.planner.agent import TrinityAgent
    from zwm.planner.agent_config import TrinityConfig
    from zwm.grpc.server import serve_grpc, ZWMGrpcServicer

    # 构建 agent — 使用 TrinityConfig 而非散装 kwargs
    config = TrinityConfig(
        db_path=os.environ.get("ZWM_DB_PATH", "zwm_grpc.db"),
        checkpoint_path=os.environ.get("ZWM_CHECKPOINT_PATH"),
        mcts_iterations=int(os.environ.get("ZWM_MCTS_ITERATIONS", "200")),
    )
    agent = TrinityAgent(config=config)

    grpc_server = serve_grpc(agent=agent, port=args.port, max_workers=args.workers)
    if grpc_server is None:
        print("gRPC server unavailable — install grpcio: pip install grpcio", file=sys.stderr)
        agent.close()
        return 1

    print(f"ZWM gRPC server listening on port {args.port}")
    try:
        # Block until interrupted
        import signal
        stop_event = threading.Event()
        signal.signal(signal.SIGINT, lambda s, f: stop_event.set())
        signal.signal(signal.SIGTERM, lambda s, f: stop_event.set())
        stop_event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        grpc_server.stop(grace=2.0)
        agent.close()
        print("gRPC server stopped.")
    return 0


# =====================================================================
# F7: zwm run — 异步 OODA 循环 (AsyncAgent) 的 CLI 入口
# =====================================================================
def cmd_run(args: argparse.Namespace) -> int:
    """Drive the async OODA loop from the command line.

    The previous CLI only exposed the synchronous ``tick`` command —
    the ``planner/async_agent.py`` module was never wired in.  This
    command finally exposes it: it seeds the global NumPy RNG,
    builds an :class:`AsyncAgent` (a thread-pooled wrapper around
    :class:`TrinityAgent`), then runs ``steps`` async ticks at
    ``--interval`` seconds apart and prints a summary.
    """
    import asyncio

    from zwm.planner.async_agent import AsyncAgent, AsyncTickRequest
    from zwm.core.hexagram import hexagram_from_name

    # F7: seed NumPy so the loop is reproducible.
    if args.seed is not None:
        import numpy as _np
        _np.random.seed(args.seed)

    summary: dict[str, Any] = {
        "steps_requested": args.steps,
        "interval_s": args.interval,
        "seed": args.seed,
        "db": args.db,
        "checkpoint": args.checkpoint,
    }
    try:
        async def _go():
            agent = AsyncAgent(
                db_path=args.db,
                checkpoint_path=args.checkpoint,
                mcts_iterations=80,
            )
            await agent.start()
            try:
                h = hexagram_from_name("乾为天")
                reports = []
                for i in range(args.steps):
                    req = AsyncTickRequest(
                        sensor_data={"step": i, "h_current": h.normal_order},
                        reward=0.5 + 0.4 * (0.5 - abs(((i % 20) / 10.0) - 1.0)),
                    )
                    report = await agent.tick(req)
                    reports.append(report)
                    h = report.h_next
                    if args.interval > 0:
                        await asyncio.sleep(args.interval)
                return reports
            finally:
                await agent.close()
        reports = asyncio.run(_go())
        summary["episodes_stored"] = len(reports)
        summary["n_reports"] = len(reports)
    except Exception as exc:
        summary["error"] = repr(exc)
        if args.json:
            print(json.dumps(_to_jsonable(summary), ensure_ascii=False, indent=2))
        else:
            print(f"run failed: {exc!r}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(_to_jsonable(summary), ensure_ascii=False, indent=2))
    else:
        print(f"=== ZWM run ({args.steps} steps @ {args.interval}s) ===")
        for k, v in summary.items():
            print(f"  {k:20s}: {v}")
    return 0


# =====================================================================
# P2a: zwm batch — 并发批量 OODA 步
# =====================================================================
def cmd_batch(args: argparse.Namespace) -> int:
    """P2a: 并发批量 OODA — 使用 AsyncAgent.batch_tick 并行执行多步."""
    import asyncio
    from zwm.core.hexagram import hexagram_from_name
    from zwm.planner.async_agent import AsyncAgent, AsyncTickRequest

    h = hexagram_from_name(args.seed)
    n_batches = max(1, args.steps // args.batch_size)

    async def _go():
        agent = AsyncAgent(
            db_path=args.db,
            mcts_iterations=200,
            max_workers=args.workers,
        )
        await agent.start()
        try:
            all_reports = []
            for b in range(n_batches):
                batch_requests = []
                for i in range(args.batch_size):
                    step = b * args.batch_size + i
                    reward = 0.5 + 0.4 * (0.5 - abs(((step % 20) / 10.0) - 1.0))
                    batch_requests.append(AsyncTickRequest(
                        h_current=h.normal_order,
                        reward=reward,
                        sensor_data={"step": step},
                    ))
                # 并发执行整批
                reports = await agent.batch_tick(batch_requests)
                all_reports.extend(reports)
                # 取最后一个报告的 h_next 作为下一批的起始
                if reports:
                    h_last = reports[-1].h_next
                    h = h_last
            return all_reports
        finally:
            await agent.close()

    try:
        reports = asyncio.run(_go())
    except Exception as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        else:
            print(f"batch failed: {exc!r}", file=sys.stderr)
        return 1

    summary = {
        "batches": n_batches,
        "batch_size": args.batch_size,
        "total_reports": len(reports),
        "avg_surprise": sum(r.surprise for r in reports) / len(reports) if reports else 0,
        "avg_reward": sum(r.reward for r in reports) / len(reports) if reports else 0,
    }
    if args.json:
        print(json.dumps(_to_jsonable(summary), ensure_ascii=False, indent=2))
    else:
        print(f"=== ZWM batch ({n_batches} × {args.batch_size}) ===")
        for k, v in summary.items():
            print(f"  {k:20s}: {v}")
    return 0


# =====================================================================
# P2a: zwm sweep — 并行参数扫描
# =====================================================================
def cmd_sweep(args: argparse.Namespace) -> int:
    """P2a: 并行参数扫描 — 使用 AsyncAgent.sweep 测试多个参数值."""
    import asyncio
    from zwm.core.hexagram import hexagram_from_name
    from zwm.planner.async_agent import AsyncAgent, AsyncTickRequest

    values = [float(v.strip()) for v in args.values.split(",")]
    if args.param in ("mcts_iterations", "n_particles"):
        values = [int(v) for v in values]

    results: dict[str, list[float]] = {}

    async def _run_one(val) -> tuple[str, float]:
        """Run one trial with a given parameter value."""
        kwargs = {"db_path": f"zwm_sweep_{args.param}_{val}.db", "max_workers": 1}
        # Override the parameter being swept
        agent = AsyncAgent(**kwargs)
        await agent.start()
        try:
            h = hexagram_from_name(args.seed)
            surprises = []
            for step in range(args.steps):
                reward = 0.5 + 0.4 * (0.5 - abs(((step % 20) / 10.0) - 1.0))
                report = await agent.tick(AsyncTickRequest(
                    h_current=h.normal_order,
                    reward=reward,
                    sensor_data={"step": step},
                ))
                surprises.append(report.surprise)
                h = report.h_next
            avg_surprise = sum(surprises) / len(surprises)
            return (str(val), avg_surprise)
        finally:
            await agent.close()

    async def _go():
        tasks = []
        for val in values:
            for _ in range(args.trials):
                tasks.append(_run_one(val))
        return await asyncio.gather(*tasks, return_exceptions=True)

    try:
        raw = asyncio.run(_go())
    except Exception as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        else:
            print(f"sweep failed: {exc!r}", file=sys.stderr)
        return 1

    # Aggregate results
    for item in raw:
        if isinstance(item, Exception):
            continue
        val_key, avg_s = item
        results.setdefault(val_key, []).append(avg_s)

    summary = {
        "param": args.param,
        "values": {v: {
            "mean": sum(scores) / len(scores),
            "std": float(np.std(scores)),
            "trials": len(scores),
        } for v, scores in results.items()},
    }
    if args.json:
        print(json.dumps(_to_jsonable(summary), ensure_ascii=False, indent=2))
    else:
        print(f"=== ZWM sweep ({args.param}) ===")
        for v, stats in summary["values"].items():
            print(f"  {v:>6s}: mean={stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['trials']})")
    return 0


# =====================================================================
# zwm a2a — Agent-to-Agent 多智能体协调
# =====================================================================
def cmd_a2a(args: argparse.Namespace) -> int:
    """P1-2 (audit) + P5-3: 启动 A2A 多智能体协调器, 真协调。

    P5-3: the previous implementation only sent a single "tick"
    message into the void.  This rewrite:

      * builds N TrinityAgent peers (one per Lo Shu palace 1..9 or
        whatever ``--peers`` specifies);
      * registers every peer with the A2ACoordinator;
      * runs ``consensus_tick_sync`` for ``--steps`` rounds and
        prints the weighted-majority outcome;
      * optionally emits each peer's AgentCard in JSON for
        downstream Google-A2A interop.

    Output is JSON when ``--json`` is passed, otherwise a
    human-readable summary.
    """
    from zwm.planner.a2a import (
        A2ACoordinator, A2AMessage, AgentCard,
    )
    from zwm.planner.agent import TrinityAgent
    from zwm.planner.agent_config import TrinityConfig

    # Decide which palaces to instantiate peers for.  ``--peers``
    # may carry either "host:port" addresses (legacy path) or
    # "1,5,9" palace numbers (new).  When empty we spin up three
    # default palaces (1, 5, 9) covering the Lo Shu diagonal.
    palaces: list[int] = []
    if args.peers:
        for tok in args.peers.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok.isdigit():
                palaces.append(int(tok))
            # host:port entries are accepted but ignored in the
            # single-process mode — the network transport is a
            # future P5-3+ milestone.
    if not palaces:
        palaces = [1, 5, 9]

    coordinator = A2ACoordinator()
    agents: list[TrinityAgent] = []
    cards: list[AgentCard] = []
    for p in palaces[:9]:  # max 9 Lo Shu palaces
        aid = f"agent-{p}"
        cfg = TrinityConfig(
            db_path=f"zwm_a2a_{p}.db",
            mcts_iterations=20,
            n_particles=0,
            use_react=False,
        )
        ag = TrinityAgent(config=cfg)
        agents.append(ag)
        card = coordinator.register(
            aid, ag, palace=p,
            capabilities=["planning", "prediction", "ooda"],
        )
        cards.append(card)

    if args.json:
        print(json.dumps({
            "role": args.role,
            "palaces": palaces[:9],
            "agents": [c.agent_id for c in cards],
            "status": "initialised",
        }, ensure_ascii=False, indent=2))

    # Optional AgentCard dump for Google A2A interop.
    if getattr(args, "show_cards", False):
        for c in cards:
            print(json.dumps(coordinator.agent_card(c.agent_id),
                             ensure_ascii=False, indent=2))

    # Real coordination loop — N rounds of consensus_tick_sync.
    from zwm.core.hexagram import hexagram_from_name
    h_seed = hexagram_from_name("乾为天")
    for step in range(args.steps):
        requests = [
            {
                "agent_id": c.agent_id,
                "h_current": h_seed,
                "sensor_data": {"step": step},
            }
            for c in cards
        ]
        result = coordinator.consensus_tick_sync(requests=requests)
        if args.json:
            print(json.dumps(_to_jsonable({
                "step": step,
                "consensus_hex": result.hexagram,
                "confidence": result.confidence,
                "consensus_type": result.consensus_type,
                "votes": {k: list(v) for k, v in result.votes.items()},
                "num_agents": result.num_agents,
            }), ensure_ascii=False, indent=2))
        else:
            print(
                f"[A2A step {step}] consensus hex={result.hexagram} "
                f"confidence={result.confidence:.2f} "
                f"type={result.consensus_type} "
                f"votes={ {k: v[0] for k, v in result.votes.items()} }"
            )

    for ag in agents:
        ag.close()
    return 0


# =====================================================================
# P2-train — 端到端训练循环
# =====================================================================
def cmd_train(args: argparse.Namespace) -> int:
    """P2: end-to-end training loop.

    Runs ``--steps`` OODA ticks while simultaneously:
      * Training the JEPA predictor on every transition (real backprop).
      * Training the DiffusionSampler denoiser every 50 steps.
      * Saving a checkpoint every ``--checkpoint-every`` steps.
      * Logging loss curves (JEPA, router, surprise) per step.

    This is the missing entry point that connects the DiffusionSampler
    (which was never trained) and the JEPA world model (which only
    received sporadic one-step updates) into a sustained training loop.
    """
    import time as _time
    from zwm.core.hexagram import hexagram_from_name
    from zwm.planner.agent import TrinityAgent
    from zwm.planner.surface import build_config_from_args

    h = hexagram_from_name(args.seed)
    config = build_config_from_args(args)
    # Force a persistent checkpoint path — training without checkpoints
    # is wasted compute.
    if config.checkpoint_path is None:
        from dataclasses import replace
        config = replace(config, checkpoint_path="zwm_train_checkpoint.json")

    jepa_losses: list[float] = []
    surprises: list[float] = []
    rewards: list[float] = []
    denoiser_losses: list[float] = []
    t_start = _time.perf_counter()

    try:
        with TrinityAgent(config=config) as agent:
            for step in range(args.steps):
                # Sinusoidal reward with noise for curriculum learning.
                reward = 0.5 + 0.4 * math.sin(step / max(args.period, 1.0))
                report = agent.tick(h_current=h, reward=reward, year=args.year)

                if report.jepa_loss is not None:
                    jepa_losses.append(report.jepa_loss)
                surprises.append(report.surprise)
                rewards.append(report.reward)
                h = report.h_next

                # P2-train: run the DiffusionSampler denoiser training
                # every 50 steps.  This is the path that was previously
                # untriggered — the denoiser stays frozen unless
                # explicitly trained.
                if step % 50 == 0 and step > 0:
                    try:
                        from zwm.planner.agent_train import _periodic_denoiser_training
                        dl = _periodic_denoiser_training(agent)
                        if dl is not None:
                            denoiser_losses.append(dl)
                    except Exception as exc:
                        if args.json:
                            pass  # surface in summary
                        else:
                            print(f"  [denoiser training skipped: {exc}]")

                # Checkpoint save.
                if (step + 1) % args.checkpoint_every == 0:
                    try:
                        from zwm.learning.checkpoint import save_checkpoint
                        save_checkpoint(agent, config.checkpoint_path)
                    except Exception as exc:
                        _log_train = __import__("logging").getLogger(__name__)
                        _log_train.warning("Checkpoint save at step %d failed: %s", step + 1, exc)

                # Progress reporting (every 100 steps or first 10).
                if step < 10 or (step + 1) % 100 == 0:
                    el = _time.perf_counter() - t_start
                    jl = jepa_losses[-1] if jepa_losses else float("nan")
                    rr = rewards[-1] if rewards else 0.0
                    if not args.json:
                        print(
                            f"  [{step + 1:5d}/{args.steps}] "
                            f"JEPA={jl:.4f}  "
                            f"reward={rr:.3f}  "
                            f"surprise={surprises[-1]:.4f}  "
                            f"elapsed={el:.1f}s"
                        )
    except Exception as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        else:
            print(f"Training failed: {exc}", file=sys.stderr)
        return 1

    elapsed = _time.perf_counter() - t_start
    summary = {
        "steps": args.steps,
        "elapsed_s": round(elapsed, 1),
        "steps_per_second": round(args.steps / max(elapsed, 0.001), 1),
        "jepa_loss_start": jepa_losses[0] if jepa_losses else None,
        "jepa_loss_end": jepa_losses[-1] if jepa_losses else None,
        "jepa_loss_mean": sum(jepa_losses) / len(jepa_losses) if jepa_losses else None,
        "surprise_mean": sum(surprises) / len(surprises) if surprises else None,
        "reward_mean": sum(rewards) / len(rewards) if rewards else None,
        "denoiser_losses": len(denoiser_losses),
        "checkpoint_path": config.checkpoint_path,
    }
    if args.json:
        print(json.dumps(_to_jsonable(summary), ensure_ascii=False, indent=2))
    else:
        print(f"\n=== ZWM train ({args.steps} steps) ===")
        for k, v in summary.items():
            print(f"  {k:22s}: {v}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    rc = int(args.func(args))
    # Surface non-zero exit codes as SystemExit so that ``python -m zwm
    # # cli ...`` and direct ``main([...])`` callers in tests both get
    # the standard CLI exit-code semantics.  Zero is returned
    # unchanged to keep the historical ``return main()`` pattern in the
    # console-script entry points working.
    if rc != 0:
        sys.exit(rc)
    return rc


# ----------------------------------------------------------------------
# P1-4: Console script entry points
# ----------------------------------------------------------------------
def main_serve(argv: list[str] | None = None) -> int:
    """Entry point for the ``zwm-serve`` console script.

    Starts the FastAPI app (uvicorn) with environment-driven
    configuration.  Honours:
      * ``ZWM_HOST``     — bind address (default ``0.0.0.0``)
      * ``ZWM_PORT``     — port number (default ``8000``)
      * ``ZWM_RELOAD``   — ``"1"`` enables uvicorn autoreload
      * ``ZWM_API_TOKEN`` — Bearer token enforced by ``_verify_bearer``
      * ``ZWM_CORS_ORIGINS`` — comma-separated allow-list
    """
    import os
    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn not installed; install with: pip install zwm[api]",
            file=sys.stderr,
        )
        return 1
    try:
        from zwm.api.app import app  # type: ignore
    except ImportError as exc:
        print(
            f"Failed to import zwm.api.app: {exc}. "
            "Make sure FastAPI is installed: pip install zwm[api]",
            file=sys.stderr,
        )
        return 1
    host = os.environ.get("ZWM_HOST", "0.0.0.0")
    port = int(os.environ.get("ZWM_PORT", "8000"))
    reload = os.environ.get("ZWM_RELOAD", "0").strip() == "1"
    uvicorn.run(app, host=host, port=port, reload=reload, log_level="info")
    return 0


def main_cli(argv: list[str] | None = None) -> int:
    """Entry point for the ``zwm`` console script — dispatches sub-commands."""
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
