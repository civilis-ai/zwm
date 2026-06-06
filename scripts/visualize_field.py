#!/usr/bin/env python3
"""ZWM 卦象场实时可视化 — 世界从未见过的动画.

展示:
  1. 64卦场实时流动 — 每个卦的6爻在翻转, 颜色从阴(蓝)到阳(红)
  2. JEPA预测 vs 现实 — 预测误差高亮显示
  3. 第一人称视角 — 中宫我 + 八方六亲 + 天地层
  4. 多场共振 — 方图/圆图/干支/元会运世四场同屏
  5. LLM解说 — agent 用自然语言描述自己看到的世界

输出: 实时OpenCV窗口 + 可选视频录制

用法:
  python scripts/visualize_field.py                  # 实时显示
  python scripts/visualize_field.py --record         # 录制视频
  python scripts/visualize_field.py --fullscreen     # 全屏
"""

import cv2, math, os, sys, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


class FieldVisualizer:
    """64卦场实时可视化引擎."""

    # 颜色映射: 阴(0)蓝 → 阳(1)红
    @staticmethod
    def yao_color(val):
        r = int(50 + 200 * val)         # 红: 50→250
        g = int(50 + 100 * (1 - abs(val - 0.5) * 2))  # 绿: 中间高
        b = int(50 + 200 * (1 - val))   # 蓝: 250→50
        return (b, g, r)

    def __init__(self, width=1200, height=800):
        self.W, self.H = width, height
        self._history = []   # 历史帧缓存
        self._step = 0

    def draw_hexagram_cell(self, canvas, y, x, yao6, highlight=1.0):
        """在画布上绘制一个卦象单元 (6条爻线)."""
        cx = 60 + x * 56
        cy = 60 + y * 56
        w, h = 48, 48

        # 背景 (高亮=预测误差大)
        alpha = min(1.0, highlight)
        bg_color = (int(50 * alpha), int(30 * alpha), int(80 * alpha))
        cv2.rectangle(canvas, (cx, cy), (cx + w, cy + h), bg_color, -1)

        # 6条爻线 (上→下)
        for yi in range(6):
            val = float(yao6[5 - yi])  # 上爻先画
            color = self.yao_color(val)
            lx, ly = cx + 6, cy + 8 + yi * 6
            lw = w - 12
            # 阳爻: 实线; 阴爻: 中间断开
            if val > 0.5:
                cv2.line(canvas, (lx, ly), (lx + lw, ly), color, 2)
            else:
                mid = lx + lw // 2
                cv2.line(canvas, (lx, ly), (mid - 4, ly), color, 2)
                cv2.line(canvas, (mid + 4, ly), (lx + lw, ly), color, 2)

        # 边框
        border = (80, 80, 80) if highlight < 1.5 else (0, 255, 255)
        cv2.rectangle(canvas, (cx, cy), (cx + w, cy + h), border, 1)

    def draw_jepa_panel(self, canvas, state):
        """JEPA 预测面板 — 右上角."""
        x0, y0 = 580, 30
        cv2.rectangle(canvas, (x0, y0), (x0 + 360, y0 + 200), (20, 20, 40), -1)
        cv2.putText(canvas, "JEPA World Model", (x0 + 10, y0 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        y = y0 + 55
        last = state
        jepa_loss = getattr(last, 'jepa_loss', 0) if last else 0
        surprise = getattr(last, 'surprise', 0) if last else 0

        # 预测误差条
        bar_w = int(min(300, jepa_loss * 800))
        bar_color = (0, 255, 0) if jepa_loss < 0.05 else (0, 255, 255) if jepa_loss < 0.1 else (0, 0, 255)
        cv2.putText(canvas, f"Pred Error: {jepa_loss:.4f}", (x0 + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.rectangle(canvas, (x0 + 10, y + 5), (x0 + 10 + bar_w, y + 17), bar_color, -1)

        # Surprise
        y += 40
        surp_w = int(min(300, surprise * 80))
        cv2.putText(canvas, f"Surprise: {surprise:.3f}", (x0 + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.rectangle(canvas, (x0 + 10, y + 5), (x0 + 10 + surp_w, y + 17), (255, 200, 0), -1)

        # 卦象
        y += 40
        hex_name = getattr(last, 'next_hexagram', '?') if last else '?'
        target = getattr(last, 'target_palace', 5) if last else 5
        cv2.putText(canvas, f"Hex: {hex_name}  Palace: {target}",
                   (x0 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    def draw_self_panel(self, canvas, ss):
        """自我面板 — 左下角."""
        x0, y0 = 580, 250
        cv2.rectangle(canvas, (x0, y0), (x0 + 360, y0 + 200), (20, 20, 40), -1)
        cv2.putText(canvas, f"Self: {ss.day_gan}·{ss.self_element} @Center",
                   (x0 + 10, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        y = y0 + 55
        # 八方六亲
        dirs = [(1, 'N'), (2, 'SW'), (3, 'E'), (4, 'SE'),
                (6, 'NW'), (7, 'W'), (8, 'NE'), (9, 'S')]
        for i, (pos, label) in enumerate(dirs):
            col = i % 4
            row = i // 4
            px = x0 + 15 + col * 90
            py = y + row * 35
            rel = ss.relation_to(pos)
            color = {'子孙': (0, 200, 0), '父母': (200, 200, 0), '妻财': (200, 150, 0),
                     '官鬼': (0, 0, 200), '兄弟': (0, 200, 200), '我': (0, 255, 0)}.get(rel, (150, 150, 150))
            cv2.putText(canvas, f"{label}:{rel}", (px, py),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 天地层
        y += 85
        cv2.putText(canvas, f"[{ss.relation_to(10)}] UP  |  MID  |  DOWN [{ss.relation_to(11)}]",
                   (x0 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 255), 1)

    def draw_multi_field(self, canvas, hex_field, tc):
        """多场共振视图 — 右下角 4个小面板."""
        x0, y0 = 580, 470
        fields = [("Square", (0, 255, 0)), ("Circular", (255, 255, 0)),
                  ("Ganzhi", (0, 255, 255)), ("Cosmic", (255, 0, 255))]
        for i, (name, color) in enumerate(fields):
            fx = x0 + (i % 2) * 180
            fy = y0 + (i // 2) * 120
            cv2.rectangle(canvas, (fx, fy), (fx + 170, fy + 110), (30, 30, 50), -1)
            cv2.putText(canvas, name, (fx + 5, fy + 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            # 简化: 用小方块表示场激活
            if hex_field is not None:
                for row in range(4):
                    for col in range(8):
                        idx = (i * 16 + row * 4 + col) % 64
                        val = float(hex_field[idx].mean()) if idx < 64 else 0.5
                        clr = self.yao_color(val)
                        cv2.rectangle(canvas,
                                     (fx + 8 + col * 5, fy + 25 + row * 5),
                                     (fx + 11 + col * 5, fy + 28 + row * 5), clr, -1)

    def render(self, hex_field=None, engine_state=None, ss=None):
        """渲染一帧."""
        canvas = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        # 标题
        cv2.putText(canvas, "ZWM · 天地人三才世界模型 · Live", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ── 左半: 64卦场 (8×8) ──
        if hex_field is not None:
            # 计算 JEPA 预测误差高亮 (如果有)
            highlights = np.ones(64, dtype=np.float32)
            if engine_state:
                hl = min(3.0, getattr(engine_state, 'jepa_loss', 0) * 30)
                highlights = np.ones(64) * hl

            for row in range(8):
                for col in range(8):
                    pos = row * 8 + col
                    self.draw_hexagram_cell(
                        canvas, row, col, hex_field[pos],
                        highlight=highlights[pos] if pos < 64 else 1.0,
                    )

        # ── 右半: 面板 ──
        if engine_state:
            self.draw_jepa_panel(canvas, engine_state)
        if ss:
            self.draw_self_panel(canvas, ss)
        if hex_field is not None:
            self.draw_multi_field(canvas, hex_field, None)

        # 底部: LLM解说
        if engine_state and hasattr(engine_state, 'agent_reply') and engine_state.agent_reply:
            reply = engine_state.agent_reply[:120]
            cv2.putText(canvas, f"ZWM: {reply}", (20, self.H - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        self._step += 1
        return canvas


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--record", action="store_true", help="录制视频文件")
    p.add_argument("--fullscreen", action="store_true", help="全屏")
    p.add_argument("--steps", type=int, default=200, help="动画步数")
    args = p.parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║  ZWM 卦象场实时可视化 — World Model Live             ║")
    print("╚══════════════════════════════════════════════════════╝")
    print("  64 hexagrams × 6 yao, evolving in real-time")
    print("  JEPA prediction vs reality, self-perspective, multi-field resonance")
    print()

    from zwm.runtime import ZWMEngine
    engine = ZWMEngine(day_gan="庚")
    engine.activate()
    ss = engine.self_state

    viz = FieldVisualizer(1200, 800)
    cv2.namedWindow("ZWM Live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ZWM Live", 1200, 800)

    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('zwm_field_animation.mp4', fourcc, 10, (1200, 800))
        print("Recording: zwm_field_animation.mp4")

    hex_field = None
    engine_state = None
    tick_count = 0
    pause = False
    frame = 0

    engine.ask  # trigger LLM auto-detect

    while frame < args.steps:
        canvas = viz.render(hex_field, engine_state, ss)

        # 每5帧自动走一步 OODA
        if not pause and frame % 8 == 0:
            try:
                s = engine.tick()
                engine_state = s
                tick_count += 1
                hex_field = getattr(engine._agent, '_last_hex_field', None)
            except Exception:
                pass

        # 第20步时让 LLM 解说
        if tick_count == 20 and engine._llm_router:
            try:
                reply = engine.ask("你现在看到了什么? 用1-2句话描述你眼前的世界状态。")
                if engine_state:
                    engine_state.agent_reply = reply
            except Exception:
                pass

        cv2.imshow("ZWM Live", canvas)
        if args.record:
            out.write(canvas)

        k = cv2.waitKey(80) & 0xFF
        if k == ord('q') or k == 27:
            break
        elif k == ord(' '):
            pause = not pause
        frame += 1

    engine.close()
    cv2.destroyAllWindows()
    if args.record:
        out.release()
        print(f"\nVideo saved: zwm_field_animation.mp4 ({args.steps} frames)")


if __name__ == "__main__":
    main()
