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
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# 中文字体路径 (Windows)
_FONT_PATH = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑
_FONT_SM = None
_FONT_MD = None
_FONT_LG = None


def _get_font(size):
    global _FONT_SM, _FONT_MD, _FONT_LG
    try:
        if size <= 14:
            if _FONT_SM is None:
                _FONT_SM = ImageFont.truetype(_FONT_PATH, 14)
            return _FONT_SM
        elif size <= 20:
            if _FONT_MD is None:
                _FONT_MD = ImageFont.truetype(_FONT_PATH, 18)
            return _FONT_MD
        else:
            if _FONT_LG is None:
                _FONT_LG = ImageFont.truetype(_FONT_PATH, 24)
            return _FONT_LG
    except Exception:
        return ImageFont.load_default()


def put_text_cn(canvas, text, pos, font_size=16, color=(0, 255, 0)):
    """用 PIL 渲染中文到 OpenCV canvas."""
    x, y = pos
    font = _get_font(font_size)
    # 测量文本
    dummy = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # 渲染到 PIL image
    pil_img = Image.new('RGB', (tw + 4, th + 4), (0, 0, 0))
    draw = ImageDraw.Draw(pil_img)
    draw.text((2, 0), text, font=font, fill=color[::-1])  # RGB→BGR

    # 转为 numpy 并叠加到 canvas
    overlay = np.array(pil_img)[:, :, ::-1].copy()  # RGB→BGR
    if overlay.shape[0] > 0 and overlay.shape[1] > 0:
        h, w = overlay.shape[:2]
        y2 = min(y + h, canvas.shape[0])
        x2 = min(x + w, canvas.shape[1])
        h2, w2 = y2 - y, x2 - x
        if h2 > 0 and w2 > 0:
            mask = (overlay[:h2, :w2] > 10).any(axis=2)
            canvas[y:y2, x:x2][mask] = overlay[:h2, :w2][mask]


class FieldVisualizer:
    """64卦场实时可视化引擎."""

    # 颜色映射: 阴(0)蓝 → 阳(1)红
    @staticmethod
    def yao_color(val):
        r = int(50 + 200 * val)
        g = int(50 + 100 * (1 - abs(val - 0.5) * 2))
        b = int(50 + 200 * (1 - val))
        return (b, g, r)

    def __init__(self, width=1600, height=900):
        self.W, self.H = width, height
        self._history = []
        self._step = 0

    def draw_hexagram_cell(self, canvas, y, x, yao6, highlight=1.0):
        """在画布上绘制一个卦象单元 (6条爻线)."""
        cell_w, cell_h = 80, 80
        gap = 6
        cx = 50 + x * (cell_w + gap)
        cy = 60 + y * (cell_h + gap)
        w, h = cell_w, cell_h

        # 背景 (高亮=预测误差大)
        alpha = min(1.0, highlight)
        bg_color = (int(50 * alpha), int(30 * alpha), int(80 * alpha))
        cv2.rectangle(canvas, (cx, cy), (cx + w, cy + h), bg_color, -1)

        # 6条爻线 (上→下), 加粗
        for yi in range(6):
            val = float(yao6[5 - yi])
            color = self.yao_color(val)
            lx, ly = cx + 10, cy + 12 + yi * 10
            lw = w - 20
            if val > 0.5:
                cv2.line(canvas, (lx, ly), (lx + lw, ly), color, 3)
            else:
                mid = lx + lw // 2
                gap = 6
                cv2.line(canvas, (lx, ly), (mid - gap, ly), color, 3)
                cv2.line(canvas, (mid + gap, ly), (lx + lw, ly), color, 3)

        # 边框
        border = (60, 60, 60) if highlight < 1.5 else (0, 255, 255)
        cv2.rectangle(canvas, (cx, cy), (cx + w, cy + h), border, 1)

    def draw_jepa_panel(self, canvas, state):
        """JEPA 预测面板 — 右上角."""
        x0, y0 = 790, 30
        pw, ph = 420, 220
        cv2.rectangle(canvas, (x0, y0), (x0 + pw, y0 + ph), (15, 15, 35), -1)
        cv2.putText(canvas, "JEPA World Model", (x0 + 15, y0 + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
        """自我面板 — 右中部."""
        x0, y0 = 790, 270
        pw, ph = 420, 220
        cv2.rectangle(canvas, (x0, y0), (x0 + pw, y0 + ph), (15, 15, 35), -1)
        put_text_cn(canvas, f"Self: {ss.day_gan}·{ss.self_element} @ Center",
                   (x0 + 15, y0 + 30), 18, (0, 255, 0))

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
            put_text_cn(canvas, f"{label}:{rel}", (px, py), 14, color)

        # 天地层
        y += 85
        put_text_cn(canvas, f"[{ss.relation_to(10)}] UP  |  MID  |  DOWN [{ss.relation_to(11)}]",
                   (x0 + 10, y), 14, (150, 150, 255))

    def draw_multi_field(self, canvas, hex_field, tc):
        """多场共振视图 — 右下部 4个小面板."""
        x0, y0 = 790, 510
        fields = [("Square Field", (0, 255, 0)), ("Circular Time", (255, 255, 0)),
                  ("Ganzhi Cycle", (0, 255, 255)), ("Cosmic Scale", (255, 0, 255))]
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
        put_text_cn(canvas, "ZWM · 天地人三才世界模型 · Live", (20, 22), 24, (0, 255, 0))

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
            put_text_cn(canvas, f"ZWM: {reply}", (20, self.H - 28), 16, (0, 255, 0))

        self._step += 1
        return canvas


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--record", action="store_true", help="录制视频文件")
    p.add_argument("--fullscreen", action="store_true", help="全屏")
    p.add_argument("--steps", type=int, default=500, help="动画步数")
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

    W, H = 1600, 900
    viz = FieldVisualizer(W, H)
    cv2.namedWindow("ZWM Live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ZWM Live", W, H)

    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('zwm_field_animation.mp4', fourcc, 10, (W, H))
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

        # 中途让 LLM 解说 (约在第 12 秒)
        if tick_count == 30 and engine._llm_router:
            try:
                reply = engine.ask("你现在看到了什么? 用1-2句话描述你眼前的世界状态。")
                if engine_state:
                    engine_state.agent_reply = reply
            except Exception:
                pass
        # 结尾再来一句
        if tick_count == 55 and engine._llm_router:
            try:
                reply = engine.ask("你探索了几个方向? 学到了什么? 用1-2句话总结。")
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
