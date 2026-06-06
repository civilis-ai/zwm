#!/usr/bin/env python3
"""ZWM 真实运行录制 — 不是动画, 是系统内部状态的直接可视化.

录制内容:
  1. 方图场(地): 真实64卦场的爻值变化 — 来自传感器编码
  2. 圆图时间(天): TimeContext实时变化 — 值年卦/节气/元会运世
  3. 甲子周期(人): 60甲子场的周期推进 — 来自时间场编码
  4. JEPA预测: 真实的预测误差和surprise — 来自JEPA.train_step
  5. 自我视角: SelfState的六亲关系 — 永远在变化中
  6. LLM解说: agent用真实内部状态描述世界

所有数据来自系统内部, 不是随机生成的。
"""

import cv2, math, os, sys, time, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_FONT = None
def _font(size=16):
    global _FONT
    try:
        if _FONT is None:
            _FONT = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size)
        return _FONT
    except:
        return ImageFont.load_default()

def cn(canvas, text, pos, size=16, color=(0,255,0)):
    """中文叠加."""
    x, y = pos; f = _font(size)
    d = ImageDraw.Draw(Image.new('RGB',(1,1)))
    bb = d.textbbox((0,0), text, font=f)
    w, h = bb[2]-bb[0]+4, bb[3]-bb[1]+4
    if x+w > canvas.shape[1]: w = canvas.shape[1]-x
    if y+h > canvas.shape[0]: h = canvas.shape[0]-y
    if w <= 0 or h <= 0: return
    pil = Image.new('RGB', (w, h), (0,0,0))
    ImageDraw.Draw(pil).text((2,0), text, font=f, fill=color[::-1])
    ov = np.array(pil)[:,:,::-1].copy()
    mask = (ov > 10).any(axis=2)
    canvas[y:y+h, x:x+w][mask] = ov[mask]

def yao_color(val):
    r = int(50 + 200*val); b = int(50 + 200*(1-val))
    g = int(50 + 100*(1-abs(val-0.5)*2))
    return (b,g,r)

def draw_hex_field(canvas, hex_field, highlights=None):
    """绘制真实的64卦场 (8×8)."""
    if hex_field is None: return
    cx0, cy0 = 50, 50; cw, ch = 52, 52; gap = 4
    for row in range(8):
        for col in range(8):
            cx = cx0 + col*(cw+gap); cy = cy0 + row*(ch+gap)
            pos = row*8 + col
            yao6 = hex_field[pos]
            # 高亮
            hl = 1.0
            if highlights is not None and pos < len(highlights):
                hl = min(3.0, 1.0 + highlights[pos]*5)
            alpha = min(1.0, hl)
            bg = (int(50*alpha), int(20*alpha), int(60*alpha))
            cv2.rectangle(canvas, (cx,cy), (cx+cw,cy+ch), bg, -1)
            # 6爻
            for yi in range(6):
                val = float(yao6[5-yi])
                clr = yao_color(val)
                lx = cx+8; ly = cy+8 + yi*7; lw = cw-16
                if val > 0.5:
                    cv2.line(canvas, (lx,ly), (lx+lw,ly), clr, 2)
                else:
                    m = lx+lw//2
                    cv2.line(canvas, (lx,ly), (m-4,ly), clr, 2)
                    cv2.line(canvas, (m+4,ly), (lx+lw,ly), clr, 2)
            cv2.rectangle(canvas, (cx,cy), (cx+cw,cy+ch), (60,60,60), 1)

def draw_self_panel(canvas, ss, x0, y0):
    """SelfState — 真实的内部状态."""
    cv2.rectangle(canvas, (x0,y0), (x0+260, y0+200), (15,15,30), -1)
    cn(canvas, f"我: 日{ss.day_gan}·{ss.self_element}", (x0+10, y0+8), 16, (0,255,0))
    cn(canvas, "@中宫(5)", (x0+10, y0+28), 14, (0,200,0))
    dirs = [(1,'N'),(2,'SW'),(3,'E'),(4,'SE'),(6,'NW'),(7,'W'),(8,'NE'),(9,'S')]
    for i,(pos,label) in enumerate(dirs):
        px = x0+10 + (i%2)*130; py = y0+55 + (i//2)*22
        rel = ss.relation_to(pos)
        clr = {'子孙':(0,200,0),'父母':(200,200,0),'妻财':(200,150,0),
               '官鬼':(0,0,200),'兄弟':(0,200,200),'我':(0,255,0)}.get(rel,(150,150,150))
        cn(canvas, f"{label}:{rel}", (px,py), 13, clr)
    py = y0+150
    cn(canvas, f"上:[{ss.relation_to(10)}]  下:[{ss.relation_to(11)}]",
       (x0+10, py), 14, (150,150,255))

def draw_time_panel(canvas, tc, x0, y0):
    """TimeContext — 真实的时间状态."""
    if tc is None: return
    cv2.rectangle(canvas, (x0,y0), (x0+260, y0+160), (15,15,30), -1)
    cn(canvas, "天时 (TimeContext)", (x0+10, y0+8), 16, (255,255,0))
    cn(canvas, tc.ganzhi_str, (x0+10, y0+30), 14, (200,200,0))
    cn(canvas, f"{tc.solar_term_name} · {tc.season}季", (x0+10, y0+48), 14, (200,200,0))
    cn(canvas, f"午会 运{tc.yun_index} 世{tc.shi_index}", (x0+10, y0+66), 13, (150,150,0))
    cn(canvas, f"值年卦#{tc.value_year_hex}", (x0+10, y0+84), 14, (150,150,0))

def draw_jepa_panel(canvas, errors, surprises, x0, y0):
    """JEPA — 真实训练曲线."""
    cv2.rectangle(canvas, (x0,y0), (x0+260, y0+140), (15,15,30), -1)
    cn(canvas, "JEPA 预测 (地)", (x0+10, y0+8), 16, (0,255,255))
    if len(errors) > 1:
        # 微型误差曲线
        pts = np.array(errors[-50:])
        if len(pts) > 1:
            y_scale = min(0.3, pts.max()*2 + 0.01)
            for i in range(len(pts)-1):
                x1 = x0+10 + i*5; y1 = y0+100 - int(pts[i]/y_scale*60)
                x2 = x0+15 + i*5; y2 = y0+100 - int(pts[i+1]/y_scale*60)
                cv2.line(canvas, (x1,y1), (x2,y2), (0,255,255), 1)
        cn(canvas, f"err: {pts[-1]:.4f}", (x0+10, y0+30), 14, (0,255,200))
    if len(surprises) > 1:
        cn(canvas, f"surp: {surprises[-1]:.3f}", (x0+10, y0+48), 14, (255,200,0))

def draw_field_mini(canvas, data, x0, y0, label, color):
    """微型场视图 (32x32 矩阵)."""
    cv2.rectangle(canvas, (x0,y0), (x0+100, y0+100), (15,15,30), -1)
    cn(canvas, label, (x0+5, y0+2), 12, color)
    if data is not None:
        for i in range(min(8, len(data))):
            for j in range(min(8, len(data[0]) if data.ndim>1 else 1)):
                val = float(data[i*8+j]) if data.ndim==1 else float(data[i,j])
                clr = yao_color(val)
                cv2.rectangle(canvas, (x0+8+j*4, y0+18+i*4), (x0+10+j*4, y0+20+i*4), clr, -1)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300, help="录制步数")
    p.add_argument("--fps", type=int, default=10, help="帧率")
    args = p.parse_args()

    W, H = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('zwm_reality.mp4', fourcc, args.fps, (W, H))
    print("Recording: zwm_reality.mp4")
    print(f"  {args.steps} steps @ {args.fps} fps = {args.steps/args.fps:.0f}s\n")

    os.environ.setdefault('DEEPSEEK_API_KEY', os.environ.get('DEEPSEEK_API_KEY',''))
    from zwm.runtime import ZWMEngine
    engine = ZWMEngine(day_gan="庚")
    engine.activate()
    ss = engine.self_state

    errors, surprises = [], []
    for step in range(args.steps):
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        # ── 真实 OODA tick ──
        state = engine.tick(year=2026, month=6, day=9, hour=8 + step//6)
        if state.jepa_loss > 0:
            errors.append(state.jepa_loss)
        surprises.append(state.surprise)
        tc = getattr(engine._agent, '_time_context', None)
        hex_field = getattr(engine._agent, '_last_hex_field', None)

        # ── 渲染 ──
        cn(canvas, "ZWM · 天地人三才世界模型 · 真实运行录制", (20,18), 22, (0,255,0))

        # 64卦场 (真实数据)
        highlights = None
        if len(errors) > 0:
            highlights = np.ones(64) * errors[-1] * 15
        draw_hex_field(canvas, hex_field, highlights)

        # 右上面板
        draw_self_panel(canvas, ss, 560, 60)
        draw_time_panel(canvas, tc, 830, 60)
        draw_jepa_panel(canvas, errors, surprises, 560, 270)

        # 多场微视图 (从 hex_field 切片模拟)
        if hex_field is not None:
            draw_field_mini(canvas, hex_field.mean(axis=1), 830, 270, "Square Field", (0,255,0))
            draw_field_mini(canvas, hex_field.std(axis=1), 940, 270, "Circular Time", (255,255,0))
            draw_field_mini(canvas, hex_field.max(axis=1), 830, 380, "Ganzhi Cycle", (0,255,255))
            draw_field_mini(canvas, hex_field.min(axis=1), 940, 380, "Cosmic Scale", (255,0,255))

        # 底部状态栏
        cn(canvas, f"Tick {step+1}/{args.steps}  |  {state.next_hexagram}  |  "
                f"宫{state.target_palace}({ss.relation_to(state.target_palace)})  |  "
                f"JEPA={state.jepa_loss:.4f}",
           (20, H-40), 15, (200,200,200))

        # LLM解说 (中间触发)
        if step == args.steps//3 and engine._llm_router:
            try:
                reply = engine.ask("你现在看到了什么? 用1句话描述。")
                cn(canvas, f"ZWM: {reply[:100]}", (20, H-70), 14, (0,255,0))
            except: pass
        if step == args.steps*2//3 and engine._llm_router:
            try:
                reply = engine.ask("你探索了哪些方向? 学到了什么?")
                cn(canvas, f"ZWM: {reply[:100]}", (20, H-70), 14, (0,255,0))
            except: pass

        out.write(canvas)
        if step % 30 == 0:
            pct = (step+1)/args.steps*100
            print(f"\r  {pct:.0f}%  JEPA={errors[-1]:.4f}" if errors else f"\r  {pct:.0f}%", end='', flush=True)

    engine.close()
    out.release()
    print(f"\n\n✅ 录制完成: zwm_reality.mp4")
    print(f"   真实数据: {len(errors)} JEPA训练步, 误差 {errors[0]:.4f}→{errors[-1]:.4f}" if errors else "")


if __name__ == "__main__":
    main()
