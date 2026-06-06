#!/usr/bin/env python3
"""ZWM 本地部署 — 摄像头 + 麦克风 + 扬声器 + 显示屏.

在你的电脑上运行一个完整的 ZWM 智能体:
  - 摄像头 → ZWMVisionField → 卦象场 → OODA
  - 麦克风 → 语音识别 → LLM 理解 → ReAct
  - 扬声器 → ZWM 语音回复
  - 显示屏 → 实时状态面板 (自我/六亲/JEPA/卦象)

用法:
    python scripts/deploy_local.py              # 默认: 庚日, 中宫
    python scripts/deploy_local.py --day-gan 甲  # 甲日, 木属性
    python scripts/deploy_local.py --no-camera   # 不用摄像头
    python scripts/deploy_local.py --no-voice    # 静默模式

按键:
    [空格]  触发一次 OODA 循环
    [T]     语音输入
    [Q]     退出
    [1-9]   手动指定目标宫位
    [S]     显示状态摘要
"""

import cv2, math, os, sys, time, threading
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


class ZWMLocalAgent:
    """带摄像头+语音的本地 ZWM 智能体."""

    def __init__(self, day_gan: str = "庚", use_camera: bool = True, use_voice: bool = True):
        self._day_gan = day_gan
        self._use_camera = use_camera
        self._use_voice = use_voice
        self._cap = None
        self._engine = None
        self._running = False
        self._last_frame = None
        self._last_hex_field = None
        self._recognizer = None
        self._input_mode = False      # GUI 输入模式
        self._input_buffer = ""       # GUI 输入缓冲
        self._display_message = ""    # GUI 状态消息
        self._display_message_timer = 0  # 消息显示计时器

    def start(self):
        """启动智能体."""
        print("╔══════════════════════════════════════════════════════════╗")
        print(f"║  ZWM 本地智能体 — 日{self._day_gan}·{'金木水火土'[{'甲':0,'丙':1,'戊':2,'庚':3,'壬':4}.get(self._day_gan,0)]} @中宫{'':<30s}║")
        print("╚══════════════════════════════════════════════════════════╝")

        # 摄像头 — Windows 需要 DSHOW 后端
        if self._use_camera:
            for cam_idx in [0, 1]:
                self._cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
                if self._cap.isOpened():
                    # 预热: 读一帧确保摄像头真的在工作
                    ret, frame = self._cap.read()
                    if ret and frame is not None and frame.size > 0:
                        h, w = frame.shape[:2]
                        print(f"📷 摄像头就绪 (index={cam_idx}, {w}x{h})")
                        break
                    self._cap.release()
                    self._cap = None
            if self._cap is None or not self._cap.isOpened():
                print("⚠️  摄像头未找到 (尝试了 index 0,1)")
                print("   检查: 相机是否被其他应用占用? 隐私设置允许访问?")
                self._use_camera = False
                self._cap = None

        # 语音
        if self._use_voice:
            try:
                import pyttsx3
                self._tts = pyttsx3.init()
                self._tts.setProperty('rate', 180)
                print("🔊 语音输出就绪")
            except Exception:
                self._tts = None
                print("⚠️  TTS未找到, 仅文本输出")
            try:
                import sounddevice as sd
                self._sd = sd
                print("🎤 麦克风就绪")
            except Exception:
                self._sd = None

        # 引擎
        print("🧠 启动 ZWMEngine...")
        from zwm.runtime import ZWMEngine
        self._engine = ZWMEngine(day_gan=self._day_gan)
        self._engine.activate()
        print(f"   {self._engine.self_state}")
        print()

        # 视觉编码器
        if self._use_camera:
            from zwm.encoder.vision_field import ZWMVisionField
            self._vision = ZWMVisionField(backbone="hexvit")
            print("👁️  视觉编码器: HexViT 就绪")

        self._running = True
        self._last_tick = time.perf_counter()

    # ── 主循环 ──

    def tick(self) -> dict:
        """一次完整感知-思考-行动循环."""
        result = {"vision": None, "thought": "", "action": "", "jepa": 0.0}

        # 1. 视觉感知
        if self._use_camera and self._cap:
            ret, frame = self._cap.read()
            if ret:
                self._last_frame = cv2.resize(frame, (224, 224))
                # ZWMVisionField: image → (64,6) hex field
                hex_field = self._vision.encode(self._last_frame)
                self._last_hex_field = hex_field
                result["vision"] = hex_field
                # 从场中心提取主导卦
                center = hex_field[31]
                from zwm.core.yao import YANG, YIN
                from zwm.core.hexagram import Hexagram
                try:
                    h = Hexagram(*[YANG if s > 0.5 else YIN for s in center])
                except Exception:
                    h = None

        # 2. OODA
        state = self._engine.tick()
        result["jepa"] = state.jepa_loss
        result["thought"] = state.llm_thought
        result["action"] = state.next_hexagram
        result["target"] = state.target_palace
        return result

    # ── 语音 ──

    def speak(self, text: str):
        """说出文本."""
        print(f"  🗣️  {text}")
        if self._tts:
            try:
                self._tts.say(text)
                self._tts.runAndWait()
            except Exception:
                pass

    def listen(self) -> str | None:
        """监听麦克风, 返回识别文本."""
        if not self._sd:
            return None
        try:
            print("  🎤 正在听...(3秒)")
            fs = 16000
            recording = self._sd.rec(int(3 * fs), samplerate=fs, channels=1, dtype='float32')
            self._sd.wait()
            # 简单音量检测
            audio = recording.flatten()
            if np.abs(audio).max() < 0.02:
                return None
            print("  📝 已录音, 需要 STT 引擎来转文字")
            return None  # 没有STT引擎时返回None, 走键盘输入
        except Exception as e:
            print(f"  ⚠️ 录音失败: {e}")
            return None

    # ── 状态面板 ──

    def status_panel(self) -> str:
        """返回当前状态摘要."""
        ss = self._engine.self_state
        n = len(self._engine.history)
        last = self._engine.history[-1] if self._engine.history else None

        lines = [
            f"┌──── ZWM 状态 ────────────────────────────┐",
            f"│ 自我: 日{ss.day_gan}·{ss.self_element}, @中宫, {n} ticks     │",
            f"│ 六亲: 北子孙 南官鬼 东妻财 西兄弟          │",
        ]
        if last:
            lines.append(f"│ 最近: →{last.next_hexagram:<6s} JEPA={last.jepa_loss:.4f}    │")
        lines.append(f"│ 访问: {ss.palace_visits}                    │" if len(str(ss.palace_visits)) < 40 else f"│ 访问: {ss.total_visits}/8 宫位                     │")
        lines.append(f"└────────────────────────────────────────────┘")
        return "\n".join(lines)

    # ── 存活循环 ──

    def _render_display(self):
        """渲染状态面板到 OpenCV 窗口."""
        # 有摄像头画面则叠加, 没有则黑底
        if self._last_frame is not None:
            display = cv2.resize(self._last_frame, (448, 448))
            # 立体参考: 十字瞄准线 (水平八方 + 垂直天地)
            h, w = 448, 448
            cv2.line(display, (w//2, 0), (w//2, h), (0,255,0), 1)  # 竖线
            cv2.line(display, (0, h//2), (w, h//2), (0,255,0), 1)  # 横线
            cv2.circle(display, (w//2, h//2), 10, (0,255,0), 1)    # 中宫
        else:
            display = np.zeros((448, 448, 3), dtype=np.uint8)

        ss = self._engine.self_state
        last = self._engine.history[-1] if self._engine.history else None

        y = 25
        cv2.putText(display, f"ZWM | {ss.day_gan}·{ss.self_element} | Center",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        y += 30
        # 立体八方 + 上下
        cv2.putText(display, f"[{ss.relation_to(10)}] N={ss.relation_to(1)} S={ss.relation_to(9)} E={ss.relation_to(3)} W={ss.relation_to(7)} [{ss.relation_to(11)}]",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
        y += 22
        cv2.putText(display, f"天(上) · 人(中:八面) · 地(下)   layer: 天地人",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150,150,150), 1)
        y += 25
        if self._last_hex_field is not None:
            act = int((self._last_hex_field.mean(axis=1) > 0.5).sum())
            cv2.putText(display, f"Active: {act}/64 hexagrams",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            y += 25
        if last:
            cv2.putText(display, f"Tick {len(self._engine.history)} | {last.next_hexagram} | Gong {last.target_palace}",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            y += 25
            cv2.putText(display, f"JEPA={last.jepa_loss:.4f}",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            y += 30
            if last.agent_reply:
                for line in self._wrap_text(last.agent_reply, 70):
                    cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
                    y += 18
        # 输入模式提示
        if self._input_mode:
            cv2.putText(display, f"> {self._input_buffer}_",
                       (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(display, "[Enter]=send [Esc]=cancel",
                       (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
        else:
            cv2.putText(display, "[SPACE]=tick [T]=talk [1-9]=gong [S]=status [Q]=quit",
                       (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)

        # 状态消息 (短暂显示)
        if self._display_message and self._display_message_timer > 0:
            cv2.putText(display, self._display_message[:60],
                       (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
            self._display_message_timer -= 1

        return display

    @staticmethod
    def _wrap_text(text: str, width: int) -> list:
        """简单文本换行."""
        lines = []
        while len(text) > width:
            split = width
            while split > 0 and text[split] not in ' ,.;;:!?':
                split -= 1
            if split == 0:
                split = width
            lines.append(text[:split].strip())
            text = text[split:].strip()
        if text:
            lines.append(text)
        return lines

    def _capture_frame(self):
        """从摄像头捕获一帧."""
        if not self._cap or not self._cap.isOpened():
            return
        ret, frame = self._cap.read()
        if ret and frame is not None and frame.size > 0:
            self._last_frame = cv2.resize(frame, (224, 224))
            # 编码为卦象场
            if self._last_frame is not None:
                try:
                    self._last_hex_field = self._vision.encode(self._last_frame)
                except Exception:
                    pass

    def run(self, headless: bool = False):
        """主循环."""
        self.start()
        has_gui = not headless

        if has_gui:
            try:
                cv2.namedWindow("ZWM", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
                cv2.resizeWindow("ZWM", 500, 500)
                # 立即渲染初始画面
                display = self._render_display()
                cv2.imshow("ZWM", display)
                cv2.waitKey(1)
            except Exception as e:
                print(f"(GUI不可用: {e}, 文本模式)")
                has_gui = False

        print("\n" + "="*56)
        print("  [SPACE]=OODA [T]=对话 [1-9]=八方 [0]=上(天) [-]=下(地)")
        print("  [S]=状态 [Q]=退出")
        print("="*56 + "\n")

        tick_count = 0
        frame_counter = 0
        key = 0
        while self._running:
            # 每5帧读一次摄像头
            frame_counter += 1
            if has_gui and frame_counter % 5 == 0:
                self._capture_frame()

            if has_gui:
                display = self._render_display()
                cv2.imshow("ZWM", display)
                k = cv2.waitKey(30)
                if k == -1:
                    continue
                key = k & 0xFF
            else:
                try:
                    cmd = input("ZWM> ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    break
                if not cmd:
                    key = ord(' ')
                elif cmd in ('q', 'quit', 'exit'):
                    key = ord('q')
                elif cmd in ('s', 'status'):
                    key = ord('s')
                elif cmd in ('t', 'talk'):
                    key = ord('t')
                elif cmd in ('1','2','3','4','5','6','7','8','9'):
                    key = ord(cmd)
                elif cmd == 'auto':
                    n = 10
                    print(f"  自动运行 {n} 步...")
                    for i in range(n):
                        result = self.tick()
                        print(f"  [{tick_count+i+1}] {result['action']} g{result['target']} JEPA={result['jepa']:.4f}")
                    tick_count += n
                    continue
                elif cmd == 'learn':
                    losses = self._engine.learn(20)
                    print(f"  JEPA: {losses[0]:.4f} -> {losses[-1]:.4f}")
                    continue
                elif cmd == 'help':
                    print("  [回车]=tick [auto]=10步 [1-9]=宫 [t]=对话 [q]=退出")
                    continue
                else:
                    key = ord('t')

            # ── 输入模式: 收集字符 ──
            if self._input_mode:
                if key == 13:  # Enter
                    text = self._input_buffer.strip()
                    self._input_buffer = ""
                    self._input_mode = False
                    if text:
                        self._display_message = f"Sending: {text[:40]}..."
                        self._display_message_timer = 30
                        state = self._engine.execute(text)
                        self._display_message = f"ZWM: {state.agent_reply[:80]}"
                        self._display_message_timer = 120
                elif key == 27:  # Esc
                    self._input_buffer = ""
                    self._input_mode = False
                elif key == 8:  # Backspace
                    self._input_buffer = self._input_buffer[:-1]
                elif 32 <= key <= 126:  # printable ASCII
                    self._input_buffer += chr(key)
                continue

            # ── 正常模式: 处理命令键 ──
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                tick_count += 1
                result = self.tick()
                self._display_message = f"Tick {tick_count}: {result['action']} g{result['target']} JEPA={result['jepa']:.4f}"
                self._display_message_timer = 60
            elif key == ord('t'):
                self._input_mode = True
                self._input_buffer = ""
                self._display_message = "Type your message, Enter to send"
                self._display_message_timer = 30
            elif key == ord('s'):
                self._display_message = self.status_panel().replace('\n', ' | ')
                self._display_message_timer = 180
            elif ord('1') <= key <= ord('9'):
                palace = key - ord('0')
                ss = self._engine.self_state
                rel = ss.relation_to(palace)
                self._display_message = f"G{palace}({rel})..."
                self._display_message_timer = 15
                state = self._engine.execute(f"探索宫位{palace}")
                self._display_message = f"G{palace}({rel}): {state.next_hexagram}"
                self._display_message_timer = 60
            elif key == ord('0'):  # 0 = 上(天)
                ss = self._engine.self_state
                self._display_message = f"Looking UP (天)..."
                self._display_message_timer = 15
                state = self._engine.execute(f"向上看")
                self._display_message = f"天(上): {state.next_hexagram}"
                self._display_message_timer = 60
            elif key == ord('-'):  # - = 下(地)
                ss = self._engine.self_state
                self._display_message = f"Looking DOWN (地)..."
                self._display_message_timer = 15
                state = self._engine.execute(f"向下看")
                self._display_message = f"地(下): {state.next_hexagram}"
                self._display_message_timer = 60

        self.stop()

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if self._engine:
            self._engine.close()
        print("\nZWM 已关闭.")


# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    p = argparse.ArgumentParser(description="ZWM 本地部署 — 摄像头+语音+智能体")
    p.add_argument("--day-gan", default="庚", help="日干 (默认: 庚=金)")
    p.add_argument("--no-camera", action="store_true", help="不使用摄像头")
    p.add_argument("--no-voice", action="store_true", help="不使用语音")
    args = p.parse_args()

    agent = ZWMLocalAgent(
        day_gan=args.day_gan,
        use_camera=not args.no_camera,
        use_voice=not args.no_voice,
    )
    try:
        agent.run()
    except KeyboardInterrupt:
        agent.stop()


if __name__ == "__main__":
    main()
