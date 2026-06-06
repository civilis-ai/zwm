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

    def start(self):
        """启动智能体."""
        print("╔══════════════════════════════════════════════════════════╗")
        print(f"║  ZWM 本地智能体 — 日{self._day_gan}·{'金木水火土'[{'甲':0,'丙':1,'戊':2,'庚':3,'壬':4}.get(self._day_gan,0)]} @中宫{'':<30s}║")
        print("╚══════════════════════════════════════════════════════════╝")

        # 摄像头
        if self._use_camera:
            self._cap = cv2.VideoCapture(0)
            if not self._cap.isOpened():
                print("⚠️  摄像头未找到, 使用模拟视觉")
                self._use_camera = False
            else:
                print("📷 摄像头就绪")

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

    def run(self):
        """主循环 — 等待按键触发."""
        self.start()

        print("\n" + "="*56)
        print("  按键: [SPACE]=OODA [T]=说话 [S]=状态 [1-9]=目标 [Q]=退出")
        print("="*56 + "\n")

        cv2.namedWindow("ZWM", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ZWM", 448, 448)

        tick_count = 0
        while self._running:
            # 显示摄像头 + 状态覆盖
            if self._last_frame is not None:
                display = self._last_frame.copy()
                # 画九宫格
                h, w = display.shape[:2]
                for i in range(1, 3):
                    cv2.line(display, (w*i//3, 0), (w*i//3, h), (0,255,0), 1)
                    cv2.line(display, (0, h*i//3), (w, h*i//3), (0,255,0), 1)
                # 状态文字
                ss = self._engine.self_state
                cv2.putText(display, f"日{ss.day_gan}·{ss.self_element} @中宫",
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                if self._last_hex_field is not None:
                    act = int((self._last_hex_field.mean(axis=1) > 0.5).sum())
                    cv2.putText(display, f"活跃卦: {act}/64",
                               (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                cv2.imshow("ZWM", display)

            key = cv2.waitKey(100) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord(' '):  # Space = tick
                tick_count += 1
                result = self.tick()
                print(f"[{tick_count}] →{result['action']:<6s} "
                      f"宫{result['target']} JEPA={result['jepa']:.4f}")
                # 每5步说一句话
                if tick_count % 5 == 0 and self._tts:
                    ss = self._engine.self_state
                    self.speak(f"第{tick_count}步, 当前卦{result['action']}, "
                              f"我在中宫, 日{ss.day_gan}·{ss.self_element}")

            elif key == ord('t'):  # T = talk
                text = input("  你说: ").strip()
                if text:
                    state = self._engine.execute(text)
                    print(f"  → {state.agent_reply}")
                    if self._tts:
                        self.speak(state.agent_reply[:100])

            elif key == ord('s'):  # S = status
                print(self.status_panel())

            elif ord('1') <= key <= ord('9'):  # 数字键 = 目标宫位
                palace = key - ord('0')
                ss = self._engine.self_state
                rel = ss.relation_to(palace)
                harmony = ss.harmony_score(palace)
                state = self._engine.execute(f"去宫{palace}")
                print(f"  目标宫{palace}({rel}, 和谐度{harmony:.1f}) → {state.next_hexagram}")

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
