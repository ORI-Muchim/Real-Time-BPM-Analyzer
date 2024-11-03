import soundcard as sc
import numpy as np
import time
import warnings
import serial
import serial.tools.list_ports
from threading import Thread, Lock, Event
from queue import Queue
from scipy.signal import find_peaks, correlate
from collections import deque
import traceback
import gc
import scipy.signal
import librosa
from typing import Optional, Dict, List, Deque
from dataclasses import dataclass

# 경고 메시지 무시
warnings.filterwarnings("ignore", category=sc.SoundcardRuntimeWarning)

@dataclass
class Config:
    SAMPLE_RATE: int = 44100
    BUFFER_DURATION: int = 5
    BUFFER_SIZE: int = SAMPLE_RATE * BUFFER_DURATION
    CHUNK_SIZE: int = 1024
    MAX_QUEUE_SIZE: int = 50
    BAUD_RATE: int = 115200
    SILENCE_THRESHOLD: float = 0.01
    SILENCE_DURATION: int = 2
    MIN_SOUND_DURATION: int = 2
    MIN_COMMAND_INTERVAL: float = 0.1
    BPM_MIN: int = 60
    BPM_MAX: int = 300
    HOP_LENGTH: int = 512

@dataclass
class BPMDeviationTracker:
    last_bpm: Optional[float] = None
    count: int = 0

class AudioProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.default_speaker = sc.default_speaker()
        self.mic = sc.get_microphone(
            id=str(self.default_speaker.name), 
            include_loopback=True
        )
        print(f"기본 스피커: {self.default_speaker.name}")
        print(f"마이크 설정 완료: {self.mic.name}")

    def compute_onset_envelope(self, data: np.ndarray) -> np.ndarray:
        return librosa.onset.onset_strength(y=data, sr=self.config.SAMPLE_RATE)

    def autocorrelation_bpm(self, onset_env: np.ndarray) -> Optional[float]:
        try:
            corr = correlate(onset_env, onset_env, mode='full')
            corr = corr[len(corr)//2:]
            
            lag_min = int(self.config.SAMPLE_RATE * 60 / self.config.BPM_MAX / self.config.HOP_LENGTH)
            lag_max = int(self.config.SAMPLE_RATE * 60 / self.config.BPM_MIN / self.config.HOP_LENGTH)
            
            corr = corr[lag_min:lag_max]
            if len(corr) == 0:
                return None
                
            peaks, _ = find_peaks(corr)
            if len(peaks) == 0:
                return None
                
            peak_lags = peaks + lag_min
            bpms = 60 * self.config.SAMPLE_RATE / (peak_lags * self.config.HOP_LENGTH)
            peak_index = np.argmax(corr[peaks])
            return float(bpms[peak_index])
        except Exception as e:
            print("autocorrelation_bpm 함수에서 오류 발생:", e)
            traceback.print_exc()
            return None

    def combined_bpm_detector(self, data: np.ndarray) -> Optional[float]:
        rms = np.sqrt(np.mean(data**2))
        if rms < self.config.SILENCE_THRESHOLD:
            return None

        try:
            onset_env = self.compute_onset_envelope(data)
            return self.autocorrelation_bpm(onset_env)
        except Exception as e:
            print("combined_bpm_detector 함수에서 오류 발생:", e)
            traceback.print_exc()
            return None

class BPMStabilizer:
    def __init__(self, config: Config):
        self.config = config
        self.bpm_history: Deque[float] = deque(maxlen=10)
        self.deviation_tracker = BPMDeviationTracker()
        self.bpm_range_groups = {
            'slow': (60, 100),
            'medium': (100, 180),
            'fast': (180, 240),
            'very_fast': (240, 300)
        }

    @staticmethod
    def remove_outliers(bpms: List[float]) -> List[float]:
        if len(bpms) < 4:
            return bpms
        q1, q3 = np.percentile(bpms, [25, 75])
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - (1.5 * iqr), q3 + (1.5 * iqr)
        return [bpm for bpm in bpms if lower_bound <= bpm <= upper_bound]

    def normalize_bpm(self, bpm: float) -> float:
        """
        BPM을 가장 가능성 높은 범위로 정규화합니다.
        2배수/절반값 관계를 고려하여 실제 BPM에 가까운 값을 반환합니다.
        """
        if bpm <= 0:
            return bpm

        # 가능한 BPM 후보들 (1/4배, 1/2배, 원본, 2배, 4배)
        candidates = [bpm/4, bpm/2, bpm, bpm*2, bpm*4]
        
        # 유효한 BPM 범위 내의 후보들만 선택
        valid_candidates = [c for c in candidates if 60 <= c <= 300]
        
        if not valid_candidates:
            return min(max(bpm, 60), 300)  # 범위 제한
            
        # 현재 히스토리가 있는 경우
        if self.bpm_history:
            current_median = np.median(self.bpm_history)
            # 히스토리 중앙값과 가장 가까운 후보 선택
            closest = min(valid_candidates, key=lambda x: abs(x - current_median))
            
            # 만약 현재 중앙값과 너무 차이가 나는 경우, 배수 관계 재검토
            if abs(closest - current_median) / current_median > 0.5:
                harmonics = [c for c in valid_candidates if abs(c/current_median - round(c/current_median)) < 0.1]
                if harmonics:
                    closest = min(harmonics, key=lambda x: abs(x - current_median))
            
            return closest
            
        # 히스토리가 없는 경우, 가장 일반적인 범위(100-180 BPM)에 있는 값 선택
        medium_candidates = [c for c in valid_candidates if 100 <= c <= 180]
        if medium_candidates:
            return medium_candidates[0]
            
        # 적절한 범위를 찾지 못한 경우, 150 BPM에 가장 가까운 값 선택
        return min(valid_candidates, key=lambda x: abs(x - 150))

    def stabilize_bpm(self, new_bpm: Optional[float]) -> Optional[float]:
        if new_bpm is None:
            return None
            
        # BPM 정규화
        normalized_bpm = self.normalize_bpm(new_bpm)
        
        if not self.bpm_history:
            self.bpm_history.append(normalized_bpm)
            print(f"초기 BPM 설정: {normalized_bpm:.2f}")
            return normalized_bpm

        stabilized_bpm = np.median(self.bpm_history)
        tolerance = 0.15  # 허용 오차를 15%로 증가

        relative_diff = abs(normalized_bpm - stabilized_bpm) / stabilized_bpm
        
        if relative_diff > tolerance:
            # 배수 관계 확인 (2배, 1/2배, 4배, 1/4배)
            ratios = [0.25, 0.5, 1, 2, 4]
            harmonic_diffs = [abs(normalized_bpm / stabilized_bpm - r) for r in ratios]
            min_harmonic_diff = min(harmonic_diffs)
            
            # 배수 관계가 명확한 경우 현재 값을 유지
            if min_harmonic_diff < 0.1:
                print(f"배수 관계가 감지되어 현재 BPM {stabilized_bpm:.2f} 유지")
                return stabilized_bpm
                
            if self.deviation_tracker.last_bpm == normalized_bpm:
                self.deviation_tracker.count += 1
            else:
                self.deviation_tracker.last_bpm = normalized_bpm
                self.deviation_tracker.count = 1

            print(f"새로운 BPM 값 {normalized_bpm:.2f}이(가) 안정화된 BPM에서 너무 벗어남. (연속 {self.deviation_tracker.count}회)")

            if self.deviation_tracker.count >= 3:
                self.bpm_history.clear()
                self.bpm_history.append(normalized_bpm)
                stabilized_bpm = normalized_bpm
                print(f"새로운 BPM 값 {normalized_bpm:.2f}이(가) 3번 연속 감지되어 안정화된 BPM으로 업데이트되었습니다.")
                self.deviation_tracker.count = 0
        else:
            self.bpm_history.append(normalized_bpm)
            self.bpm_history = deque(self.remove_outliers(list(self.bpm_history)), maxlen=10)
            stabilized_bpm = float(np.median(self.bpm_history))
            print(f"새로운 BPM: {normalized_bpm:.2f}, 안정화된 BPM: {stabilized_bpm:.2f}")
            self.deviation_tracker.count = 0

        return stabilized_bpm

class ArduinoManager:
    def __init__(self):
        self.arduino = self._connect()
        self.lock = Lock()
        self.last_command_time = 0
        self.last_command = None

    def _find_arduino_port(self) -> Optional[str]:
        try:
            ports = [
                p.device
                for p in serial.tools.list_ports.comports()
                if 'Arduino' in p.description or 'CH340' in p.description
            ]
            if not ports:
                print("아두이노를 찾을 수 없습니다.")
                return None
            if len(ports) > 1:
                print("여러 개의 아두이노가 감지되었습니다. 첫 번째 포트를 사용합니다.")
            return ports[0]
        except Exception as e:
            print("find_arduino_port 함수에서 오류 발생:", e)
            traceback.print_exc()
            return None

    def _connect(self) -> Optional[serial.Serial]:
        try:
            port = self._find_arduino_port()
            if not port:
                return None

            for attempt in range(3):
                try:
                    arduino = serial.Serial(port, Config.BAUD_RATE, timeout=1, write_timeout=1)
                    time.sleep(2)
                    print(f"아두이노와 연결되었습니다. ({port})")
                    return arduino
                except serial.SerialException as e:
                    print(f"아두이노 연결 실패 (시도 {attempt + 1}/3): {str(e)}")
                    if "Access is denied" in str(e) or "에러 코드 31" in str(e):
                        time.sleep(2)
                    else:
                        break
            print("아두이노 연결 실패: 최대 시도 횟수 초과")
        except Exception as e:
            print("connect_arduino 함수에서 오류 발생:", e)
            traceback.print_exc()
        return None

    def reconnect(self) -> None:
        with self.lock:
            if self.arduino and self.arduino.is_open:
                self.arduino.close()
            self.arduino = self._connect()

    def send_command(self, command: str) -> None:
        if command == self.last_command:
            return

        current_time = time.time()
        if current_time - self.last_command_time >= Config.MIN_COMMAND_INTERVAL:
            with self.lock:
                try:
                    if self.arduino and self.arduino.is_open:
                        self.arduino.write(command.encode())
                        self.last_command_time = current_time
                        self.last_command = command
                except serial.SerialException as e:
                    print(f"시리얼 통신 오류: {str(e)}")
                    traceback.print_exc()

class AudioCapture(Thread):
    def __init__(self, config: Config, audio_processor: AudioProcessor):
        super().__init__()
        self.config = config
        self.audio_processor = audio_processor
        self.audio_queue = Queue(maxsize=config.MAX_QUEUE_SIZE)
        self.rms_history: Deque[float] = deque(maxlen=5)
        self.sound_lock = Lock()
        self.stop_event = Event()
        self.daemon = True

    def run(self) -> None:
        with self.audio_processor.mic.recorder(samplerate=self.config.SAMPLE_RATE) as rec:
            print("오디오 캡처 스레드 시작")
            while not self.stop_event.is_set():
                try:
                    data = rec.record(numframes=self.config.CHUNK_SIZE)
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    
                    rms = np.sqrt(np.mean(data**2))
                    self.rms_history.append(rms)
                    
                    if np.max(np.abs(data)) > 0:
                        data = data / np.max(np.abs(data))
                        
                    if self.audio_queue.qsize() < self.config.MAX_QUEUE_SIZE:
                        self.audio_queue.put(data)
                    else:
                        try:
                            self.audio_queue.get_nowait()
                        except Queue.Empty:
                            pass
                        self.audio_queue.put(data)
                except Exception as e:
                    print("오디오 캡처 중 오류 발생:", e)
                    traceback.print_exc()
                    time.sleep(0.1)

class BPMAnalyzer(Thread):
    def __init__(self, config: Config, audio_processor: AudioProcessor, audio_capture: AudioCapture):
        super().__init__()
        self.config = config
        self.audio_processor = audio_processor
        self.audio_capture = audio_capture
        self.bpm_queue = Queue()
        self.stop_event = Event()
        self.daemon = True

    def run(self) -> None:
        audio_buffer = deque(maxlen=self.config.BUFFER_SIZE)
        last_analysis_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                while not self.audio_capture.audio_queue.empty():
                    audio_buffer.extend(self.audio_capture.audio_queue.get_nowait())

                if (len(audio_buffer) >= self.config.BUFFER_SIZE and 
                    current_time - last_analysis_time > 1):
                    data = np.array(audio_buffer)
                    bpm = self.audio_processor.combined_bpm_detector(data)
                    
                    if bpm:
                        self.bpm_queue.put(bpm)
                    last_analysis_time = current_time
                    
                    del data
                    gc.collect()
                time.sleep(0.1)
            except Exception as e:
                print("BPM 분석 중 오류 발생:", e)
                traceback.print_exc()
                time.sleep(1)

class LightController(Thread):
    def __init__(self, arduino_manager: ArduinoManager):
        super().__init__()
        self.arduino_manager = arduino_manager
        self.bpm: Optional[float] = None
        self.last_beat_time = time.time()
        self.stop_event = Event()
        self.is_silent = False
        self.lock = Lock()
        self.light_on = False
        self.daemon = True

    def update_bpm(self, bpm: Optional[float]) -> None:
        with self.lock:
            if bpm is not None:  # BPM이 유효한 경우에만 업데이트
                self.bpm = bpm
                self.is_silent = False  # BPM 업데이트 시 자동으로 silent 모드 해제

    def set_silent(self, is_silent: bool) -> None:
        with self.lock:
            self.is_silent = is_silent
            if is_silent:
                self.bpm = None  # silent 모드일 때는 BPM도 초기화

    def run(self) -> None:
        while not self.stop_event.is_set():
            with self.lock:
                bpm = self.bpm
                is_silent = self.is_silent

            current_time = time.time()

            if is_silent or bpm is None:
                if self.light_on:
                    self.arduino_manager.send_command('0\n')
                    self.light_on = False
                time.sleep(0.1)
                continue

            beat_interval = 60.0 / bpm
            time_since_last_beat = current_time - self.last_beat_time

            if time_since_last_beat >= beat_interval:
                if not self.light_on:
                    self.arduino_manager.send_command('1\n')
                    self.light_on = True
                    print(f"\r💡 ON (BPM: {bpm:.2f})", end='', flush=True)

                off_duration = min(0.1, beat_interval / 2)
                time.sleep(off_duration)

                if self.light_on:
                    self.arduino_manager.send_command('0\n')
                    self.light_on = False
                    print(f"\r💡 OFF (BPM: {bpm:.2f})", end='', flush=True)

                self.last_beat_time = current_time
            else:
                time.sleep(min(beat_interval - time_since_last_beat, 0.1))

    def stop(self) -> None:
        self.stop_event.set()
        if self.light_on:
            self.arduino_manager.send_command('0\n')
                
class ArduinoMonitor(Thread):
    def __init__(self, arduino_manager: ArduinoManager):
        super().__init__()
        self.arduino_manager = arduino_manager
        self.stop_event = Event()
        self.daemon = True

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                if (self.arduino_manager.arduino is None or 
                    not self.arduino_manager.arduino.is_open):
                    print("아두이노가 연결되어 있지 않습니다. 재연결을 시도합니다.")
                    self.arduino_manager.reconnect()
            except Exception as e:
                print("아두이노 모니터링 중 오류 발생:", e)
                traceback.print_exc()
            time.sleep(5)

class AudioAnalysisSystem:
    def __init__(self):
        self.config = Config()
        self.audio_processor = AudioProcessor(self.config)
        self.arduino_manager = ArduinoManager()
        self.bpm_stabilizer = BPMStabilizer(self.config)
        
        # 스레드 초기화
        self.audio_capture = AudioCapture(self.config, self.audio_processor)
        self.bpm_analyzer = BPMAnalyzer(self.config, self.audio_processor, self.audio_capture)
        self.light_controller = LightController(self.arduino_manager)
        self.arduino_monitor = ArduinoMonitor(self.arduino_manager)
        
        # 상태 관리 변수
        self.sound_start_time: Optional[float] = None
        self.last_sound_time = time.time()
        self.last_gc_time = time.time()
        self.silence_start_time: Optional[float] = None
        self.is_song_playing = False

    def start(self) -> None:
        """모든 스레드를 시작합니다."""
        print(f"'{self.audio_processor.default_speaker.name}'에서 출력되는 소리를 실시간으로 분석하여 BPM을 계산합니다.")
        
        # 모든 스레드 시작
        self.audio_capture.start()
        self.bpm_analyzer.start()
        self.light_controller.start()
        self.arduino_monitor.start()

    def stop(self) -> None:
        """모든 스레드를 안전하게 종료하고 정리합니다."""
        # 모든 스레드 정지
        self.audio_capture.stop_event.set()
        self.bpm_analyzer.stop_event.set()
        self.light_controller.stop_event.set()
        self.arduino_monitor.stop_event.set()

        # 스레드 종료 대기
        self.audio_capture.join()
        self.bpm_analyzer.join()
        self.light_controller.join()
        self.arduino_monitor.join()

        # 아두이노 연결 종료
        with self.arduino_manager.lock:
            if self.arduino_manager.arduino and self.arduino_manager.arduino.is_open:
                self.arduino_manager.send_command('0\n')
                self.arduino_manager.arduino.close()
                print("\n아두이노 연결을 안전하게 종료했습니다.")

    def reset_analysis(self) -> None:
        """모든 분석 데이터를 초기화합니다."""
        print("\n💫 노래가 종료되어 모든 분석 데이터를 초기화합니다.")
        
        # BPM 관련 데이터 초기화
        self.bpm_stabilizer.bpm_history.clear()
        self.bpm_stabilizer.deviation_tracker = BPMDeviationTracker()
        
        # 타이밍 관련 변수 초기화
        self.sound_start_time = None
        self.silence_start_time = None
        self.is_song_playing = False
        
        # 버퍼 및 큐 초기화
        while not self.bpm_analyzer.bpm_queue.empty():
            self.bpm_analyzer.bpm_queue.get()
            
        # 오디오 버퍼 초기화
        while not self.audio_capture.audio_queue.empty():
            self.audio_capture.audio_queue.get()
        
        # LED 끄기
        self.light_controller.update_bpm(None)
        self.light_controller.set_silent(True)
        self.arduino_manager.send_command('0\n')

    def prepare_for_new_song(self) -> None:
        """새로운 노래 분석을 위한 준비를 합니다."""
        print("\n🎵 새로운 노래가 감지되었습니다.")
        
        # 기존 데이터 초기화
        self.bpm_stabilizer.bpm_history.clear()
        self.bpm_stabilizer.deviation_tracker = BPMDeviationTracker()
        
        # 버퍼 초기화
        while not self.bpm_analyzer.bpm_queue.empty():
            self.bpm_analyzer.bpm_queue.get()
            
        while not self.audio_capture.audio_queue.empty():
            self.audio_capture.audio_queue.get()
            
        # 상태 변수 초기화 및 설정
        self.is_song_playing = True
        self.sound_start_time = time.time()
        self.silence_start_time = None
        self.light_controller.set_silent(False)

    def check_silence_and_reset(self, current_time: float, is_sound: bool) -> bool:
        """
        침묵 상태를 체크하고 필요한 경우 초기화를 수행합니다.
        Returns:
            bool: 사운드 처리가 필요한지 여부
        """
        if not is_sound:
            if self.silence_start_time is None:
                self.silence_start_time = current_time
            elif (current_time - self.silence_start_time > self.config.SILENCE_DURATION and 
                  self.is_song_playing):
                self.reset_analysis()
                return False
        else:
            if not self.is_song_playing:
                self.prepare_for_new_song()
            elif self.silence_start_time is not None:
                # 짧은 침묵 후 다시 소리가 감지된 경우
                silence_duration = current_time - self.silence_start_time
                if silence_duration > 0.5:  # 0.5초 이상의 침묵이 있었다면
                    self.prepare_for_new_song()  # 새로운 노래로 간주
            
            self.silence_start_time = None
            self.last_sound_time = current_time
        
        return True

    def run(self) -> None:
        """메인 실행 루프를 시작합니다."""
        try:
            self.start()
            while True:
                current_time = time.time()

                # 주기적인 가비지 컬렉션
                if current_time - self.last_gc_time > 60:
                    gc.collect()
                    self.last_gc_time = current_time

                # 사운드 상태 확인
                with self.audio_capture.sound_lock:
                    is_sound = np.mean(self.audio_capture.rms_history) > self.config.SILENCE_THRESHOLD
                
                # 침묵 상태 체크 및 초기화 처리
                should_process_sound = self.check_silence_and_reset(current_time, is_sound)

                # BPM 분석 및 업데이트
                if should_process_sound and is_sound:
                    if self.sound_start_time and current_time - self.sound_start_time >= self.config.MIN_SOUND_DURATION:
                        while not self.bpm_analyzer.bpm_queue.empty():
                            try:
                                new_bpm = self.bpm_analyzer.bpm_queue.get_nowait()
                                if self.config.BPM_MIN <= new_bpm <= self.config.BPM_MAX:
                                    current_bpm = self.bpm_stabilizer.stabilize_bpm(new_bpm)
                                    if current_bpm is not None:
                                        self.light_controller.update_bpm(current_bpm)
                                        self.light_controller.set_silent(False)  # LED 제어 확실히 활성화
                            except Exception as e:
                                print("BPM 큐 처리 중 오류 발생:", e)
                                traceback.print_exc()

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n프로그램 종료")
        except Exception as e:
            print("메인 루프 중 오류 발생:", e)
            traceback.print_exc()
        finally:
            self.stop()

    def run(self) -> None:
        try:
            self.start()
            while True:
                current_time = time.time()

                # 주기적인 가비지 컬렉션
                if current_time - self.last_gc_time > 60:
                    gc.collect()
                    self.last_gc_time = current_time

                # 사운드 상태 확인
                with self.audio_capture.sound_lock:
                    is_sound = np.mean(self.audio_capture.rms_history) > self.config.SILENCE_THRESHOLD
                
                # 침묵 상태 체크 및 초기화 처리
                should_process_sound = self.check_silence_and_reset(current_time, is_sound)

                # BPM 분석 및 업데이트
                if should_process_sound and is_sound:
                    if self.sound_start_time and current_time - self.sound_start_time >= self.config.MIN_SOUND_DURATION:
                        while not self.bpm_analyzer.bpm_queue.empty():
                            try:
                                new_bpm = self.bpm_analyzer.bpm_queue.get_nowait()
                                if self.config.BPM_MIN <= new_bpm <= self.config.BPM_MAX:
                                    current_bpm = self.bpm_stabilizer.stabilize_bpm(new_bpm)
                                    if current_bpm is not None:
                                        self.light_controller.update_bpm(current_bpm)
                                        self.light_controller.set_silent(False)  # LED 제어 확실히 활성화
                            except Exception as e:
                                print("BPM 큐 처리 중 오류 발생:", e)
                                traceback.print_exc()

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n프로그램 종료")
        except Exception as e:
            print("메인 루프 중 오류 발생:", e)
            traceback.print_exc()
        finally:
            self.stop()

def main():
    system = AudioAnalysisSystem()
    system.run()

if __name__ == "__main__":
    main()
