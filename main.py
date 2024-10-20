import soundcard as sc
import numpy as np
import time
import warnings
import serial
import serial.tools.list_ports
from threading import Thread, Lock
from queue import Queue
from scipy.signal import find_peaks, correlate
from collections import deque
import traceback
import gc
import scipy.signal
import librosa

# 경고 메시지 무시
warnings.filterwarnings("ignore", category=sc.SoundcardRuntimeWarning)

# 설정 변수
SAMPLE_RATE = 44100
BUFFER_DURATION = 5
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION)
CHUNK_SIZE = 1024
MAX_QUEUE_SIZE = 50
BAUD_RATE = 115200
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 2
MIN_COMMAND_INTERVAL = 0.1  # 아두이노 명령 전송 간격 (초)

# 기본 스피커 설정
default_speaker = sc.default_speaker()
print(f"기본 스피커: {default_speaker.name}")

# 마이크 (스피커 루프백 포함)
mic = sc.get_microphone(id=str(default_speaker.name), include_loopback=True)
print(f"마이크 설정 완료: {mic.name}")

# BPM 분석 함수들
def compute_onset_envelope(data, sr):
    frame_size = 512
    hop_length = 128
    num_frames = (len(data) - frame_size) // hop_length + 1
    frames = np.lib.stride_tricks.as_strided(
        data,
        shape=(num_frames, frame_size),
        strides=(data.strides[0]*hop_length, data.strides[0])
    )
    window = np.hanning(frame_size)
    stft = np.abs(np.fft.rfft(frames * window[None, :], axis=1))
    spectral_flux = np.diff(stft, axis=0)
    spectral_flux = np.maximum(spectral_flux, 0)
    onset_envelope = np.sum(spectral_flux, axis=1)
    onset_envelope = scipy.signal.medfilt(onset_envelope, kernel_size=3)
    onset_envelope /= np.max(onset_envelope) + 1e-6
    return onset_envelope, hop_length

def autocorrelation_bpm(data, sr):
    try:
        onset_env, hop_length = compute_onset_envelope(data, sr)
        corr = correlate(onset_env, onset_env, mode='full')
        corr = corr[len(corr)//2:]
        bpm_min, bpm_max = 60, 300
        lag_min = int(sr * 60 / bpm_max / hop_length)
        lag_max = int(sr * 60 / bpm_min / hop_length)
        corr = corr[lag_min:lag_max]
        if len(corr) == 0:
            return None
        peaks, _ = find_peaks(corr)
        if len(peaks) == 0:
            return None
        peak_lags = peaks + lag_min
        bpms = 60 * sr / (peak_lags * hop_length)
        peak_index = np.argmax(corr[peaks])
        bpm = bpms[peak_index]
        return float(bpm)
    except Exception as e:
        print("autocorrelation_bpm 함수에서 오류 발생:", e)
        traceback.print_exc()
    return None

def fft_based_bpm_detector(data, sr):
    try:
        onset_env, hop_length = compute_onset_envelope(data, sr)
        fft_vals = np.abs(np.fft.rfft(onset_env))
        freqs = np.fft.rfftfreq(len(onset_env), d=hop_length / sr)
        min_bpm = 60
        max_bpm = 300
        min_freq = min_bpm / 60
        max_freq = max_bpm / 60
        mask = (freqs >= min_freq) & (freqs <= max_freq)
        fft_filtered = fft_vals[mask]
        freqs_filtered = freqs[mask]
        if len(fft_filtered) == 0:
            return None
        peak_index = np.argmax(fft_filtered)
        peak_freq = freqs_filtered[peak_index]
        bpm = peak_freq * 60
        return float(bpm)
    except Exception as e:
        print("fft_based_bpm_detector 함수에서 오류 발생:", e)
        traceback.print_exc()
    return None

def librosa_bpm_detector(data, sr):
    try:
        tempo, _ = librosa.beat.beat_track(y=data, sr=sr, start_bpm=150, tightness=100)
        return float(tempo)
    except Exception as e:
        print("librosa_bpm_detector 함수에서 오류 발생:", e)
        traceback.print_exc()
    return None

def combined_bpm_detector(data, sr):
    rms = np.sqrt(np.mean(data**2))
    if rms < SILENCE_THRESHOLD:
        return None
    
    bpm_estimates = []
    try:
        librosa_bpm = librosa_bpm_detector(data, sr)
        if librosa_bpm:
            bpm_estimates.append(librosa_bpm)
        autocorr_bpm = autocorrelation_bpm(data, sr)
        if autocorr_bpm:
            bpm_estimates.append(autocorr_bpm)
        fft_bpm = fft_based_bpm_detector(data, sr)
        if fft_bpm:
            bpm_estimates.append(fft_bpm)
        bpm_estimates = [bpm for bpm in bpm_estimates if bpm is not None]
        if not bpm_estimates:
            return None
        return float(np.median(bpm_estimates))
    except Exception as e:
        print("combined_bpm_detector 함수에서 오류 발생:", e)
        traceback.print_exc()
    return None

def remove_outliers(bpms):
    try:
        if len(bpms) < 4:
            return bpms
        q1, q3 = np.percentile(bpms, [25, 75])
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - (1.5 * iqr), q3 + (1.5 * iqr)
        return [bpm for bpm in bpms if lower_bound <= bpm <= upper_bound]
    except Exception as e:
        print("remove_outliers 함수에서 오류 발생:", e)
        traceback.print_exc()
    return bpms

def stabilize_bpm(new_bpm, bpm_history):
    try:
        if new_bpm is None:
            return None
        if not bpm_history:
            bpm_history.append(new_bpm)
            stabilized_bpm = new_bpm
            print(f"초기 BPM 설정: {stabilized_bpm:.2f}")
            return stabilized_bpm
        stabilized_bpm = np.median(bpm_history)
        tolerance = 0.1
        if abs(new_bpm - stabilized_bpm) / stabilized_bpm > tolerance:
            print(f"새로운 BPM 값 {new_bpm:.2f}이(가) 안정화된 BPM에서 너무 벗어남.")
            return stabilized_bpm
        else:
            bpm_history.append(new_bpm)
            bpm_history = deque(remove_outliers(bpm_history), maxlen=10)
            stabilized_bpm = float(np.median(bpm_history))
            print(f"새로운 BPM: {new_bpm:.2f}, 안정화된 BPM: {stabilized_bpm:.2f}")
            return stabilized_bpm
    except Exception as e:
        print("stabilize_bpm 함수에서 오류 발생:", e)
        traceback.print_exc()
    return new_bpm

# 아두이노 통신 관리 클래스
class ArduinoManager:
    def __init__(self, arduino):
        self.arduino = arduino
        self.last_command_time = 0
        self.last_command = None
        self.command_interval = MIN_COMMAND_INTERVAL

    def send_command(self, command):
        current_time = time.time()
        if command == self.last_command:
            return
        if current_time - self.last_command_time >= self.command_interval:
            self._send_command_to_arduino(command)
            self.last_command_time = current_time
            self.last_command = command

    def _send_command_to_arduino(self, command):
        try:
            if self.arduino and self.arduino.is_open:
                self.arduino.write(command.encode())
        except serial.SerialTimeoutException as e:
            print(f"Write timeout 발생: {str(e)}")
            traceback.print_exc()
        except serial.SerialException as e:
            print(f"시리얼 통신 오류: {str(e)}")
            traceback.print_exc()

# 조명 제어 스레드 클래스
class LightController(Thread):
    def __init__(self, arduino_manager):
        super().__init__()
        self.arduino_manager = arduino_manager
        self.bpm = None
        self.last_beat_time = time.time()
        self.running = True
        self.is_silent = False
        self.lock = Lock()
        self.light_on = False

    def update_bpm(self, bpm):
        with self.lock:
            self.bpm = bpm

    def set_silent(self, is_silent):
        with self.lock:
            self.is_silent = is_silent

    def run(self):
        while self.running:
            with self.lock:
                bpm = self.bpm
                is_silent = self.is_silent

            current_time = time.time()

            if is_silent or bpm is None:
                if self.light_on:
                    if self.arduino_manager:
                        self.arduino_manager.send_command('0\n')
                    self.light_on = False
                time.sleep(0.1)
                continue

            beat_interval = 60.0 / bpm
            time_since_last_beat = current_time - self.last_beat_time

            if time_since_last_beat >= beat_interval:
                if not self.light_on:
                    if self.arduino_manager:
                        self.arduino_manager.send_command('1\n')
                    self.light_on = True
                    print(f"\r💡 ON (BPM: {bpm:.2f})", end='', flush=True)

                off_duration = min(0.1, beat_interval / 2)
                time.sleep(off_duration)

                if self.light_on:
                    if self.arduino_manager:
                        self.arduino_manager.send_command('0\n')
                    self.light_on = False
                    print(f"\r💡 OFF (BPM: {bpm:.2f})", end='', flush=True)

                self.last_beat_time = current_time
            else:
                sleep_time = beat_interval - time_since_last_beat
                time.sleep(min(sleep_time, 0.1))

    def stop(self):
        self.running = False

def find_arduino_port():
    try:
        arduino_ports = [
            p.device
            for p in serial.tools.list_ports.comports()
            if 'Arduino' in p.description or 'CH340' in p.description
        ]
        if not arduino_ports:
            print("아두이노를 찾을 수 없습니다.")
            return None
        if len(arduino_ports) > 1:
            print("여러 개의 아두이노가 감지되었습니다. 첫 번째 포트를 사용합니다.")
        return arduino_ports[0]
    except Exception as e:
        print("find_arduino_port 함수에서 오류 발생:", e)
        traceback.print_exc()
    return None

def connect_arduino():
    try:
        arduino_port = find_arduino_port()
        if arduino_port:
            for attempt in range(3):
                try:
                    arduino = serial.Serial(arduino_port, BAUD_RATE, timeout=1, write_timeout=1)
                    time.sleep(2)
                    print(f"아두이노와 연결되었습니다. ({arduino_port})")
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

def reconnect_arduino(current_arduino):
    if current_arduino and current_arduino.is_open:
        current_arduino.close()
    return connect_arduino()

def audio_capture(mic, audio_queue, rms_history, sound_lock):
    with mic.recorder(samplerate=SAMPLE_RATE) as rec:
        print("오디오 캡처 스레드 시작")
        while True:
            try:
                data = rec.record(numframes=CHUNK_SIZE)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                rms = np.sqrt(np.mean(data**2))
                rms_history.append(rms)
                rms_average = np.mean(rms_history)
                with sound_lock:
                    sound_detected = rms_average > SILENCE_THRESHOLD
                if np.max(np.abs(data)) > 0:
                    data = data / np.max(np.abs(data))
                if audio_queue.qsize() < MAX_QUEUE_SIZE:
                    audio_queue.put(data)
                else:
                    try:
                        audio_queue.get_nowait()
                    except Queue.Empty:
                        pass
                    audio_queue.put(data)
            except Exception as e:
                print("오디오 캡처 중 오류 발생:", e)
                traceback.print_exc()
                time.sleep(0.1)

def bpm_analysis(audio_queue, bpm_queue):
    audio_buffer = deque(maxlen=BUFFER_SIZE)
    last_analysis_time = time.time()
    while True:
        try:
            current_time = time.time()
            while not audio_queue.empty():
                new_data = audio_queue.get()
                audio_buffer.extend(new_data)

            if len(audio_buffer) >= BUFFER_SIZE and current_time - last_analysis_time > 1:
                data = np.array(audio_buffer)

                # **여기서 입력 데이터의 RMS 값을 계산하여 무음 여부를 판단합니다.**
                rms = np.sqrt(np.mean(data**2))
                if rms < SILENCE_THRESHOLD:
                    # 무음인 경우 BPM 분석을 건너뜁니다.
                    bpm = None
                else:
                    bpm = combined_bpm_detector(data, SAMPLE_RATE)

                if bpm:
                    bpm_queue.put(bpm)
                last_analysis_time = current_time
                del data
                gc.collect()
            time.sleep(0.1)
        except Exception as e:
            print("BPM 분석 중 오류 발생:", e)
            traceback.print_exc()
            time.sleep(1)

def monitor_arduino(arduino_lock, arduino, arduino_manager):
    while True:
        try:
            with arduino_lock:
                if arduino is None or not arduino.is_open:
                    print("아두이노가 연결되어 있지 않습니다. 재연결을 시도합니다.")
                    arduino = reconnect_arduino(arduino)
                    arduino_manager = ArduinoManager(arduino) if arduino else None
        except Exception as e:
            print("아두이노 모니터링 중 오류 발생:", e)
            traceback.print_exc()
        time.sleep(5)

def main():
    arduino_lock = Lock()
    arduino = connect_arduino()
    arduino_manager = ArduinoManager(arduino) if arduino else None

    audio_queue = Queue(maxsize=MAX_QUEUE_SIZE)
    bpm_queue = Queue()
    rms_history = deque(maxlen=5)
    sound_lock = Lock()

    audio_thread = Thread(target=audio_capture, args=(mic, audio_queue, rms_history, sound_lock))
    audio_thread.daemon = True
    audio_thread.start()

    bpm_thread = Thread(target=bpm_analysis, args=(audio_queue, bpm_queue))
    bpm_thread.daemon = True
    bpm_thread.start()

    arduino_monitor_thread = Thread(target=monitor_arduino, args=(arduino_lock, arduino, arduino_manager))
    arduino_monitor_thread.daemon = True
    arduino_monitor_thread.start()

    print(f"'{default_speaker.name}'에서 출력되는 소리를 실시간으로 분석하여 BPM을 계산합니다.")

    last_sound_time = time.time()
    current_bpm = None
    bpm_history = deque(maxlen=10)
    silence_start_time = None
    last_gc_time = time.time()

    light_controller = LightController(arduino_manager)
    light_controller.daemon = True
    light_controller.start()

    try:
        while True:
            current_time = time.time()

            if current_time - last_gc_time > 60:
                gc.collect()
                last_gc_time = current_time

            with sound_lock:
                is_sound = np.mean(rms_history) > SILENCE_THRESHOLD

            if is_sound:
                last_sound_time = current_time
                silence_start_time = None
            else:
                if silence_start_time is None:
                    silence_start_time = current_time
                elif current_time - silence_start_time > SILENCE_DURATION:
                    bpm_history.clear()
                    current_bpm = None
                    print("\r💡 침묵이 감지되어 BPM 정보가 초기화되었습니다.", end='', flush=True)

            is_silent = (current_time - last_sound_time) > SILENCE_DURATION

            if is_sound and not bpm_queue.empty():
                try:
                    new_bpm = bpm_queue.get()
                    if 60 <= new_bpm <= 300:
                        current_bpm = stabilize_bpm(new_bpm, bpm_history)
                        last_sound_time = current_time
                except Exception as e:
                    print("BPM 큐 처리 중 오류 발생:", e)
                    traceback.print_exc()

            light_controller.update_bpm(current_bpm)
            light_controller.set_silent(is_silent)

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n프로그램 종료")
    except Exception as e:
        print("메인 루프 중 오류 발생:", e)
        traceback.print_exc()
    finally:
        light_controller.stop()
        light_controller.join()
        with arduino_lock:
            if arduino and arduino.is_open:
                if arduino_manager:
                    arduino_manager.send_command('0\n')
                arduino.close()
                print("\n아두이노 연결을 안전하게 종료했습니다.")

if __name__ == "__main__":
    main()
