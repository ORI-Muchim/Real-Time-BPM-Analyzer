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
import logging
import gc
import scipy.signal
import librosa
import numba

# Numba 로그 레벨 설정
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

# 로그 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

warnings.filterwarnings("ignore", category=sc.SoundcardRuntimeWarning)

# 설정 변수
SAMPLE_RATE = 44100
BUFFER_DURATION = 5  # 초 단위
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION)
CHUNK_SIZE = 1024
MAX_QUEUE_SIZE = 50  # 오디오 큐의 최대 크기

ARDUINO_PORT = 'COM6'  # 아두이노 포트 (자동으로 설정함)
BAUD_RATE = 9600

SILENCE_THRESHOLD = 0.01  # RMS 기준 침묵 임계값
SILENCE_DURATION = 2

# 아두이노 통신 관련 상수
MIN_COMMAND_INTERVAL = 0.05  # 최소 명령 간격 (초)
COMMAND_BUFFER_SIZE = 10     # 명령 버퍼 크기

# 기본 스피커 설정
default_speaker = sc.default_speaker()
print(f"기본 스피커: {default_speaker.name}")
logging.info(f"기본 스피커 설정: {default_speaker.name}")

# 마이크 (스피커 루프백 포함)
mic = sc.get_microphone(id=str(default_speaker.name), include_loopback=True)
logging.info(f"마이크 설정 완료: {mic.name}")

# BPM 분석 함수들
def compute_onset_envelope(data, sr):
    frame_size = 512
    hop_length = 128
    # 프레임 분할
    num_frames = (len(data) - frame_size) // hop_length + 1
    frames = np.lib.stride_tricks.as_strided(
        data,
        shape=(num_frames, frame_size),
        strides=(data.strides[0]*hop_length, data.strides[0])
    )
    # STFT 계산
    window = np.hanning(frame_size)
    stft = np.abs(np.fft.rfft(frames * window[None, :], axis=1))
    # 스펙트럴 플럭스 계산
    spectral_flux = np.diff(stft, axis=0)
    spectral_flux = np.maximum(spectral_flux, 0)
    onset_envelope = np.sum(spectral_flux, axis=1)
    # 스무딩 필터 적용
    onset_envelope = scipy.signal.medfilt(onset_envelope, kernel_size=3)
    # 엔벨로프 정규화
    onset_envelope /= np.max(onset_envelope) + 1e-6
    return onset_envelope, hop_length

def autocorrelation_bpm(data, sr):
    try:
        # 온셋 엔벨로프 계산
        onset_env, hop_length = compute_onset_envelope(data, sr)
        # 오토코릴레이션 계산
        corr = correlate(onset_env, onset_env, mode='full')
        corr = corr[len(corr)//2:]
        bpm_min, bpm_max = 60, 300  # 최대 BPM을 늘림
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
        # 가장 강한 피크 선택
        peak_index = np.argmax(corr[peaks])
        bpm = bpms[peak_index]
        return float(bpm)
    except Exception:
        logging.error("autocorrelation_bpm 함수에서 오류 발생:")
        traceback.print_exc()
    return None

def fft_based_bpm_detector(data, sr):
    try:
        # 온셋 엔벨로프 계산
        onset_env, hop_length = compute_onset_envelope(data, sr)
        # FFT 계산
        fft_vals = np.abs(np.fft.rfft(onset_env))
        freqs = np.fft.rfftfreq(len(onset_env), d=hop_length / sr)
        min_bpm = 60
        max_bpm = 300
        min_freq = min_bpm / 60  # BPM을 Hz로 변환
        max_freq = max_bpm / 60
        mask = (freqs >= min_freq) & (freqs <= max_freq)
        fft_filtered = fft_vals[mask]
        freqs_filtered = freqs[mask]
        if len(fft_filtered) == 0:
            return None
        # 가장 강한 피크 찾기
        peak_index = np.argmax(fft_filtered)
        peak_freq = freqs_filtered[peak_index]
        bpm = peak_freq * 60  # Hz를 BPM으로 변환
        return float(bpm)
    except Exception:
        logging.error("fft_based_bpm_detector 함수에서 오류 발생:")
        traceback.print_exc()
    return None

def librosa_bpm_detector(data, sr):
    try:
        tempo, _ = librosa.beat.beat_track(y=data, sr=sr, start_bpm=150, tightness=100)
        return float(tempo)
    except Exception:
        logging.error("librosa_bpm_detector 함수에서 오류 발생:")
        traceback.print_exc()
    return None

def combined_bpm_detector(data, sr):
    bpm_estimates = []
    try:
        # librosa를 사용한 BPM 감지
        librosa_bpm = librosa_bpm_detector(data, sr)
        if librosa_bpm:
            bpm_estimates.append(librosa_bpm)
        # 추가적인 방법들을 사용하여 BPM 추정
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
    except Exception:
        logging.error("combined_bpm_detector 함수에서 오류 발생:")
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
    except Exception:
        logging.error("remove_outliers 함수에서 오류 발생:")
        traceback.print_exc()
    return bpms

def stabilize_bpm(new_bpm, bpm_history):
    try:
        if new_bpm is None:
            return None
        if not bpm_history:
            bpm_history.append(new_bpm)
            stabilized_bpm = new_bpm
            logging.info(f"초기 BPM 설정: {stabilized_bpm:.2f}")
            return stabilized_bpm

        stabilized_bpm = np.median(bpm_history)
        # 허용 오차율 설정: 10%
        tolerance = 0.1
        if abs(new_bpm - stabilized_bpm) / stabilized_bpm > tolerance:
            logging.info(f"새로운 BPM 값 {new_bpm:.2f}이(가) 안정화된 BPM에서 너무 벗어남.")
            return stabilized_bpm  # 기존 안정화된 BPM 유지
        else:
            bpm_history.append(new_bpm)
            bpm_history = deque(remove_outliers(bpm_history), maxlen=10)
            stabilized_bpm = float(np.median(bpm_history))
            logging.info(f"새로운 BPM: {new_bpm:.2f}, 안정화된 BPM: {stabilized_bpm:.2f}")
            return stabilized_bpm
    except Exception:
        logging.error("stabilize_bpm 함수에서 오류 발생:")
        traceback.print_exc()
    return new_bpm

# 아두이노 통신 관리 클래스
class ArduinoManager:
    def __init__(self, arduino):
        self.arduino = arduino
        self.last_command_time = 0
        self.command_buffer = deque(maxlen=COMMAND_BUFFER_SIZE)
        self.retry_limit = 3  # 재시도 횟수 설정

    def send_command(self, command):
        current_time = time.time()
        if current_time - self.last_command_time < MIN_COMMAND_INTERVAL:
            self.command_buffer.append(command)
        else:
            self._send_command_to_arduino_with_retry(command)
            self.last_command_time = current_time

    def process_buffer(self):
        current_time = time.time()
        if self.command_buffer and current_time - self.last_command_time >= MIN_COMMAND_INTERVAL:
            command = self.command_buffer.popleft()
            self._send_command_to_arduino_with_retry(command)
            self.last_command_time = current_time

    def _send_command_to_arduino_with_retry(self, command):
        for attempt in range(self.retry_limit):
            try:
                if self.arduino and self.arduino.is_open:
                    self.arduino.write(command.encode())
                    return  # 성공 시 종료
            except serial.SerialTimeoutException as e:
                logging.error(f"Arduino write timeout (시도 {attempt + 1}/{self.retry_limit}): {str(e)}")
                time.sleep(0.5)
            except serial.SerialException as e:
                logging.error(f"Arduino 통신 오류: {str(e)}")
                break  # 다른 시리얼 오류 발생 시 중단
        logging.error("Arduino로 명령 전송 실패: 최대 재시도 횟수 초과")

# 조명 상태 변수
light_on = False
light_lock = Lock()

def control_lights(bpm, last_beat_time, arduino_manager, is_silent=False):
    global light_on
    current_time = time.time()
    try:
        with light_lock:
            if is_silent:
                if light_on:
                    if arduino_manager:
                        arduino_manager.send_command('0\n')
                    print("\r💡 소리가 감지되지 않아 조명이 꺼졌습니다.", end='', flush=True)
                    light_on = False
                return current_time, False

            if bpm is None:
                if light_on:
                    if arduino_manager:
                        arduino_manager.send_command('0\n')
                    print("\r💡 BPM을 측정할 수 없어 조명이 꺼졌습니다.", end='', flush=True)
                    light_on = False
                return current_time, False

            # 비트 간격 계산
            beat_interval = 60.0 / bpm
            time_since_last_beat = current_time - last_beat_time

            if time_since_last_beat >= beat_interval:
                if not light_on:
                    if arduino_manager:
                        arduino_manager.send_command('1\n')
                    print(f"\r💡 ON (BPM: {bpm:.2f})", end='', flush=True)
                    light_on = True

                time.sleep(min(0.1, beat_interval / 2))
                if light_on:
                    if arduino_manager:
                        arduino_manager.send_command('0\n')
                    print(f"\r💡 OFF (BPM: {bpm:.2f})", end='', flush=True)
                    light_on = False

                return current_time, True

            return last_beat_time, False
    except Exception:
        logging.error("control_lights 함수에서 오류 발생:")
        traceback.print_exc()
    return last_beat_time, False

def find_arduino_port():
    try:
        arduino_ports = [
            p.device
            for p in serial.tools.list_ports.comports()
            if 'Arduino' in p.description or 'CH340' in p.description
        ]
        if not arduino_ports:
            logging.warning("아두이노를 찾을 수 없습니다.")
            print("아두이노를 찾을 수 없습니다.")
            return None
        if len(arduino_ports) > 1:
            logging.warning("여러 개의 아두이노가 감지되었습니다. 첫 번째 포트를 사용합니다.")
            print("여러 개의 아두이노가 감지되었습니다. 첫 번째 포트를 사용합니다.")
        return arduino_ports[0]
    except Exception:
        logging.error("find_arduino_port 함수에서 오류 발생:")
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
                    logging.info(f"아두이노와 연결되었습니다. ({arduino_port})")
                    print(f"아두이노와 연결되었습니다. ({arduino_port})")
                    return arduino
                except serial.SerialException as e:
                    logging.error(f"아두이노 연결 실패 (시도 {attempt + 1}/3): {str(e)}")
                    if "Access is denied" in str(e) or "에러 코드 31" in str(e):
                        time.sleep(2)
                    else:
                        break
            logging.error("아두이노 연결 실패: 최대 시도 횟수 초과")
    except Exception as e:
        logging.error(f"connect_arduino 함수에서 오류 발생: {str(e)}")
        traceback.print_exc()
    return None

def reconnect_arduino(current_arduino):
    if current_arduino and current_arduino.is_open:
        current_arduino.close()
    return connect_arduino()

# Arduino 연결 설정
arduino_lock = Lock()
arduino = connect_arduino()
arduino_manager = ArduinoManager(arduino) if arduino else None

# 큐 및 잠금 객체 초기화
audio_queue = Queue(maxsize=MAX_QUEUE_SIZE)
bpm_queue = Queue()

# 공유 변수와 잠금 객체 초기화
sound_detected = False
sound_lock = Lock()
rms_history = deque(maxlen=5)  # RMS 값의 이동 평균 계산을 위한 큐

def audio_capture():
    global sound_detected
    with mic.recorder(samplerate=SAMPLE_RATE) as rec:
        logging.info("오디오 캡처 스레드 시작")
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
            except Exception:
                logging.error("오디오 캡처 중 오류 발생:")
                traceback.print_exc()
                time.sleep(0.1)

def bpm_analysis():
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
                bpm = combined_bpm_detector(data, SAMPLE_RATE)
                if bpm:
                    bpm_queue.put(bpm)
                last_analysis_time = current_time

                del data
                gc.collect()

            time.sleep(0.1)
        except Exception:
            logging.error("BPM 분석 중 오류 발생:")
            traceback.print_exc()
            time.sleep(1)

def monitor_arduino():
    global arduino, arduino_manager
    while True:
        try:
            with arduino_lock:
                if arduino is None or not arduino.is_open:
                    logging.info("아두이노가 연결되어 있지 않습니다. 재연결을 시도합니다.")
                    print("아두이노가 연결되어 있지 않습니다. 재연결을 시도합니다.")
                    arduino = reconnect_arduino(arduino)
                    arduino_manager = ArduinoManager(arduino) if arduino else None
        except Exception:
            logging.error("아두이노 모니터링 중 오류 발생:")
            traceback.print_exc()
        time.sleep(5)

def main():
    global arduino, arduino_manager
    try:
        audio_thread = Thread(target=audio_capture, name="AudioCaptureThread")
        audio_thread.daemon = True
        audio_thread.start()

        bpm_thread = Thread(target=bpm_analysis, name="BPMAnalysisThread")
        bpm_thread.daemon = True
        bpm_thread.start()

        arduino_monitor_thread = Thread(target=monitor_arduino, name="ArduinoMonitorThread")
        arduino_monitor_thread.daemon = True
        arduino_monitor_thread.start()

        print(f"'{default_speaker.name}'에서 출력되는 소리를 실시간으로 분석하여 BPM을 계산합니다.")
        logging.info(f"'{default_speaker.name}'에서 출력되는 소리를 실시간으로 분석하여 BPM을 계산합니다.")

        last_beat_time = time.time()
        last_sound_time = time.time()
        current_bpm = None
        bpm_history = deque(maxlen=10)
        silence_start_time = None
        last_gc_time = time.time()

        while True:
            try:
                current_time = time.time()

                if current_time - last_gc_time > 60:
                    gc.collect()
                    last_gc_time = current_time

                with sound_lock:
                    is_sound = sound_detected

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
                        logging.info("침묵이 감지되어 BPM 정보가 초기화되었습니다.")

                is_silent = (current_time - last_sound_time) > SILENCE_DURATION

                if not bpm_queue.empty():
                    try:
                        new_bpm = bpm_queue.get()
                        if 60 <= new_bpm <= 300:
                            current_bpm = stabilize_bpm(new_bpm, bpm_history)
                            print(f"\r새로운 BPM: {new_bpm:.2f}, 안정화된 BPM: {current_bpm:.2f}", end='', flush=True)
                            logging.info(f"새로운 BPM: {new_bpm:.2f}, 안정화된 BPM: {current_bpm:.2f}")
                            last_sound_time = current_time
                    except Exception:
                        logging.error("BPM 큐 처리 중 오류 발생:")
                        traceback.print_exc()

                if current_bpm is not None and not is_silent:
                    with arduino_lock:
                        last_beat_time, _ = control_lights(current_bpm, last_beat_time, arduino_manager, is_silent)
                else:
                    with arduino_lock:
                        last_beat_time, _ = control_lights(None, last_beat_time, arduino_manager, is_silent)

                if arduino_manager:
                    arduino_manager.process_buffer()

                time.sleep(0.01)
            except Exception as e:
                logging.error(f"메인 루프 중 오류 발생: {str(e)}")
                traceback.print_exc()
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n프로그램 종료")
        logging.info("프로그램이 사용자에 의해 종료되었습니다.")
    except Exception:
        logging.error("메인 함수에서 예상치 못한 오류 발생:")
        traceback.print_exc()
    finally:
        try:
            with arduino_lock:
                if arduino and arduino.is_open:
                    if arduino_manager:
                        arduino_manager.send_command('0\n')
                    arduino.close()
                    print("\n아두이노 연결을 안전하게 종료했습니다.")
                    logging.info("아두이노 연결을 안전하게 종료했습니다.")
        except Exception:
            logging.error("아두이노 종료 중 오류 발생:")
            traceback.print_exc()

if __name__ == "__main__":
    main()
