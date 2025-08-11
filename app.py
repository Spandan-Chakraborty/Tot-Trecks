from flask import Flask, render_template, jsonify, Response
from flask_socketio import SocketIO, emit
import cv2
import sounddevice as sd
import numpy as np
import queue
import librosa
from tensorflow.keras.models import load_model
import threading
import time
from datetime import datetime
from twilio.rest import Client
import sqlite3
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'baby_monitor_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# ================== CONFIGURATION ==================
TWILIO_ACCOUNT_SID = '<your_account_sid>'
TWILIO_AUTH_TOKEN = '<your_auth_token>'
FROM_WHATSAPP = 'whatsapp:+14155238886'
TO_WHATSAPP = 'whatsapp:+919836364257'

SAMPLE_RATE = 16000
AUDIO_DURATION = 2
AUDIO_BLOCK_SIZE = SAMPLE_RATE * AUDIO_DURATION

MOTION_THRESHOLD = 4000
ALERT_THRESHOLD = 12000
CRITICAL_THRESHOLD = 25000
SLEEP_TIME_THRESHOLD = 600
ALERT_COOLDOWN = 300

# ================== GLOBAL VARIABLES ==================
audio_queue = queue.Queue()
monitoring_active = False
cry_model = None
twilio_client = None
last_motion_time = time.time()
baby_sleeping = False
last_alert_time = {}
current_frame = None
latest_events = []
statistics = {
    'total_cries': 0,
    'total_movements': 0,
    'sleep_sessions': 0,
    'alerts_sent': 0,
    'uptime_start': datetime.now()
}

# ================== DATABASE SETUP ==================
def init_database():
    conn = sqlite3.connect('baby_monitor.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            event_type TEXT NOT NULL,
            description TEXT NOT NULL,
            confidence REAL,
            alert_sent BOOLEAN DEFAULT FALSE
        )
    ''')
    conn.commit()
    conn.close()

# ================== AUDIO MONITORING ==================
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio status: {status}")
    audio_queue.put(indata.copy())

def extract_mfcc(audio):
    audio = audio.flatten()
    if len(audio) < AUDIO_BLOCK_SIZE:
        audio = np.pad(audio, (0, AUDIO_BLOCK_SIZE - len(audio)))
    mfcc = librosa.feature.mfcc(y=audio.astype(np.float32), sr=SAMPLE_RATE, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean.reshape(1, -1)

def audio_monitoring_thread():
    global monitoring_active, cry_model, statistics
    try:
        if cry_model is None:
            print("⚠️ Cry detection model not loaded.")
            return
        with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=audio_callback, blocksize=AUDIO_BLOCK_SIZE):
            print("🎤 Audio monitoring started...")
            while monitoring_active:
                try:
                    audio_data = audio_queue.get(timeout=1)

                    rms = np.sqrt(np.mean(audio_data**2))  # Raw RMS
                    normalized = min(rms / 0.1, 1.0)       # Normalize (adjust 0.1 based on real input)
                    socketio.emit('audio_level', normalized)

                    features = extract_mfcc(audio_data)
                    prediction = cry_model.predict(features, verbose=0)
                    label = np.argmax(prediction)
                    confidence = float(np.max(prediction))

                    status = 'Cry' if label == 1 else 'No Cry'
                    print(f"Detected: {status} (Confidence: {confidence:.2f})")

                    # Emit detection result to frontend
                    socketio.emit('cry_detection', {
                        'status': status,
                        'confidence': confidence,
                        'timestamp': datetime.now().isoformat()
                    })

                    if label == 1 and confidence > 0.7:
                        event = {
                            'type': 'cry',
                            'description': f'Baby cry detected (confidence: {confidence:.2f})',
                            'confidence': confidence,
                            'timestamp': datetime.now().isoformat()
                        }
                        log_event(event)
                        statistics['total_cries'] += 1
                        if confidence > 0.9:
                            send_alert(f"CRITICAL: Strong baby cry detected! Confidence: {confidence:.2f}")
                except queue.Empty:
                    continue
    except Exception as e:
        print(f"Audio monitoring error: {e}")


# ================== VIDEO MONITORING ==================

def video_monitoring_thread():
    import mediapipe as mp
    import math
    global monitoring_active, current_frame, baby_sleeping, last_motion_time, statistics

    # EAR configuration
    EAR_THRESHOLD = 0.2
    EYE_CLOSED_SLEEP_DURATION = 10  # seconds
    HYPOTONIA_MOTION_LIMIT = 1000
    HYPOTONIA_DURATION = 15  # seconds

    # Eye indices
    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

    # State tracking
    eye_closed_start = None
    low_motion_start = None
    pending_alert = None

    # MediaPipe setup
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    # EAR helper functions
    def euclidean(p1, p2):
        return math.dist(p1, p2)

    def calculate_ear(landmarks, eye_indices):
        p = [landmarks[i] for i in eye_indices]
        vertical1 = euclidean(p[1], p[5])
        vertical2 = euclidean(p[2], p[4])
        horizontal = euclidean(p[0], p[3])
        return (vertical1 + vertical2) / (2.0 * horizontal)

    # Camera setup
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access webcam")
        return

    ret, frame1 = cap.read()
    if not ret:
        print("❌ Failed to read from camera")
        return

    gray1 = cv2.GaussianBlur(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), (21, 21), 0)

    while monitoring_active:
        ret, frame2 = cap.read()
        if not ret:
            break

        current_frame = frame2.copy()
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.GaussianBlur(gray, (21, 21), 0)
        diff = cv2.absdiff(gray1, gray2)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        motion_pixels = cv2.countNonZero(thresh)
        current_time = time.time()

        if motion_pixels > MOTION_THRESHOLD:
            last_motion_time = current_time

        # Face and Eye EAR Detection
        frame_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            h, w, _ = frame2.shape
            landmarks = results.multi_face_landmarks[0].landmark
            landmark_points = [(int(pt.x * w), int(pt.y * h)) for pt in landmarks]

            left_ear = calculate_ear(landmark_points, LEFT_EYE_IDX)
            right_ear = calculate_ear(landmark_points, RIGHT_EYE_IDX)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                if eye_closed_start is None:
                    eye_closed_start = current_time
                elif current_time - eye_closed_start >= EYE_CLOSED_SLEEP_DURATION:
                    if not baby_sleeping:
                        baby_sleeping = True
                        statistics['sleep_sessions'] += 1
                        log_event({
                            'type': 'sleep',
                            'description': 'Baby appears to be sleeping (Eyes closed)',
                            'timestamp': datetime.now().isoformat()
                        })

                    if pending_alert is None:
                        if motion_pixels > CRITICAL_THRESHOLD:
                            pending_alert = "CRITICAL: Spasm/seizure-like movement during sleep"
                        elif motion_pixels > ALERT_THRESHOLD:
                            pending_alert = "ALERT: Strong movement during sleep"
            else:
                eye_closed_start = None
                if baby_sleeping:
                    baby_sleeping = False
                    log_event({
                        'type': 'wake',
                        'description': 'Baby woke up (Eyes opened)',
                        'timestamp': datetime.now().isoformat()
                    })
                    if pending_alert:
                        send_alert(pending_alert)
                        pending_alert = None
        else:
            eye_closed_start = None

        # Hypotonia Detection
        if not baby_sleeping:
            if motion_pixels < HYPOTONIA_MOTION_LIMIT:
                if low_motion_start is None:
                    low_motion_start = current_time
                elif current_time - low_motion_start >= HYPOTONIA_DURATION:
                    log_event({
                        'type': 'warning',
                        'description': 'Low movement detected while awake (Possible hypotonia)',
                        'timestamp': datetime.now().isoformat()
                    })
            else:
                low_motion_start = None
        else:
            low_motion_start = None

        # Movement Alerts (awake)
        if not baby_sleeping:
            if motion_pixels > CRITICAL_THRESHOLD:
                send_alert("CRITICAL: Sudden abnormal movement (Possible seizure)")
                log_event({'type': 'critical', 'description': 'Abnormal movement', 'timestamp': datetime.now().isoformat()})
            elif motion_pixels > ALERT_THRESHOLD:
                send_alert("ALERT: Strong movement (Possible discomfort)")
                log_event({'type': 'alert', 'description': 'Strong movement', 'timestamp': datetime.now().isoformat()})

        gray1 = gray2
        time.sleep(0.1)

    cap.release()



# ================== EVENT LOGGING ==================
def log_event(event):
    latest_events.insert(0, event)
    if len(latest_events) > 100:
        latest_events.pop()
    conn = sqlite3.connect('baby_monitor.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO events (event_type, description, confidence, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (event['type'], event['description'], event.get('confidence'), event['timestamp']))
    conn.commit()
    conn.close()
    socketio.emit('new_event', event)
    print(f"[{event['timestamp']}] {event['type'].upper()}: {event['description']}")

# ================== ALERTING ==================
def send_alert(message):
    global statistics, last_alert_time, twilio_client
    now = time.time()
    key = message[:30]

    if key in last_alert_time and now - last_alert_time[key] < 10:
        print(f"⏳ Skipping duplicate alert: {message}")
        return

    last_alert_time[key] = now
    statistics['alerts_sent'] += 1

    try:
        if twilio_client:
            twilio_client.messages.create(
                from_=FROM_WHATSAPP,
                body=f"🚨 Baby Monitor Alert 🚨\n{message}\nTime: {datetime.now().strftime('%H:%M:%S')}",
                to=TO_WHATSAPP
            )
            print(f"📱 WhatsApp alert sent: {message}")
    except Exception as e:
        print(f"❌ Failed to send WhatsApp alert: {e}")

    socketio.emit('alert', {
        'message': message,
        'timestamp': datetime.now().isoformat()
    })


# ================== FLASK ROUTES ==================
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    uptime = datetime.now() - statistics['uptime_start']
    return jsonify({
        'monitoring_active': monitoring_active,
        'baby_sleeping': baby_sleeping,
        'statistics': {
            **statistics,
            'uptime': str(uptime).split('.')[0]
        }
    })

@app.route('/api/events')
def get_events():
    return jsonify(latest_events[:20])

@app.route('/api/start', methods=['GET', 'POST'])
def start_monitoring():
    global monitoring_active
    if not monitoring_active:
        monitoring_active = True
        threading.Thread(target=audio_monitoring_thread, daemon=True).start()
        threading.Thread(target=video_monitoring_thread, daemon=True).start()
        return jsonify({'status': 'started'})
    return jsonify({'status': 'already_running'})

@app.route('/api/stop', methods=['GET', 'POST'])
def stop_monitoring():
    global monitoring_active
    monitoring_active = False
    return jsonify({'status': 'stopped'})


def generate_video_stream():
    global current_frame
    while True:
        if current_frame is not None:
            ret, buffer = cv2.imencode('.jpg', current_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ================== SOCKET EVENTS ==================
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status_update', {
        'monitoring_active': monitoring_active,
        'baby_sleeping': baby_sleeping
    })

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

# ================== INITIALIZATION ==================
def initialize_system():
    global cry_model, twilio_client
    print("🚀 Initializing system...")
    init_database()
    try:
        if os.path.exists("baby_cry_model.h5"):
            cry_model = load_model("baby_cry_model.h5")
            print("✅ Cry model loaded")
        else:
            print("⚠️ Cry model not found")
    except Exception as e:
        print(f"❌ Model load error: {e}")
    try:
        twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        print("✅ Twilio client initialized")
    except Exception as e:
        print(f"❌ Twilio init error: {e}")

if __name__ == '__main__':
    initialize_system()
    print("\n" + "=" * 50)
    print("🍼 SMART BABY MONITOR SYSTEM")
    print("=" * 50)
    print("🌐 Dashboard: http://localhost:5000")
    print("📹 Video Feed: http://localhost:5000/video_feed")
    print("=" * 50)
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
