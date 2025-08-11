# Tot Trecks : AI-Powered Baby Cry Monitor

An intelligent baby monitoring app that uses deep learning to detect an infant's cry in real-time, giving parents peace of mind via a live web dashboard and event logging.

---

## 🧭 Problem — The Parent Anxiety Gap
Over **60%** of working parents report high anxiety about their baby’s safety.  
Traditional monitors give audio/video but lack intelligence and actionable insights, leaving parents unsure, anxious, and sometimes unable to intervene during critical moments.

**Key pain points**
- No real-time visibility into baby’s emotional/physical state.  
- Uncertainty about caregiver attentiveness.  
- No detection of critical events (seizures, abnormal stillness).  
- Delayed or no alerts for distressing behavior.  
- No centralized history/log for trend monitoring.

---

## ✨ Solution — Intelligent & Actionable Monitoring
This system bridges that gap by adding AI-powered detection and instant alerts on top of live audio/video.

**Core capabilities**
- 🎯 Real-time cry detection (audio) using a Keras/TensorFlow model.  
- 🛡️ Detection of abnormal movement / prolonged stillness (extendable).  
- 📲 Instant notifications for critical events (WhatsApp, push, etc. — optional).  
- 📊 Live web dashboard showing current status + live feed.  
- 🗄️ Local event logging (SQLite) for history and trend review.

---

## 🚀 Features
- **Real-Time Audio Processing** — continuous microphone analysis.  
- **Deep Learning Model** — `baby_cry_model.h5` (Keras/TensorFlow).  
- **Web Dashboard** — Flask + HTML/CSS/JS for live updates.  
- **Event Logging** — SQLite (`baby_monitor.db`) with timestamps.  
- **Lightweight & Extensible** — modular design to add video analysis, notifications, caregiver accountability features.

---

## 🛠️ Tech Stack
- **Backend:** Python, Flask  
- **ML:** TensorFlow, Keras, Librosa  
- **Audio:** PyAudio  
- **Database:** SQLite  
- **Frontend:** HTML, CSS, JavaScript  

---

## ⚙️ Setup & Installation

> **Prerequisites**
> - Python 3.8+  
> - `pip`  
> - `portaudio` (required for PyAudio)  
>
> macOS:  
> ```bash
> brew install portaudio
> ```
> Debian/Ubuntu:  
> ```bash
> sudo apt-get install portaudio19-dev
> ```

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Spandan-Chakraborty/AI-Baby-Monitor.git
cd AI-Baby-Monitor
```
> Replace the repo URL if different.

### 2️⃣ Create & activate a virtual environment
```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# Windows (cmd)
venv\Scripts\activate
```

### 3️⃣ Install dependencies
Create `requirements.txt` with:
```
Flask
tensorflow
numpy
librosa
pyaudio
```
Then run:
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the application
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser to view the dashboard.

---

## 📖 How to Use
1. Connect and enable a microphone on the machine running the server.  
2. Launch `app.py`.  
3. Open the dashboard URL in your browser.  
4. The app listens continuously — status updates between **Calm** and **Crying**.  
5. Cry events are recorded in `baby_monitor.db` with timestamps.

---

## 📂 Project Structure
```
.
├── app.py                  # Main Flask application logic
├── baby_cry_model.h5       # Pre-trained Keras model for cry detection
├── baby_monitor.db         # SQLite DB for event logging (auto-created)
├── templates/
│   └── dashboard.html      # Frontend HTML for the dashboard
├── static/
│   └── style.css           # CSS for styling the dashboard
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

---

## 🔧 Extending the Project
- 📹 Add video-based movement / seizure detection.  
- 📩 Integrate WhatsApp / push notifications for critical events.  
- 🔐 Add user auth + multi-device monitoring.  
- ☁️ Cloud storage for long-term analytics and trend dashboards.  
- 📱 Mobile app for push alerts and live viewing.

---

## 🤝 Contributing
Contributions welcome!  
1. Fork the repo.  
2. Create a feature branch:  
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit changes & push.  
4. Open a Pull Request describing your changes.

Please open issues for bugs or feature requests.

---

## 📄 License
This project is released under the **MIT License**. See the `LICENSE` file for details.

---

## ⚡ Credits
Created by **Spandan Chakraborty**.
