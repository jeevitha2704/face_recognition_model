# 🎓 Face Recognition Attendance System
**Stack:** Python · OpenCV · Flask · SQLite · openpyxl

---

## 📁 Project Structure
```
attendance_system/
├── app.py               ← Main Flask app (all logic here)
├── requirements.txt     ← pip dependencies
├── attendance.db        ← SQLite DB (auto-created)
├── encodings.pkl        ← Face encodings store (auto-created)
├── templates/
│   └── index.html       ← Full web UI
├── faces_db/            ← (reserved for future raw face saves)
└── reports/             ← Excel exports saved here
```

---

## ⚙️ Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```

Open → **http://localhost:5000**

---

## 🚀 How to Use

### Step 1 — Register Students
1. Go to **Register Student** tab
2. Fill in Name, Roll No, Department → click **Add Student**
3. In the **Capture Face** section, select the student
4. Click **Start Face Capture** — student must face the webcam for ~3 seconds

### Step 2 — Mark Attendance (Automatic)
1. Stay on **Dashboard** tab — live feed is always running
2. When a registered face enters the frame → attendance is auto-marked
3. A **5-second cooldown** prevents double-marking

### Step 3 — View & Export
- **Attendance tab** → pick any date → click View
- Click **Export Excel** to download a formatted `.xlsx` report

---

## 🔧 Configuration (in app.py)

| Variable | Default | Description |
|---|---|---|
| `COOLDOWN` | 5 sec | Time before re-marking same person |
| `threshold` | 0.45 | Recognition sensitivity (lower = stricter) |
| Camera index | `0` | Change `VideoCapture(0)` for external webcam |

---

## 🧠 How Recognition Works

This system uses **OpenCV Haar Cascades** for face detection and a **pixel-intensity feature vector** (64×64 grayscale normalized) with **Euclidean distance** for recognition.

- No dlib/face_recognition library needed
- Pure OpenCV + NumPy — runs on any machine
- ~20 face samples captured per student for better accuracy

---

## 📊 Excel Report Format
- Title + generation timestamp
- All students listed (present & absent)
- Green rows = Present, Red rows = Absent
- Summary section (total / present / absent)

---

## 💡 Tips for Better Accuracy
- Capture faces in **good lighting**
- Face the camera **straight on** during registration
- Avoid extreme angles or glasses changes between registration and recognition
- If accuracy drops, re-capture the student's face
