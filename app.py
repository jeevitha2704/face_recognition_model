import os
import cv2
import pickle
import sqlite3
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, send_file
from datetime import datetime, date
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
import base64
import threading
import time

app = Flask(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, "attendance.db")
FACES_DIR  = os.path.join(BASE_DIR, "faces_db")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
ENCODINGS_PATH = os.path.join(BASE_DIR, "encodings.pkl")

os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Database ────────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS students (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT NOT NULL,
            roll_no  TEXT UNIQUE NOT NULL,
            dept     TEXT,
            created  TEXT DEFAULT (datetime('now','localtime'))
        );
        CREATE TABLE IF NOT EXISTS attendance (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            date       TEXT NOT NULL,
            time_in    TEXT NOT NULL,
            status     TEXT DEFAULT 'Present',
            FOREIGN KEY(student_id) REFERENCES students(id),
            UNIQUE(student_id, date)
        );
    """)
    conn.commit()
    conn.close()

init_db()

# ── Face Recognition State ──────────────────────────────────────────────────────
known_encodings = []
known_ids       = []
recognition_lock = threading.Lock()

def load_encodings():
    global known_encodings, known_ids
    if os.path.exists(ENCODINGS_PATH):
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
        known_encodings = data.get("encodings", [])
        known_ids       = data.get("ids", [])
    else:
        known_encodings = []
        known_ids       = []

def save_encodings():
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump({"encodings": known_encodings, "ids": known_ids}, f)

load_encodings()

# ── Camera & Recognition Thread ─────────────────────────────────────────────────
camera      = None
camera_lock = threading.Lock()
last_frame  = None
last_recognized = {}   # student_id -> timestamp
COOLDOWN    = 5        # seconds between re-marking same person

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def get_camera():
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

def release_camera():
    global camera
    with camera_lock:
        if camera and camera.isOpened():
            camera.release()
        camera = None

def compute_encoding(face_img):
    """Simple LBP-style mean encoding (no dlib dependency)."""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
    resized = cv2.resize(gray, (64, 64))
    # Normalize pixel values as feature vector
    return resized.flatten().astype(np.float32) / 255.0

def compare_encodings(known_encs, candidate, threshold=0.45):
    """Return index of best match, or -1."""
    if not known_encs:
        return -1, 1.0
    diffs = [np.linalg.norm(np.array(e) - candidate) / np.sqrt(len(candidate))
             for e in known_encs]
    idx   = int(np.argmin(diffs))
    return (idx, diffs[idx]) if diffs[idx] < threshold else (-1, diffs[idx])

def mark_attendance(student_id):
    now  = datetime.now()
    today = now.strftime("%Y-%m-%d")
    t     = now.strftime("%H:%M:%S")
    conn  = get_db()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO attendance (student_id, date, time_in) VALUES (?,?,?)",
            (student_id, today, t)
        )
        conn.commit()
    finally:
        conn.close()

def gen_frames():
    global last_frame, last_recognized
    cam = get_camera()
    while True:
        ok, frame = cam.read()
        if not ok:
            time.sleep(0.05)
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            enc       = compute_encoding(face_crop)

            with recognition_lock:
                idx, dist = compare_encodings(known_encodings, enc)

            label = "Unknown"
            color = (0, 0, 220)

            if idx != -1:
                sid = known_ids[idx]
                conn = get_db()
                row  = conn.execute("SELECT name, roll_no FROM students WHERE id=?", (sid,)).fetchone()
                conn.close()
                if row:
                    label = f"{row['name']} ({row['roll_no']})"
                    color = (0, 200, 60)
                    now_ts = time.time()
                    if sid not in last_recognized or (now_ts - last_recognized[sid]) > COOLDOWN:
                        last_recognized[sid] = now_ts
                        mark_attendance(sid)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(frame, (x, y-30), (x+w, y), color, -1)
            cv2.putText(frame, label, (x+4, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

        # Timestamp overlay
        ts = datetime.now().strftime("%d-%m-%Y  %H:%M:%S")
        cv2.putText(frame, ts, (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        last_frame = frame.copy()
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/students", methods=["GET"])
def get_students():
    conn = get_db()
    rows = conn.execute("SELECT * FROM students ORDER BY name").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/students", methods=["POST"])
def add_student():
    data    = request.json
    name    = data.get("name","").strip()
    roll_no = data.get("roll_no","").strip()
    dept    = data.get("dept","").strip()
    if not name or not roll_no:
        return jsonify({"error": "Name and Roll No are required"}), 400
    conn = get_db()
    try:
        cur = conn.execute(
            "INSERT INTO students (name, roll_no, dept) VALUES (?,?,?)",
            (name, roll_no, dept)
        )
        conn.commit()
        sid = cur.lastrowid
        return jsonify({"success": True, "id": sid})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Roll number already exists"}), 409
    finally:
        conn.close()

@app.route("/api/capture", methods=["POST"])
def capture_face():
    """Capture face samples from current camera frame for a student."""
    data = request.json
    sid  = data.get("student_id")
    if sid is None:
        return jsonify({"error": "student_id required"}), 400

    cam = get_camera()
    samples = []
    attempts = 0
    while len(samples) < 20 and attempts < 60:
        ok, frame = cam.read()
        attempts += 1
        if not ok:
            time.sleep(0.05)
            continue
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            enc = compute_encoding(face_crop)
            samples.append(enc)
            if len(samples) >= 20:
                break
        time.sleep(0.05)

    if len(samples) < 5:
        return jsonify({"error": "Could not capture enough face samples. Ensure face is visible."}), 400

    with recognition_lock:
        # Remove existing encodings for this student
        new_encs = []
        new_ids  = []
        for e, i in zip(known_encodings, known_ids):
            if i != sid:
                new_encs.append(e)
                new_ids.append(i)
        for s in samples:
            new_encs.append(s.tolist())
            new_ids.append(sid)
        known_encodings.clear(); known_encodings.extend(new_encs)
        known_ids.clear();       known_ids.extend(new_ids)
        save_encodings()

    return jsonify({"success": True, "samples": len(samples)})

@app.route("/api/attendance", methods=["GET"])
def get_attendance():
    target = request.args.get("date", date.today().isoformat())
    conn   = get_db()
    rows   = conn.execute("""
        SELECT s.name, s.roll_no, s.dept, a.time_in, a.status
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        WHERE a.date = ?
        ORDER BY a.time_in
    """, (target,)).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/attendance/stats", methods=["GET"])
def attendance_stats():
    today = date.today().isoformat()
    conn  = get_db()
    total   = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    present = conn.execute(
        "SELECT COUNT(*) FROM attendance WHERE date=?", (today,)
    ).fetchone()[0]
    conn.close()
    return jsonify({"total": total, "present": present, "absent": total - present, "date": today})

@app.route("/api/export", methods=["GET"])
def export_excel():
    target = request.args.get("date", date.today().isoformat())
    conn   = get_db()
    rows   = conn.execute("""
        SELECT s.name, s.roll_no, s.dept, a.time_in, a.status
        FROM attendance a JOIN students s ON a.student_id = s.id
        WHERE a.date=? ORDER BY a.time_in
    """, (target,)).fetchall()
    all_students = conn.execute("SELECT name, roll_no, dept FROM students ORDER BY name").fetchall()
    conn.close()

    present_rolls = {r["roll_no"] for r in rows}
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = f"Attendance {target}"

    # ── Styles
    hdr_fill  = PatternFill("solid", fgColor="1A237E")
    hdr_font  = Font(color="FFFFFF", bold=True, size=11)
    alt_fill  = PatternFill("solid", fgColor="E8EAF6")
    pres_fill = PatternFill("solid", fgColor="C8E6C9")
    abs_fill  = PatternFill("solid", fgColor="FFCDD2")
    center    = Alignment(horizontal="center", vertical="center")
    thin      = Side(style="thin", color="BDBDBD")
    border    = Border(left=thin, right=thin, top=thin, bottom=thin)

    # Title
    ws.merge_cells("A1:E1")
    ws["A1"] = "ATTENDANCE REPORT"
    ws["A1"].font = Font(bold=True, size=14, color="1A237E")
    ws["A1"].alignment = center

    ws.merge_cells("A2:E2")
    ws["A2"] = f"Date: {target}   |   Generated: {datetime.now().strftime('%d-%m-%Y %H:%M')}"
    ws["A2"].alignment = center
    ws["A2"].font = Font(italic=True, color="555555")

    ws.append([])  # blank row

    headers = ["#", "Name", "Roll No", "Department", "Status / Time In"]
    ws.append(headers)
    hdr_row = ws.max_row
    for col, _ in enumerate(headers, 1):
        cell = ws.cell(hdr_row, col)
        cell.fill = hdr_fill
        cell.font = hdr_font
        cell.alignment = center
        cell.border = border

    present_map = {r["roll_no"]: r for r in rows}
    for i, stu in enumerate(all_students, 1):
        if stu["roll_no"] in present_map:
            r = present_map[stu["roll_no"]]
            row_data = [i, stu["name"], stu["roll_no"], stu["dept"] or "—", f"Present  {r['time_in']}"]
            fill = pres_fill
        else:
            row_data = [i, stu["name"], stu["roll_no"], stu["dept"] or "—", "Absent"]
            fill = abs_fill

        ws.append(row_data)
        cur_row = ws.max_row
        for col in range(1, 6):
            cell = ws.cell(cur_row, col)
            cell.fill = fill if col == 5 else (alt_fill if i % 2 == 0 else PatternFill())
            cell.alignment = center
            cell.border = border

    # Column widths
    for col, width in zip("ABCDE", [5, 22, 14, 18, 22]):
        ws.column_dimensions[col].width = width

    # Summary
    ws.append([])
    ws.append(["", "Total Students", len(all_students), "", ""])
    ws.append(["", "Present", len(present_rolls), "", ""])
    ws.append(["", "Absent", len(all_students) - len(present_rolls), "", ""])

    path = os.path.join(REPORTS_DIR, f"attendance_{target}.xlsx")
    wb.save(path)
    return send_file(path, as_attachment=True,
                     download_name=f"attendance_{target}.xlsx")

@app.route("/api/students/<int:sid>", methods=["DELETE"])
def delete_student(sid):
    conn = get_db()
    conn.execute("DELETE FROM attendance WHERE student_id=?", (sid,))
    conn.execute("DELETE FROM students WHERE id=?", (sid,))
    conn.commit()
    conn.close()
    with recognition_lock:
        indices = [i for i,x in enumerate(known_ids) if x != sid]
        new_encs = [known_encodings[i] for i in indices]
        new_ids  = [known_ids[i] for i in indices]
        known_encodings.clear(); known_encodings.extend(new_encs)
        known_ids.clear();       known_ids.extend(new_ids)
        save_encodings()
    return jsonify({"success": True})

if __name__ == "__main__":
    print("🎓 Attendance System running → http://localhost:5000")
    app.run(debug=False, threaded=True, host="0.0.0.0", port=5000)
