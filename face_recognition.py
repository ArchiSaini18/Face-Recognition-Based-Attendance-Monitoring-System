import streamlit as st
import numpy as np
import pandas as pd
import cv2
import os
import csv
import datetime
import time
from PIL import Image
import io
import base64

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Recognition Attendance System",
    page_icon="🎓",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CSS  —  Elegant Dark Theme · Luxury Fintech style (matching diabetes.py)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

/* ── Variables ── */
:root {
    --bg:         #060b10;
    --surface:    #0b1319;
    --surface2:   #101a22;
    --border:     #182430;
    --border-hi:  #1f3040;
    --blue:       #38bdf8;
    --blue-light: #7dd3fc;
    --blue-dim:   rgba(56,189,248,0.12);
    --red:        #f43f5e;
    --green:      #10b981;
    --amber:      #f59e0b;
    --text:       #dceeff;
    --muted:      #4a6070;
    --head-font:  'Playfair Display', Georgia, serif;
    --mono-font:  'IBM Plex Mono', monospace;
    --glow-blue:  0 0 28px rgba(56,189,248,0.20);
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {
    background: var(--bg) !important;
    font-family: var(--mono-font) !important;
    color: var(--text) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 4px; }

/* ── HERO ── */
.hero {
    position: relative;
    padding: 3rem 3.5rem 2.5rem;
    margin-bottom: 2.5rem;
    background: linear-gradient(135deg, #060b10 0%, #0b1c2a 60%, #060b10 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background:
        radial-gradient(ellipse 60% 50% at 90% 50%, rgba(56,189,248,0.07) 0%, transparent 70%),
        radial-gradient(ellipse 40% 60% at 10% 80%, rgba(56,189,248,0.04) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--blue), var(--blue-light), transparent);
}
.hero-tag {
    display: inline-block;
    font-family: var(--mono-font);
    font-size: 0.68rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--blue);
    background: var(--blue-dim);
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 4px;
    padding: 0.25rem 0.8rem;
    margin-bottom: 1rem;
}
.hero h1 {
    font-family: var(--head-font) !important;
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    line-height: 1.1 !important;
    letter-spacing: -0.01em;
    margin: 0 0 0.6rem 0 !important;
}
.hero h1 span {
    background: linear-gradient(135deg, var(--blue), var(--blue-light));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub { font-size: 0.82rem; color: var(--muted); letter-spacing: 0.04em; }
.hero-badge {
    position: absolute;
    right: 3.5rem; top: 50%;
    transform: translateY(-50%);
    width: 80px; height: 80px;
    border-radius: 50%;
    background: var(--blue-dim);
    border: 1px solid rgba(56,189,248,0.28);
    display: flex; align-items: center; justify-content: center;
    font-size: 2.2rem;
    box-shadow: var(--glow-blue);
}
.status-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--blue);
    box-shadow: 0 0 6px var(--blue);
    margin-right: 0.5rem;
    vertical-align: middle;
}
.status-badge { font-size: 0.7rem; letter-spacing: 0.1em; color: var(--muted); text-transform: uppercase; }

/* ── Stat Chips ── */
.acc-row { display: flex; gap: 0.8rem; margin-bottom: 2.5rem; flex-wrap: wrap; }
.acc-chip {
    background: var(--surface2);
    border: 1px solid var(--border-hi);
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 0.05em;
}
.acc-chip span {
    font-family: var(--head-font);
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--blue);
    margin-right: 0.35rem;
}

/* ── Section Labels ── */
.section-label {
    font-family: var(--mono-font);
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--blue);
    border-left: 2px solid var(--blue);
    padding-left: 0.7rem;
    margin-bottom: 1.4rem;
}

/* ── Form Panel ── */
.form-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.8rem 1.6rem;
    margin-bottom: 1rem;
}

/* ── Streamlit widget overrides ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: var(--surface2) !important;
    border: 1px solid var(--border-hi) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
    font-family: var(--mono-font) !important;
}
[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label {
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="stButton"] button {
    background: linear-gradient(135deg, #0369a1, #0ea5e9) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--mono-font) !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.4rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
[data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #0284c7, #38bdf8) !important;
    box-shadow: var(--glow-blue) !important;
}
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 10px !important; }
[data-testid="stAlert"] { border-radius: 8px !important; font-family: var(--mono-font) !important; }

/* ── Result Cards ── */
.result-wrap { margin-top: 0; }
.result-success {
    background: linear-gradient(135deg, #05101a 0%, #071e2e 100%);
    border: 1px solid rgba(16,185,129,0.30);
    border-top: 3px solid var(--green);
    border-radius: 14px;
    padding: 2.5rem 2.5rem 2rem;
    position: relative; overflow: hidden;
}
.result-warning {
    background: linear-gradient(135deg, #100b05 0%, #1c1505 100%);
    border: 1px solid rgba(245,158,11,0.30);
    border-top: 3px solid var(--amber);
    border-radius: 14px;
    padding: 2.5rem 2.5rem 2rem;
}
.result-icon { font-size: 3rem; margin-bottom: 1rem; display: block; }
.result-verdict {
    font-family: var(--head-font);
    font-size: 1.6rem; font-weight: 700;
    margin-bottom: 0.5rem; line-height: 1.1;
}
.result-verdict.green { color: var(--green); }
.result-verdict.amber { color: var(--amber); }
.result-verdict.blue  { color: var(--blue); }
.result-note {
    font-size: 0.78rem; color: var(--muted);
    margin-top: 1.2rem; line-height: 1.65;
    border-top: 1px solid rgba(255,255,255,0.05);
    padding-top: 1rem;
}

/* ── Idle Card ── */
.idle-card {
    background: var(--surface);
    border: 1px dashed var(--border-hi);
    border-radius: 14px;
    padding: 3.5rem 2rem;
    text-align: center; color: var(--muted);
}
.idle-icon { font-size: 2.8rem; margin-bottom: 1rem; opacity: 0.45; }
.idle-head {
    font-family: var(--head-font);
    font-size: 1.2rem; color: #1e3a52; margin-bottom: 0.4rem;
}
.idle-body { font-size: 0.8rem; line-height: 1.6; }

/* ── Attendance Table ── */
.att-table-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem 1.6rem;
    margin-top: 1rem;
}

/* ── Layout helpers ── */
[data-testid="stHorizontalBlock"] { gap: 1.4rem !important; align-items: stretch !important; }
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
}
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 2rem 0 !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {
    font-family: var(--mono-font) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--blue) !important;
    border-bottom-color: var(--blue) !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS  (filesystem-safe, cross-platform paths)
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR    = os.getcwd()
STUDENT_DIR = os.path.join(BASE_DIR, "StudentDetails")
TRAIN_DIR   = os.path.join(BASE_DIR, "TrainingImage")
LABEL_DIR   = os.path.join(BASE_DIR, "TrainingImageLabel")
ATT_DIR     = os.path.join(BASE_DIR, "Attendance")
CASCADE     = "haarcascade_frontalface_default.xml"

def assure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def student_csv_path():
    return os.path.join(STUDENT_DIR, "StudentDetails.csv")

def trainner_path():
    return os.path.join(LABEL_DIR, "Trainner.yml")

def attendance_csv_path():
    ts   = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    return os.path.join(ATT_DIR, f"Attendance_{date}.csv")

def count_registrations():
    f = student_csv_path()
    if not os.path.isfile(f):
        return 0
    with open(f, 'r') as cf:
        rows = sum(1 for _ in csv.reader(cf))
    return max(0, (rows // 2) - 1)

def get_next_serial():
    f = student_csv_path()
    if not os.path.isfile(f):
        return 1
    with open(f, 'r') as cf:
        rows = sum(1 for _ in csv.reader(cf))
    return rows // 2

def cascade_available():
    return os.path.isfile(CASCADE)

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    faces, ids = [], []
    for ip in image_paths:
        try:
            pil_img = Image.open(ip).convert('L')
            img_np  = np.array(pil_img, 'uint8')
            sid     = int(os.path.split(ip)[-1].split(".")[1])
            faces.append(img_np)
            ids.append(sid)
        except Exception:
            pass
    return faces, ids

def load_attendance_today():
    f = attendance_csv_path()
    if not os.path.isfile(f):
        return pd.DataFrame(columns=["ID", "Name", "Date", "Time"])
    try:
        df = pd.read_csv(f, header=0, usecols=[0, 2, 4, 6],
                         names=["ID", "Name", "Date", "Time"], skiprows=1)
        return df
    except Exception:
        return pd.DataFrame(columns=["ID", "Name", "Date", "Time"])

# ── Hero ──────────────────────────────────────────────────────────────────────
total_reg = count_registrations()
today_att = load_attendance_today()
cascade_ok = cascade_available()

st.markdown(f"""
<div class="hero">
    <div class="hero-tag">⬡ AI Campus Intelligence</div>
    <h1>Face Recognition <span>Attendance</span><br>Monitoring System</h1>
    <p class="hero-sub">
        OpenCV · LBPH Face Recognizer &nbsp;·&nbsp;
        <span class="status-dot"></span>
        <span class="status-badge">{"System Active" if cascade_ok else "Cascade File Missing"}</span>
    </p>
    <div class="hero-badge">🎓</div>
</div>
<div class="acc-row">
    <div class="acc-chip"><span>{total_reg}</span> Registrations</div>
    <div class="acc-chip"><span>{len(today_att)}</span> Today's Attendance</div>
    <div class="acc-chip"><span>LBPH</span> Recognition Model</div>
    <div class="acc-chip"><span>{"✓" if cascade_ok else "✗"}</span> Haar Cascade</div>
</div>
""", unsafe_allow_html=True)

if not cascade_ok:
    st.error(
        "⚠️ **haarcascade_frontalface_default.xml** not found in the working directory. "
        "Download it from the OpenCV GitHub repo and place it alongside this script."
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📸  Register Student", "🎯  Take Attendance", "📋  View Records"])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — REGISTER STUDENT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    reg_col, reg_res = st.columns([1.05, 0.95], gap="medium")

    with reg_col:
        st.markdown('<div class="section-label">01 — Student Details</div>', unsafe_allow_html=True)
        st.markdown('<div class="form-panel">', unsafe_allow_html=True)

        r_id   = st.text_input("Student ID", placeholder="e.g. 2024001")
        r_name = st.text_input("Full Name",  placeholder="e.g. Rahul Sharma")

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        num_frames = st.slider("Number of face samples to capture", 30, 150, 80, step=10)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        take_btn  = st.button("📷  Capture Face Images")
        train_btn = st.button("💾  Save Profile (Train Model)")

    with reg_res:
        st.markdown('<div class="section-label">02 — Registration Status</div>', unsafe_allow_html=True)

        # ── Capture Images ────────────────────────────────────────────────────
        if take_btn:
            if not cascade_ok:
                st.error("Haar cascade file missing — cannot detect faces.")
            elif not r_id.strip():
                st.warning("Please enter a Student ID.")
            elif not r_name.strip() or not all(c.isalpha() or c == ' ' for c in r_name.strip()):
                st.warning("Please enter a valid name (letters only).")
            else:
                assure_path_exists(STUDENT_DIR)
                assure_path_exists(TRAIN_DIR)

                serial  = get_next_serial()
                columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
                csv_f   = student_csv_path()
                if not os.path.isfile(csv_f):
                    with open(csv_f, 'a+') as cf:
                        csv.writer(cf).writerow(columns)

                detector = cv2.CascadeClassifier(CASCADE)
                cam      = cv2.VideoCapture(0)

                if not cam.isOpened():
                    st.error("❌ Could not open webcam. Make sure a camera is connected.")
                else:
                    progress   = st.progress(0, text="Capturing face samples…")
                    frame_slot = st.empty()
                    sample_num = 0

                    while True:
                        ret, img = cam.read()
                        if not ret:
                            break
                        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = detector.detectMultiScale(gray, 1.3, 5)
                        for (x, y, w, h) in faces:
                            sample_num += 1
                            cv2.rectangle(img, (x, y), (x+w, y+h), (56, 189, 248), 2)
                            cv2.imwrite(
                                os.path.join(TRAIN_DIR,
                                    f"{r_name.strip()}.{serial}.{r_id.strip()}.{sample_num}.jpg"),
                                gray[y:y+h, x:x+w]
                            )
                            cv2.putText(img, f"Sample {sample_num}/{num_frames}",
                                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6, (56, 189, 248), 2)

                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        frame_slot.image(rgb, channels="RGB", use_container_width=True)
                        pct = min(int(sample_num / num_frames * 100), 100)
                        progress.progress(pct, text=f"Captured {sample_num}/{num_frames} samples…")

                        if sample_num >= num_frames:
                            break

                    cam.release()
                    cv2.destroyAllWindows()
                    frame_slot.empty()
                    progress.empty()

                    # Save to CSV
                    with open(csv_f, 'a+') as cf:
                        csv.writer(cf).writerow([serial, '', r_id.strip(), '', r_name.strip()])

                    st.markdown(f"""
                    <div class="result-wrap">
                    <div class="result-success">
                        <span class="result-icon">✅</span>
                        <div class="result-verdict green">Registration Successful</div>
                        <div style="font-size:0.85rem;color:#8a9aaa;margin-top:0.4rem">
                            {sample_num} face samples captured for <strong style="color:#38bdf8">{r_name.strip()}</strong>
                            (ID: {r_id.strip()})
                        </div>
                        <div class="result-note">
                            Serial #<strong>{serial}</strong> assigned. Now click
                            <em>Save Profile</em> to train the recognition model.
                        </div>
                    </div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Train Model ───────────────────────────────────────────────────────
        elif train_btn:
            if not cascade_ok:
                st.error("Haar cascade file missing.")
            else:
                assure_path_exists(LABEL_DIR)
                faces, ids = get_images_and_labels(TRAIN_DIR)
                if not faces:
                    st.warning("No training images found. Please capture images first.")
                else:
                    with st.spinner("Training LBPH face recogniser…"):
                        recognizer = cv2.face.LBPHFaceRecognizer_create()
                        recognizer.train(faces, np.array(ids))
                        recognizer.save(trainner_path())
                    st.markdown(f"""
                    <div class="result-wrap">
                    <div class="result-success">
                        <span class="result-icon">🧠</span>
                        <div class="result-verdict green">Profile Saved</div>
                        <div style="font-size:0.85rem;color:#8a9aaa;margin-top:0.4rem">
                            Model trained on <strong style="color:#38bdf8">{len(faces)}</strong> face samples
                            across <strong style="color:#38bdf8">{len(set(ids))}</strong> registered student(s).
                        </div>
                        <div class="result-note">
                            The recognition model is ready. Switch to the
                            <em>Take Attendance</em> tab to begin.
                        </div>
                    </div>
                    </div>
                    """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="idle-card">
                <div class="idle-icon">◈</div>
                <div class="idle-head">Awaiting Registration</div>
                <div class="idle-body">
                    Enter student ID and name on the left,<br>
                    capture face images, then save the profile<br>
                    to train the recognition model.
                </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — TAKE ATTENDANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    att_col, att_res = st.columns([1.05, 0.95], gap="medium")

    with att_col:
        st.markdown('<div class="section-label">01 — Recognition Settings</div>', unsafe_allow_html=True)
        st.markdown('<div class="form-panel">', unsafe_allow_html=True)

        conf_thresh = st.slider(
            "Confidence threshold (lower = stricter)", 30, 80, 50,
            help="Faces with LBPH confidence below this value are recognised. "
                 "Raise if too many 'Unknown'; lower if getting false matches."
        )
        st.markdown(f"""
        <div style="font-size:0.76rem;color:var(--muted);margin-top:0.4rem;line-height:1.6">
            Threshold: <strong style="color:var(--blue)">{conf_thresh}</strong> &nbsp;·&nbsp;
            Strict at <strong>30</strong>, lenient at <strong>80</strong>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        att_btn = st.button("🎯  Start Attendance (press Q in camera window to stop)")

    with att_res:
        st.markdown('<div class="section-label">02 — Attendance Result</div>', unsafe_allow_html=True)

        if att_btn:
            if not cascade_ok:
                st.error("Haar cascade file missing.")
            elif not os.path.isfile(trainner_path()):
                st.warning("No trained model found. Please register students and save the profile first.")
            elif not os.path.isfile(student_csv_path()):
                st.warning("No student records found. Please register students first.")
            else:
                assure_path_exists(ATT_DIR)
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                recognizer.read(trainner_path())
                cascade    = cv2.CascadeClassifier(CASCADE)
                df_students = pd.read_csv(student_csv_path())

                cam  = cv2.VideoCapture(0)
                if not cam.isOpened():
                    st.error("❌ Could not open webcam.")
                else:
                    col_names  = ['Id', '', 'Name', '', 'Date', '', 'Time']
                    attendance = []
                    live_slot  = st.empty()
                    info_slot  = st.empty()

                    while True:
                        ret, frame = cam.read()
                        if not ret:
                            break
                        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = cascade.detectMultiScale(gray, 1.2, 5)

                        for (x, y, w, h) in faces:
                            serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
                            ts    = time.time()
                            date  = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                            tstr  = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

                            if conf < conf_thresh:
                                name_rows = df_students.loc[
                                    df_students['SERIAL NO.'] == serial, 'NAME'].values
                                id_rows   = df_students.loc[
                                    df_students['SERIAL NO.'] == serial, 'ID'].values
                                name = str(name_rows[0]) if len(name_rows) else "Unknown"
                                uid  = str(id_rows[0])   if len(id_rows)   else "—"
                                attendance = [str(uid), '', name, '', date, '', tstr]
                                color = (56, 189, 248)
                                label = f"{name} ({uid})"
                            else:
                                name  = "Unknown"
                                uid   = "—"
                                color = (244, 63, 94)
                                label = "Unknown"

                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(frame, label, (x, y+h+20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        live_slot.image(rgb, channels="RGB", use_container_width=True)
                        info_slot.info("📷 Camera running — press **Q** in the camera window to stop")

                        if cv2.waitKey(1) == ord('q'):
                            break

                    cam.release()
                    cv2.destroyAllWindows()
                    live_slot.empty()
                    info_slot.empty()

                    # Write attendance CSV
                    if attendance and attendance[2] != "Unknown":
                        att_file = attendance_csv_path()
                        file_exists = os.path.isfile(att_file)
                        with open(att_file, 'a+') as cf:
                            w = csv.writer(cf)
                            if not file_exists:
                                w.writerow(col_names)
                            w.writerow(attendance)

                        st.markdown(f"""
                        <div class="result-wrap">
                        <div class="result-success">
                            <span class="result-icon">✅</span>
                            <div class="result-verdict green">Attendance Marked</div>
                            <div style="font-size:0.85rem;color:#8a9aaa;margin-top:0.4rem">
                                <strong style="color:#38bdf8">{attendance[2]}</strong>
                                (ID: {attendance[0]}) &nbsp;·&nbsp; {attendance[4]} &nbsp;·&nbsp; {attendance[6]}
                            </div>
                            <div class="result-note">
                                Attendance successfully recorded. Switch to
                                <em>View Records</em> to see today's full log.
                            </div>
                        </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="result-warning">
                            <span class="result-icon">⚠️</span>
                            <div class="result-verdict amber">No Face Recognised</div>
                            <div style="font-size:0.85rem;color:#8a9aaa;margin-top:0.4rem">
                                No known face was detected with sufficient confidence.
                            </div>
                            <div class="result-note">
                                Try lowering the confidence threshold, ensure good lighting,
                                or re-register the student with more face samples.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="idle-card">
                <div class="idle-icon">◈</div>
                <div class="idle-head">Ready to Scan</div>
                <div class="idle-body">
                    Adjust the confidence threshold if needed,<br>
                    then click <em>Start Attendance</em> to open<br>
                    the camera and recognise faces.
                </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — VIEW RECORDS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    view_col, dl_col = st.columns([1.3, 0.7], gap="medium")

    with view_col:
        st.markdown('<div class="section-label">01 — Today\'s Attendance Log</div>',
                    unsafe_allow_html=True)
        df_today = load_attendance_today()
        if df_today.empty:
            st.markdown("""
            <div class="idle-card">
                <div class="idle-icon">📋</div>
                <div class="idle-head">No Records Yet</div>
                <div class="idle-body">
                    No attendance has been recorded today.<br>
                    Use the <em>Take Attendance</em> tab to begin.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.dataframe(df_today, use_container_width=True, hide_index=True)

    with dl_col:
        st.markdown('<div class="section-label">02 — Download & Manage</div>',
                    unsafe_allow_html=True)

        # Download today's CSV
        if not df_today.empty:
            csv_bytes = df_today.to_csv(index=False).encode()
            st.download_button(
                label="⬇️  Download Today's Attendance",
                data=csv_bytes,
                file_name=f"Attendance_{datetime.datetime.now().strftime('%d-%m-%Y')}.csv",
                mime="text/csv",
            )

        # Registered students
        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">03 — Registered Students</div>',
                    unsafe_allow_html=True)
        if os.path.isfile(student_csv_path()):
            df_students = pd.read_csv(student_csv_path())
            df_display  = df_students[["SERIAL NO.", "ID", "NAME"]].dropna()
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            st.download_button(
                label="⬇️  Download Student List",
                data=df_students.to_csv(index=False).encode(),
                file_name="StudentDetails.csv",
                mime="text/csv",
            )
        else:
            st.markdown("""
            <div class="idle-card" style="padding:2rem 1.5rem">
                <div class="idle-body">No students registered yet.</div>
            </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:2rem;font-size:0.72rem;color:#1e3a52;
            border-top:1px solid #182430;padding-top:1rem;line-height:1.7">
    This system is for institutional use only. Facial recognition data should be
    handled in accordance with applicable privacy regulations and institutional policies.
    Ensure informed consent is obtained before enrolling students.
</div>
""", unsafe_allow_html=True)