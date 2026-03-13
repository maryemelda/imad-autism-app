import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
import json
import os
from datetime import datetime
from PIL import Image
import cv2

# ---------------- CONFIG ----------------

st.set_page_config(page_title="iMAD", layout="centered")

# ---------------- STYLE + WATERMARK ----------------

st.markdown("""
<style>
.watermark {
 position: fixed;
 bottom: 10px;
 right: 20px;
 font-size: 46px;
 font-style: italic;
 color: rgba(0,0,0,0.07);
 font-weight: 900;
 z-index: 0;
}

.big {font-size:42px;font-weight:800;text-align:center;color:#1565c0;}

.stButton>button {
 background: linear-gradient(90deg,#42a5f5,#1e88e5);
 color:white;
 font-weight:700;
 border-radius:12px;
 height:48px;
}
</style>

<div class="watermark">iMAD</div>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------

@st.cache_resource
def load_models():
    return (
        tf.keras.models.load_model("audio_emotion_model.keras", compile=False),
        tf.keras.models.load_model("gaze_geo_model.keras", compile=False),
        tf.keras.models.load_model("asd_image_model_fixed.keras", compile=False)
    )

audio_model, gaze_model, image_model = load_models()

# ---------------- EYE DETECTOR ----------------

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# ---------------- SESSION FOLDER HELPER ----------------

def get_session_folder():
    """
    Creates and returns a unique folder for this session.
    Folder: sessions/<ChildName>_<YYYY-MM-DD_HH-MM-SS>/
    All audio, images, and report.json are saved inside.
    """
    if "session_folder" not in st.session_state:
        name = st.session_state.get("child_name", "Unknown").replace(" ", "_")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = os.path.join("sessions", f"{name}_{timestamp}")
        os.makedirs(folder, exist_ok=True)
        st.session_state.session_folder = folder
    return st.session_state.session_folder

# ---------------- STATE ----------------

if "screen" not in st.session_state:
    st.session_state.screen = "login"

if "scores" not in st.session_state:
    st.session_state.scores = dict(q=0, g=0, a=0, v=0, z=0)

if "game_correct" not in st.session_state:
    st.session_state.game_correct = 0

if "child_name" not in st.session_state:
    st.session_state.child_name = "Unknown"

if "child_info" not in st.session_state:
    st.session_state.child_info = {}

# ---------------- GOTO ----------------

def goto(x):
    if x == "g":
        st.session_state.game_correct = 0
        if "game_initialized" in st.session_state:
            del st.session_state["game_initialized"]
    st.session_state.screen = x
    st.rerun()

# =====================================================
# LOGIN
# =====================================================

if st.session_state.screen == "login":

    st.markdown('<div class="big">iMAD</div>', unsafe_allow_html=True)

    st.image(
        "https://images.unsplash.com/photo-1503454537195-1dcabb73ffb9",
        use_container_width=True
    )

    st.subheader("Login")
    st.text_input("Email")
    st.text_input("Password", type="password")

    if st.button("Login"):
        goto("child")

# =====================================================
# CHILD INFO
# =====================================================

elif st.session_state.screen == "child":

    st.header("Child Information")

    name        = st.text_input("Name")
    age         = st.number_input("Age", 1, 18, 5)
    gender      = st.selectbox("Gender", ["Male", "Female", "Other"])
    nationality = st.text_input("Nationality")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.text_input("Session Time", value=now, disabled=True)

    if st.button("Continue"):
        st.session_state.child_name = name if name else "Unknown"
        st.session_state.child_info = {
            "name":         name,
            "age":          int(age),
            "gender":       gender,
            "nationality":  nationality,
            "session_time": now
        }
        # Create the session folder right after child info is filled
        get_session_folder()
        goto("select")

# =====================================================
# SELECT TEST
# =====================================================

elif st.session_state.screen == "select":

    st.markdown('<div class="big">Select Test</div>', unsafe_allow_html=True)

    if st.button("📋 Questionnaire"): goto("q")
    if st.button("🎮 Game"):          goto("g")
    if st.button("🎤 Audio"):         goto("a")
    if st.button("📷 Video"):         goto("v")
    if st.button("👁 Gaze"):          goto("z")
    if st.button("📊 Final Results"): goto("results")

# =====================================================
# QUESTIONNAIRE
# =====================================================

elif st.session_state.screen == "q":

    st.header("Questionnaire")

    qs = [
        "Maintains eye contact",
        "Responds to name",
        "Uses gestures",
        "Pretend play",
        "Shows emotions"
    ]

    score = sum(st.radio(q, ["Yes", "No"], key=q) == "No" for q in qs)

    if st.button("Evaluate Questionnaire"):
        risk = score / len(qs)
        st.session_state.scores["q"] = risk
        st.success(f"Risk Score = {risk:.2f}")

    if st.button("Back"):
        goto("select")

# =====================================================
# GAME — VISUAL
# =====================================================

elif st.session_state.screen == "g":

    if "game_initialized" not in st.session_state:
        st.session_state.game_correct = 0
        st.session_state.game_initialized = True

    st.header("Visual Identification Game")

    questions = [

        ("Select apple",
         [("https://upload.wikimedia.org/wikipedia/commons/1/15/Red_Apple.jpg", "apple"),
          ("https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Single.jpg", "banana"),
          ("https://upload.wikimedia.org/wikipedia/commons/9/90/Hapus_Mango.jpg", "mango"),
          ("https://upload.wikimedia.org/wikipedia/commons/c/c4/Orange-Fruit-Pieces.jpg", "orange")],
         "apple"),

        ("Select red colour",
         [("https://singlecolorimage.com/get/ff0000/200x200", "red"),
          ("https://singlecolorimage.com/get/0000ff/200x200", "blue"),
          ("https://singlecolorimage.com/get/00ff00/200x200", "green"),
          ("https://singlecolorimage.com/get/ffff00/200x200", "yellow")],
         "red"),

        ("Select square",
         [("https://via.assets.so/img.jpg?w=200&h=200&tc=black&bg=black&t=■", "square"),
          ("https://via.assets.so/img.jpg?w=200&h=200&tc=black&bg=white&t=●", "circle"),
          ("https://via.assets.so/img.jpg?w=200&h=200&tc=black&bg=white&t=▲", "triangle"),
          ("https://via.assets.so/img.jpg?w=200&h=200&tc=black&bg=white&t=▬", "rectangle")],
         "square"),

        ("Select chair",
         [("https://cdn-icons-png.flaticon.com/512/2271/2271334.png", "chair"),
          ("https://cdn-icons-png.flaticon.com/512/1581/1581811.png", "table"),
          ("https://cdn-icons-png.flaticon.com/512/2631/2631611.png", "sofa"),
          ("https://cdn-icons-png.flaticon.com/512/4149/4149676.png", "bed")],
         "chair"),

        ("Select happy face",
         [("https://cdn-icons-png.flaticon.com/512/166/166538.png", "happy"),
          ("https://cdn-icons-png.flaticon.com/512/166/166543.png", "sad"),
          ("https://cdn-icons-png.flaticon.com/512/166/166549.png", "angry"),
          ("https://cdn-icons-png.flaticon.com/512/166/166553.png", "surprise")],
         "happy"),
    ]

    for qi, (q, opts, ans) in enumerate(questions):
        st.subheader(q)
        cols = st.columns(4)
        for i, (url, label) in enumerate(opts):
            with cols[i]:
                st.image(url)
                if st.button("Select", key=f"{qi}-{i}"):
                    if label == ans:
                        st.success("Correct")
                        st.session_state.game_correct += 1
                    else:
                        st.error("Wrong")

    if st.button("Evaluate Game"):
        risk = 1 - st.session_state.game_correct / len(questions)
        st.session_state.scores["g"] = risk
        st.success(f"Game Risk = {risk:.2f}")

    if st.button("Back"):
        goto("select")

# =====================================================
# AUDIO
# =====================================================

elif st.session_state.screen == "a":

    st.header("Audio Recording")

    audio_bytes = st.audio_input("Record voice")

    if audio_bytes:
        st.audio(audio_bytes)

        if st.button("Evaluate Audio"):

            y, sr = librosa.load(audio_bytes, sr=16000)

            # ---- SAVE AUDIO ----
            folder     = get_session_folder()
            audio_path = os.path.join(folder, "audio_recording.wav")
            sf.write(audio_path, y, sr)
            st.session_state["saved_audio_path"] = audio_path
            # --------------------

            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
            mel = librosa.power_to_db(mel)

            if mel.shape[1] < 128:
                mel = np.pad(mel, ((0, 0), (0, 128 - mel.shape[1])), mode='constant')
            else:
                mel = mel[:, :128]

            mel   = (mel - mel.mean()) / (mel.std() + 1e-6)
            mel   = mel[..., None][None, ...]
            probs = audio_model.predict(mel, verbose=0)[0]
            risk  = float(1 - np.max(probs))

            st.session_state.scores["a"] = risk
            st.success(f"Audio Risk = {risk:.2f}")
            st.info(f"✅ Audio saved → {audio_path}")

    if st.button("Back"):
        goto("select")

# =====================================================
# VIDEO
# =====================================================

elif st.session_state.screen == "v":

    st.header("Video Session — Capture 3 Frames")

    if "video_frames" not in st.session_state:
        st.session_state.video_frames     = [None, None, None]
        st.session_state.video_frames_pil = [None, None, None]

    for i in range(3):
        img = st.camera_input(f"Frame {i+1}", key=f"vf{i}")
        if img:
            pil_img = Image.open(img)
            arr     = np.array(pil_img.resize((224, 224))) / 255.0
            st.session_state.video_frames[i]     = arr
            st.session_state.video_frames_pil[i] = pil_img.copy()

    captured = sum(f is not None for f in st.session_state.video_frames)
    st.info(f"Frames captured: {captured}/3")

    if st.button("Evaluate Video"):
        frames     = st.session_state.video_frames
        frames_pil = st.session_state.video_frames_pil

        if any(f is None for f in frames):
            st.error("Capture all 3 frames first")
        else:
            preds = [image_model.predict(f[None, ...], verbose=0)[0][0] for f in frames]
            risk  = float(np.mean(preds))
            st.session_state.scores["v"] = risk

            # ---- SAVE VIDEO FRAMES ----
            folder       = get_session_folder()
            video_folder = os.path.join(folder, "video_frames")
            os.makedirs(video_folder, exist_ok=True)
            for idx, pil_img in enumerate(frames_pil):
                pil_img.save(os.path.join(video_folder, f"frame_{idx+1}.jpg"))
            st.session_state["saved_video_folder"] = video_folder
            # ---------------------------

            st.session_state.video_frames     = [None, None, None]
            st.session_state.video_frames_pil = [None, None, None]
            st.success(f"Video Risk = {risk:.2f}")
            st.info(f"✅ Frames saved → {video_folder}")

    if st.button("Back"):
        st.session_state.video_frames     = [None, None, None]
        st.session_state.video_frames_pil = [None, None, None]
        goto("select")

# =====================================================
# GAZE
# =====================================================

elif st.session_state.screen == "z":

    st.header("Eye Capture")

    img_file = st.camera_input("Capture eye close-up")

    if img_file:

        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        frame      = cv2.imdecode(file_bytes, 1)
        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes       = eye_cascade.detectMultiScale(gray, 1.3, 5)

        if len(eyes) == 0:
            st.error("No eyes detected — try better lighting or move camera closer.")
        else:
            (x, y, w, h) = eyes[0]
            eye = gray[y:y+h, x:x+w]

            # ---- SAVE GAZE IMAGES ----
            folder    = get_session_folder()
            gaze_path = os.path.join(folder, "gaze_capture.jpg")
            eye_path  = os.path.join(folder, "eye_crop.jpg")
            cv2.imwrite(gaze_path, frame)   # Full face frame
            cv2.imwrite(eye_path,  eye)     # Cropped eye region
            st.session_state["saved_gaze_path"] = gaze_path
            # --------------------------

            feats = [
                np.mean(eye), np.std(eye), w, h,
                np.max(eye),  np.min(eye),
                np.percentile(eye, 25), np.percentile(eye, 75)
            ]

            risk = float(gaze_model.predict(np.array(feats).reshape(1, -1))[0][0])
            st.session_state.scores["z"] = risk
            st.success(f"Gaze Risk = {risk:.2f}")
            st.info(f"✅ Gaze image saved → {gaze_path}")

    if st.button("Back"):
        goto("select")

# =====================================================
# RESULTS
# =====================================================

elif st.session_state.screen == "results":

    s     = st.session_state.scores
    final = .2*s["q"] + .2*s["g"] + .2*s["a"] + .2*s["v"] + .2*s["z"]

    if final < .33:
        band, color = "LOW RISK",  "#2ecc71"
    elif final < .66:
        band, color = "MODERATE",  "#f39c12"
    else:
        band, color = "HIGH RISK", "#e74c3c"

    st.markdown(f"""
    <div style='background:{color};padding:20px;border-radius:16px;
    color:white;font-size:28px;font-weight:800;text-align:center'>
    Final Risk Score {final:.2f}<br>{band}
    </div>
    """, unsafe_allow_html=True)

    for name, key, c in [
        ("Questionnaire", "q", "#8e44ad"),
        ("Game",          "g", "#2980b9"),
        ("Audio",         "a", "#16a085"),
        ("Video",         "v", "#d35400"),
        ("Gaze",          "z", "#c0392b"),
    ]:
        st.markdown(f"""
        <div style='background:{c};padding:14px;border-radius:12px;
        color:white;font-size:18px;font-weight:700;margin:6px 0'>
        {name}: {s[key]:.2f}
        </div>
        """, unsafe_allow_html=True)

    # ---- SAVE FULL REPORT as JSON ----
    folder = get_session_folder()
    report = {
        "child_info": st.session_state.child_info,
        "scores": {
            "questionnaire": round(s["q"], 4),
            "game":          round(s["g"], 4),
            "audio":         round(s["a"], 4),
            "video":         round(s["v"], 4),
            "gaze":          round(s["z"], 4),
            "final":         round(final,  4),
        },
        "risk_band": band,
        "saved_files": {
            "audio":        st.session_state.get("saved_audio_path",   "Not recorded"),
            "video_frames": st.session_state.get("saved_video_folder", "Not captured"),
            "gaze_image":   st.session_state.get("saved_gaze_path",    "Not captured"),
        },
        "report_generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    report_path = os.path.join(folder, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    # ----------------------------------

    st.markdown("---")
    st.success(f"✅ Full session saved to:  **{folder}**")
    st.caption("📁 Folder contains:  audio_recording.wav  |  video_frames/frame_1,2,3.jpg  |  gaze_capture.jpg  |  eye_crop.jpg  |  report.json")

    with st.expander("📄 View Full Report (JSON)"):
        st.json(report)

    if st.button("Back"):
        goto("select")
