import os
import time
from typing import Generator, Optional, Tuple
import cv2
import numpy as np
import torch
from flask import Flask, Response, flash, redirect, render_template, request, send_from_directory, url_for
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE_DIR  # model files are alongside this script
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
FACES_DIR = os.path.join(UPLOAD_DIR, "faces_without_mask")
os.makedirs(FACES_DIR, exist_ok=True)

# Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

# Shared live state for saving faces from the live stream
LIVE_STATE = {"frame": None, "results": []}
# Continuous capture control/state for saving faces without mask from live stream
CAPTURE_STATE = {"enabled": False, "last_save": 0.0, "min_interval": 0.7, "counter": 0}


def load_model(model_dir: str, device: torch.device):
    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(model_dir, local_files_only=True)
    model.to(device)
    model.eval()
    # Normalize id2label keys to int
    raw = model.config.id2label
    id2label = {}
    for k, v in raw.items():
        try:
            id2label[int(k)] = v
        except Exception:
            id2label[k] = v
    return processor, model, id2label


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = get_device()
PROCESSOR, MODEL, ID2LABEL = load_model(MODEL_DIR, DEVICE)

# Use half precision if CUDA available
USE_AMP = DEVICE.type == "cuda"
if USE_AMP:
    try:
        # Optimizations for CUDA
        MODEL.half()
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
    except Exception:
        pass
if DEVICE.type == "cuda":
    try:
        # Channels-last can improve performance on some GPUs
        MODEL.to(memory_format=torch.channels_last)
    except Exception:
        pass

# Face detector
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
if not os.path.exists(CASCADE_PATH):
    raise FileNotFoundError("Haar Cascade not found. Please reinstall opencv-python.")
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)


# ---------------------- Helpers ----------------------

def color_for_label(label: str) -> Tuple[int, int, int]:
    lab_norm = str(label).strip().replace(" ", "").lower()
    is_with_mask = lab_norm in ("withmask", "mask", "with_mask")
    return (0, 200, 0) if is_with_mask else (0, 0, 255)


def draw_label(frame, text: str, x: int, y: int, color=(0, 255, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (x, y - th - baseline - 4), (x + tw + 4, y), color, -1)
    cv2.putText(frame, text, (x + 2, y - 4), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def detect_and_classify(frame: np.ndarray, scale: float = 0.5, minNeighbors: int = 5):
    """Detect faces and classify mask status.
    - scale: downscale factor used for face detection (smaller = faster, more delay reduction)
    - minNeighbors: cascade stability (higher reduces jitter)
    """
    # Downscale for faster detection
    fh, fw = frame.shape[:2]
    small = cv2.resize(frame, (max(1, int(fw*scale)), max(1, int(fh*scale))))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces_small = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.07, minNeighbors=minNeighbors, minSize=(24, 24)
    )

    results = []  # (x,y,w,h,label,score,color)
    boxes = []
    faces_rgb = []
    for (sx, sy, sw, sh) in faces_small:
        x, y, w, h = int(sx/scale), int(sy/scale), int(sw/scale), int(sh/scale)
        face_bgr = frame[y:y+h, x:x+w]
        if face_bgr.size == 0:
            continue
        boxes.append((x, y, w, h))
        faces_rgb.append(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))

    if faces_rgb:
        inputs = PROCESSOR(images=faces_rgb, return_tensors="pt")
        if USE_AMP:
            inputs = {k: v.half().to(DEVICE) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            if USE_AMP:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = MODEL(**inputs)
            else:
                outputs = MODEL(**inputs)
            logits = outputs.logits  # shape [N, C]
            probs = torch.softmax(logits, dim=-1)
            pred_ids = probs.argmax(dim=-1).tolist()
            scores = probs.max(dim=-1).values.tolist()

        for (x, y, w, h), pred_id, score in zip(boxes, pred_ids, scores):
            label = ID2LABEL.get(int(pred_id), str(pred_id))
            color = color_for_label(label)
            results.append((x, y, w, h, label, float(score), color))

    return results


def annotate_frame(frame: np.ndarray, results):
    for (x, y, w, h, label, score, color) in results:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        draw_label(frame, f"{label} ({score*100:.1f}%)", x, y, color)
    return frame


# ---------------------- Routes ----------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict_image", methods=["POST"])
def predict_image():
    file = request.files.get("image")
    if not file or file.filename == "":
        flash("Please upload an image.")
        return redirect(url_for("index"))

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(save_path)

    img = cv2.imdecode(np.fromfile(save_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        # Fallback: try cv2.imread if np.fromfile failed due to path
        img = cv2.imread(save_path)
    if img is None:
        flash("Could not read the uploaded image.")
        return redirect(url_for("index"))

    results = detect_and_classify(img)
    out = annotate_frame(img.copy(), results)

    out_name = f"annotated_{int(time.time())}.jpg"
    out_path = os.path.join(UPLOAD_DIR, out_name)
    cv2.imwrite(out_path, out)

    return send_from_directory(UPLOAD_DIR, out_name, as_attachment=False)


@app.route("/upload_video", methods=["POST"])
def upload_video():
    file = request.files.get("video")
    if not file or file.filename == "":
        flash("Please upload a video.")
        return redirect(url_for("index"))
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(save_path)
    return redirect(url_for("video", src="file", path=os.path.basename(save_path)))


@app.route("/video")
def video():
    """Render a page with a streaming player. Query params:
    - src=live|file
    - path=<filename> (for file)
    """
    src = request.args.get("src", "live")
    path = request.args.get("path", "")
    return render_template("video.html", src=src, video_path=path)


@app.route("/video_feed")
def video_feed():
    src = request.args.get("src", "live")
    path = request.args.get("path")

    def gen() -> Generator[bytes, None, None]:
        # Open capture
        is_file = bool(src == "file" and path)
        if is_file:
            cap = cv2.VideoCapture(os.path.join(UPLOAD_DIR, path))
        else:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            # Use 640x480 @ 30 FPS to reduce device buffering and latency
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        if not cap.isOpened():
            return

        # Shared state for threads
        latest = {"frame": None, "stopped": False, "results": []}

        # If playing a file, determine FPS to preserve real-time speed
        file_fps = 30.0
        read_interval = 1.0 / file_fps
        if is_file:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 0 and fps < 120:
                file_fps = float(fps)
            else:
                file_fps = 30.0
            read_interval = 1.0 / file_fps

        # Threaded frame grabber (non-blocking)
        def grab_frames():
            if is_file:
                next_read = time.time()
            while not latest["stopped"]:
                if is_file:
                    # Throttle reads to file FPS to avoid fast playback
                    now = time.time()
                    if now < next_read:
                        time.sleep(min(0.005, next_read - now))
                        continue
                    next_read += read_interval
                ok, frm = cap.read()
                if not ok:
                    time.sleep(0.005)
                    continue
                latest["frame"] = frm

        # Background detection worker (runs at fixed rate)
        DETECT_FPS = 30.0  # default detection loop rate
        if is_file:
            DETECT_FPS = min(30.0, max(5.0, file_fps))
        DETECT_INTERVAL = 1.0 / DETECT_FPS

        def detect_worker():
            last_time = 0.0
            while not latest["stopped"]:
                now = time.time()
                if now - last_time < DETECT_INTERVAL:
                    time.sleep(0.002)
                    continue
                last_time = now
                frm = latest["frame"]
                if frm is None:
                    continue
                # Work on a copy to avoid partial modifications while streaming
                # Use a moderate downscale to keep detection fast and responsive
                results = detect_and_classify(frm.copy(), scale=0.50, minNeighbors=4)
                latest["results"] = results

                # If continuous capture is enabled, periodically save faces without mask
                if CAPTURE_STATE["enabled"] and (time.time() - CAPTURE_STATE["last_save"]) >= CAPTURE_STATE["min_interval"]:
                    saved_any = False
                    for (x, y, w, h, label, score, color) in results:
                        lab_norm = str(label).strip().replace(" ", "").lower()
                        if lab_norm in ("withoutmask", "no_mask", "nomask", "without_mask"):
                            crop = frm[max(0,y):max(0,y)+max(1,h), max(0,x):max(0,x)+max(1,w)]
                            if crop.size == 0:
                                continue
                            ts = int(time.time()*1000)
                            idx = CAPTURE_STATE["counter"]
                            fname = f"wm_live_{ts}_{idx}.jpg"
                            fpath = os.path.join(FACES_DIR, fname)
                            cv2.imwrite(fpath, crop)
                            CAPTURE_STATE["counter"] += 1
                            saved_any = True
                    if saved_any:
                        CAPTURE_STATE["last_save"] = time.time()

        import threading
        t_cap = threading.Thread(target=grab_frames, daemon=True)
        t_det = threading.Thread(target=detect_worker, daemon=True)
        t_cap.start()
        t_det.start()

        # Streaming/render loop at target FPS
        STREAM_FPS = 30.0
        if is_file:
            STREAM_FPS = min(30.0, max(5.0, file_fps))
        STREAM_INTERVAL = 1.0 / STREAM_FPS
        try:
            last = 0.0
            while True:
                now = time.time()
                if now - last < STREAM_INTERVAL:
                    time.sleep(0.002)
                    continue
                last = now
                frm = latest["frame"]
                if frm is None:
                    continue
                annotated = annotate_frame(frm.copy(), latest["results"])
                # Update global live state for saving
                LIVE_STATE["frame"] = frm
                LIVE_STATE["results"] = latest["results"]
                # Slightly lower JPEG quality to reduce network/buffer latency
                ok, jpeg = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if not ok:
                    continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        finally:
            latest["stopped"] = True
            cap.release()

    return Response(
        gen(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
            'Pragma': 'no-cache'
        }
    )


@app.route('/toggle_capture', methods=['POST'])
def toggle_capture():
    """Enable/disable continuous capture of faces without mask from live stream."""
    try:
        data = request.get_json(silent=True) or {}
    except Exception:
        data = {}
    enable = data.get('enable')
    if enable is None:
        # Toggle if not explicitly set
        CAPTURE_STATE['enabled'] = not CAPTURE_STATE['enabled']
    else:
        CAPTURE_STATE['enabled'] = bool(enable)
    return {"ok": True, "enabled": CAPTURE_STATE['enabled']}


@app.route('/capture_status', methods=['GET'])
def capture_status():
    return {"ok": True, "enabled": CAPTURE_STATE['enabled'], "last_save": CAPTURE_STATE['last_save'], "dir": "/uploads/faces_without_mask/"}


@app.route('/save_without_mask', methods=['POST'])
def save_without_mask():
    """Save all faces labeled WithoutMask from the current live frame.
    Returns JSON with count and saved filenames.
    """
    frame = LIVE_STATE.get("frame")
    results = LIVE_STATE.get("results", [])
    if frame is None:
        return {"ok": False, "message": "No live frame available."}, 400

    saved = []
    ts = int(time.time())
    idx = 0
    for (x, y, w, h, label, score, color) in results:
        lab_norm = str(label).strip().replace(" ", "").lower()
        if lab_norm in ("withoutmask", "no_mask", "nomask", "without_mask"):
            crop = frame[max(0,y):max(0,y)+max(1,h), max(0,x):max(0,x)+max(1,w)]
            if crop.size == 0:
                continue
            fname = f"wm_face_{ts}_{idx}.jpg"
            fpath = os.path.join(FACES_DIR, fname)
            cv2.imwrite(fpath, crop)
            saved.append(fname)
            idx += 1

    return {"ok": True, "count": len(saved), "files": saved, "dir": "/uploads/faces_without_mask/"}


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    # For production, consider running via: waitress-serve --host=0.0.0.0 --port=8000 app:app
    app.run(host="0.0.0.0", port=5000, debug=True)
