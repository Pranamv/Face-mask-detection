import os
import time
from typing import Tuple

import cv2
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification


def load_model(model_dir: str, device: torch.device):

    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(
        model_dir, local_files_only=True
    )
    model.to(device)
    model.eval()
    return processor, model


def draw_label(frame, text: str, topleft: Tuple[int, int], color=(0, 255, 0)):
    x, y = topleft
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    # Background for text
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame, (x, y - th - baseline - 4), (x + tw + 4, y), color, -1)
    cv2.putText(frame, text, (x + 2, y - 4), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def main():
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = script_dir  # model files are in the same folder

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    try:
        processor, model = load_model(model_dir, device)
    except Exception as e:
        print("Failed to load model. Ensure config.json, preprocessor_config.json, and model.safetensors are present in this folder.")
        raise e

    # Normalize id2label to have integer keys to avoid showing 0/1
    raw_id2label = model.config.id2label
    id2label = {}
    for k, v in raw_id2label.items():
        try:
            ik = int(k)
        except Exception:
            ik = k
        id2label[ik] = v

    # Load a face detector (OpenCV Haar Cascade)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(
            "OpenCV Haar Cascade not found. Please ensure opencv-python is installed correctly."
        )
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Open webcam (DirectShow on Windows can reduce latency)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # Fallback
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. If you have multiple cameras, change the index in VideoCapture(0).")

    # Reduce capture resolution for higher FPS
    target_w, target_h = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # Try MJPG to lower latency on some webcams
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    except Exception:
        pass
    # Reduce internal buffering if supported
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    # Threaded frame grabber to decouple capture from processing
    latest = {"frame": None, "stopped": False, "results": [], "ts": 0.0}

    def grab_frames():
        while not latest["stopped"]:
            ret, frm = cap.read()
            if not ret:
                continue
            latest["frame"] = frm

    import threading
    t = threading.Thread(target=grab_frames, daemon=True)
    t.start()

    # Background worker for detection + inference
    DETECT_FPS = 10.0  # run detection/inference ~10 times per second
    DETECT_INTERVAL = 1.0 / DETECT_FPS

    def detect_and_classify():
        use_amp = device.type == "cuda"
        if use_amp:
            try:
                model.half()
            except Exception:
                pass
        last_time = 0.0
        while not latest["stopped"]:
            now = time.time()
            if now - last_time < DETECT_INTERVAL:
                time.sleep(0.001)
                continue
            last_time = now

            frame = latest["frame"]
            if frame is None:
                time.sleep(0.001)
                continue

            # Detect faces on a downscaled copy for speed, then map boxes back
            fh, fw = frame.shape[:2]
            scale = 0.5
            small = cv2.resize(frame, (int(fw*scale), int(fh*scale)))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces_small = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            results = []  # (x, y, w, h, label, score, color)
            for (sx, sy, sw, sh) in faces_small:
                x, y, w, h = int(sx/scale), int(sy/scale), int(sw/scale), int(sh/scale)

                face_bgr = frame[y:y+h, x:x+w]
                if face_bgr.size == 0:
                    continue
                face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

                inputs = processor(images=face_rgb, return_tensors="pt")
                if use_amp:
                    inputs = {k: v.half().to(device) for k, v in inputs.items()}
                else:
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    if use_amp:
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            outputs = model(**inputs)
                    else:
                        outputs = model(**inputs)
                    logits = outputs.logits
                    pred_id = int(torch.argmax(logits, dim=-1).item())
                    score = float(torch.softmax(logits, dim=-1)[0, pred_id].item())

                label = id2label.get(pred_id, str(pred_id))
                # Determine color explicitly: WithMask -> green, WithoutMask -> red
                lab_norm = str(label).strip().replace(" ", "").lower()
                is_with_mask = lab_norm in ("withmask", "mask", "with_mask")
                color = (0, 200, 0) if is_with_mask else (0, 0, 255)
                results.append((x, y, w, h, label, score, color))

            latest["results"] = results
            latest["ts"] = now

    worker = threading.Thread(target=detect_and_classify, daemon=True)
    worker.start()

    print("Press 'q' to quit.")
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            frame = latest["frame"]
            if frame is None:
                # camera thread not yet delivered a frame
                time.sleep(0.005)
                continue

            # Compute FPS (smoothed)
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Draw latest detection results without blocking
            for (x, y, w, h, label, score, color) in latest["results"]:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                if label:
                    draw_label(frame, f"{label} ({score*100:.1f}%)", (x, y), color=color)

            # Overlay FPS
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Face Mask Detection (ViT)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        latest["stopped"] = True
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
