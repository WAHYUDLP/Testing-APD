import cv2
import requests
import os
import time
import math
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from ultralytics import YOLO
from collections import defaultdict

# --- KONFIGURASI API ---
# Daftar gratis di api.imgbb.com untuk dapat kuncinya
IMGBB_API_KEY = "158ee9e068a89b28e5b374a664a8e192" 

# Info lokasi kejadian (isi sesuai lokasi proyek)
SITE_LOCATION = "Area Proyek A"
SITE_LAT = ""
SITE_LON = ""
TIMEZONE_NAME = "Asia/Jakarta"

# URL Backend Temanmu
# PERHATIAN: Kalau beda laptop, ganti "localhost" dengan IP WiFi temanmu! (misal: 192.168.1.10)
URL_BACKEND = "http://localhost:8000/report-violation"

# --- INISIALISASI MODEL ---
model = YOLO('best.pt')
# Fallback ID (kalau auto-resolve gagal)
APD_CLASS_MAP = {0: "helmet", 1: "mask", 7: "vest"}
PERSON_CLASS_ID = 5

def _norm_label(text):
    return str(text).strip().lower().replace("-", "_").replace(" ", "_")

def _build_name_map(names_obj):
    if isinstance(names_obj, dict):
        return {_norm_label(name): int(cid) for cid, name in names_obj.items()}
    if isinstance(names_obj, list):
        return {_norm_label(name): idx for idx, name in enumerate(names_obj)}
    return {}

def _find_class_id(name_to_id, aliases):
    for alias in aliases:
        normalized = _norm_label(alias)
        if normalized in name_to_id:
            return name_to_id[normalized]
    for alias in aliases:
        normalized = _norm_label(alias)
        for label, cid in name_to_id.items():
            if normalized in label:
                return cid
    return None

name_to_id = _build_name_map(model.model.names)
resolved_person_id = _find_class_id(name_to_id, ["person", "people", "worker"])
resolved_helmet_id = _find_class_id(name_to_id, ["helmet", "hardhat", "hard_hat"])
resolved_mask_id = _find_class_id(name_to_id, ["mask", "face_mask", "facemask"])
resolved_vest_id = _find_class_id(name_to_id, ["vest", "safety_vest", "safetyvest"])

if resolved_person_id is not None:
    PERSON_CLASS_ID = resolved_person_id

dynamic_apd_map = {}
if resolved_helmet_id is not None:
    dynamic_apd_map[resolved_helmet_id] = "helmet"
if resolved_mask_id is not None:
    dynamic_apd_map[resolved_mask_id] = "mask"
if resolved_vest_id is not None:
    dynamic_apd_map[resolved_vest_id] = "vest"

if dynamic_apd_map:
    APD_CLASS_MAP = dynamic_apd_map

print(f"INFO: person class id = {PERSON_CLASS_ID}")
print(f"INFO: APD class map = {APD_CLASS_MAP}")

WINDOW_NAME = "Monitor K3 - YOLO11s"

def now_local_str():
    try:
        current = datetime.now(ZoneInfo(TIMEZONE_NAME))
    except Exception:
        current = datetime.now()
    return current.strftime("%Y-%m-%d %H:%M:%S")

def format_location_text():
    if SITE_LAT and SITE_LON:
        return f"{SITE_LOCATION} ({SITE_LAT}, {SITE_LON})"
    return SITE_LOCATION

def format_violation_type_id(vtype):
    mapping = {
        "not_wearing_helmet": "Tidak Memakai Helm",
        "not_wearing_vest": "Tidak Memakai Vest",
        "not_wearing_mask": "Tidak Memakai Masker",
        "not_wearing_any_apd": "Tidak Memakai APD Lengkap",
        "attempt_remove_helmet": "Melepas Helm",
        "attempt_remove_vest": "Melepas Vest",
        "attempt_remove_mask": "Melepas Masker",
    }
    return mapping.get(vtype, vtype.replace("_", " ").title())

# Ambang overlap APD ke person
APD_PERSON_IOU_THRESHOLD = 0.01

# Turunkan conf agar objek kecil seperti masker lebih mudah tertangkap
CONF_THRESHOLD = 0.25
PERSON_CONF_THRESHOLD = 0.25
APD_CONF_THRESHOLD = {
    "helmet": 0.50,
    "vest": 0.50,
    "mask": 0.25,
}

# Turunkan resolusi kamera dan frame inferensi untuk mengurangi lag di laptop
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360
INFERENCE_DOWNSCALE = 0.5  
FULLSCREEN_VIEW = False
DISPLAY_UPSCALE = 2.0  

MISSING_FRAMES_THRESHOLD = 10
NEVER_WEAR_FRAMES = 5
TELEGRAM_RENOTIFY_INTERVAL_SEC = 300
ALERT_SPATIAL_DISTANCE_PX = 100
MIN_SEEN_FRAMES_FOR_REMOVE_ALERT = 3

tracked_states = {}
last_violation_notification = {}
recent_alert_locations = defaultdict(list)
recent_violations = []

def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

def bbox_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def buka_kamera(index_opsi=(0, 1, 2)):
    for idx in index_opsi:
        # On Windows, DirectShow is often more stable than MSMF for webcam selection.
        if os.name == "nt":
            cap_uji = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        else:
            cap_uji = cv2.VideoCapture(idx)
        if not cap_uji.isOpened():
            cap_uji.release()
            continue
        ok, frame_uji = cap_uji.read()
        if ok and frame_uji is not None:
            print(f"Kamera aktif di index: {idx}")
            return cap_uji
        cap_uji.release()
    return None

cap = buka_kamera()
if cap is None:
    print("ERROR: Kamera tidak terdeteksi.")
    raise SystemExit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
if CAMERA_HEIGHT is not None:
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"INFO: resolusi kamera aktif = {actual_w}x{actual_h}")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
if FULLSCREEN_VIEW:
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
pelanggar_tercatat = set()

split_view = False
monitoring_enabled = True

print("Kontrol runtime: 'p' pause/resume, 's' split view, 'q' quit")
print("=== SISTEM MONITORING K3 AKTIF ===")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("ERROR: Gagal membaca frame dari kamera.")
        break

    if 0 < INFERENCE_DOWNSCALE < 1.0:
        inf_w = max(1, int(frame.shape[1] * INFERENCE_DOWNSCALE))
        inf_h = max(1, int(frame.shape[0] * INFERENCE_DOWNSCALE))
        infer_frame = cv2.resize(frame, (inf_w, inf_h), interpolation=cv2.INTER_AREA)
    else:
        infer_frame = frame

    if not monitoring_enabled:
        frame_paused = frame.copy()
        cv2.putText(frame_paused, "PAUSED - tekan 'p' untuk resume", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(WINDOW_NAME, frame_paused)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            monitoring_enabled = not monitoring_enabled
            print("Monitoring paused" if not monitoring_enabled else "Monitoring resumed")
        continue

    results = model.track(infer_frame, persist=True, tracker="bytetrack.yaml", conf=CONF_THRESHOLD, verbose=False)
    
    person_boxes = {}
    apd_boxes = []

    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf_score = float(box.conf[0]) if box.conf is not None else 0.0
        xyxy = tuple(float(v) for v in box.xyxy[0].tolist())

        if cls == PERSON_CLASS_ID and conf_score >= PERSON_CONF_THRESHOLD and box.id is not None:
            person_track_id = int(box.id[0])
            person_boxes[person_track_id] = {
                "bbox": xyxy,
                "conf": conf_score,
            }

        if cls in APD_CLASS_MAP:
            apd_name = APD_CLASS_MAP[cls]
            min_apd_conf = APD_CONF_THRESHOLD.get(apd_name, CONF_THRESHOLD)
            if conf_score >= min_apd_conf:
                apd_boxes.append({
                    "name": apd_name,
                    "bbox": xyxy,
                    "conf": conf_score,
                })

    detected_apd_per_person = defaultdict(set)
    for apd_item in apd_boxes:
        apd_name = apd_item["name"]
        apd_box = apd_item["bbox"]
        best_person_id = None
        best_iou = 0.0
        for person_id, person_item in person_boxes.items():
            iou = bbox_iou(apd_box, person_item["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_person_id = person_id
        if best_person_id is not None and best_iou >= APD_PERSON_IOU_THRESHOLD:
            detected_apd_per_person[best_person_id].add(apd_name)

    # =========================================================================
    # FUNGSI REPORT: UPLOAD KE IMGBB LALU LEMPAR JSON KE BACKEND  YAHYAAAAAA;
    # =========================================================================
    def report_violation(tid, vtype, frame_img, current_present_apd, person_bbox):
        key_violation = (tid, vtype)
        now_ts = time.time()
        last_sent_ts = last_violation_notification.get(key_violation)
        
        # Cek Anti Spam Waktu
        if last_sent_ts is not None and (now_ts - last_sent_ts) < TELEGRAM_RENOTIFY_INTERVAL_SEC:
            return

        # Cek Anti Spam Lokasi Tracker ID
        center_x, center_y = bbox_center(person_bbox)
        pruned = []
        suppress_by_location = False
        for ts_prev, px_prev, py_prev in recent_alert_locations[vtype]:
            if (now_ts - ts_prev) <= TELEGRAM_RENOTIFY_INTERVAL_SEC:
                pruned.append((ts_prev, px_prev, py_prev))
                if math.hypot(center_x - px_prev, center_y - py_prev) <= ALERT_SPATIAL_DISTANCE_PX:
                    suppress_by_location = True
        recent_alert_locations[vtype] = pruned
        if suppress_by_location:
            return

        event_time = now_local_str()
        print(f"🚨 Pelanggaran K3: {vtype} (ID {tid}) | {event_time}")

        # 1. Upload foto ke ImgBB
        temp_filename = f"temp_pelanggar_{tid}_{vtype}.jpg"
        cv2.imwrite(temp_filename, frame_img)
        
        image_url = ""
        try:
            with open(temp_filename, "rb") as file:
                payload_imgbb = {"key": IMGBB_API_KEY}
                files = {"image": file}
                res_cloud = requests.post("https://api.imgbb.com/1/upload", params=payload_imgbb, files=files)
            
            if res_cloud.status_code == 200:
                image_url = res_cloud.json()["data"]["url"]
                print(f"✅ Foto ter-upload ke ImgBB: {image_url}")
            else:
                print("⚠️ Gagal upload ke ImgBB!")
        except Exception as e:
            print(f"❌ Error Upload ImgBB: {e}")
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

        # 2. Lempar data (termasuk link ImgBB) ke Backend Temanmu
        if image_url: # Pastikan fotonya berhasil ke-upload dulu
            payload_be = {
                "camera_id": SITE_LOCATION,
                "label": vtype,
                "image_path": image_url, # <-- Link ini yang bakal ditangkap BE
                "id_pekerja": str(tid)
            }

            try:
                res = requests.post(URL_BACKEND, json=payload_be, timeout=5)
                if res.status_code == 200:
                    print(f"✅ JSON berhasil dikirim ke Backend Teman!")
                else:
                    print(f"⚠️ Backend menolak! Status: {res.status_code}")
            except Exception as e:
                print(f"❌ Gagal koneksi ke Backend ({URL_BACKEND}): {e}")
                

        # 3. Update State UI AI
        pelanggar_tercatat.add(key_violation)
        last_violation_notification[key_violation] = now_ts
        recent_alert_locations[vtype].append((now_ts, center_x, center_y))
        recent_violations.insert(0, (now_ts, f"{event_time} | ID {tid} | {vtype}"))
        if len(recent_violations) > 20:
            recent_violations.pop()
    # =========================================================================

    for person_id in person_boxes:
        state = tracked_states.setdefault(person_id, {
            "age": 0,
            "ever": set(),
            "seen_counts": defaultdict(int),
            "missing_counts": defaultdict(int),
        })
        state["age"] += 1
        current_present = detected_apd_per_person.get(person_id, set())

        for apd_name in APD_CLASS_MAP.values():
            if apd_name in current_present:
                state["ever"].add(apd_name)
                state["seen_counts"][apd_name] += 1
                state["missing_counts"][apd_name] = 0
            else:
                state["missing_counts"][apd_name] += 1

        for apd_name in APD_CLASS_MAP.values():
            if (
                state["missing_counts"][apd_name] >= MISSING_FRAMES_THRESHOLD
                and apd_name in state["ever"]
                and state["seen_counts"][apd_name] >= MIN_SEEN_FRAMES_FOR_REMOVE_ALERT
            ):
                report_violation(person_id, f"attempt_remove_{apd_name}", frame, current_present, person_boxes[person_id]["bbox"])

        if state["age"] >= NEVER_WEAR_FRAMES and not any(apd in state["ever"] for apd in APD_CLASS_MAP.values()):
            report_violation(person_id, "not_wearing_any_apd", frame, current_present, person_boxes[person_id]["bbox"])
        else:
            for apd_name in APD_CLASS_MAP.values():
                if state["age"] >= NEVER_WEAR_FRAMES and apd_name not in state["ever"]:
                    report_violation(person_id, f"not_wearing_{apd_name}", frame, current_present, person_boxes[person_id]["bbox"])

    display_img = results[0].plot()

    if DISPLAY_UPSCALE > 1.0:
        up_w = int(display_img.shape[1] * DISPLAY_UPSCALE)
        up_h = int(display_img.shape[0] * DISPLAY_UPSCALE)
        display_img = cv2.resize(display_img, (up_w, up_h), interpolation=cv2.INTER_CUBIC)

    if split_view:
        panel_w = 380
        h = display_img.shape[0]
        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)
        y = 30
        cv2.putText(panel, "Recent Violations", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        for ts, msg in recent_violations[:12]:
            timestr = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            line = f"{timestr} {msg.split('|',1)[1]}"
            cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y += 20
            if y > h - 20:
                break
        try:
            composite = np.hstack((display_img, panel))
        except Exception:
            composite = display_img
        cv2.imshow(WINDOW_NAME, composite)
    else:
        cv2.imshow(WINDOW_NAME, display_img)
        
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('p'):
        monitoring_enabled = not monitoring_enabled
        print("Monitoring paused" if not monitoring_enabled else "Monitoring resumed")
    if key == ord('s'):
        split_view = not split_view
        print("Split view ON" if split_view else "Split view OFF")

cap.release()
cv2.destroyAllWindows()