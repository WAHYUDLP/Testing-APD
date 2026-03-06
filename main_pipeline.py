import cv2
import requests
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from ultralytics import YOLO
from collections import defaultdict

# --- KONFIGURASI API ---
# Minta ini ke temanmu yang sudah berhasil nyoba Telegram
TELEGRAM_BOT_TOKEN = "TOKEN_DARI_TEMANMU"
TELEGRAM_CHAT_ID = "CHAT_ID_GRUP"

# Daftar gratis di api.imgbb.com untuk dapat kuncinya
IMGBB_API_KEY = "158ee9e068a89b28e5b374a664a8e192" 

# Info lokasi kejadian (isi sesuai lokasi proyek)
SITE_LOCATION = "Area Proyek A"
SITE_LAT = ""
SITE_LON = ""
TIMEZONE_NAME = "Asia/Jakarta"

# Aktifkan kirim Telegram jika token/chat id sudah valid
ENABLE_TELEGRAM = True

# URL Database Dashboard (Misal pakai REST API yang dibuat tim web/SI)
# DATABASE_API_URL = "https://api-proyek-k3.com/pelanggaran" 

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
    # Exact match dulu
    for alias in aliases:
        normalized = _norm_label(alias)
        if normalized in name_to_id:
            return name_to_id[normalized]
    # Lalu contains match sebagai fallback
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


def telegram_is_configured():
    return (
        ENABLE_TELEGRAM
        and TELEGRAM_BOT_TOKEN
        and TELEGRAM_CHAT_ID
        and TELEGRAM_BOT_TOKEN != "TOKEN_DARI_TEMANMU"
        and TELEGRAM_CHAT_ID != "CHAT_ID_GRUP"
    )

# ======================= KAMUS KONFIGURASI =======================
# CONF_THRESHOLD
# - Minimal confidence deteksi dari YOLO (0.0 - 1.0).
# - Lebih kecil: lebih sensitif (objek kecil lebih mungkin terdeteksi), tapi false positive bisa naik.
# - Lebih besar: deteksi lebih ketat, tapi objek kecil/jauh bisa terlewat.
# - Catatan: ini dipakai sebagai batas global model.track, lalu APD difilter lagi per item.
#
# APD_PERSON_IOU_THRESHOLD
# - Minimal IoU agar box APD dianggap milik box person.
# - Karena APD kecil (mask/helmet), threshold dibuat kecil.
#
# INFERENCE_DOWNSCALE
# - Skala resize frame sebelum inferensi YOLO.
# - 1.0 = ukuran asli, 0.75 = 75%, 0.5 = 50%.
# - Lebih kecil: FPS naik, akurasi bisa turun.
#
# MISSING_FRAMES_THRESHOLD
# - Jumlah frame APD hilang berturut-turut (padahal pernah terlihat)
#   sebelum dianggap "attempt_remove_<apd>".
#
# NEVER_WEAR_FRAMES
# - Umur minimal track (dalam frame) sebelum sistem menilai
#   "not_wearing_any_apd" atau "not_wearing_<apd>".
# - Tujuannya menghindari vonis terlalu cepat saat track baru muncul.
#
# TELEGRAM_RENOTIFY_INTERVAL_SEC
# - Interval kirim ulang notifikasi untuk pelanggaran yang sama (anti-spam).
# - Contoh 300 = 5 menit.
# ================================================================

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
INFERENCE_DOWNSCALE = 1.0  # set 0.75 atau 0.5 jika masih patah-patah
FULLSCREEN_VIEW = True

# Ambang frame dibuat kecil supaya respons lebih cepat
MISSING_FRAMES_THRESHOLD = 10
NEVER_WEAR_FRAMES = 5
TELEGRAM_RENOTIFY_INTERVAL_SEC = 300

# State per track: umur, pernah_terlihat(set), hitungan_missing per apd
tracked_states = {}
last_violation_notification = {}


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


def buka_kamera(index_opsi=(0, 1, 2)):
    """Coba beberapa index kamera, lalu pilih yang benar-benar bisa baca frame."""
    for idx in index_opsi:
        cap_uji = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
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
    print("ERROR: Kamera tidak terdeteksi, sedang dipakai aplikasi lain, atau index kamera tidak cocok.")
    raise SystemExit(1)

# Minta kamera kirim resolusi lebih kecil agar proses lebih ringan
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"INFO: resolusi kamera aktif = {actual_w}x{actual_h}")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
if FULLSCREEN_VIEW:
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
pelanggar_tercatat = set()

# Runtime toggles
monitoring_enabled = True
notifications_enabled = telegram_is_configured()

print("Kontrol runtime: 'p' pause/resume, 't' toggle Telegram, 'q' quit")

print("=== SISTEM MONITORING K3 AKTIF ===")
print(
    "INFO: Logika pelanggaran berbasis kelas person."
    " APD akan di-associate ke person via overlap bounding box (IoU)."
)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("ERROR: Gagal membaca frame dari kamera.")
        break

    # Optional downscale sebelum inferensi untuk menurunkan beban komputasi
    if 0 < INFERENCE_DOWNSCALE < 1.0:
        inf_w = max(1, int(frame.shape[1] * INFERENCE_DOWNSCALE))
        inf_h = max(1, int(frame.shape[0] * INFERENCE_DOWNSCALE))
        infer_frame = cv2.resize(frame, (inf_w, inf_h), interpolation=cv2.INTER_AREA)
    else:
        infer_frame = frame

    # Jika monitoring dipause, lewati deteksi tapi tetap tampilkan frame dan tangani tombol
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
        if key == ord('t'):
            notifications_enabled = not notifications_enabled
            print("Notifications ON" if notifications_enabled else "Notifications OFF")
        continue

    # 1. DETEKSI: Jalankan YOLO11s dengan mode tracking
    results = model.track(infer_frame, persist=True, tracker="bytetrack.yaml", conf=CONF_THRESHOLD, verbose=False)
    # Kumpulkan person dan APD dari frame saat ini
    person_boxes = {}
    apd_boxes = []

    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf_score = float(box.conf[0]) if box.conf is not None else 0.0
        xyxy = tuple(float(v) for v in box.xyxy[0].tolist())

        if cls == PERSON_CLASS_ID and conf_score >= PERSON_CONF_THRESHOLD and box.id is not None:
            person_track_id = int(box.id[0])
            person_boxes[person_track_id] = xyxy

        if cls in APD_CLASS_MAP:
            apd_name = APD_CLASS_MAP[cls]
            min_apd_conf = APD_CONF_THRESHOLD.get(apd_name, CONF_THRESHOLD)
            if conf_score >= min_apd_conf:
                apd_boxes.append((apd_name, xyxy))

    # Associate APD -> person berdasarkan IoU terbesar
    detected_apd_per_person = defaultdict(set)
    for apd_name, apd_box in apd_boxes:
        best_person_id = None
        best_iou = 0.0
        for person_id, person_box in person_boxes.items():
            iou = bbox_iou(apd_box, person_box)
            if iou > best_iou:
                best_iou = iou
                best_person_id = person_id
        if best_person_id is not None and best_iou >= APD_PERSON_IOU_THRESHOLD:
            detected_apd_per_person[best_person_id].add(apd_name)

    # Helper: laporkan pelanggaran (simpan, upload, catat)
    def report_violation(tid, vtype, frame_img, current_present_apd):
        key_violation = (tid, vtype)
        now_ts = time.time()
        last_sent_ts = last_violation_notification.get(key_violation)
        if last_sent_ts is not None and (now_ts - last_sent_ts) < TELEGRAM_RENOTIFY_INTERVAL_SEC:
            return

        event_time = now_local_str()
        event_location = format_location_text()
        print(f"🚨 Pelanggaran K3: {vtype} (ID {tid}) | {event_time} | {event_location}")
        apd_summary = ", ".join(
            f"{apd}: {'yes' if apd in current_present_apd else 'no'}"
            for apd in sorted(set(APD_CLASS_MAP.values()))
        )
        temp_filename = f"temp_pelanggar_{tid}_{vtype}.jpg"
        cv2.imwrite(temp_filename, frame_img)
        try:
            with open(temp_filename, "rb") as file:
                payload = {"key": IMGBB_API_KEY}
                files = {"image": file}
                res_cloud = requests.post("https://api.imgbb.com/1/upload", params=payload, files=files)
            if res_cloud.status_code == 200:
                image_url = res_cloud.json()["data"]["url"]
                print(f"✅ Foto berhasil diupload ke Cloud: {image_url}")
                data_db = {
                    "waktu": event_time,
                    "pekerja_id": tid,
                    "jenis_pelanggaran": vtype,
                    "lokasi": event_location,
                    "foto_url": image_url,
                }
                # requests.post(DATABASE_API_URL, json=data_db)

                if notifications_enabled:
                    pesan_tele = (
                        "⚠️ PELANGGARAN K3!\n"
                        f"Waktu: {event_time}\n"
                        f"ID Pekerja: {tid}\n"
                        f"Jenis: {vtype}\n"
                        f"Status APD saat ini: {apd_summary}\n"
                        f"Notifikasi ulang: tiap {TELEGRAM_RENOTIFY_INTERVAL_SEC // 60} menit jika masih melanggar\n"
                        f"Lokasi: {event_location}\n"
                        f"Bukti: {image_url}"
                    )
                    url_tele = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                    try:
                        requests.post(url_tele, data={"chat_id": TELEGRAM_CHAT_ID, "text": pesan_tele}, timeout=8)
                    except Exception:
                        pass

                pelanggar_tercatat.add(key_violation)
                last_violation_notification[key_violation] = now_ts
        except Exception as e:
            print(f"Terjadi kesalahan saat upload/kirim: {e}")
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    # Update state hanya untuk person yang benar-benar terlihat pada frame ini
    for person_id in person_boxes:
        state = tracked_states.setdefault(person_id, {
            "age": 0,
            "ever": set(),
            "missing_counts": defaultdict(int),
        })
        state["age"] += 1
        current_present = detected_apd_per_person.get(person_id, set())

        # Update missing counts / ever-seen
        for apd_name in APD_CLASS_MAP.values():
            if apd_name in current_present:
                state["ever"].add(apd_name)
                state["missing_counts"][apd_name] = 0
            else:
                state["missing_counts"][apd_name] += 1

        # Deteksi: mencoba melepas (pernah kelihatan, lalu hilang beberapa frame)
        for apd_name in APD_CLASS_MAP.values():
            if state["missing_counts"][apd_name] >= MISSING_FRAMES_THRESHOLD and apd_name in state["ever"]:
                report_violation(person_id, f"attempt_remove_{apd_name}", frame, current_present)

        # Deteksi: tidak mengenakan semua APD (helm, rompi, masker)
        if state["age"] >= NEVER_WEAR_FRAMES and not any(apd in state["ever"] for apd in APD_CLASS_MAP.values()):
            report_violation(person_id, "not_wearing_any_apd", frame, current_present)
        else:
            # Deteksi per-APD jika hanya sebagian yang tidak pernah terlihat
            for apd_name in APD_CLASS_MAP.values():
                if state["age"] >= NEVER_WEAR_FRAMES and apd_name not in state["ever"]:
                    report_violation(person_id, f"not_wearing_{apd_name}", frame, current_present)

    # Tampilkan hasil secara visual di layar laptop
    cv2.imshow(WINDOW_NAME, results[0].plot())
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('p'):
        monitoring_enabled = not monitoring_enabled
        print("Monitoring paused" if not monitoring_enabled else "Monitoring resumed")
    if key == ord('t'):
        notifications_enabled = not notifications_enabled
        print("Notifications ON" if notifications_enabled else "Notifications OFF")

cap.release()
cv2.destroyAllWindows()