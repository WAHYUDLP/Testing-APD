# Monitoring APD + Upload ImgBB + Kirim ke Backend

## Ringkasan
Project ini melakukan monitoring APD (helm, masker, vest) dari kamera real-time menggunakan YOLO, lalu:
1. Mendeteksi pelanggaran per pekerja (berbasis tracking ID).
2. Mengambil frame bukti pelanggaran.
3. Upload bukti ke ImgBB.
4. Mengirim data pelanggaran + URL gambar ke backend.

File utama runtime ada di `mainWithLinkImgb.py`.

## Alur Data End-to-End
1. Kamera dibuka dan dibaca frame-nya.
2. Model YOLO melakukan deteksi + tracking (ByteTrack).
3. Sistem menentukan APD apa saja yang dipakai tiap person.
4. Jika terdeteksi pelanggaran, frame disimpan sementara lalu di-upload ke ImgBB.
5. Setelah dapat URL gambar dari ImgBB, sistem POST JSON ke endpoint backend.
6. Backend menerima payload, lalu bisa disimpan ke DB, diteruskan ke dashboard, notifikasi, dll.

Skema singkat:

Kamera -> YOLO/Tracking -> Rule Pelanggaran -> Upload ImgBB -> POST ke Backend -> DB/Dashboard

## Fitur Utama
- Auto map class dari model names, dengan fallback ID.
- Tracking person per ID (persist antar frame).
- Rule pelanggaran:
  - `not_wearing_any_apd`
  - `not_wearing_helmet`
  - `not_wearing_vest`
  - `not_wearing_mask`
  - `attempt_remove_helmet`
  - `attempt_remove_vest`
  - `attempt_remove_mask`
- Anti spam notifikasi:
  - Interval waktu
  - Filter jarak spasial
- Overlay tampilan deteksi + mode split view riwayat pelanggaran.

## Payload ke Backend
Script mengirim payload JSON berikut:

```json
{
  "camera_id": "Area Proyek A",
  "label": "not_wearing_any_apd",
  "image_path": "https://i.ibb.co/...jpg",
  "id_pekerja": "1"
}
```

Keterangan field:
- `camera_id`: lokasi kamera/sumber kejadian.
- `label`: jenis pelanggaran.
- `image_path`: URL hasil upload ImgBB.
- `id_pekerja`: tracking ID pekerja (string).

## Konfigurasi Penting di Script
Di `mainWithLinkImgb.py`:
- `IMGBB_API_KEY`: API key ImgBB.
- `SITE_LOCATION`, `SITE_LAT`, `SITE_LON`, `TIMEZONE_NAME`: metadata lokasi.
- `URL_BACKEND`: endpoint backend tujuan.
- Threshold deteksi:
  - `CONF_THRESHOLD`
  - `PERSON_CONF_THRESHOLD`
  - `APD_CONF_THRESHOLD`
- Rule timing/anti spam:
  - `MISSING_FRAMES_THRESHOLD`
  - `NEVER_WEAR_FRAMES`
  - `TELEGRAM_RENOTIFY_INTERVAL_SEC`
  - `ALERT_SPATIAL_DISTANCE_PX`

## Requirement
- Python 3.10+ (disarankan)
- Webcam
- Model file `best.pt`
- Internet (untuk upload ImgBB)

Python package:
- ultralytics
- opencv-python
- requests
- numpy

## Setup dan Run
1. Buka terminal di folder project.
2. Buat virtual env.
3. Aktifkan virtual env.
4. Install dependency.
5. Jalankan script.

Contoh PowerShell (Windows):

```powershell
cd "C:\Users\ASUS\Downloads\Testing APD"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install ultralytics opencv-python requests numpy
python mainWithLinkImgb.py
```

Kontrol saat runtime:
- `p`: pause/resume
- `s`: split view ON/OFF
- `q`: quit

## Cara Nyambung ke Backend
### 1) Backend di laptop yang sama
Pakai:
- `URL_BACKEND = "http://localhost:8000/report-violation"`

Syarat:
- Service backend harus aktif.
- Service listen di port 8000.
- Route `POST /report-violation` tersedia.

### 2) Backend di laptop berbeda, satu WiFi/LAN
Pakai IP laptop backend, misalnya:
- `URL_BACKEND = "http://192.168.1.10:8000/report-violation"`

Syarat di laptop backend:
- Server bind ke `0.0.0.0`, bukan `127.0.0.1`.
- Firewall mengizinkan inbound port 8000.

### 3) Backend beda WiFi/jaringan internet
`localhost` dan IP lokal tidak bisa langsung dipakai.
Pilih salah satu:
- Deploy backend ke server publik.
- Gunakan tunnel (ngrok/Cloudflare Tunnel).
- Gunakan VPN mesh (Tailscale/ZeroTier), lalu pakai IP VPN.

## Cek Konektivitas
Dari laptop pengirim:

```powershell
Test-NetConnection -ComputerName localhost -Port 8000
```

Atau jika backend beda laptop:

```powershell
Test-NetConnection -ComputerName 192.168.1.10 -Port 8000
```

Jika `TcpTestSucceeded : False`, berarti endpoint belum bisa diakses dari jaringan saat ini.

## Contoh Backend Minimal (FastAPI)
Gunakan ini jika ingin validasi integrasi cepat:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ViolationPayload(BaseModel):
    camera_id: str
    label: str
    image_path: str
    id_pekerja: str

@app.post("/report-violation")
def report_violation(payload: ViolationPayload):
    print("Incoming:", payload.model_dump())
    return {"status": "ok", "message": "data diterima"}
```

Jalankan:

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Troubleshooting
### Error koneksi backend ditolak (WinError 10061)
Penyebab umum:
- Backend belum jalan.
- Port backend bukan 8000.
- Server hanya bind ke localhost padahal diakses dari laptop lain.
- Firewall blok port.

### Warning kamera OpenCV
Sudah ditangani dengan backend kamera DirectShow di Windows. Jika masih muncul:
- Tutup aplikasi lain yang pakai webcam.
- Coba ganti urutan index kamera.

### Upload ImgBB gagal
- API key invalid/expired.
- Koneksi internet bermasalah.
- Rate limit ImgBB.

## Catatan Keamanan
- Jangan commit API key ke repo publik.
- Disarankan pindahkan `IMGBB_API_KEY` ke environment variable.

Contoh:

```powershell
$env:IMGBB_API_KEY="your_key_here"
```

Lalu di script, ambil dari `os.getenv("IMGBB_API_KEY")`.

## Checklist Integrasi
- `best.pt` tersedia.
- Kamera terbaca.
- ImgBB upload sukses.
- Backend aktif dan bisa diakses.
- Endpoint `POST /report-violation` cocok dengan payload.
- Backend merespons status 200 untuk request valid.
