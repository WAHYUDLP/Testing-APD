
# ======================= KONFIGURASI =======================
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
#
# ALERT_SPATIAL_DISTANCE_PX
# - Radius piksel untuk menganggap pelanggaran baru sebagai orang yang sama
#   (berguna saat track ID ganti dari tracker).
#
# MIN_SEEN_FRAMES_FOR_REMOVE_ALERT
# - APD harus terlihat minimal N frame sebelum boleh memicu
#   "attempt_remove_<apd>", agar tidak mudah false alarm.
# ================================================================

