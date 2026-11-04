# Panduan Setup Kredensial Kaggle (`beta-kaggle`)

Sistem **Beta Kaggle Auto-Server** memungkinkan aplikasi **AIChat** untuk:
1. Menyalakan server LLM (`llama-server` Dual GPU T4 + `cloudflared`) di Kaggle secara otomatis dari terminal tanpa perlu membuka browser web Kaggle.
2. Mengambil URL publik Cloudflare Tunnel (`https://xxxx.trycloudflare.com`) secara otomatis begitu model selesai dimuat.
3. Mematikan server Kaggle secara otomatis saat kamu menekan `Ctrl + C` di terminal AIChat (atau otomatis mati sendiri setelah **30 menit idle** tanpa aktivitas chat/heartbeat agar kuota GPU mingguan kamu tidak terbuang).

---

## 1. Syarat Wajib Akun Kaggle

Sebelum menggunakan fitur ini, pastikan akun Kaggle kamu sudah memenuhi **2 syarat wajib** dari Kaggle:
1. **Verifikasi Nomor Telepon (Phone Verified)**:
   - Buka [https://www.kaggle.com/settings](https://www.kaggle.com/settings)
   - Pada bagian **Phone Verification**, pastikan statusnya sudah terverifikasi. Tanpa verifikasi nomor HP, Kaggle **menolak** menyalakan GPU T4 dan koneksi Internet di dalam Kernel.
2. **Sisa Kuota GPU Mingguan**:
   - Kaggle memberikan sekitar **30 jam GPU per minggu**. Pastikan kuota GPU mingguan kamu belum habis.

---

## 2. Cara Mendapatkan File `kaggle.json`

1. Login ke akun Kaggle kamu di browser dan buka halaman pengaturan akun:
   👉 **[https://www.kaggle.com/settings](https://www.kaggle.com/settings)**
2. Gulir ke bawah sampai menemukan bagian **API**.
3. Klik tombol **"Create New Token"** (atau **"Generate New Token"**).
   - Browser akan mengunduh file bernama `kaggle.json` (atau menampilkan token API).
4. Salin / pindahkan file `kaggle.json` tersebut ke dalam folder ini:
   ```text
   kaggle-kernel/config/kaggle.json
   ```
5. Pastikan isi file `kaggle-kernel/config/kaggle.json` berformat seperti ini (lihat `kaggle.json.example`):
   ```json
   {
     "username": "nama_user_kaggle_kamu",
     "key": "kunci_api_rahasia_kaggle_kamu"
   }
   ```
   *(Catatan: Jika kamu menggunakan token format baru `KGAT_...` dari `kaggle auth login`, sistem juga mendukung pembacaan otomatis dari `%USERPROFILE%\.kaggle\kaggle.json` atau `~/.kaggle/kaggle.json`).*

> **Keamanan:** File `kaggle-kernel/config/kaggle.json` sudah dimasukkan ke dalam `.gitignore` sehingga token rahasia kamu tidak akan pernah ikut ter-upload ke GitHub.

---

## 3. Prasyarat Paket Python `kaggle` (Opsional / Fallback)

Modul orchestrator Node.js dapat berkomunikasi langsung dengan REST API Kaggle **maupun** menggunakan paket Python `kaggle` (seperti pada project `voicex`). Jika ingin menggunakan CLI Python `kaggle`, pastikan terpasang:

```powershell
pip install kaggle
```

---

## 4. Cara Menjalankan Server (`npm run beta-kaggle`)

Buka terminal di folder utama project **AI Chat**, lalu jalankan:

```powershell
npm run beta-kaggle
```

### Apa yang akan terjadi setelah dijalankan?
1. **`[1/4] Validasi Kredensial`**: Membaca `kaggle-kernel/config/kaggle.json`.
2. **`[2/4] Siapkan Dataset Tunnel Privat`**: Memastikan dataset privat `<username>/aichat-kaggle-tunnel` tersedia di akun Kaggle kamu sebagai saluran pengiriman URL.
3. **`[3/4] Push Kernel ke Kaggle GPU`**: Mengirim script `kaggle-kernel/kernel/main.py` ke Kaggle untuk dijalankan dengan GPU T4 x2 dan Internet menyala.
4. **`[4/4] Menunggu Server & Mengambil URL Cloudflare`**:
   - Mesin Kaggle mengunduh `llama-server` + model GGUF dan menyalakan `cloudflared`.
   - Begitu URL `https://xxxx.trycloudflare.com` siap, mesin Kaggle meng-commit URL tersebut ke dataset privat `<username>/aichat-kaggle-tunnel`.
   - Aplikasi AIChat mendeteksi URL baru tersebut, menyimpannya ke `.env` (`BASE_URL`), dan langsung siap digunakan untuk chatting!

---

## 5. Cara Mematikan Server Kaggle

- **Cara Utama (Langsung Mati Detik Itu Juga)**:
  Cukup tekan **`Ctrl + C`** di terminal tempat `npm run beta-kaggle` berjalan.
  Sebelum aplikasi AIChat tertutup, aplikasi otomatis mengirim sinyal `/__aichat_admin/shutdown` ke server Kaggle. Server Kaggle akan langsung membersihkan RAM/VRAM dan berhenti (`COMPLETE`) sehingga kuota GPU kamu langsung berhenti terpakai.
- **Pengaman Otomatis (Auto-Idle Watchdog 30 Menit)**:
  Jika HP/laptop kamu mati mendadak atau koneksi terputus tanpa sempat menekan `Ctrl + C`, server di Kaggle akan otomatis mematikan dirinya sendiri apabila tidak ada sinyal *heartbeat* dan tidak ada aktivitas chat selama **30 menit**.
