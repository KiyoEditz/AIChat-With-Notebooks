# AICHAT DENGAN COLAB/KAGGLE - AICHAT WITH COLAB/KAGGLE

- Aplikasi obrolan roleplay yang terhubung ke server LLM mandiri (llama.cpp / `llama-server`) yang kamu jalankan sendiri di Google Colab atau Kaggle, lalu diekspos publik lewat Cloudflare Tunnel.
--------
- A roleplay chat application that connects to a self-hosted LLM server (llama.cpp / `llama-server`) you run yourself on Google Colab or Kaggle, exposed publicly via Cloudflare Tunnel.

## DESKRIPSI SINGKAT - BRIEF DESCRIPTION

- Proyek ini memungkinkan Anda menjalankan server LLM mandiri di GPU gratis Colab/Kaggle dan menghubungkannya ke klien obrolan lokal ini. Jalur aktif proyek: **AIChat → Cloudflare Tunnel → `llama-server` (llama.cpp) → model GGUF** — tanpa lapisan proxy tambahan. Salin skrip `Collab-Llama.py` (Colab) atau `Kaggle-Llama.py` (Kaggle) ke sel notebook, jalankan, lalu tempel `BASE_URL`/`API_KEY` yang dicetak ke `.env` lokal Anda.

  Stack lama (Ollama + LiteLLM, via `colab_llm_server.py`/`kaggle_llm_server.py`) masih ada di repo ini sebagai alternatif opsional — lihat bagian "Pilihan Stack Server LLM" di bawah — tapi bukan lagi jalur default proyek ini.
--------
- This project lets you run a self-hosted LLM server on Colab/Kaggle's free GPU and connect it to this local chat client. This project's active path: **AIChat → Cloudflare Tunnel → `llama-server` (llama.cpp) → GGUF model** — no extra proxy layer. Copy `Collab-Llama.py` (Colab) or `Kaggle-Llama.py` (Kaggle) into a notebook cell, run it, then paste the printed `BASE_URL`/`API_KEY` into your local `.env`.

  The older stack (Ollama + LiteLLM, via `colab_llm_server.py`/`kaggle_llm_server.py`) is still in this repo as an optional alternative — see "LLM Server Stack Options" below — but it is no longer this project's default path.

## SPESIFIKASI MINIMUM - MINIMUM SPECIFICATIONS

- **RAM:** 12 GB (Bawaan standar tier gratis Google Colab)
- **VRAM:** 16 GB (Tersedia pada runtime GPU gratis Google Colab)
--------
- **RAM:** 12 GB (Standard default on Google Colab free tier)
- **VRAM:** 16 GB (Available on Google Colab free GPU runtime)

## PERSYARATAN - REQUIREMENTS

- Akun Google (untuk Colab) atau akun Kaggle, untuk mengakses GPU gratis.
- Koneksi internet yang stabil.
- [Node.js](https://nodejs.org/) terinstal pada perangkat lokal Anda.
--------
- A Google account (for Colab) or a Kaggle account, to access free GPU compute.
- A stable internet connection.
- [Node.js](https://nodejs.org/) installed on your local machine.

## PILIHAN STACK SERVER LLM - LLM SERVER STACK OPTIONS

Ada dua pilihan stack untuk menjalankan server LLM di Colab/Kaggle. Keduanya menghasilkan `BASE_URL` + `API_KEY` yang bisa langsung ditempel ke `.env` — pilih salah satu, tidak perlu keduanya.
--------
There are two stack options for running the LLM server on Colab/Kaggle. Both produce a `BASE_URL` + `API_KEY` you paste straight into `.env` — pick one, you don't need both.

| | `colab_llm_server.py` / `kaggle_llm_server.py` (Ollama + LiteLLM) | `Collab-Llama.py` / `Kaggle-Llama.py` (llama.cpp langsung) |
|---|---|---|
| Setup | Lebih cepat (binary Ollama siap pakai) — *faster (prebuilt Ollama binary)* | Perlu compile dari source pertama kali, ~5-10 menit ekstra — *needs a source build first run, ~5-10 min extra* |
| Kontrol parameter sampling | Lewat 2 lapis translasi (LiteLLM → Ollama), bisa ambigu — *through 2 translation layers, can be ambiguous* | Native, langsung ke model, tanpa translasi — *native, direct to the model, no translation* |
| Kontrol repetisi teks panjang | `frequency_penalty`/`presence_penalty` gaya OpenAI, terakumulasi di seluruh riwayat chat — bisa menekan kata umum ("the", "it", "is") makin lama makin sering di chat panjang — *OpenAI-style, accumulates over the whole chat history — can suppress common words the longer a chat gets* | Punya **DRY sampling** — menghukum pengulangan frasa/pola, bukan kata umum individual; lebih cocok untuk roleplay panjang — *has **DRY sampling** — penalizes repeated phrases/patterns, not individual common words; better suited for long roleplay* |
| Jumlah proses | 2 (Ollama + LiteLLM) | 1 (llama-server saja) — *1 (llama-server only)* |
| Rekomendasi | Kalau ingin setup tercepat — *if you want the fastest setup* | Kalau chat roleplay-mu panjang dan mulai terasa repetitif — *if your roleplay chats run long and start feeling repetitive* |

Cloudflare Tunnel dipakai di kedua stack dengan cara yang sama persis — tinggal tunnel port HTTP lokal, tidak ada perbedaan cara kerja.
--------
Cloudflare Tunnel is used identically in both stacks — it just tunnels a local HTTP port either way, no difference in how it works.

## PANDUAN INSTALASI: SERVER LLM (GOOGLE COLAB, OLLAMA + LITELLM) - INSTALLATION GUIDE: LLM SERVER (GOOGLE COLAB, OLLAMA + LITELLM)

1. **Persiapan Skrip:** Unggah berkas `colab_llm_server.ipynb` ke [Google Colab](https://colab.research.google.com/), **atau** salin seluruh isi skrip dari `colab_llm_server.py` ke dalam sel notebook baru di environment Colab Anda.
2. **Konfigurasi Runtime:** Pada bilah menu Google Colab, navigasikan ke **Runtime** > **Ubah jenis runtime** (Change runtime type) dan pastikan akselerator perangkat keras disetel ke **GPU**.
3. **Eksekusi:** Jalankan (Run) sel yang berisi skrip tersebut untuk memulai proses instalasi dependensi dan menginisialisasi server. Tunggu hingga proses ini selesai sepenuhnya.
4. **Pengambilan Kredensial:** Setelah server berhasil berjalan, sistem akan mencetak nilai `API_KEY` dan `BASE_URL` pada output terminal/sel.
5. **Konfigurasi Lokal:** Buka berkas `.env.example` yang terdapat pada direktori proyek lokal Anda. Perbarui variabel environment yang relevan dengan menempelkan nilai `API_KEY` dan `BASE_URL` yang telah didapatkan. Terakhir, ubah nama berkas `.env.example` menjadi `.env`.
--------
1. **Script Preparation:** Upload the `colab_llm_server.ipynb` file to [Google Colab](https://colab.research.google.com/), **or** copy the entire script contents from `colab_llm_server.py` into a new notebook cell in your Colab environment.
2. **Runtime Configuration:** In the Google Colab menu bar, navigate to **Runtime** > **Change runtime type** and ensure the hardware accelerator is set to **GPU**.
3. **Execution:** Run the cell containing the script to initiate the dependency installation and server initialization process. Wait for this process to complete entirely.
4. **Credential Retrieval:** Once the server is successfully running, the system will output the `API_KEY` and `BASE_URL` values in the cell/terminal output.
5. **Local Configuration:** Open the `.env.example` file located in your local project directory. Update the relevant environment variables by pasting the acquired `API_KEY` and `BASE_URL` values. Finally, rename the `.env.example` file to `.env`.

## PANDUAN INSTALASI: SERVER LLM (KAGGLE, OLLAMA + LITELLM, DUAL T4) - INSTALLATION GUIDE: LLM SERVER (KAGGLE, OLLAMA + LITELLM, DUAL T4)

1. **Persiapan Skrip:** Buat notebook baru di [Kaggle](https://www.kaggle.com/code) dan salin seluruh isi skrip dari `kaggle_llm_server.py` ke dalam sel notebook.
2. **Konfigurasi Akselerator:** Di panel **Notebook options** (sisi kanan), atur **Accelerator** ke **GPU T4 x2**, dan pastikan **Internet** dalam posisi **On**.
3. **Eksekusi:** Jalankan sel tersebut. Model 32B yang dipakai butuh kedua T4 untuk muat di VRAM — proses pull model + load pertama kali cukup lama.
4. **Pengambilan Kredensial & Konfigurasi Lokal:** Sama seperti panduan Colab di atas — salin `BASE_URL`/`API_KEY` dari output ke `.env`.
5. **Pembersihan Otomatis:** Skrip ini punya cleanup otomatis (RAM/VRAM) saat sel dihentikan atau kernel di-kill dari UI Kaggle — tidak perlu langkah manual tambahan.
--------
1. **Script Preparation:** Create a new notebook on [Kaggle](https://www.kaggle.com/code) and copy the entire script contents from `kaggle_llm_server.py` into a notebook cell.
2. **Accelerator Configuration:** In the **Notebook options** panel (right side), set **Accelerator** to **GPU T4 x2**, and make sure **Internet** is **On**.
3. **Execution:** Run the cell. The 32B model used needs both T4s to fit in VRAM — the first pull + load takes a while.
4. **Credential Retrieval & Local Configuration:** Same as the Colab guide above — copy `BASE_URL`/`API_KEY` from the output into `.env`.
5. **Automatic Cleanup:** This script has automatic RAM/VRAM cleanup when the cell is stopped or the kernel is killed from the Kaggle UI — no extra manual step needed.

## PANDUAN INSTALASI: SERVER LLM (COLAB, LLAMA.CPP LANGSUNG) - INSTALLATION GUIDE: LLM SERVER (COLAB, LLAMA.CPP DIRECT)

1. **Persiapan Skrip:** Salin seluruh isi skrip dari `Collab-Llama.py` ke dalam sel notebook baru di Google Colab.
2. **Konfigurasi Runtime:** **Runtime** > **Change runtime type** > **GPU (T4)**.
3. **Eksekusi:** Jalankan sel. Pada run pertama, skrip akan **meng-compile llama.cpp dari source** (~5-10 menit) sebelum mengunduh model — run berikutnya melewati langkah compile ini kalau kernel tidak di-restart.
4. **Pengambilan Kredensial & Konfigurasi Lokal:** Sama seperti panduan lain — salin `BASE_URL`/`API_KEY` dari output ke `.env`.
--------
1. **Script Preparation:** Copy the entire script contents from `Collab-Llama.py` into a new notebook cell in Google Colab.
2. **Runtime Configuration:** **Runtime** > **Change runtime type** > **GPU (T4)**.
3. **Execution:** Run the cell. On the first run, the script **compiles llama.cpp from source** (~5-10 min) before downloading the model — subsequent runs skip the compile step as long as the kernel isn't restarted.
4. **Credential Retrieval & Local Configuration:** Same as the other guides — copy `BASE_URL`/`API_KEY` from the output into `.env`.

## PANDUAN INSTALASI: SERVER LLM (KAGGLE, LLAMA.CPP LANGSUNG, DUAL T4) - INSTALLATION GUIDE: LLM SERVER (KAGGLE, LLAMA.CPP DIRECT, DUAL T4)

1. **Persiapan Skrip:** Buat notebook baru di [Kaggle](https://www.kaggle.com/code) dan salin seluruh isi skrip dari `Kaggle-Llama.py`.
2. **Konfigurasi Akselerator:** **Accelerator** → **GPU T4 x2**, **Internet** → **On**.
3. **Eksekusi:** Jalankan sel. Model default (32B, Q6_K) butuh kedua T4 — skrip otomatis memverifikasi keduanya benar-benar terpakai setelah model dimuat.
4. **Pengambilan Kredensial & Konfigurasi Lokal:** Sama seperti panduan lain — salin `BASE_URL`/`API_KEY` dari output ke `.env`.
5. **Pembersihan Otomatis:** Cleanup RAM/VRAM otomatis saat sel dihentikan, sama seperti `kaggle_llm_server.py`.
--------
1. **Script Preparation:** Create a new notebook on [Kaggle](https://www.kaggle.com/code) and copy the entire script contents from `Kaggle-Llama.py`.
2. **Accelerator Configuration:** **Accelerator** → **GPU T4 x2**, **Internet** → **On**.
3. **Execution:** Run the cell. The default model (32B, Q6_K) needs both T4s — the script automatically verifies both are actually being used after the model loads.
4. **Credential Retrieval & Local Configuration:** Same as the other guides — copy `BASE_URL`/`API_KEY` from the output into `.env`.
5. **Automatic Cleanup:** Automatic RAM/VRAM cleanup when the cell is stopped, same as `kaggle_llm_server.py`.

## PANDUAN INSTALASI: SERVER WEB LOKAL - INSTALLATION GUIDE: LOCAL WEB SERVER

1. **Prasyarat:** Pastikan lingkungan sistem Anda telah memiliki [Node.js](https://nodejs.org/).
2. **Instalasi Dependensi:** Buka terminal atau command prompt dan arahkan ke direktori root proyek ini. Jalankan perintah `npm install` untuk mengunduh seluruh pustaka yang diperlukan.
3. **Menjalankan Server:** Eksekusi perintah `npm start` untuk menghidupkan server web lokal.
4. **Akses Aplikasi (Lokal):** Buka peramban web (browser) favorit Anda dan kunjungi `http://localhost:3000`.
5. **Akses Aplikasi (Jaringan WiFi Sama):** Aplikasi ini dapat diakses oleh perangkat apa pun (seperti ponsel pintar atau tablet) yang terhubung ke jaringan WiFi yang sama dengan komputer host. Cukup masukkan alamat IP lokal dari komputer host diikuti dengan port server pada peramban perangkat tersebut (contoh: `http://192.168.1.15:3000`).
--------
1. **Prerequisites:** Ensure your system environment has [Node.js](https://nodejs.org/) installed.
2. **Dependency Installation:** Open a terminal or command prompt and navigate to this project's root directory. Run the `npm install` command to download all required libraries.
3. **Starting the Server:** Execute the `npm start` command to boot up the local web server.
4. **Accessing the Application (Local):** Open your preferred web browser and navigate to `http://localhost:3000`.
5. **Accessing the Application (Same WiFi Network):** This application can be accessed by any device (such as a smartphone or tablet) connected to the same WiFi network as the host computer. Simply enter the local IP address of the host computer followed by the server port into the device's browser (e.g., `http://192.168.1.15:3000`).

## DUKUNGAN & DONASI - SUPPORT & DONATION

Jika Anda menyukai proyek ini dan ingin mendukung pengembangannya, Anda dapat memberikan apresiasi melalui platform donasi berikut:

- ☕ **Ko-fi:** [ko-fi.com/kiyoeditz](https://ko-fi.com/kiyoeditz)
- 💛 **Saweria:** [saweria.co/KiyoEditz](https://saweria.co/KiyoEditz)
- 💖 **GitHub Sponsors:** *(Masih dalam peninjauan / Under review)*

Dukungan Anda sangat berarti untuk membantu pemeliharaan dan penambahan fitur-fitur baru ke depannya! Terima kasih banyak!
--------
If you enjoy this project and want to support its ongoing development, feel free to show some love through the following donation channels:

- ☕ **Ko-fi:** [ko-fi.com/kiyoeditz](https://ko-fi.com/kiyoeditz)
- 💛 **Saweria:** [saweria.co/KiyoEditz](https://saweria.co/KiyoEditz)
- 💖 **GitHub Sponsors:** *(Currently under review)*

Your support means a lot and helps sustain maintenance and future feature improvements! Thank you so much!

## CHANGELOG

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of changes.

## LISENSI - LICENSE

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.
This means you are free to use it for individual or non-commercial purposes, but commercial/corporate use requires a separate arrangement.

See the [LICENSE](LICENSE) file for the full text.
