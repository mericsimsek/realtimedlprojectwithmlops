# 1. TEMEL AŞAMA
# "trixie" yerine "bookworm" (Debian 12) kullanarak
# paket isimlerinin stabil olmasını sağlıyoruz.
FROM python:3.10-slim-bookworm

# 2. ORTAM AŞAMASI
WORKDIR /app

# 3. BAĞIMLILIK AŞAMASI

# 3.1: SİSTEM ARAÇLARINI KUR
# pip'in numpy, pandas, opencv gibi paketleri derleyebilmesi için
# gerekli olan temel Linux araçlarını yüklüyoruz.
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    # OpenCV'nin çalışması için gereken grafik kütüphaneleri
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# 3.2: PYTHON KÜTÜPHANELERİNİ KUR
# Artık sistem araçları kurulu olduğuna göre pip install çalışacaktır.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. KOD AŞAMASI
COPY . .

# 5. VERİ AŞAMASI
# Dışarıdan DagsHub token'ını al
ARG DAGSHUB_TOKEN

# DVC'ye şifre sormamasını, verdiğimiz token'ı kullanmasını söyle
RUN dvc remote modify origin --local ask_password false
RUN dvc remote modify origin --local password ${DAGSHUB_TOKEN}

# DVC'ye ".git klasörü olmadan" çalışmasını söyle (Global ayar)
#
RUN dvc config core.no_scm true

# Veriyi çek
RUN dvc pull

# 6. ÇALIŞTIRMA AŞAMASI
CMD ["python", "train.py"]