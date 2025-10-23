# 1. TEMEL AŞAMA: Hangi işletim sistemini/programı baz alıyoruz?
# Python 3.9'un "slim" (minimum) versiyonunu alıyoruz.
FROM python:3.9-slim


#Tüm kodlarımızı /app klasörüne koyacağız.
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# .dockerignore'da olmayan her şeyi (.) konteynerin içine (/app'e) kopyala.
COPY . .

RUN dvc pull

RUN dvc pull

# 6. ÇALIŞTIRMA AŞAMASI: Bu "kutu" başlatıldığında ne yapsın?
# Bizim eğitim script'imizi çalıştırsın.
CMD ["python", "train.py"]