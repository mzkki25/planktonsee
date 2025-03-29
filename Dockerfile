# Gunakan image Python yang kompatibel
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VENV_PATH="/opt/venv"

# Install dependencies sistem yang diperlukan oleh OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Buat virtual environment
RUN python -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Buat direktori kerja
WORKDIR /app

# Salin semua file ke dalam container
COPY . /app/

# Install dependencies Python dari requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose port 8080 untuk Railway
EXPOSE 8080

# Perintah untuk menjalankan Flask dengan Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app"]
