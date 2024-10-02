# Gunakan image python resmi
FROM python:3.9-slim

# Set working directory di dalam container
WORKDIR /app

# Copy file requirements dan install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy semua file aplikasi ke container
COPY . .

# Tentukan port aplikasi
ENV PORT=8080

# Jalankan aplikasi
CMD ["python", "app.py"]
