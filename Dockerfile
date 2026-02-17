FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# نصب Nginx برای لود بالانسینگ
RUN apt-get update && apt-get install -y nginx supervisor

# تنظیمات Nginx
COPY nginx.conf /etc/nginx/nginx.conf

# Supervisor برای مدیریت چندین worker
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 80 443 8000

CMD ["/usr/bin/supervisord"]
