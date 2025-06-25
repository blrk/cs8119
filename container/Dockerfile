FROM  python:3.12.11-bullseye
WORKDIR /app
COPY container/requirements.txt .
RUN apt-get update && \
    apt-get install -y --no-install-recommends
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
COPY ../app.py .

EXPOSE 5000
CMD ["python", "app.py"]