FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY speech_api.py speech_model.joblib ./
CMD gunicorn -b 0.0.0.0:${PORT:-5001} speech_api:app