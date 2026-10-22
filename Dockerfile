# Works on Hugging Face Spaces (SDK: Docker, app_port 7860) and any container host.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Keep the model inside the image. HF_HOME=/tmp would be empty on every restart.
ENV HF_HOME=/app/.cache PORT=7860
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"
EXPOSE 7860
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0"]
