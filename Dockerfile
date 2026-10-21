# Works on Hugging Face Spaces (SDK: Docker, app_port 7860) and any container host.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# pre-download the default Whisper model so the first request isn't slow
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"
ENV PORT=7860 HF_HOME=/tmp/hf
EXPOSE 7860
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0"]
