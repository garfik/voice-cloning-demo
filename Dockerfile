FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg curl espeak supervisor build-essential cmake \
  && rm -rf /var/lib/apt/lists/*

ENV HF_HOME=/cache \
    TTS_HOME=/cache \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PIP_PREFER_BINARY=1 \
    COQUI_TOS_AGREED=1 \
    ENABLED_WORKERS=coqui,neutts

WORKDIR /app

# Create three separate virtual environments
RUN python -m venv /venv-app && \
    python -m venv /venv-coqui && \
    python -m venv /venv-neutts

# Create required directories
RUN mkdir -p /tmp/tts_queue/coqui /tmp/tts_queue/neutts /tmp/tts_out /tmp/tts_done /tmp/tts_in

# Install base dependencies
RUN /venv-app/bin/pip install --upgrade pip wheel && \
    /venv-coqui/bin/pip install --upgrade pip wheel && \
    /venv-neutts/bin/pip install --upgrade pip wheel

# Install requirements for each environment
COPY requirements-gateway.txt .
RUN /venv-app/bin/pip install --no-cache-dir -r requirements-gateway.txt

# Install Coqui requirements with CPU-specific PyTorch
COPY requirements-coqui.txt .
RUN /venv-coqui/bin/pip install --no-cache-dir "setuptools<81" && \
    /venv-coqui/bin/pip install --no-cache-dir torch==2.2.2 torchaudio==2.2.2 --extra-index-url=https://download.pytorch.org/whl/cpu && \
    /venv-coqui/bin/pip install --no-cache-dir -r requirements-coqui.txt

# Install NeuTTS requirements
COPY requirements-neutts.txt .
RUN /venv-neutts/bin/pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 --extra-index-url=https://download.pytorch.org/whl/cpu && \
    /venv-neutts/bin/pip install --no-cache-dir llama-cpp-python && \
    /venv-neutts/bin/pip install --no-cache-dir -r requirements-neutts.txt

# Copy application code
COPY app/ .

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 8000

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
