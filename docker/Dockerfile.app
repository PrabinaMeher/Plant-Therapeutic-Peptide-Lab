# ptplab-app — Streamlit container (no heavy bioinformatics tools).
# This container talks to ptplab-haddock / ptplab-gromacs / ptplab-esmfold
# via `docker exec`, so it needs the Docker CLI + access to the host's
# Docker socket (mounted in docker-compose.yml).
FROM python:3.10-slim

LABEL org.opencontainers.image.title="ptplab-app"
LABEL org.opencontainers.image.description="Streamlit frontend for Plant Therapeutic Peptide Lab"

# Docker CLI so this container can `docker exec` into the compute containers
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates gnupg build-essential && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    chmod a+r /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian bookworm stable" \
        > /etc/apt/sources.list.d/docker.list && \
    apt-get update && apt-get install -y --no-install-recommends docker-ce-cli && \
    apt-get purge -y gnupg && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better Docker layer caching —
# this step only re-runs if requirements_api.txt changes, not on every
# code edit)
COPY requirements_app.txt /app/requirements_app.txt
RUN pip install --no-cache-dir -r requirements_app.txt

# Copy the actual application code and everything it depends on
COPY app.py /app/app.py
COPY predictor.py /app/predictor.py
COPY run_predictor.py /app/run_predictor.py
COPY pipeline.py /app/pipeline.py
COPY embeddings.py /app/embeddings.py
COPY utils /app/utils
COPY models /app/models
COPY scalers /app/scalers
COPY receptors /app/receptors
COPY PTPLab_User_Guide.pdf /app/PTPLab_User_Guide.pdf
COPY Photo /app/Photo

ENV PTPLAB_JOBS_DIR=/ptplab_jobs
ENV OLLAMA_URL=http://ollama:11434/api/chat
RUN mkdir -p /ptplab_jobs

EXPOSE 8502

CMD ["streamlit", "run", "app.py", "--server.port=8502", "--server.address=0.0.0.0"]
