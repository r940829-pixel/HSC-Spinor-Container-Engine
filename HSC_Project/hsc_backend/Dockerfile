
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_PORT=5000


RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    numpy \
    scipy \
    cupy-cuda11x \
    redis \
    fastapi \
    pydantic \
    uvicorn


WORKDIR /app

COPY hsc_qubit.py /app/

# Expose the designated local runtime communication port
EXPOSE 5000


CMD ["uvicorn", "hsc_qubit:app", "--host", "0.0.0.0", "--port", "5000"]
