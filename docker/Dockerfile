FROM python:3.11-slim AS test

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Minimal build deps; numpy/biopython wheels are available on linux
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install package and test deps
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --upgrade pip \
    && pip install . \
    && pip install pytest \
    && pip cache purge

# Copy tests and minimal fixtures
COPY test /app/test
COPY test_data /app/test_data

## Run tests inside image
#RUN pytest -q


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install only runtime deps and package
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --upgrade pip \
    && pip install . \
    && pip cache purge

# Sensible default; override in deployments
CMD ["alphajudge", "--help"]


