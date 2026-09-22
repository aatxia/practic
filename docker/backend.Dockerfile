FROM python:3.12-slim

WORKDIR /app

# mediapipe (ml/preprocessing/landmarks.py) dlopen()s a native shared library
# that needs these -- python:3.12-slim ships none of them, so without this
# HandLandmarker.create_from_options() fails at runtime with a bare
# "libEGL.so.1: cannot open shared object file" (Phase 9-10: this only
# started mattering once real inference needed the CV pipeline to actually run
# in the container, not just import cleanly).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libgles2 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend ./backend
COPY ml ./ml
COPY models ./models

WORKDIR /app/backend

EXPOSE 8000
# Shell form (not exec-array) so $PORT expands -- most PaaS hosts (Render,
# Railway, ...) inject PORT at runtime and expect the container to bind to
# it rather than a fixed port; `exec` keeps uvicorn as PID 1 so it still
# receives SIGTERM directly (a plain shell-form CMD without exec would
# leave a shell as PID 1 and swallow the signal on container stop/restart).
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
