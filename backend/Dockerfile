FROM python:3.13-slim
WORKDIR /workspace
COPY backend/requirements.txt /workspace/backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend/app /workspace/backend/app
# Model imports remain available; optional torch is installed separately when serving a checkpoint.
COPY ktbench /workspace/ktbench
WORKDIR /workspace/backend
RUN useradd --create-home learner
USER learner
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
