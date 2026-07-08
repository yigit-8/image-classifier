FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# CPU-only torch keeps the image small (default Linux wheels bundle CUDA)
RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Model weights are mounted at runtime (training is too heavy for image build)
RUN mkdir -p data

EXPOSE 8000

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
