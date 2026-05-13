FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY rl/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The protobuf generation is handled at runtime or via volumes during dev, 
# but for a production container we would generate them here.
COPY . .
WORKDIR /app/rl
RUN bash generate.sh

EXPOSE 50051
CMD ["python", "grpc_stub/server.py"]
