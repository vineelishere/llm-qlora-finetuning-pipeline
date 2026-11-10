
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 


RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app


COPY requirements.txt . 
RUN pip3 install --no-cache-dir -r requirements.txt


COPY . .


ENV MODEL_TYPE=llm
ENV MODEL_PATH=outputs/merged_model
ENV LOAD_IN_4BIT=true
ENV PYTHONUNBUFFERED=1


EXPOSE 8000


CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


