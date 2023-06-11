FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq

COPY requirements.txt /
RUN pip --no-cache-dir install -r /requirements.txt
RUN pip --no-cache-dir install dvc[gdrive]

COPY static ./static
COPY templates ./templates
COPY femaleShadowKNN.joblib ./
COPY maleShadowKNN.joblib ./
COPY *.py ./

EXPOSE 8000
ENTRYPOINT [ "python3", "app.py"]
