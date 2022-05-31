FROM ubuntu:20.04

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND=noninteractive

ARG CURL_CA_BUNDLE

COPY . /app/

RUN apt-get update -y \
  && apt-get install -y python3.9 python3.9-dev python3-pip build-essential \
  && python3.9 -mpip install --upgrade --no-cache-dir pip setuptools \
  && python3.9 -mpip install --no-cache-dir . \
  && apt-get purge -y python3.9-dev

RUN larch-server-dryrun

EXPOSE 80/tcp

CMD ["larch-server", "--port", "80"]
