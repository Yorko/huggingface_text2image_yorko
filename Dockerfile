FROM python:3.6-stretch
MAINTAINER Yury Kashnitsky <yury.kashnitsky@gmail.com>

# install build utilities
#RUN apt-get update && \
#	apt-get install -y gcc make apt-transport-https ca-certificates build-essential

# check our python environment
RUN python3 --version
RUN pip3 --version

# set the working directory for containers
WORKDIR /usr/src/huggingface_text2image_yorko
COPY scripts/ /src/scripts/
COPY training_logs/ /src/training_logs/
COPY config.yml /src/
RUN ls -la /src/*

# Installing python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Running Python Application
CMD ["python3", "/src/scripts/text2image_model.py"]

