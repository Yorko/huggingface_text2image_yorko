FROM python:3.7

MAINTAINER Yury Kashnitsky <yury.kashnitsky@gmail.com>

EXPOSE 8501

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .