FROM python:3.7.3-slim

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER author@sample.com

RUN mkdir /physionet2019
COPY ./ /physionet2019
WORKDIR /physionet2019
RUN pip install -r requirements.txt