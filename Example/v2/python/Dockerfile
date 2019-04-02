FROM python:3.7.3-slim

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER author@sample.com

RUN pip install -r requirements.txt

## ADD
ADD get_sepsis_score.py /physionet2019
WORKDIR /physionet2019
