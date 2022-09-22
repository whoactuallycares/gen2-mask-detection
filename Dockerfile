FROM ghcr.io/luxonis/robothub-base-app:ubuntu-depthai-main

RUN pip3 install -U numpy opencv-contrib-python-headless
ARG FILE=app.py

ADD script.py .
ADD format_script.py .

ADD face-detection-retail-0004.blob .
ADD face-detection-retail-0004.json .
ADD sbd_mask_classification_224x224.blob .
add sbd_mask_classification_224x224.json .

ADD $FILE run.py
