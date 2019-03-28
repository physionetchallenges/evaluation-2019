# Python prediction and evaluation code

`get_sepsis_score.py` makes predictions on the input data, and `evaluate_sepsis_score.py` evaluates the predictions.

## Steps
1) We use TensorFlow v1.13.1 as the reference version here. Pull the docker container of tf v1.13.1 from docker hub

        docker pull tensorflow/tensorflow:1.13.1

2) Start the container. (This will change based on Annie's requirements)

        docker run -it -w /physionet2019 -v $PWD:/mnt -e HOST_PERMS="$(id -u):$(id -g)" tensorflow/tensorflow:1.13.1 bash
        cp /mnt/get_sepsis_score.py /physionet2019

3) For predictions, run the Python prediction script, which takes `/mnt/training_100.zip` as input and returns `/mnt/predictions_100.zip` as output.

        python get_sepsis_score.py /mnt/training_100.zip /mnt/predictions_100.zip

4) For evaluation, run the Python evaluation script, which takes `/mnt/data_100.zip` and `/mnt/predictions_100.zip` as inputs and returns `/mnt/score_100.psv` as output.

        python evaluate_sepsis_score.py /mnt/labels_100.zip /mnt/predictions_100.zip /mnt/score_100.psv
