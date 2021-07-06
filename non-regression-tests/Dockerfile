# FROM nvidia/cuda:11.4.0-base-ubuntu20.04 
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
ARG SRC_DIR
ARG PIAF_ML_SRC=$SRC_DIR/piaf-ml
RUN apt-get update
RUN apt-get install --yes git
RUN git clone https://github.com/etalab-ia/piaf-ml.git $PIAF_ML_SRC
RUN echo $SRC_DIR $PIAF_ML_SRC
WORKDIR $PIAF_ML_SRC
ARG TEST_BRANCH
RUN git checkout $TEST_BRANCH

# Add timezone to prevent tzdata installation from blocking
# https://dev.to/setevoy/docker-configure-tzdata-and-timezone-during-build-20bk
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get install --yes gcc make python3-pip python3-dev
RUN pip install -r requirements.txt

ARG NON_REG_TESTS_SRC=$SRC_DIR/non-regression-tests
ARG BIN_DIR
COPY non-regression-tests $BIN_DIR/
COPY non-regression-tests.py $NON_REG_TESTS_SRC/
COPY .env $NON_REG_TESTS_SRC/
COPY crontab $NON_REG_TESTS_SRC/

RUN apt-get install --yes cron
RUN crontab $NON_REG_TESTS_SRC/crontab
ARG LOG_DIR
RUN mkdir -p $LOG_DIR/non-regression-tests


# Configure ssh access to mlflow server
ARG SSH_CONFIG_DIR
RUN mkdir -p $SSH_CONFIG_DIR
ARG MLFLOW_SSH_USERNAME
ARG MLFLOW_SSH_HOSTNAME
RUN echo "Host $MLFLOW_SSH_HOSTNAME\n\
    User $MLFLOW_SSH_USERNAME" > $SSH_CONFIG_DIR/config
RUN ln -s /run/secrets/.ssh/id_rsa $SSH_CONFIG_DIR/id_rsa
COPY known_hosts $SSH_CONFIG_DIR/known_hosts

CMD cron -f
