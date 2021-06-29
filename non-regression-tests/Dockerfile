FROM ubuntu:21.04
WORKDIR "/usr/local/src"
RUN apt-get update
RUN apt-get install --yes git
RUN git clone https://github.com/etalab-ia/piaf-ml.git
WORKDIR "/usr/local/src/piaf-ml"
# TODO AFTERMERGE use main branch
RUN git checkout non-regression-test

# Add timezone to prevent tzdata installation from blocking
# https://dev.to/setevoy/docker-configure-tzdata-and-timezone-during-build-20bk
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get install --yes gcc make python3-pip python3-dev
RUN pip install -r requirements.txt

COPY non-regression-tests /usr/local/bin/
COPY non-regression-tests.py /usr/local/src/non-regression-tests/
COPY .env /usr/local/src/non-regression-tests/
COPY crontab /usr/local/src/non-regression-tests/

RUN apt-get install --yes cron
RUN crontab /usr/local/src/non-regression-tests/crontab
RUN mkdir -p /var/log/non-regression-tests

ENV HOME=/root

# Configure ssh access to mlflow server
RUN mkdir -p $HOME/.ssh
ARG MLFLOW_SSH_USERNAME
ARG MLFLOW_SSH_HOSTNAME
RUN echo "Host $MLFLOW_SSH_HOSTNAME\n\
    User $MLFLOW_SSH_USERNAME" > $HOME/.ssh/config
RUN ln -s /run/secrets/.ssh/id_rsa $HOME/.ssh/id_rsa
COPY known_hosts $HOME/.ssh/known_hosts

CMD cron -f
