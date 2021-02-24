# Note: You can use any Debian/Ubuntu based image you want. 
FROM nvidia/cuda:10.2-base

RUN apt-get update \
    && apt-get install -y python3.7 python3.7-dev python3.7-distutils python3-pip curl git pkg-config cmake \
    # Clean up
    && apt-get autoremove -y && apt-get clean -y \
    && pip3 install --no-cache-dir numpy scipy Cython \
    && pip3 install --upgrade pip

# RUN apt-get install wget \
#     && wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.6.2-amd64.deb \
#     && wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.6.2-amd64.deb.sha512 \
#     && shasum -a 512 -c elasticsearch-7.6.2-amd64.deb.sha512 \
#     && dpkg -i elasticsearch-7.6.2-amd64.deb \
#     && service elasticsearch start

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# # Set default Python version
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
# RUN update-alternatives --set python3 /usr/bin/python3.7

COPY . .
RUN pip3 install -r requirements_simplified.txt

ENTRYPOINT [ "python3" ]
CMD [ "-m", "src.evaluation.retriever_reader.retriever_reader_eval_squad" ]

# Setting the ENTRYPOINT to docker-init.sh will configure non-root access 
# to the Docker socket. The script will also execute CMD as needed.
# ENTRYPOINT [  "/bin/bash", "-c" ]
# CMD ["sleep infinity" ]

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>