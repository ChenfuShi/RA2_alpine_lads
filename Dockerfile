FROM continuumio/miniconda3@sha256:6c979670684d970f8ba934bf9b7bf42e77c30a22eb96af1f30a039b484719159

ENV PYTHONDONTWRITEBYTECODE=true
ARG GIT_HASH=unspecified
ENV CURR_VERSION=$GIT_HASH

COPY run.sh /run.sh
RUN  mkdir /train \
    && mkdir /test \
    && mkdir /output \
    && chmod 775 /run.sh

RUN /opt/conda/bin/conda install --yes --freeze-installed \
    nomkl \
    pandas \
    tensorflow-gpu==2.0.0 \
    opencv \
    matplotlib \
    pillow \
    && pip install tensorflow-addons==0.6.0 \
    && conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete

COPY /ra_joint_predictions /usr/local/bin/ra_joint_predictions
RUN chmod -R 775 /usr/local/bin/ra_joint_predictions

ENTRYPOINT ["/bin/bash", "/run.sh"]
