FROM continuumio/miniconda3@sha256:6c979670684d970f8ba934bf9b7bf42e77c30a22eb96af1f30a039b484719159

ENV PYTHONDONTWRITEBYTECODE=true
ARG GIT_HASH=unspecified
ENV CURR_VERSION=$GIT_HASH

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

ADD /ra_joint_predictions /usr/local/bin/ra_joint_predictions

# Required for GPU: run.sh defines PATHs to find GPU drivers, see run.sh for specific commands
COPY run.sh /run.sh

# Required: Create /train /test and /output directories 
RUN mkdir /train \
    && mkdir /test \
    && mkdir /output \
    && chmod 775 /run.sh \
    && chmod 775 /usr/local/bin/ra_joint_predictions

# Required: define an entrypoint. run.sh will run the model for us, but in a different configuration
# you could simply call the model file directly as an entrypoint 
ENTRYPOINT ["/bin/bash", "/run.sh"]
