FROM nvcr.io/nvidia/pytorch:20.01-py3

RUN mkdir /opt/app
WORKDIR /opt/app

COPY env.yml /opt/app
RUN conda init bash
RUN conda env create -f env.yml

SHELL ["conda", "run", "-n", "mixed_precision", "/bin/bash", "-c"]
RUN git clone https://github.com/NVIDIA/apex && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex

COPY main.py /opt/app
COPY networks.py /opt/app

ENV PATH /opt/conda/envs/mixed_precision/bin:$PATH
CMD /bin/bash