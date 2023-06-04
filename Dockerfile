FROM tensorflow/tensorflow:2.4.1-gpu-jupyter

RUN pip install --upgrade \
    pip \
    setuptools

RUN pip3 install jupyterlab==3.0.10 \
                 pandas==1.1.5 \
                 numpy==1.19.5 \
                 scikit-learn==0.24.1 \
                 tqdm==4.59.0 \
                 matplotlib==3.3.4 \
                 seaborn==0.11.1 \
                 scipy==1.5.4 \
                 Pillow==8.1.2 \
                 PyYAML==5.4.1

RUN pip3 install jedi==0.17.2

RUN pip3 install pytest \
                 pytest-black \
                 pytest-flake8
