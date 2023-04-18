# Use the official Python base image for x86_64
FROM --platform=linux/x86_64 python:3.9

# Download QEMU for cross-compilation
ADD https://github.com/multiarch/qemu-user-static/releases/download/v6.1.0-8/qemu-x86_64-static /usr/bin/qemu-x86_64-static
RUN chmod +x /usr/bin/qemu-x86_64-static

# Install required dependencies for FSL and Freesurfer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    git \
    tcsh \
    xfonts-base \
    gfortran \
    libjpeg62 \
    libtiff5-dev \
    libpng-dev \
    unzip \
    libxext6 \
    libx11-6 \
    libxmu6 \
    libglib2.0-0 \
    libxft2 \
    libxrender1 \
    libxt6

# Install FSL
ENV FSLDIR="/opt/fsl" \
    FSLOUTPUTTYPE=NIFTI_GZ \
    PATH="/opt/fsl/bin:$PATH"

RUN curl -sL https://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py -o fslinstaller.py && python3 fslinstaller.py -d /usr/local/fsl -o && echo ". /usr/local/fsl/fsl/etc/fslconf/fsl.sh" >> ~/.bashrc

# Install Freesurfer
RUN wget -qO- https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/${FREESURFER_VERSION}/freesurfer-linux-ubuntu16-${FREESURFER_VERSION}.tar.gz | tar zxv -C /usr/local
ENV FREESURFER_HOME=/usr/local/freesurfer
ENV PATH=$FREESURFER_HOME/bin:$PATH

# Install BIDS application and its dependencies
RUN pip3 install numpy nibabel pybids bids-validator

# Clone and install petprep_hmc from GitHub
RUN git clone https://github.com/mnoergaard/petprep_hmc.git /opt/petprep_hmc
RUN pip3 install --no-cache-dir -e /opt/petprep_hmc

# BIDS App entry point
ENTRYPOINT ["python3", "/opt/petprep_hmc/run.py"]
CMD ["--help"]
