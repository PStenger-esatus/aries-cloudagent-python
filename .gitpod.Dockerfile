FROM gitpod/workspace-base

RUN sudo apt-get update -y && \
	sudo apt-get install -y --no-install-recommends \
	python3 \
	python3-pip \
	python3-setuptools \
	libsodium23 


