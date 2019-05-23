############################################################
# Dockerfile for Diting
############################################################

# Base image
FROM tensorflow/tensorflow:1.12.0-devel-gpu-py3

# Maintainer
MAINTAINER Wenlong Shen <shenwl1988@gmail.com>

# Install prerequisites
RUN apt-get update && apt-get install -y \
		graphviz tzdata && \
	apt-get clean && apt-get purge && rm -rf /var/lib/apt/lists/* /tmp/*

# Set timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
	echo "Asia/Shanghai" > /etc/timezone && \
	dpkg-reconfigure -f noninteractive tzdata
	
# Install python packages
RUN pip install --no-cache-dir keras bayesian-optimization seaborn pubmed_lookup gensim xgboost pydot weblogo
