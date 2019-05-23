#!/bin/bash

docker run -d --runtime=nvidia -v /home/archer/Desktop/Diting/:/Diting/ -w /Diting -p 8888:8888 wenlongshen/diting:0.1 jupyter notebook --notebook-dir=/Diting --ip 0.0.0.0 --no-browser --allow-root
