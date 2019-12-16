#!/bin/sh

# Download CelebA dataset with 64x64 images

if ! [ -d "data/celeba/img_align_celeba64" ] ; then
    mkdir -p data/celeba
    cd data/celeba
    wget http://webia.lip6.fr/~robert/cours/rdfia/celeba64.zip
    unzip celeba64.zip
    rm -r celeba64.zip
fi
