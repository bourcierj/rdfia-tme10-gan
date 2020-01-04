#!/usr/bin/env sh

# Download CelebA dataset with 32x32 images

if ! [ -d "data/celeba/img_align_celeba" ] ; then
    mkdir -p data/celeba
    cd data/celeba
    wget http://webia.lip6.fr/~robert/cours/rdfia/celeba.zip
    unzip celeba.zip
    rm -r celeba.zip
fi
