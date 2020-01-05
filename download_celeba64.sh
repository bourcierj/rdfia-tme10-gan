#!/usr/bin/env sh

# Download CelebA dataset with 64x64 images

if ! [ -d "data/celeba64/img_align_celeba" ] ; then
    mkdir -p data/celeba64
    cd data/celeba64
    wget http://webia.lip6.fr/~robert/cours/rdfia/celeba64.zip
    unzip celeba64.zip
    rm -r celeba64.zip
fi
