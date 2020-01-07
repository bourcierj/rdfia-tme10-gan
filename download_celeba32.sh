#!/usr/bin/env sh

# Download CelebA dataset with 32x32 images
#@TODO: The link is dead now!!! Host it somewhere else

if ! [ -d "data/celeba32/img_align_celeba" ] ; then
    mkdir -p data/celeba32
    cd data/celeba32
    wget http://webia.lip6.fr/~robert/cours/rdfia/celeba32.zip
    unzip celeba32.zip
    rm -r celeba32.zip
fi
