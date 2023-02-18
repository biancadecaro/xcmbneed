#!/bin/bash

cd /tmp
wget https://mirror.kumi.systems/gnu/gsl/gsl-2.7.1.tar.gz
tar -zxf gsl-2.7.1.tar.gz
cd gsl-2.7.1
mkdir $HOME/gsl
./configure CC=gcc F77=gfortran --enable-threads --enable-openmp --enable-mpi MPICC=mpicc --prefix=$HOME/gsl/gsl-2.7.1
make -j 8
make install
   
