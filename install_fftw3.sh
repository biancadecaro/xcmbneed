#!/bin/bash

wget http://fftw.org/fftw-3.3.10.tar.gz
tar -zxf fftw-3.3.10.tar.gz
cd fftw-3.3.10
mkdir $HOME/fftw3
./configure CC=gcc F77=gfortran --enable-shared --enable-threads --enable-openmp --enable-mpi MPICC=mpicc --prefix=$HOME/fftw3/fftw-3.3.10-double
make -j 8
make install
