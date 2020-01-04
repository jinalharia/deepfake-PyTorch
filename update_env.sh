#!/bin/bash
# Script to remove conda env directory and reload when new packages are used
# keeps conda env clean for project

cd `dirname $0`
rm -r env
conda env create -f env-spec.yml -p env

cd -
