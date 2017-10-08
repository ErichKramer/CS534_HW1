#!/bin/bash

datPath="./income-data/"
declare -a fnames=("income.dev.txt" "income.test.txt" "income.train.txt")
for i in {1..9}
do
    cat ${fnames[@]/#/$datPath} | awk -v param=$i -F ',' '{ print $param }' | sed 's/ //g' | sort -g| uniq
    printf "\n"
done
