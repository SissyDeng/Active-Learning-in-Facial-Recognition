#!/bin/bash

max=400000
n_step=10
cur=0
for dir in data/*
do
  if [ -d "data/amazon-reviews/active_learning" ];then
    rm -r $dir/active_learning
  fi
  mkdir $dir/active_learning
  dir_cur=$dir/active_learning
  touch $dir_cur/test_result.txt
  cp $dir/train.ft.txt $dir_cur/unlabeled.txt
  shuf $dir_cur/unlabeled.txt | head -80000 > $dir_cur/training.txt
  cat $dir_cur/unlabeled.txt $dir_cur/training.txt | sort | uniq -u > $dir_cur/temp.txt
  rm $dir_cur/unlabeled.txt
  mv $dir_cur/temp.txt $dir_cur/unlabeled.txt
  while [ $cur -lt $n_step ]
  do
    cat $dir_cur/training.txt | wc
    cat $dir_cur/unlabeled.txt | wc
    ./fastText/fasttext supervised -input $dir_cur/training.txt -output $dir/model -verbose 3
    ./fastText/fasttext test-label $dir/model.bin $dir/test.ft.txt >> $dir_cur/test_result.txt
    ./fastText/fasttext predict-prob $dir/model.bin $dir_cur/unlabeled.txt > $dir_cur/prob.txt
    awk '{print $2}' $dir_cur/prob.txt > $dir_cur/prob_temp.txt
    paste $dir_cur/unlabeled.txt $dir_cur/prob_temp.txt | sort -t $'\t' -k2 -n | awk  '{$NF="";print}' |sed 's/[ \t]*$//g'| head -80000 >> $dir_cur/training.txt
    rm $dir_cur/prob.txt
    rm $dir_cur/prob_temp.txt
    sort $dir_cur/unlabeled.txt $dir_cur/training.txt $dir_cur/training.txt| uniq -u > $dir_cur/temp.txt
    rm $dir_cur/unlabeled.txt
    mv $dir_cur/temp.txt $dir_cur/unlabeled.txt
    cur=$((cur + 1))
  done
done
