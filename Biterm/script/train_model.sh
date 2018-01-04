#!/bin/bash
# run an toy example for BTM
for K in 26 27 28 29
# for K in $(seq 4 5)
do
    alpha=`echo "scale=3;1/$K"|bc`
    beta=0.01
    niter=500
    save_step=501

    # input_dir=../../intermediate_data/LDA_BTM_comparison/
    input_dir=../sample-data/
    output_dir=../output/${K}tp/
    model_dir=${output_dir}model/
    mkdir -p $output_dir/model

    # the input docs for training
    doc_pt=${input_dir}hpv_tweets.txt
    # doc_pt=${input_dir}lda_BTM_comparison_traning_data.txt
    echo "=============== Index Docs ============="
    # docs after indexing
    dwid_pt=${output_dir}doc_wids.txt
    # vocabulary file
    voca_pt=${output_dir}voca.txt
    python indexDocs.py $doc_pt $dwid_pt $voca_pt

    ## learning parameters p(z) and p(w|z)
    echo "=============== Topic Learning ============="
    W=`wc -l < $voca_pt` # vocabulary size
    make -C ../src
    echo "../src/btm est $K $W $alpha $beta $niter $save_step $dwid_pt $model_dir"
    ../src/btm est $K $W $alpha $beta $niter $save_step $dwid_pt $model_dir

    ## infer p(z|d) for each doc
    echo "================ Infer P(z|d)==============="
    echo "../src/btm inf sum_b $K $dwid_pt $model_dir"
    ../src/btm inf sum_b $K $dwid_pt $model_dir

    # ## output top words of each topic
    # echo "================ Topic Display ============="
    # python topicDisplay.py $model_dir $K $voca_pt
done
