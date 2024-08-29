#!/bin/bash

# 性能结果文件
results_file="SD_cross_validation_results.txt"

# 清空之前的结果
> $results_file

# YAML 文件的原始路径
yaml_file="hparams/train_custom1.yaml"

# 检查点文件夹的路径
checkpoint_dir="/media/super514/2TSSD/Mark/SER_crossvalidation/results/train_with_wav2ser_trim_SDcrossvalidation"

# 循环5次，第一次使用1999作为随机种子，之后使用新的随机种子
for i in {1..5}
do
    # 如果是第一次迭代，使用1999作为种子，否则生成新的随机种子
    if [ $i -eq 1 ]; then
        new_seed=1999
    else
        new_seed=$RANDOM
    fi

    echo "Running training with random seed $new_seed for iteration $i"

    # 构建新的 share feature audio 路径，包含当前的随机种子
    sharefeature_dir="/media/super514/2TSSD/Mark/SER_crossvalidation/output_audios_train_with_wav2ser_trim_SDcrossvalidation_$new_seed"

    # 构建新的 result路径，包含当前的随机种子
    result_dir="/media/super514/2TSSD/Mark/SER_crossvalidation/results/train_with_wav2ser_trim_SDcrossvalidation/$new_seed"

    # 删除旧的检查点
    rm -rf $checkpoint_dir/*

    # 删除旧的混和音档
    rm -rf $sharefeature_dir/*

    # 备份原始 YAML 文件
    cp $yaml_file "${yaml_file}.bak"

    # 更新 YAML 配置文件中的 random seed
    sed -i "s/seed: [0-9]*/seed: $new_seed/" $yaml_file

    # 运行训练程序
    python train_with_wav2SER_newshare3_trim.py $yaml_file


    # 計算性能指標
    output=$(python plot_cmatrix.py "$result_dir")

    # 提取 UA 和 WA
    ua=$(echo $output | grep -oP 'UA accuracy: \K[0-9.]+')
    wa=$(echo $output | grep -oP 'WA accuracy: \K[0-9.]+')

    # 将性能记录到结果文件
    echo "Iteration=$i with random seed $new_seed: UA=$ua, WA=$wa" >> $results_file

    # 恢复原始 YAML 文件
    mv "${yaml_file}.bak" $yaml_file
done

# 计算平均性能
awk -F 'UA=|, WA=' '{ua_sum += $2; wa_sum += $3} END {print "Average UA: " ua_sum / NR; print "Average WA: " wa_sum / NR}' $results_file
