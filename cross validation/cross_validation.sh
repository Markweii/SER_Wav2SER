#!/bin/bash

# 性能结果文件
results_file="cross_validation_results.txt"

# 清空之前的结果
> $results_file

# YAML 文件的原始路径
yaml_file="hparams/train_custom.yaml"

# 检查点文件夹的路径
checkpoint_dir="/media/super514/2TSSD/Mark/SER_crossvalidation/results/train_with_wav2ser_trim_crossvalidation"

# share feature audio路徑
sharefeature_dir="/media/super514/2TSSD/Mark/SER_crossvalidation/output_audios_train_with_wav2ser_trim_crossvalidation_1999"

# 循环10次，每次更改 test_spk_id
for i in {1..10}
do
    echo "Running training for test_spk_id=$i"

    # 删除旧的检查点
    rm -rf $checkpoint_dir/*

    # 删除旧的混和音檔
    rm -rf $sharefeature_dir/*

    # 备份原始 YAML 文件
    cp $yaml_file "${yaml_file}.bak"

    # 更新 YAML 配置文件中的 test_spk_id
    sed -i "s/test_spk_id: [0-9]*/test_spk_id: $i/" $yaml_file

    # 运行训练程序
    python train_with_wav2SER_newshare3_trim.py $yaml_file

    # 计算性能指标，假设 plotc.py 输出含有 UA 和 WA 的字符串
    output=$(python plot_cmatrix.py --fpath "output_folder_$i")

    # 提取 UA 和 WA
    ua=$(echo $output | grep -oP 'UA accuracy: \K[0-9.]+')
    wa=$(echo $output | grep -oP 'WA accuracy: \K[0-9.]+')

    # 将性能记录到结果文件
    echo "test_spk_id=$i: UA=$ua, WA=$wa" >> $results_file

    # 恢复原始 YAML 文件
    mv "${yaml_file}.bak" $yaml_file
done

# 计算平均性能
awk -F 'UA=|, WA=' '{ua_sum += $2; wa_sum += $3} END {print "Average UA: " ua_sum / NR; print "Average WA: " wa_sum / NR}' $results_file
