#!/bin/bash

# 输入文件名（原始CSV）
input_file="test.csv"
# 输出文件名
output_file="a.csv"

# 清空输出文件
> "$output_file"

# 逐行读取输入CSV
while IFS=, read -r path value; do
    # 从路径中提取下划线后面、斜杠前的三个数字
    # 例如：data/video2img_2fps/0_110/output.mp4 -> 110
    num=$(echo "$path" | grep -oP '_\K\d{3}(?=/)')
    # 如果匹配成功，就写入新行
    if [[ -n "$num" ]]; then
        echo "$path,$num" >> "$output_file"
    else
        # 如果没有匹配，保持原样
        echo "$path,$value" >> "$output_file"
    fi
done < "$input_file"

echo "✅ 处理完成！输出文件为：$output_file"
