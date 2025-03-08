#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

directory="$1"

# 检查 black 和 isort 是否已安装
if ! command -v black &> /dev/null; then
    echo "Error: black is not installed. Install it using: pip install black"
    exit 1
fi

if ! command -v isort &> /dev/null; then
    echo "Error: isort is not installed. Install it using: pip install isort"
    exit 1
fi

# 查找指定目录下的所有 .py 文件并格式化
find "$directory" -type f -name "*.py" | while read -r file; do
    echo "Formatting: $file"
    black --line-length 120 "$file"
    isort "$file"
done

echo "Formatting completed."