#!/bin/bash

# 确保已经初始化了git并且关联了远程仓库
# 遍历所有的切片文件
for file in model_chunks/part_*; do
    if [ -f "$file" ]; then
        echo "正在处理: $file"
        
        # 仅添加当前这个切片
        git add "$file"
        git commit -m "Upload chunk $file"
        
        # 推送到远程仓库 (假设分支名为 main)
        git push origin main
        
        # 暂停 3 秒，防止被 GitHub 判定为恶意攻击或滥用 API
        sleep 3 
    fi
done

echo "所有切片上传完成！"
