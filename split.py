import os

def split_file(input_file, output_dir, chunk_size=95 * 1024):
    """将大文件切分为指定大小的小块"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 确保文件存在
    if not os.path.exists(input_file):
        print(f"错误：找不到文件 {input_file}")
        return

    print(f"开始切片：{input_file} ...")
    with open(input_file, 'rb') as f:
        index = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            # 使用四位数字作为索引，例如 part_0000, part_0001
            output_filename = os.path.join(output_dir, f"part_{index:04d}")
            with open(output_filename, 'wb') as chunk_file:
                chunk_file.write(chunk)
            index += 1

    print(f"切片完成！共生成了 {index} 个切片文件，保存在 {output_dir} 目录中。")

# 使用示例
input_pth = "your_model.pth"  # 替换为你的 pth 文件名
output_folder = "model_chunks"
split_file(input_pth, output_folder)
