import cv2
import os


def canny_edge_extraction(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 构建完整的文件路径
        input_path = os.path.join(input_folder, filename)

        # 读取图像
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Skipping non-image file: {filename}")
            continue

        # 应用 Canny 边缘检测
        edges = cv2.Canny(image, 80, 150)

        # 构建输出文件路径
        output_path = os.path.join(output_folder, filename)

        # 保存边缘检测后的图像
        cv2.imwrite(output_path, edges)
        print(f"Processed and saved: {output_path}")


# 主函数
if __name__ == "__main__":
    input_folder = "obj5/renderext"  # 输入文件夹路径
    output_folder = "obj5/edgesext"  # 输出文件夹路径

    # 调用函数进行 Canny 边缘提取
    canny_edge_extraction(input_folder, output_folder)