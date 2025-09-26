import os.path

import numpy as np
import cv2

#######  读取相机的内参 #######3
def read_camera_params(file_path):
    """
    读取相机参数，包括RGB和深度相机的内参、外参
    """
    params = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith('[ColorIntrinsic]'):
                params['fx_color'] = float(lines[i + 1].split('=')[1].strip())
                params['fy_color'] = float(lines[i + 2].split('=')[1].strip())
                params['cx_color'] = float(lines[i + 3].split('=')[1].strip())
                params['cy_color'] = float(lines[i + 4].split('=')[1].strip())
            elif line.startswith('[DepthIntrinsic]'):
                params['fx_depth'] = float(lines[i + 1].split('=')[1].strip())
                params['fy_depth'] = float(lines[i + 2].split('=')[1].strip())
                params['cx_depth'] = float(lines[i + 3].split('=')[1].strip())
                params['cy_depth'] = float(lines[i + 4].split('=')[1].strip())

        # 提取D2CTransformParam参数，读取旋转矩阵和位移向量
        params['R'] = np.array([
            [float(lines[37].split('=')[1].strip()), float(lines[38].split('=')[1].strip()),
             float(lines[39].split('=')[1].strip())],
            [float(lines[40].split('=')[1].strip()), float(lines[41].split('=')[1].strip()),
             float(lines[42].split('=')[1].strip())],
            [float(lines[43].split('=')[1].strip()), float(lines[44].split('=')[1].strip()),
             float(lines[45].split('=')[1].strip())]
        ])
        params['t'] = np.array([
            float(lines[46].split('=')[1].strip()),
            float(lines[47].split('=')[1].strip()),
            float(lines[48].split('=')[1].strip())
        ])

    return params

def align_depth_to_rgb(depth_image, rgb_image, K_depth, K_rgb, R, t):
    """
    对齐RGB图像和深度图像
    :param depth_image: 输入的深度图像
    :param rgb_image: 输入的RGB图像
    :param K_depth: 深度相机的内参矩阵
    :param K_rgb: RGB相机的内参矩阵
    :param R: 旋转矩阵，表示从深度相机坐标系到RGB相机坐标系的旋转
    :param t: 平移向量，表示从深度相机坐标系到RGB相机坐标系的平移
    :return: 对齐后的深度图像
    """
    # 获取图像尺寸
    height, width = rgb_image.shape[:2]

    # 创建输出的对齐深度图
    aligned_depth = np.zeros_like(depth_image)

    # 遍历深度图中的每个像素
    for v in range(height):
        for u in range(width):
            # 获取深度值
            Z = depth_image[v, u]
            if Z == 0:  # 如果深度值为0，跳过
                continue

            # 将深度图像中的像素 (u, v, Z) 转换为相机坐标系中的三维点
            uv1 = np.array([u, v, 1])
            inv_K_depth = np.linalg.inv(K_depth)
            xyz_camera = inv_K_depth @ uv1 * Z  # 计算三维点

            # 将三维点从深度图相机坐标系转换到RGB图像相机坐标系
            xyz_rgb = R @ xyz_camera + t

            # 将三维点投影回RGB图像平面
            uv_rgb = K_rgb @ xyz_rgb
            u_rgb, v_rgb = uv_rgb[0] / uv_rgb[2], uv_rgb[1] / uv_rgb[2]

            # 检查投影是否在RGB图像内
            if 0 <= u_rgb < width and 0 <= v_rgb < height:
                u_rgb, v_rgb = int(u_rgb), int(v_rgb)

                # 将深度值赋给RGB图像对应的像素
                aligned_depth[v_rgb, u_rgb] = Z

    return aligned_depth


# 示例：读取相机内外参和图像
camera_params = read_camera_params('盐砖内参.ini')

from glob import glob

imgs=glob("/home/essh-yc/桌面/数据集备份/体重/信阳11月/complete/2024-11-11/weight_ID/*/color_images/*.png")
for i in imgs:
    # 读取RGB和深度图像
    rgb_image = cv2.imread(i)  # 读取RGB图像
    depth_image = cv2.imread(i.replace("color_images","depth_PNG"), cv2.IMREAD_UNCHANGED)  # 读取深度图像

    #深度转为彩色
    # depth_color1 = cv2.applyColorMap(depth_image.astype(np.uint8), cv2.COLORMAP_JET)
    #
    # # 将深度图像与可见光图像叠加
    # # 可以根据需要调整叠加的透明度
    # alpha = 0.4  # 控制叠加的透明度
    # blended_image1 = cv2.addWeighted(rgb_image, 1 - alpha, depth_color1, alpha, 0)
    # cv2.destroyAllWindows()
    # 显示结果


    # 相机内参矩阵
    K_depth = np.array([[camera_params['fx_depth'], 0, camera_params['cx_depth']],
                        [0, camera_params['fy_depth'], camera_params['cy_depth']],
                        [0, 0, 1]])  # 深度相机内参矩阵

    K_rgb = np.array([[camera_params['fx_color'], 0, camera_params['cx_color']],
                      [0, camera_params['fy_color'], camera_params['cy_color']],
                      [0, 0, 1]])  # RGB相机内参矩阵

    # 对齐RGB和深度图像
    aligned_depth = align_depth_to_rgb(depth_image, rgb_image, K_depth, K_rgb, camera_params['R'], camera_params['t'])
    if not os.path.exists(i.split("color_images")[0]+"/depth_PNG_align"):
        os.makedirs(i.split("color_images")[0]+"/depth_PNG_align")
    cv2.imwrite(i.replace("color_images","depth_PNG_align"),aligned_depth)


    # 显示深度图
    # aligned_depth[aligned_depth>5000]=0
    # aligned_depth=cv2.convertScaleAbs(aligned_depth,alpha=255.0/aligned_depth.max())

    #深度转为彩色
    # depth_color = cv2.applyColorMap(aligned_depth.astype(np.uint8), cv2.COLORMAP_JET)

    # 将深度图像与可见光图像叠加
    # 可以根据需要调整叠加的透明度
    # alpha = 0.7  # 控制叠加的透明度
    # blended_image = cv2.addWeighted(rgb_image, 1 - alpha, depth_color, alpha, 0)
    # cv2.imwrite(i.replace("color_images","new"),blended_image)
    # cv2.imwrite(i.replace("color_images","ori"),blended_image1)

    # cv2.destroyAllWindows()
    # # 显示结果
    # cv2.imshow('Blended Image', blended_image)
    # cv2.imshow('Blended Image1', blended_image1)
    # # 显示对齐后的图像
    # cv2.imshow("Aligned Depth Image", aligned_depth)
    # cv2.imshow("RGB Image", rgb_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

