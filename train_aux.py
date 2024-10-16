import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import numpy as np

# 假设文件夹路径是'bounding_boxes_folder'
folder_path = ''
# 初始化列表来存储所有目标框的宽度和高度
all_widths = []
all_heights = []

# 遍历文件夹中的所有文本文件
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):  # 确保只处理文本文件
        file_path = os.path.join(folder_path, filename)

        # 读取文本文件并提取目标框尺寸
        with open(file_path, 'r') as file:
            for line in file:
                # 假设每行的格式为：类别编号, 归一化左上角x, 归一化左上角y, 归一化右下角x, 归一化右下角y
                class_id, normalized_x1, normalized_y1, normalized_x2, normalized_y2 = map(float,
                                                                                           line.strip().split(' '))
                # 计算目标框的宽度和高度
                # width = abs(normalized_x1 - normalized_x2)
                # height = abs(normalized_y1 - normalized_y2)
                # 将宽度和高度添加到列表中
                all_widths.append(normalized_x2)
                all_heights.append(normalized_y2)

# 将宽度和高度列表合并成一个二维数组
all_boxes = np.array(all_widths).reshape(-1, 1), np.array(all_heights).reshape(-1, 1)

# 使用K-Means算法进行聚类，假设我们想要聚类成5个群组
kmeans = KMeans(n_clusters=7, random_state=0)
kmeans.fit(np.hstack((all_boxes[0], all_boxes[1])))

# 获取聚类中心
centroids = kmeans.cluster_centers_

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(all_boxes[0], all_boxes[1], color=(0.7,0.85,0.95),alpha=0.5, label='Object Boxes',s=7)

# 在散点图上用特定标记画出聚类中心
for centroid in centroids:
    plt.plot(centroid[0], centroid[1], 'r^', markersize=10)  # r*表示红色的五角星

# 设置标题和轴标签
# plt.title('Distribution of Object Box Sizes with Cluster Centers')
plt.xlabel('Width')
plt.ylabel('Height')

# 添加图例
plt.legend()
plt.savefig('cluster_plot.svg', format='svg')
# 显示图形
plt.show()