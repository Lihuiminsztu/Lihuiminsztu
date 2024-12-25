import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 实验数据
weights_data = {
    'true': [0.4, 0.35, 0.25],
    'estimates': [
        [0.398, 0.260, 0.342],
        [0.403, 0.250, 0.347],
        [0.394, 0.243, 0.363],
        [0.420, 0.261, 0.319],
        [0.424, 0.203, 0.373],
        [0.426, 0.247, 0.327],
        [0.382, 0.243, 0.375],
        [0.407, 0.206, 0.387],
        [0.390, 0.202, 0.408],
        [0.403, 0.262, 0.335]
    ]
}

means_data = {
    'true': [0, 5, 10],
    'estimates': [
        [0.036, 5.158, 10.199],
        [0.062, 5.189, 10.121],
        [0.090, 5.304, 10.522],
        [0.006, 4.891, 9.766],
        [0.047, 5.118, 10.125],
        [0.019, 5.128, 10.088],
        [-0.008, 5.079, 10.315],
        [0.126, 5.208, 10.172],
        [-0.020, 5.141, 10.326],
        [-0.058, 5.026, 10.406]
    ]
}

stds_data = {
    'true': [1.0, 1.5, 2.0],
    'estimates': [
        [1.031, 1.634, 1.882],
        [0.993, 1.487, 1.880],
        [1.091, 1.466, 1.816],
        [1.019, 1.337, 1.660],
        [0.989, 1.449, 1.812],
        [0.954, 1.500, 1.663],
        [1.004, 1.611, 1.779],
        [1.087, 1.517, 1.873],
        [0.988, 1.588, 1.857],
        [1.008, 1.462, 1.832]
    ]
}

metrics_data = {
    'log_likelihood': [-2686.62, -2653.80, -2691.76, -2619.27, -2607.18, 
                      -2615.92, -2676.59, -2656.17, -2644.55, -2679.15],
    'aic': [5389.25, 5323.60, 5399.51, 5254.55, 5230.36, 
            5247.84, 5369.19, 5328.34, 5305.09, 5374.30],
    'bic': [5428.51, 5362.86, 5438.77, 5293.81, 5269.62, 
            5287.10, 5408.45, 5367.60, 5344.36, 5413.57]
}

def plot_parameter_estimates():
    """绘制参数估计结果的比较图"""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 10})
    
    # 1. 权重估计
    ax1 = fig.add_subplot(gs[0, 0])
    weights_est = np.array(weights_data['estimates'])
    bp1 = ax1.boxplot([weights_est[:, i] for i in range(3)], 
                      patch_artist=True,
                      boxprops=dict(facecolor='lightblue', alpha=0.5))
    ax1.plot([1, 2, 3], weights_data['true'], 'r*', markersize=10, label='真实值')
    ax1.set_title('权重估计分布', pad=10, fontsize=12)
    ax1.set_xticklabels(['组分1', '组分2', '组分3'])
    ax1.set_ylabel('权重值')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. 均值估计
    ax2 = fig.add_subplot(gs[0, 1])
    means_est = np.array(means_data['estimates'])
    bp2 = ax2.boxplot([means_est[:, i] for i in range(3)],
                      patch_artist=True,
                      boxprops=dict(facecolor='lightgreen', alpha=0.5))
    ax2.plot([1, 2, 3], means_data['true'], 'r*', markersize=10, label='真实值')
    ax2.set_title('均值估计分布', pad=10, fontsize=12)
    ax2.set_xticklabels(['组分1', '组分2', '组分3'])
    ax2.set_ylabel('均值')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. 标准差估计
    ax3 = fig.add_subplot(gs[1, 0])
    stds_est = np.array(stds_data['estimates'])
    bp3 = ax3.boxplot([stds_est[:, i] for i in range(3)],
                      patch_artist=True,
                      boxprops=dict(facecolor='lightsalmon', alpha=0.5))
    ax3.plot([1, 2, 3], stds_data['true'], 'r*', markersize=10, label='真实值')
    ax3.set_title('标准差估计分布', pad=10, fontsize=12)
    ax3.set_xticklabels(['组分1', '组分2', '组分3'])
    ax3.set_ylabel('标准差')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. 评价指标
    ax4 = fig.add_subplot(gs[1, 1])
    experiments = range(1, 11)
    ax4.plot(experiments, metrics_data['log_likelihood'], 'b-', label='对数似然值', linewidth=2)
    ax4.plot(experiments, metrics_data['aic'], 'r-', label='AIC', linewidth=2)
    ax4.plot(experiments, metrics_data['bic'], 'g-', label='BIC', linewidth=2)
    ax4.set_title('模型评价指标', pad=10, fontsize=12)
    ax4.set_xlabel('实验次数')
    ax4.set_ylabel('指标值')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('GMM_analysis_results1.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_experiments():
    """将所有实验结果整合到一张图中"""
    plt.figure(figsize=(15, 12))
    
    for i in range(10):
        plt.subplot(4, 3, i+1)
        img = plt.imread(f'GMM_test_figure{i+1}.png')
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'实验 {i+1}', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('GMM_all_experiments1.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # 生成参数估计和评价指标的可视化
    plot_parameter_estimates()
    
    # 整合所有实验的拟合结果
    plot_all_experiments()
    
    print("分析完成！生成了两个文件：")
    print("1. GMM_analysis_results1.png - 参数估计和评价指标的可视化")
    print("2. GMM_all_experiments1.png - 所有实验的拟合结果整合") 