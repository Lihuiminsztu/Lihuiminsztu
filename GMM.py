import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

class GMMImplementation:
    """
    高斯混合模型(GMM)的实现类
    包含数据生成、参数估计和可视化评估功能
    """
    
    def __init__(self, n_components=3, n_samples=1000):
        """
        初始化GMM参数
        
        参数:
            n_components: 高斯分布的数量，默认为3
            n_samples: 样本数量，默认为1000
        """
        self.n_components = n_components
        self.n_samples = n_samples
        
        # 初始化真实参数
        self.true_weights = np.array([0.4, 0.35, 0.25])
        self.true_means = np.array([0, 5, 10])
        self.true_stds = np.array([1, 1.5, 2])
        
        # 存储生成的数据和估计的参数
        self.data = None
        self.gmm = None
        
    def generate_data(self):
        """
        生成混合高斯分布的数据
        
        返回:
            numpy.ndarray: 生成的数据样本
        """
        data = []
        n_samples_per_component = np.random.multinomial(
            self.n_samples, 
            self.true_weights
        )
        
        for i in range(self.n_components):
            samples = np.random.normal(
                loc=self.true_means[i],
                scale=self.true_stds[i],
                size=n_samples_per_component[i]
            )
            data.extend(samples)
            
        self.data = np.array(data).reshape(-1, 1)
        return self.data
    
    def fit_gmm(self):
        """
        使用EM算法拟合GMM模型
        
        返回:
            GaussianMixture: 拟合后的GMM模型
        """
        if self.data is None:
            raise ValueError("请先生成数据")
            
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            random_state=42
        )
        self.gmm.fit(self.data)
        return self.gmm
    
    def plot_results(self):
        """
        可视化真实分布和估计分布的对比
        """
        if self.data is None or self.gmm is None:
            raise ValueError("请先生成数据并拟合模型")
            
        # 使用 matplotlib 内置的样式替代 seaborn
        plt.style.use('default')  # 或者使用其他内置样式如 'classic', 'bmh', 'ggplot'
        plt.figure(figsize=(12, 7))
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 绘制数据直方图，使用更柔和的颜色
        plt.hist(
            self.data, 
            bins=50, 
            density=True, 
            alpha=0.6,
            color='lightblue',
            label='数据分布'
        )
        
        # 绘制真实分布
        x = np.linspace(min(self.data)-1, max(self.data)+1, 1000).reshape(-1, 1)
        true_density = np.zeros_like(x)
        
        # 绘制各个组分（使用不同的颜色）
        colors = ['#FF9999', '#99FF99', '#9999FF']
        for i in range(self.n_components):
            component_density = self.true_weights[i] * norm.pdf(
                x,
                self.true_means[i],
                self.true_stds[i]
            )
            true_density += component_density
            plt.plot(x, component_density, '--', 
                    color=colors[i], alpha=0.5,
                    label=f'组分 {i+1}')
        
        # 绘制真实总分布
        plt.plot(x, true_density, 'r-', 
                label='真实分布', linewidth=2)
        
        # 绘制估计分布
        estimated_density = np.exp(self.gmm.score_samples(x))
        plt.plot(x, estimated_density, 'g--', 
                label='估计分布', linewidth=2)
        
        plt.title('高斯混合模型拟合结果', fontsize=14, pad=20)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('密度', fontsize=12)
        
        # 优化图例位置和样式
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1),
                  frameon=True, fancybox=True, shadow=True)
        
        # 设置网格
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 设置坐标轴范围
        plt.xlim(min(self.data)-1, max(self.data)+1)
        
        # 添加评价指标文本框
        metrics = self.evaluate_model()
        metrics_text = (
            f"模型评价指标:\n"
            f"对数似然值: {metrics['log_likelihood']:.2f}\n"
            f"AIC: {metrics['aic']:.2f}\n"
            f"BIC: {metrics['bic']:.2f}\n"
            f"参数数量: {metrics['n_parameters']}"
        )
        
        plt.text(0.98, 0.02, metrics_text,
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                 verticalalignment='bottom',
                 horizontalalignment='right',
                 fontsize=10)
        
        # 添加边距，防止图像被裁剪
        plt.tight_layout()
        
    def print_parameters(self):
        """
        打印真实参数和估计参数的对比
        """
        if self.gmm is None:
            raise ValueError("请先拟合模型")
            
        print("参数对比：")
        print("\n权重:")
        print(f"真实值: {self.true_weights}")
        print(f"估计值: {self.gmm.weights_}")
        
        print("\n均值:")
        print(f"真实值: {self.true_means}")
        print(f"估计值: {self.gmm.means_.flatten()}")
        
        print("\n标准差:")
        print(f"真实值: {self.true_stds}")
        print(f"估计值: {np.sqrt(self.gmm.covariances_).flatten()}")
        
        # 打印评价指标
        metrics = self.evaluate_model()
        print("\n模型评价指标:")
        print(f"对数似然值: {metrics['log_likelihood']:.2f}")
        print(f"AIC: {metrics['aic']:.2f}")
        print(f"BIC: {metrics['bic']:.2f}")
        print(f"参数数量: {metrics['n_parameters']}")
    
    def evaluate_model(self):
        """
        计算模型的评价指标
        
        返回:
            dict: 包含各项评价指标的字典
            - log_likelihood: 对数似然值
            - aic: 赤池信息准则
            - bic: 贝叶斯信息准则
            - n_parameters: 模型参数数量
        """
        if self.data is None or self.gmm is None:
            raise ValueError("请先生成数据并拟合模型")
        
        # 计算对数似然值
        log_likelihood = self.gmm.score(self.data) * len(self.data)
        
        # 计算模型参数数量
        # 对于每个组分，需要估计：均值(1)、方差(1)、权重(1)
        # 注意：权重有一个约束条件，所以实际自由参数少1
        n_parameters = self.n_components * 2 + (self.n_components - 1)
        
        # 计算AIC和BIC
        aic = -2 * log_likelihood + 2 * n_parameters
        bic = -2 * log_likelihood + np.log(len(self.data)) * n_parameters
        
        return {
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'n_parameters': n_parameters
        }
    
    def verify_estimator_properties(self, n_experiments=1000):
        """
        验证估计量的统计性质
        
        参数:
            n_experiments: 重复实验次数
            
        返回:
            dict: 包含验证结果的字典
        """
        means_estimates = np.zeros((n_experiments, self.n_components))
        weights_estimates = np.zeros((n_experiments, self.n_components))
        vars_estimates = np.zeros((n_experiments, self.n_components))
        
        for i in range(n_experiments):
            # 生成新数据并拟合
            self.generate_data()
            self.fit_gmm()
            
            # 记录估计值
            means_estimates[i] = self.gmm.means_.flatten()
            weights_estimates[i] = self.gmm.weights_
            vars_estimates[i] = np.sqrt(self.gmm.covariances_).flatten()
        
        # 计算偏差
        means_bias = np.mean(means_estimates, axis=0) - self.true_means
        weights_bias = np.mean(weights_estimates, axis=0) - self.true_weights
        vars_bias = np.mean(vars_estimates, axis=0) - self.true_stds
        
        # 计算方差
        means_var = np.var(means_estimates, axis=0)
        weights_var = np.var(weights_estimates, axis=0)
        vars_var = np.var(vars_estimates, axis=0)
        
        return {
            'means_bias': means_bias,
            'weights_bias': weights_bias,
            'vars_bias': vars_bias,
            'means_var': means_var,
            'weights_var': weights_var,
            'vars_var': vars_var
        }

def main():
    """
    主函数，运行完整的GMM实现流程
    """
    # 创建GMM实例
    gmm_impl = GMMImplementation()
    
    # 生成数据
    print("生成数据...")
    data = gmm_impl.generate_data()
    
    # 拟合模型
    print("拟合GMM模型...")
    gmm_impl.fit_gmm()
    
    # 打印参数对比
    gmm_impl.print_parameters()
    
    # 可视化结果
    print("绘制结果...")
    gmm_impl.plot_results()
    plt.show()

if __name__ == "__main__":
    main()
