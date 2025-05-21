import torch
from torch.optim.lr_scheduler import LRScheduler
import warnings
import math
class WarmupCosineLR(LRScheduler):
    """
    基于步数的预热+余弦退火学习率调度器，专为强化学习设计
    
    参数:
        optimizer: PyTorch优化器
        warmup_steps: 预热阶段的步数
        max_steps: 最大训练步数
        warmup_start_factor: 预热开始时的学习率因子（相对于基础学习率）
        lr_min: 最小学习率
        last_epoch: 上次更新的步数 (-1表示从头开始)
    """
    
    def __init__(self, optimizer, warmup_steps, max_steps, 
                 warmup_start_factor=0.1, lr_min=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.warmup_start_factor = warmup_start_factor
        self.lr_min = lr_min
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")
            
        current_step = self.last_epoch
        
        if current_step < self.warmup_steps:
            # 预热阶段 - 线性增长
            factor = self.warmup_start_factor + (1 - self.warmup_start_factor) * (current_step / self.warmup_steps)
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            cosine_steps = self.max_steps - self.warmup_steps
            cosine_current = current_step - self.warmup_steps
            
            # 确保不会超出余弦周期
            cosine_current = min(cosine_current, cosine_steps)
            
            # 应用余弦衰减
            cosine_factor = 0.5 * (1 + math.cos(math.pi * cosine_current / cosine_steps))
            factor = self.lr_min + (1 - self.lr_min) * cosine_factor
            
            return [base_lr * factor for base_lr in self.base_lrs]


# 使用示例
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import math
    
    # 创建一个简单的模型和优化器
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 设置调度器参数
    warmup_steps = 1000
    max_steps = 10000
    
    # 创建学习率调度器
    scheduler = WarmupCosineLR(
        optimizer, 
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        warmup_start_factor=0.1,
        lr_min=1e-6
    )
    
    # 记录学习率变化
    lrs = []
    for step in range(max_steps):
        # 在强化学习中，这里通常是与环境交互、更新模型的地方
        optimizer.step()
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        
        if step % 1000 == 0:
            print(f"Step {step}, Learning Rate: {current_lr:.6f}")
    
    # 可视化学习率变化（实际使用时可以注释掉）
    plt.figure(figsize=(10, 5))
    plt.plot(lrs)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Warmup + Cosine Annealing LR Schedule')
    plt.axvline(x=warmup_steps, color='r', linestyle='--', label='End of Warmup')
    plt.legend()
    plt.grid(True)
    plt.savefig('lr_schedule.png')
    plt.close()
    
    print("学习率调度完成")


# 不依赖matplotlib的简单示例
def simple_test():
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    scheduler = WarmupCosineLR(
        optimizer, 
        warmup_steps=1000,
        max_steps=10000,
        warmup_start_factor=0.1,
        lr_min=1e-6
    )
    
    # 模拟训练循环
    for step in range(10000):
        # 模拟与环境交互并更新模型
        loss = torch.rand(1).requires_grad_(True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 更新学习率
        scheduler.step()