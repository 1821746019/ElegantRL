import torch
import math
import warnings
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineLR(LRScheduler):
    """
    基于步数的预热+余弦退火学习率调度器，专为强化学习设计
    
    参数:
        optimizer: PyTorch优化器
        warmup_steps_pct: 预热阶段步数占总步数的比例
        total_steps: 总训练步数
        warmup_start_factor: 预热开始时的学习率因子（相对于基础学习率）
        cosine_end_factor: 余弦退火结束时的学习率因子（相对于基础学习率）
        last_epoch: 上次更新的步数 (-1表示从头开始)
    """
    
    def __init__(self, optimizer, warmup_steps_pct, total_steps, 
                 warmup_start_factor=0.1, cosine_end_factor=0, last_epoch=-1):
        self.warmup_steps = int(warmup_steps_pct * total_steps)
        self.total_steps = total_steps
        self.warmup_start_factor = warmup_start_factor
        # 方案1：直接用因子（推荐）
        self.cosine_end_factor = cosine_end_factor
        
        # 方案2：如果坚持用绝对值
        # self.lr_min_values = [cosine_end_factor * base_lr for base_lr in self.base_lrs]
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")
            
        current_step = self.last_epoch
        
        if current_step < self.warmup_steps:
            # 预热阶段 - 线性增长
            if self.warmup_steps == 0:  # 防止除零
                factor = 1.0
            else:
                factor = self.warmup_start_factor + (1 - self.warmup_start_factor) * (current_step / self.warmup_steps)
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            cosine_steps = self.total_steps - self.warmup_steps
            cosine_current = current_step - self.warmup_steps
            
            # 确保不会超出余弦周期
            cosine_current = min(cosine_current, cosine_steps)
            
            if cosine_steps == 0:  # 防止除零
                cosine_factor = 0
            else:
                # 余弦衰减：从1衰减到0
                cosine_factor = 0.5 * (1 + math.cos(math.pi * cosine_current / cosine_steps))
            
            # 方案1：因子空间计算（当前实现）
            factor = self.cosine_end_factor + (1 - self.cosine_end_factor) * cosine_factor
            return [base_lr * factor for base_lr in self.base_lrs]
            
            # 方案2：绝对值空间计算（如果用lr_min_values）
            # return [lr_min + (base_lr - lr_min) * cosine_factor 
            #         for base_lr, lr_min in zip(self.base_lrs, self.lr_min_values)]
# 使用示例
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import math
    
    # 创建一个简单的模型和优化器
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 设置调度器参数
    warmup_steps_pct = 0.1
    total_steps = 10000
    
    # 创建学习率调度器
    scheduler = WarmupCosineLR(
        optimizer, 
        warmup_steps_pct=warmup_steps_pct,
        total_steps=total_steps,
        warmup_start_factor=0.1,
        cosine_end_factor=0.1
    )
    
    # 记录学习率变化
    lrs = []
    extra_steps = 1000
    for step in range(1,total_steps+1+extra_steps):
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        
        if step==1 or step % 1000 == 0:
            print(f"Step {step}, Learning Rate: {current_lr:.6f}")
        # 在强化学习中，这里通常是与环境交互、更新模型的地方
        optimizer.step()
        # 更新学习率
        scheduler.step()
    
    # 可视化学习率变化（实际使用时可以注释掉）
    plt.figure(figsize=(10, 5))
    plt.plot(lrs)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Warmup + Cosine Annealing LR Schedule')
    plt.axvline(x=warmup_steps_pct*total_steps, color='r', linestyle='--', label='End of Warmup')
    plt.legend()
    plt.grid(True)
    plt.savefig('lr_schedule.png')
    plt.close()
    
    print("学习率调度完成")