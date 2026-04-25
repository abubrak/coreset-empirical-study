# CSReL 原始实现分析报告

## 1. 核心算法概述

### 1.1 Reducible Loss 的定义

**关键公式**：
```
Reducible Loss (ReL) = Loss_current_model - Loss_reference_model
```

其中：
- `Loss_current_model`: 当前在子集上训练的模型对样本的损失
- `Loss_reference_model`: 在全量数据上训练的参考模型对样本的损失

**物理意义**：
- ReL 越高，说明该样本包含越多当前模型缺失的信息
- 参考模型代表"理想状态"，在全量数据上训练后损失较低
- 当前模型损失减去参考损失 = 可以通过添加该样本减少的损失

### 1.2 核心选择策略

```python
# 核心选择逻辑 (select_by_loss_diff)
loss_diff = loss_current_model - loss_reference_model
sorted_loss_diffs = sorted(loss_diffs.items(), key=lambda x: x[1], reverse=True)
# 选择 loss_diff 最大的样本
```

**关键点**：
1. **确定性选择**：直接选择 loss_diff 最大的样本，不使用随机采样
2. **增量式选择**：分多轮逐步添加样本
3. **每轮重新训练**：每次添加样本后重新训练模型

---

## 2. 原始实现核心代码分析

### 2.1 参考损失计算 (compute_loss_dic)

**位置**：`utils.py` 第 140-180 行

```python
def compute_loss_dic(ref_model, data_loader, aug_iters, use_cuda, loss_params):
    ref_model.eval()
    loss_fn = loss_functions.CompliedLoss(
        ce_factor=loss_params['ce_factor'],
        mse_factor=loss_params['mse_factor'],
        reduction='none'  # 关键：返回每个样本的损失，不是均值
    )
    loss_dic = {}
    with torch.no_grad():
        for i in range(aug_iters):  # 数据增强迭代次数
            for data in data_loader:
                # 计算损失
                loss = loss_fn(ref_model(sps), labs, logit)
                # 存储每个样本的损失
                for j in range(batch_size):
                    d_id = int(d_ids[j].numpy())
                    if d_id not in loss_dic:
                        loss_dic[d_id] = [loss[j]]
                    else:
                        loss_dic[d_id].append(loss[j])
    # 对多次增强求平均
    for d_id in loss_dic.keys():
        loss_dic[d_id] = float(np.mean(loss_dic[d_id]))
    return loss_dic
```

**关键点**：
- `aug_iters=1`：通常只使用 1 次增强迭代
- `reduction='none'`：返回每个样本的损失，不是批次均值
- 返回字典 `{sample_id: average_loss}`

### 2.2 样本选择 (select_by_loss_diff)

**位置**：`coreset_selection/coreset_selection_functions.py` 第 48-142 行

```python
def select_by_loss_diff(ref_loss_dic, rand_data, model, incremental_size,
                        transforms, on_cuda, loss_params, class_sizes=None):
    model.eval()
    loss_fn = loss_functions.CompliedLoss(
        ce_factor=loss_params['ce_factor'],
        mse_factor=loss_params['mse_factor'],
        reduction='none'
    )

    loss_diffs = {}
    # 批量计算当前模型损失
    with torch.no_grad():
        for i, di in enumerate(rand_data):
            # 计算当前模型损失
            loss = loss_fn(x=model(sps), y=labs, logits=lab_logits)
            # 计算损失差值
            loss_dif = float(loss[j] - ref_loss_dic[did])  # 核心公式
            loss_diffs[did] = loss_dif

    # 按损失差值降序排序
    sorted_loss_diffs = sorted(loss_diffs.items(), key=lambda x: x[1], reverse=True)

    # 确定性选择 Top-K
    selected_data = []
    for i in range(len(sorted_loss_diffs)):
        d_id = sorted_loss_diffs[i][0]
        # 类别平衡处理
        if class_sizes is not None:
            lab = int(di[2])
            if class_cnt[lab] == class_sizes[lab]:
                continue
            class_cnt[lab] += 1
        selected_data.append(new_di)
        if len(selected_data) == incremental_size:
            break

    return selected_data, id2loss_dif
```

**关键点**：
1. **确定性选择**：直接选择 loss_diff 最大的样本
2. **类别平衡**：支持按类别平衡选择
3. **批量计算**：使用 batch_size=32 提高效率
4. **无温度参数**：不使用 softmax 或温度缩放

### 2.3 增量选择流程 (incremental_selection)

**位置**：`coreset_selection/selection_agent.py` 第 103-267 行

```python
def incremental_selection(self, x, y, select_size, ...):
    # 1. 训练参考模型（全量数据）
    if self.ref_model is None:
        self.train_ref_model(x=x, y=y)

    # 2. 计算参考损失
    ref_loss_dic = utils.compute_loss_dic(
        ref_model=self.ref_model,
        data_loader=temp_loader,
        aug_iters=1,  # 通常使用 1
        use_cuda=self.train_params['use_cuda'],
        loss_params=ref_loss_params
    )

    # 3. 初始化模型和选择
    init_model = utils.build_model(model_params=self.model_params)

    # 4. 如果有初始大小，先随机选择
    if self.init_size > 0:
        init_ids = random.sample(full_ids, self.init_size)
        # 训练初始模型
        init_model = train_methods_for_selection.train_model(...)

    # 5. 增量选择循环
    while len(all_selected_ids) < select_size:
        incremental_size = max(int(select_size / self.selection_steps), 1)

        # 选择新样本
        selected_data, _ = coreset_selection_functions.select_by_loss_diff(
            ref_loss_dic=ref_loss_dic,
            rand_data=rand_data,
            model=init_model,
            incremental_size=incremental_size,
            ...
        )

        # 添加到训练集
        coreset_selection_functions.add_new_data(
            data_file=self.cur_train_file,
            new_data=selected_data
        )

        # 重新训练模型
        init_model = train_methods_for_selection.train_model(...)
```

**关键点**：
1. **参考模型固定**：参考模型只在开始时训练一次，后续不再更新
2. **增量式选择**：分多轮选择，每轮选择 `select_size / selection_steps` 个样本
3. **每轮重新训练**：每次添加样本后重新训练当前模型
4. **类别平衡**：支持 class_balance 模式

### 2.4 损失函数 (CompliedLoss)

**位置**：`functions/loss_functions.py`

```python
class CompliedLoss(torch.nn.Module):
    def __init__(self, ce_factor, mse_factor, reduction='mean', kd_mode='mse'):
        super(CompliedLoss, self).__init__()
        self.reduction = reduction
        self.ce_factor = ce_factor
        self.mse_factor = mse_factor
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction=reduction)
        if self.kd_mode == 'mse':
            self.mse_loss = torch.nn.MSELoss(reduction=reduction)

    def forward(self, x, y, logits=None):
        loss_c = self.ce_factor * self.ce_loss(x, y)
        if self.mse_factor > 0 and logits is not None:
            loss_m = self.mse_loss(x, logits)
            if self.reduction == 'none':
                loss_m = torch.mean(loss_m, dim=-1)
            loss = self.ce_factor * loss_c + self.mse_factor * loss_m
            return loss
        else:
            return self.ce_factor * loss_c
```

**关键点**：
- 支持 CE loss 和 MSE loss（知识蒸馏）的组合
- `reduction='none'` 时返回每个样本的损失
- 当使用 MSE 时，会对 logits 的维度求平均

---

## 3. 与我们实现的对比

### 3.1 主要差异

| 方面 | 原始实现 | 我们的实现 | 状态 |
|------|---------|-----------|------|
| **损失计算** | `loss_current - loss_reference` | 仅使用 `loss_current` | ❌ 错误 |
| **选择策略** | 确定性选择 Top-K | Softmax + 随机采样 | ❌ 错误 |
| **温度参数** | 无 | 有 temperature=1.0 | ⚠️ 不必要 |
| **参考模型** | 在全量数据上训练一次 | 未实现 | ❌ 缺失 |
| **增量选择** | 分多轮，每轮重训练 | 单次选择 | ❌ 不完整 |
| **类别平衡** | 支持 | 未实现 | ⚠️ 可选 |
| **权重归一化** | 无权重概念 | 有权重归一化 | ⚠️ 不必要 |

### 3.2 关键 Bug 识别

#### Bug 1: 错误的 Reducible Loss 定义

**我们的实现**：
```python
def _compute_reducible_losses(self, dataset, model):
    # 使用当前模型在样本上的损失作为代理
    logits = model(batch_x)
    probs = torch.softmax(logits, dim=1)
    loss = -probs[range(len(batch_y)), batch_y].log()
    return loss  # ❌ 这只是普通的交叉熵损失
```

**正确实现**：
```python
# 需要 reference model
def compute_reducible_loss(current_model, ref_model, dataset):
    ref_loss = ref_model(dataset)  # 在全量数据上训练的模型
    cur_loss = current_model(dataset)  # 在子集上训练的模型
    return cur_loss - ref_loss  # ✓ 正确的 reducible loss
```

#### Bug 2: 错误的选择策略

**我们的实现**：
```python
# 使用 Softmax 将可约损失转换为采样概率
probs = torch.softmax(rel_losses / self.temperature, dim=0)
# 随机采样
selected_indices = torch.multinomial(probs, num_samples=self.memory_budget, replacement=False)
```

**正确实现**：
```python
# 直接选择 Top-K，确定性选择
sorted_loss_diffs = sorted(loss_diffs.items(), key=lambda x: x[1], reverse=True)
selected_data = sorted_loss_diffs[:select_size]  # ✓ 直接选择最大的
```

#### Bug 3: 缺少参考模型

**我们的实现**：
```python
def __init__(self, memory_budget: int, device=None, temperature: float = 1.0):
    super().__init__(memory_budget, device)
    self.temperature = temperature  # ❌ 不需要温度参数
    # ❌ 没有参考模型
```

**正确实现**：
```python
def __init__(self, memory_budget, device, ref_model=None):
    super().__init__(memory_budget, device)
    self.ref_model = ref_model  # ✓ 需要参考模型
```

#### Bug 4: 缺少增量选择流程

**我们的实现**：
```python
def select_coreset(self, dataset, model, task_id, previous_coresets=None):
    # 单次选择所有样本
    rel_losses = self._compute_reducible_losses(dataset, model)
    selected_indices = ...
    return selected_indices, weights  # ❌ 一次性选择
```

**正确实现**：
```python
def incremental_selection(self, x, y, select_size, selection_steps):
    # 分多轮选择
    while len(all_selected_ids) < select_size:
        incremental_size = max(int(select_size / selection_steps), 1)
        # 选择一小批
        selected_data = select_by_loss_diff(...)
        # 重新训练模型
        model = train_model(model, selected_data)
        # 继续下一轮
```

### 3.3 实现细节差异

#### 损失计算

**原始实现**：
```python
# 使用 CompliedLoss，支持 CE + MSE
loss_fn = loss_functions.CompliedLoss(
    ce_factor=1.0,
    mse_factor=0.0,
    reduction='none'  # 关键
)
loss = loss_fn(model(sps), labs, logit)
```

**我们的实现**：
```python
# 手动计算
logits = model(batch_x)
probs = torch.softmax(logits, dim=1)
loss = -probs[range(len(batch_y)), batch_y].log()
```
⚠️ 功能相同，但原始实现更通用

#### 批量处理

**原始实现**：
```python
# 使用 batch_size=32 提高效率
batch_size = 32
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    loss = model(batch)
```

**我们的实现**：
```python
# 逐个样本计算
for idx in indices:
    x, y = dataset.dataset[idx]
    loss = model(x)
```
⚠️ 效率较低

---

## 4. 核心算法总结

### 4.1 CSReL 算法伪代码

```
Input: 全量数据 D_full, 核心集大小 M, 选择步数 S
Output: 核心集 C

1. 训练参考模型 M_ref 在 D_full 上
2. 计算 L_ref = Loss(M_ref, D_full)

3. 初始化: 随机选择 C_init (init_size 个样本)
4. 训练当前模型 M_cur 在 C_init 上

5. for s = 1 to S:
6.     计算 L_cur = Loss(M_cur, D_full)
7.
8.     for each sample x in D_full \ C:
9.         ReL(x) = L_cur(x) - L_ref(x)  # Reducible Loss
10.
11.    # 选择 ReL 最大的样本
12.    C_new = Top-K(ReL, k=M/S)
13.
14.    # 更新核心集
15.    C = C ∪ C_new
16.
17.    # 重新训练模型
18.    M_cur = Train(M_cur, C)
19.
20. return C
```

### 4.2 关键特性

1. **确定性选择**：直接选择 ReL 最大的样本，不使用随机采样
2. **增量式**：分多轮选择，每轮选择少量样本
3. **动态更新**：每轮重新训练模型，更新当前损失估计
4. **参考固定**：参考模型在开始时训练一次，后续不再更新
5. **类别平衡**：支持按类别平衡选择（可选）

---

## 5. 修复建议

### 5.1 高优先级修复

1. **实现参考模型**
   ```python
   def train_reference_model(self, full_dataset):
       # 在全量数据上训练
       model = build_model()
       train(model, full_dataset, epochs=ref_epochs)
       return model
   ```

2. **修正 Reducible Loss 计算**
   ```python
   def compute_reducible_loss(self, dataset, current_model, ref_model):
       ref_loss = compute_loss(ref_model, dataset)
       cur_loss = compute_loss(current_model, dataset)
       return cur_loss - ref_loss
   ```

3. **改为确定性选择**
   ```python
   # 移除 softmax 和随机采样
   sorted_indices = torch.argsort(rel_losses, descending=True)
   selected_indices = sorted_indices[:self.memory_budget].tolist()
   ```

4. **实现增量选择**
   ```python
   def incremental_selection(self, dataset, select_size, selection_steps):
       incremental_size = select_size // selection_steps
       for step in range(selection_steps):
           # 选择一批
           batch = self.select_batch(dataset, incremental_size)
           # 添加到核心集
           self.coreset.extend(batch)
           # 重新训练
           self.model = train(self.model, self.coreset)
   ```

### 5.2 中优先级优化

1. **移除温度参数**：原始实现没有温度参数
2. **添加类别平衡**：支持 class_balance 选项
3. **优化批量计算**：使用 batch_size 提高效率
4. **添加权重支持**：如果需要，可以添加样本权重

### 5.3 低优先级增强

1. **支持知识蒸馏**：添加 MSE loss 支持
2. **添加早停机制**：防止过拟合
3. **支持数据增强**：在计算损失时使用 aug_iters

---

## 6. 代码片段示例

### 6.1 完整的选择函数

```python
def select_by_loss_diff(ref_loss_dic, rand_data, model, incremental_size,
                        transforms, on_cuda, loss_params, class_sizes=None):
    """
    基于损失差值选择样本

    Args:
        ref_loss_dic: 参考模型损失字典 {sample_id: loss}
        rand_data: 候选样本列表
        model: 当前模型
        incremental_size: 本次选择的样本数
        transforms: 数据变换
        on_cuda: 是否使用 CUDA
        loss_params: 损失参数
        class_sizes: 类别大小限制（可选）

    Returns:
        selected_data: 选中的样本列表
        id2loss_dif: 损失差值字典
    """
    model.eval()
    loss_fn = CompliedLoss(
        ce_factor=loss_params['ce_factor'],
        mse_factor=loss_params['mse_factor'],
        reduction='none'
    )

    loss_diffs = {}
    id2pos = {}
    id2logits = {}

    # 批量计算当前模型损失
    batch_ids = []
    batch_sps = []
    batch_labs = []

    with torch.no_grad():
        for i, di in enumerate(rand_data):
            d_id, sp, lab = di[:3]
            logit = di[3] if len(di) > 3 else None

            id2pos[d_id] = i
            batch_ids.append(d_id)
            batch_sps.append(sp)
            batch_labs.append(lab)

            # 每 32 个样本计算一次
            if len(batch_ids) == 32 or i == len(rand_data) - 1:
                sps = torch.stack(batch_sps, dim=0)
                labs = torch.tensor(batch_labs, dtype=torch.long)

                if on_cuda:
                    sps = sps.cuda()
                    labs = labs.cuda()

                # 计算损失
                loss = loss_fn(model(sps), labs, None)
                loss = loss.cpu().numpy()

                # 计算损失差值
                for j, did in enumerate(batch_ids):
                    loss_dif = float(loss[j] - ref_loss_dic[did])
                    loss_diffs[did] = loss_dif

                batch_ids.clear()
                batch_sps.clear()
                batch_labs.clear()

    # 按损失差值降序排序
    sorted_loss_diffs = sorted(loss_diffs.items(), key=lambda x: x[1], reverse=True)

    # 选择 Top-K
    selected_data = []
    id2loss_dif = {}
    class_cnt = {}

    if class_sizes is not None:
        for ci in class_sizes.keys():
            class_cnt[ci] = 0

    for d_id, loss_diff in sorted_loss_diffs:
        pos = id2pos[d_id]
        di = rand_data[pos]

        # 类别平衡检查
        if class_sizes is not None:
            lab = int(di[2])
            if class_cnt[lab] >= class_sizes[lab]:
                continue
            class_cnt[lab] += 1

        selected_data.append(copy.deepcopy(di))
        id2loss_dif[d_id] = loss_diff

        if len(selected_data) == incremental_size:
            break

    return selected_data, id2loss_dif
```

### 6.2 增量选择主循环

```python
def incremental_selection(self, x, y, select_size, selection_steps):
    """
    增量式选择核心集

    Args:
        x: 数据特征
        y: 数据标签
        select_size: 目标核心集大小
        selection_steps: 选择步数
    """
    # 1. 训练参考模型
    if self.ref_model is None:
        self.train_ref_model(x, y)

    # 2. 计算参考损失
    ref_loss_dic = compute_loss_dic(
        self.ref_model,
        make_data_loader(x, y),
        aug_iters=1,
        use_cuda=self.use_cuda,
        loss_params={'ce_factor': 1.0, 'mse_factor': 0.0}
    )

    # 3. 初始化
    all_selected_ids = set()
    incremental_size = max(int(select_size / selection_steps), 1)

    # 4. 如果有初始大小，先随机选择
    if self.init_size > 0:
        init_ids = random.sample(list(range(len(x))), self.init_size)
        all_selected_ids.update(init_ids)
        # 训练初始模型
        self.current_model = train_model(
            self.current_model,
            make_subset(x, y, init_ids)
        )

    # 5. 增量选择循环
    while len(all_selected_ids) < select_size:
        # 候选样本池
        id_pool = set(range(len(x))) - all_selected_ids

        # 选择新样本
        rand_data = make_subset(x, y, id_pool)
        selected_data, _ = select_by_loss_diff(
            ref_loss_dic,
            rand_data,
            self.current_model,
            min(incremental_size, select_size - len(all_selected_ids)),
            self.transforms,
            self.use_cuda,
            {'ce_factor': 1.0, 'mse_factor': 0.0}
        )

        # 添加到核心集
        for di in selected_data:
            all_selected_ids.add(int(di[0]))

        # 重新训练模型
        self.current_model = train_model(
            self.current_model,
            make_subset(x, y, all_selected_ids)
        )

        print(f'Selected {len(all_selected_ids)}/{select_size} samples')

    return list(all_selected_ids)
```

---

## 7. 总结

### 7.1 核心发现

1. **Reducible Loss 的正确理解**
   - 不是简单的"当前损失"
   - 是"当前模型损失 - 参考模型损失"的差值
   - 参考模型代表全量数据训练的"理想模型"

2. **选择策略**
   - 确定性选择 Top-K，不是随机采样
   - 不需要温度参数
   - 不需要 softmax

3. **增量式选择**
   - 分多轮选择，每轮选择少量样本
   - 每轮重新训练模型，更新损失估计
   - 参考模型固定不变

4. **实现细节**
   - 使用 `reduction='none'` 获取每个样本的损失
   - 批量计算提高效率
   - 支持类别平衡（可选）

### 7.2 需要修复的关键问题

1. ❌ **实现参考模型训练**
2. ❌ **修正 Reducible Loss 计算**
3. ❌ **改为确定性选择**
4. ❌ **实现增量选择流程**
5. ⚠️ **移除不必要的温度参数**
6. ⚠️ **优化批量计算效率**
7. ⚠️ **添加类别平衡支持**

### 7.3 修复优先级

**高优先级**（核心功能错误）：
1. 实现参考模型
2. 修正 Reducible Loss 计算
3. 改为确定性选择
4. 实现增量选择

**中优先级**（性能优化）：
1. 优化批量计算
2. 添加类别平衡
3. 移除温度参数

**低优先级**（可选功能）：
1. 支持知识蒸馏
2. 添加数据增强
3. 添加早停机制

---

## 附录：原始实现文件路径

- `/tmp/CSReL-Coreset-CL/coreset_selection/selection_agent.py` - 主要选择代理
- `/tmp/CSReL-Coreset-CL/coreset_selection/coreset_selection_functions.py` - 核心选择函数
- `/tmp/CSReL-Coreset-CL/utils.py` - 工具函数（包含 compute_loss_dic）
- `/tmp/CSReL-Coreset-CL/functions/loss_functions.py` - 损失函数
- `/tmp/CSReL-Coreset-CL/coreset_selection/train_methods_for_selection.py` - 训练方法
