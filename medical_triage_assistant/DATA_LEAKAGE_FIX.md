# 数据泄漏修复方案

## 🚨 问题确认

当前系统存在严重的数据泄漏问题：

- FAISS 索引包含 189,780 个案例（全量数据）
- 测试时检索到的"相似案例"可能包含测试集数据
- 导致模型性能被人为高估

## 🔧 修复步骤

### 1. 修改前端应用的 FAISS 索引使用

**文件**: `triage_frontend/app.py`

```python
# 当前问题代码（第139-147行）：
train_data, _ = train_test_split(data, test_size=0.2, random_state=42, stratify=data['acuity'])
index_path = "../../shared_rag_train_index"  # 这个索引包含全量数据！
llama_rag_system.initialize_system(train_data, save_path=index_path)

# 修复方案：
train_data, _ = train_test_split(data, test_size=0.2, random_state=42, stratify=data['acuity'])
# 使用只包含训练数据的索引
train_only_index_path = "../../train_only_rag_index"
llama_rag_system.initialize_system(train_data, save_path=train_only_index_path, force_rebuild=True)
```

### 2. 修改评估流程确保数据分离

**文件**: `llama_rag/llama_rag_triage.py`

```python
def run_full_pipeline(self, data_path: str, test_size: int = 100, force_rebuild: bool = False):
    # 1. 加载数据
    data = pd.read_csv(data_path, low_memory=False)
    data = data[data["complaint_keywords"].notna()].copy()

    # 2. 分割训练和测试数据（使用固定随机种子确保一致性）
    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data['acuity']
    )

    # 3. ⚠️ 关键修复：使用只包含训练数据的索引名称
    train_only_index_path = f"train_only_rag_index_{len(train_data)}"  # 加入训练数据量避免混淆

    # 4. 初始化系统（只使用训练数据，强制重建确保不使用旧的全量索引）
    self.initialize_system(train_data, save_path=train_only_index_path, force_rebuild=True)

    # 5. 评估系统（测试数据完全分离）
    evaluation_results = self.evaluate_system(test_data, sample_size=test_size)
```

### 3. 创建严格的数据验证机制

**文件**: `vector_store/faiss_store.py`

```python
def build_or_load_faiss_index(self, case_database: pd.DataFrame, save_path: str = "faiss_index",
                             force_rebuild: bool = False, validate_separation: bool = True):
    """构建FAISS索引，包含数据分离验证"""

    if validate_separation:
        # 验证当前数据不包含测试集标识
        self._validate_no_test_data_leakage(case_database)

    # ... 现有索引构建逻辑

def _validate_no_test_data_leakage(self, case_database: pd.DataFrame):
    """验证数据集中不包含测试数据（基于数据量和分布检查）"""
    total_cases = len(case_database)

    # 检查是否接近全量数据（189,780个案例）
    if total_cases > 180000:  # 如果超过18万，可能包含测试数据
        raise ValueError(f"⚠️ 数据泄漏风险：当前数据量({total_cases})接近全量，可能包含测试集")

    # 验证应该是大约80%的数据（约151,824个案例）
    expected_train_size = int(189780 * 0.8)
    if abs(total_cases - expected_train_size) > 1000:
        print(f"⚠️ 警告：训练数据量({total_cases})与预期({expected_train_size})差异较大")
```

### 4. 重新评估系统性能

修复后需要重新运行评估：

```bash
# 删除可能存在数据泄漏的旧索引
rm shared_rag_train_index*
rm azure_rag_train_index*

# 重新运行评估
cd methods/llama_rag
python llama_rag_triage.py --force-rebuild
```

## 📊 预期影响

修复数据泄漏后，预期性能指标会下降：

- **当前性能可能被高估了 10-20%**
- **真实性能**应该通过严格分离的测试集来评估
- **特别是高分诊等级的召回率**可能会显著下降

## ✅ 验证清单

修复完成后验证：

- [ ] FAISS 索引只包含训练数据（约 151,824 个案例）
- [ ] 测试过程中检索的相似案例全部来自训练集
- [ ] 重新评估的性能指标更加真实
- [ ] 测试集完全独立，未被系统访问过

