# 测试编写指南

## 🎯 单元测试 vs 集成测试

### 单元测试 (Unit Tests)

- **目标**: 测试单个函数或方法
- **特点**: 快速、独立、可重复
- **使用场景**: 业务逻辑、数据处理函数、工具方法
- **Mock 策略**: 隔离所有外部依赖

### 集成测试 (Integration Tests)

- **目标**: 测试组件间的交互
- **特点**: 更真实、较慢、端到端
- **使用场景**: 工作流程、模型集成、系统接口
- **Mock 策略**: 最小化 Mock，使用真实组件

## 📋 测试编写检查清单

### ✅ 单元测试检查清单

- [ ] 测试函数命名清晰 (`test_function_does_what`)
- [ ] 包含正常情况、边界情况、异常情况
- [ ] 使用 `@pytest.mark.unit` 标记
- [ ] Mock 所有外部依赖
- [ ] 每个测试只验证一个功能点
- [ ] 使用合适的断言 (`assert`, `pytest.raises`)
- [ ] 测试执行时间 < 1 秒

### ✅ 集成测试检查清单

- [ ] 测试完整的工作流程
- [ ] 使用 `@pytest.mark.integration` 标记
- [ ] 使用真实或接近真实的数据
- [ ] 测试组件间的接口
- [ ] 验证端到端的功能
- [ ] 包含错误处理和降级逻辑
- [ ] 可以标记为 `@pytest.mark.slow` 如果耗时

## 🛠️ 常用的测试模式

### 1. AAA 模式 (Arrange-Act-Assert)

```python
def test_function():
    # Arrange: 准备测试数据
    input_data = "test input"
    expected_output = "expected result"

    # Act: 执行被测试的功能
    result = function_under_test(input_data)

    # Assert: 验证结果
    assert result == expected_output
```

### 2. 参数化测试

```python
@pytest.mark.parametrize("input,expected", [
    ("input1", "output1"),
    ("input2", "output2"),
    ("input3", "output3"),
])
def test_multiple_cases(input, expected):
    assert function(input) == expected
```

### 3. 异常测试

```python
def test_function_raises_exception():
    with pytest.raises(ValueError, match="Invalid input"):
        function_that_should_fail("invalid_input")
```

### 4. Mock 使用

```python
@patch('module.external_dependency')
def test_with_mock(mock_dependency):
    mock_dependency.return_value = "mocked_value"
    result = function_using_dependency()
    assert result == "expected_result"
    mock_dependency.assert_called_once()
```

## 📊 测试数据管理

### Fixtures 使用

```python
@pytest.fixture
def sample_data():
    return {"key": "value", "numbers": [1, 2, 3]}

@pytest.fixture(scope="session")
def expensive_resource():
    # 只在测试会话开始时创建一次
    resource = create_expensive_resource()
    yield resource
    resource.cleanup()
```

### 测试数据文件

- 将测试数据放在 `tests/fixtures/` 目录
- 使用 JSON、CSV 等格式存储测试数据
- 避免在测试代码中硬编码大量数据

## 🚀 运行测试的方法

### 基本运行

```bash
# 运行所有测试
pytest

# 运行单元测试
pytest tests/unit/ -v

# 运行集成测试
pytest tests/integration/ -v

# 运行特定测试文件
pytest tests/unit/test_specific.py -v

# 运行特定测试方法
pytest tests/unit/test_file.py::TestClass::test_method -v
```

### 使用标记过滤

```bash
# 只运行单元测试
pytest -m unit

# 只运行集成测试
pytest -m integration

# 跳过慢速测试
pytest -m "not slow"

# 运行特定功能的测试
pytest -m "unit and preprocessing"
```

### 生成报告

```bash
# 生成覆盖率报告
pytest --cov=your_module --cov-report=html

# 生成JUnit报告
pytest --junit-xml=report.xml

# 并行运行测试
pytest -n auto
```

## 💡 测试编写技巧

### 1. 测试命名

- **好的命名**: `test_user_login_with_valid_credentials_returns_success`
- **不好的命名**: `test_login`, `test_user`

### 2. 断言消息

```python
# 提供清晰的错误消息
assert result > 0, f"Expected positive result, got {result}"
assert "expected_text" in response, f"Response missing expected text: {response}"
```

### 3. 测试数据

```python
# 使用有意义的测试数据
patient_data = {
    "age": 65,  # 老年患者
    "chiefcomplaint": "chest pain",  # 高优先级症状
    "temperature": 98.6  # 正常体温
}
```

### 4. 测试组织

```python
class TestUserAuthentication:
    """用户认证相关的所有测试"""

    def setup_method(self):
        """每个测试方法前的设置"""
        self.user = create_test_user()

    def test_valid_login(self):
        """测试有效登录"""
        pass

    def test_invalid_password(self):
        """测试无效密码"""
        pass
```

## ⚠️ 常见陷阱

### 1. 测试间的依赖

```python
# ❌ 错误：测试间有依赖
def test_create_user():
    global created_user_id
    created_user_id = create_user()

def test_update_user():
    update_user(created_user_id)  # 依赖上一个测试

# ✅ 正确：每个测试独立
def test_update_user():
    user_id = create_user()  # 每个测试创建自己的数据
    update_user(user_id)
```

### 2. 过度 Mock

```python
# ❌ 错误：Mock了太多东西，测试失去意义
@patch('module.function1')
@patch('module.function2')
@patch('module.function3')
def test_complex_workflow(mock1, mock2, mock3):
    # 测试变成了测试Mock而不是真实逻辑
    pass

# ✅ 正确：只Mock必要的外部依赖
@patch('module.external_api_call')
def test_workflow(mock_api):
    # 只Mock外部API调用，其他逻辑使用真实实现
    pass
```

### 3. 测试过于复杂

```python
# ❌ 错误：一个测试验证太多东西
def test_everything():
    # 测试数据处理
    # 测试模型预测
    # 测试结果保存
    # 测试通知发送
    pass

# ✅ 正确：每个测试专注一个方面
def test_data_processing():
    pass

def test_model_prediction():
    pass

def test_result_saving():
    pass
```

## 📈 测试质量指标

### 覆盖率目标

- **单元测试覆盖率**: > 80%
- **集成测试覆盖率**: > 60%
- **关键路径覆盖率**: 100%

### 性能指标

- **单元测试**: < 1 秒/测试
- **集成测试**: < 10 秒/测试
- **总测试时间**: < 5 分钟

### 质量指标

- **测试通过率**: > 95%
- **测试稳定性**: 无随机失败
- **测试维护性**: 易于理解和修改

