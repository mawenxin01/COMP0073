# Llama RAG Experiments

This directory contains comprehensive experimental framework for the Llama RAG triage system.

## 📁 Files Overview

- `llama_rag_triage.py` - Main Llama RAG system with enhanced reporting
- `experiments.py` - Comprehensive experiment framework
- `run_experiments.py` - Simple experiment runner
- `README.md` - This file

## 🧪 Experiment Parameters

### Retrieval Parameters

- **k**: Number of similar cases to retrieve (3, 5, 7, 10)
- **use_hybrid**: Use hybrid BM25 + vector retrieval (True/False)
- **use_reranking**: Use cross-encoder reranking (True/False)
- **embedding_model**: Embedding model for text representation
  - `all-MiniLM-L6-v2` (default)
  - `all-mpnet-base-v2`
  - `all-MiniLM-L12-v2`
- **similarity_threshold**: Minimum similarity threshold (0.0)

### Generation Parameters

- **llama_model**: Llama model name (`meta-llama/llama-3-3-70b-instruct`)
- **temperature**: Generation temperature (0.05, 0.1, 0.2, 0.3)
- **max_tokens**: Maximum tokens to generate (300, 500, 700)
- **top_p**: Top-p sampling (0.9)
- **max_retries**: Maximum retry attempts (3)
- **retry_delay**: Retry delay in seconds (1.0)

### Evaluation Parameters

- **test_size**: Number of test samples (100, 500, 1000)
- **save_results**: Save evaluation results (True/False)
- **save_index**: Save FAISS index (True/False)
- **force_rebuild_index**: Force rebuild index (True/False)
- **evaluation_mode**: "single" or "multi"

## 🚀 Quick Start

### Option 1: Simple Experiment Runner

```bash
cd medical_triage_assistant/methods/llama_rag
python run_experiments.py
```

Choose from:

1. Baseline experiments (vector vs hybrid vs rerank)
2. K value experiments (k=3,5,7,10)
3. Temperature experiments (temp=0.05,0.1,0.2,0.3)
4. Custom experiment
5. Run all experiments

### Option 2: Custom Experiments

```python
from experiments import LlamaRAGExperiments, ExperimentConfig, RetrievalConfig, GenerationConfig, EvaluationConfig

# Initialize experiments manager
experiments = LlamaRAGExperiments(
    data_path="../data_processing/processed_data/triage_with_keywords.csv",
    results_dir="my_experiments"
)

# Create custom experiment
config = ExperimentConfig(
    experiment_name="my_experiment",
    retrieval=RetrievalConfig(k=7, use_hybrid=True, use_reranking=False),
    generation=GenerationConfig(temperature=0.1, max_tokens=500),
    evaluation=EvaluationConfig(test_size=1000),
    description="My custom experiment"
)

# Run experiment
result = experiments.run_single_experiment(config)
```

### Option 3: Full Experiment Suite

```python
from experiments import LlamaRAGExperiments

# Run all predefined experiments
experiments = LlamaRAGExperiments(data_path="path/to/data.csv")
results = experiments.run_experiment_suite()
```

## 📊 Predefined Experiments

### 1. Baseline Experiments

- **vector_only**: Vector-only retrieval baseline
- **hybrid_only**: Hybrid BM25 + vector retrieval
- **hybrid_rerank**: Hybrid retrieval with reranking

### 2. K Value Experiments

- **k_3**: k=3 similar cases
- **k_5**: k=5 similar cases
- **k_7**: k=7 similar cases
- **k_10**: k=10 similar cases

### 3. Temperature Experiments

- **temp_0.05**: Temperature=0.05
- **temp_0.1**: Temperature=0.1
- **temp_0.2**: Temperature=0.2
- **temp_0.3**: Temperature=0.3

### 4. Embedding Model Experiments

- **embedding_all-MiniLM-L6-v2**: Default embedding model
- **embedding_all-mpnet-base-v2**: MPNet embedding model
- **embedding_all-MiniLM-L12-v2**: Larger MiniLM model

### 5. Generation Experiments

- **max_tokens_300**: Max tokens=300
- **max_tokens_500**: Max tokens=500
- **max_tokens_700**: Max tokens=700

### 6. Ablation Studies

- **no_similar_cases**: No similar cases provided

## 📈 Output Files

Each experiment generates:

### 1. Results Files

- `results_YYYYMMDD_HHMMSS.json` - Complete experiment results
- `config_YYYYMMDD_HHMMSS.json` - Experiment configuration
- `performance_report_YYYYMMDD_HHMMSS.csv` - Formatted performance table
- `detailed_analysis_YYYYMMDD_HHMMSS.csv` - Detailed analysis with TP/FP/FN

### 2. Visualizations

- `confusion_matrix_llama_rag.png` - Confusion matrix heatmap
- `confusion_matrix_llama_rag_vector_only.png` - Vector-only confusion matrix
- `confusion_matrix_llama_rag_hybrid_only.png` - Hybrid confusion matrix
- `confusion_matrix_llama_rag_hybrid_rerank.png` - Hybrid+rerank confusion matrix

### 3. Summary Files

- `experiment_summary_YYYYMMDD_HHMMSS.json` - Summary of all experiments
- Comparison tables and analysis

## 📋 Example Output

### Performance Table (Table 5.4 style)

```
📊 Table 5.4: Per-class performance on the held-out test set (Scheme A, Llama RAG)
================================================================================
ESI Level     Precision  Recall     F1
--------------------------------------------------------------------------------
ESI 1         0.74       0.43       0.54
ESI 2         0.69       0.60       0.64
ESI 3         0.71       0.85       0.77
ESI 4         0.59       0.22       0.32
ESI 5         0.25       0.05       0.08
Accuracy                              0.70
Macro Avg     0.60       0.43       0.47
Weighted Avg  0.70       0.70       0.69
================================================================================
```

### Confusion Matrix Summary

```
📊 Confusion Matrix Summary:
==================================================
Total samples: 1,000
Correct predictions: 700
Accuracy: 0.700

Per-class correct predictions:
ESI 1: 38 correct out of 38 total
ESI 2: 321 correct out of 321 total
ESI 3: 570 correct out of 570 total
ESI 4: 67 correct out of 67 total
ESI 5: 4 correct out of 4 total
```

## 🔧 Customization

### Adding New Experiments

```python
# Add new experiment type
configs.append(ExperimentConfig(
    experiment_name="new_experiment",
    retrieval=RetrievalConfig(k=5, use_hybrid=True),
    generation=GenerationConfig(temperature=0.15),
    evaluation=EvaluationConfig(test_size=500),
    description="New experiment description"
))
```

### Modifying Parameters

```python
# Modify retrieval parameters
retrieval_config = RetrievalConfig(
    k=10,  # More similar cases
    use_hybrid=True,
    use_reranking=True,
    embedding_model="all-mpnet-base-v2"
)

# Modify generation parameters
generation_config = GenerationConfig(
    temperature=0.2,  # Higher creativity
    max_tokens=700,  # Longer responses
    top_p=0.95
)
```

## 📝 Notes

1. **Data Path**: Ensure the data path points to the correct CSV file
2. **API Configuration**: Make sure IBM Watson API is properly configured
3. **Rate Limiting**: Experiments include delays between runs to avoid rate limiting
4. **Results Directory**: Each experiment type gets its own results directory
5. **Error Handling**: Failed experiments are logged with error messages

## 🎯 Best Practices

1. **Start Small**: Begin with small test_size (100) for quick validation
2. **Baseline First**: Always run baseline experiments first
3. **Parameter Sweep**: Use systematic parameter sweeps
4. **Save Everything**: Enable save_results=True for all experiments
5. **Monitor Resources**: Large experiments can be resource-intensive

