#!/usr/bin/env python3
"""
FAISS Vector Store Module
Manages vector indexing and retrieval for medical triage cases
"""

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import pickle
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import for hybrid retrieval
from sklearn.feature_extraction.text import TfidfVectorizer

# Optional imports for hybrid retrieval
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    print("⚠️ rank-bm25 not available. Run: pip install rank-bm25")
    BM25Okapi = None
    BM25_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
    
    # Download necessary NLTK data (run once)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
        
except ImportError:
    print("⚠️ NLTK not available. Run: pip install nltk")
    word_tokenize = None
    stopwords = None
    NLTK_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    print("⚠️ CrossEncoder not available. Install sentence-transformers")
    CrossEncoder = None
    CROSS_ENCODER_AVAILABLE = False


def safe_numeric_convert(value, default=0.0):
    """
    Safely convert value to float, return default if conversion fails
    
    Args:
        value: Value to convert
        default: Default value if conversion fails (can be None)
        
    Returns:
        float or None: Converted value or default
    """
    if value is None or value == '' or pd.isna(value):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


class FAISSVectorStore:
    """FAISS Vector Store Manager with Hybrid Retrieval"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", 
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize FAISS vector store with hybrid retrieval capabilities
        
        Args:
            embedding_model: Model used for text embeddings
            cross_encoder_model: Cross encoder for reranking
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.faiss_index = None
        self.case_database = None
        self.scaler = None
        
        # Hybrid retrieval components
        self.bm25_index = None
        self.cross_encoder = None
        self.stop_words = None
        
        # Check component availability
        self.hybrid_available = BM25_AVAILABLE and NLTK_AVAILABLE
        
        if NLTK_AVAILABLE and stopwords:
            try:
                self.stop_words = set(stopwords.words('english'))
            except Exception:
                self.stop_words = set()
        else:
            self.stop_words = set()
        
        # Retrieval weights (tunable hyperparameters)
        self.text_weight = 0.7    # α - text similarity weight
        self.numeric_weight = 0.3  # β - numeric similarity weight
        self.bm25_weight = 0.3     # BM25 weight in hybrid retrieval
        self.vector_weight = 0.7   # Vector weight in hybrid retrieval
        
        print(f"✅ FAISS Vector Store initialized successfully")
        print(f"   Embedding model: {embedding_model}")
        print(f"   Hybrid retrieval available: {self.hybrid_available}")
        
        # Initialize cross encoder (lazy loading to avoid startup delays)
        if CROSS_ENCODER_AVAILABLE and CrossEncoder:
            try:
                self.cross_encoder = CrossEncoder(cross_encoder_model)
                print(f"   Cross encoder loaded successfully")
            except Exception as e:
                print(f"   ⚠️ Cross encoder loading failed: {e}")
                self.cross_encoder = None
        else:
            print(f"   ⚠️ Cross encoder not available")
    
    def prepare_case_database(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare case database"""
        
        print("📚 Preparing case database...")
        
        # Create comprehensive case descriptions
        data = data.copy()
        
        # Build complete case descriptions
        data['case_description'] = data.apply(self._create_case_description, axis=1)
        
        # Create numerical feature vectors (for similarity calculation), including time features
        numerical_features = ['age_at_visit', 'heartrate', 'sbp', 'dbp', 'temperature', 'o2sat', 'pain', 
                             'time_period']
        
        # Check which features exist
        available_features = [feat for feat in numerical_features if feat in data.columns]
        print(f"Available numerical features: {available_features}")
        
        # Data preprocessing with better error handling
        for feat in available_features:
            print(f"Processing feature: {feat}")
            original_count = len(data)
            
            # Convert to numeric, coerce errors to NaN
            data[feat] = pd.to_numeric(data[feat], errors='coerce')
            
            # Count how many values were coerced to NaN
            nan_count = data[feat].isna().sum()
            if nan_count > 0:
                print(f"  ⚠️ Warning: {nan_count}/{original_count} values in '{feat}' could not be converted to numeric")
                
                # Show some examples of problematic values for debugging
                if feat in data.columns:
                    try:
                        # Try to find the original data file using absolute path
                        project_root = os.path.join(os.path.dirname(__file__), '../../..')
                        data_path = os.path.join(project_root, "data_processing", "processed_data", "triage_with_keywords.csv")
                        
                        if os.path.exists(data_path):
                            original_data = pd.read_csv(data_path, low_memory=False)
                            if feat in original_data.columns:
                                problem_mask = data[feat].isna() & original_data[feat].notna()
                                if problem_mask.any():
                                    problem_examples = original_data.loc[problem_mask, feat].head(3).tolist()
                                    print(f"  Examples of problematic values: {problem_examples}")
                        else:
                            print(f"  Could not load original data for examples from: {data_path}")
                    except Exception as e:
                        print(f"  Could not show problematic value examples: {e}")
        
        # Intelligent missing value imputation with median + missing indicators
        print(f"🔧 Applying intelligent missing value imputation...")
        
        # Calculate medians for each feature
        medians = {}
        missing_indicators = {}
        
        for feat in available_features:
            # Calculate median (excluding NaN)
            median_val = data[feat].median()
            medians[feat] = median_val
            
            # Create missing indicator (1 if missing, 0 if present)
            missing_col = f"{feat}_missing"
            data[missing_col] = data[feat].isna().astype(int)
            missing_indicators[feat] = missing_col
            
            # Fill missing values with median
            missing_count = data[feat].isna().sum()
            if missing_count > 0:
                data[feat].fillna(median_val, inplace=True)
                print(f"  {feat}: filled {missing_count} missing values with median {median_val:.1f}")
            else:
                print(f"  {feat}: no missing values")
        
        # Store imputation parameters for later use
        self.imputation_params = {
            'medians': medians,
            'missing_indicators': missing_indicators
        }
        
        print(f"✅ Data preprocessing completed")
        print(f"   Total cases: {len(data)}")
        print(f"   Features processed: {available_features}")
        print(f"   Missing indicators added: {list(missing_indicators.values())}")
        
        # Update available features to include missing indicators
        all_features = available_features + list(missing_indicators.values())
        
        # Create numerical vectors with missing indicators
        if all_features:
            data['numerical_vector'] = data[all_features].apply(lambda x: x.values, axis=1)
        else:
            # If no features, create zero vector (16 dimensions: 8 base + 8 missing indicators)
            data['numerical_vector'] = data.apply(lambda x: np.zeros(16), axis=1)
        
        return data
    

    
    def _create_case_description(self, row: pd.Series) -> str:
        """Create textual description of medical case"""
        
        description = f"Chief Complaint: {row.get('chiefcomplaint', 'Unknown')}\n"
        description += f"Age: {row.get('age_at_visit', 'Unknown')} years\n"
        description += f"Gender: {row.get('gender', 'Unknown')}\n"
        
        # Vital signs
        description += "Vital Signs:\n"
        description += f"- Heart Rate: {row.get('heartrate', 'Unknown')} bpm\n"
        description += f"- Blood Pressure: {row.get('sbp', 'Unknown')}/{row.get('dbp', 'Unknown')} mmHg\n"
        description += f"- Temperature: {row.get('temperature', 'Unknown')} °F\n"
        description += f"- Oxygen Saturation: {row.get('o2sat', 'Unknown')}%\n"
        description += f"- Pain Score: {row.get('pain', 'Unknown')}/10\n"
        
        # Arrival method
        if 'arrival_transport' in row and pd.notna(row['arrival_transport']):
            description += f"Arrival Method: {row['arrival_transport']}\n"
        
        # Time period information
        if 'time_period' in row and pd.notna(row['time_period']):
            time_period_map = {0: "Night (0-6AM)", 1: "Morning (6AM-12PM)", 
                             2: "Afternoon (12PM-6PM)", 3: "Evening (6PM-12AM)"}
            time_period_name = time_period_map.get(int(row['time_period']), "Unknown")
            description += f"Arrival Time Period: {time_period_name}\n"
            
        # Symptom keywords
        if 'complaint_keywords' in row and pd.notna(row['complaint_keywords']):
            description += f"Symptom Keywords: {row['complaint_keywords']}\n"
            
        # Actual acuity (for training data)
        if 'acuity' in row and pd.notna(row['acuity']):
            description += f"Actual Triage Level: ESI {row['acuity']}\n"
            
        return description
    
    def build_or_load_faiss_index(self, case_database: pd.DataFrame, save_path: str = "faiss_index", 
                                 force_rebuild: bool = False):
        """Build or load FAISS index with intelligent caching mechanism"""
        
        # Convert to absolute path to avoid working directory issues
        if not os.path.isabs(save_path):
            # Get the directory of the current file (faiss_store.py)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to methods/llama_rag directory
            llama_rag_dir = os.path.dirname(current_dir)
            save_path = os.path.join(llama_rag_dir, save_path)
        
        index_file = f"{save_path}_faiss.index"
        metadata_file = f"{save_path}_metadata.pkl"
        
        print(f"🔍 Looking for index files:")
        print(f"   Index: {index_file}")
        print(f"   Metadata: {metadata_file}")
        
        # If force rebuild, skip loading
        if force_rebuild:
            print("🔄 Force rebuilding index...")
        elif os.path.exists(index_file) and os.path.exists(metadata_file):
            print(f"📖 Found existing index files, validating...")
            try:
                # Load and validate index
                if self._load_and_validate_index(save_path, case_database):
                    print(f"✅ Index validation passed, using cached index with {self.faiss_index.ntotal} cases")
                    print("💡 To rebuild index, set force_rebuild=True")
                    return True
                else:
                    print(f"⚠️ Index validation failed, data mismatch, rebuilding...")
            except Exception as e:
                print(f"⚠️ Index loading failed: {e}, rebuilding...")
        else:
            print("🔍 No index files found, building new index...")
        
        # If index doesn't exist, validation fails, or loading fails, rebuild
        print("🔍 Building new FAISS vector index...")
        
        # Generate text embeddings
        descriptions = case_database['case_description'].tolist()
        embeddings = self.embedding_model.encode(descriptions, show_progress_bar=True)
        
        # Ensure embeddings are float32 type
        embeddings = embeddings.astype('float32')
        
        # Standardize numerical features
        numerical_features = np.vstack(case_database['numerical_vector'].values)
        self.scaler = StandardScaler()
        numerical_features_scaled = self.scaler.fit_transform(numerical_features)
        
        # Ensure numerical features are float32 type
        numerical_features_scaled = numerical_features_scaled.astype('float32')
        
        # Combine text embeddings and numerical features
        combined_features = np.hstack([embeddings, numerical_features_scaled])
        
        # Ensure combined features are float32 type
        combined_features = combined_features.astype('float32')
        
        # Create FAISS index
        dimension = combined_features.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product similarity
        
        # Normalize features to use cosine similarity
        faiss.normalize_L2(combined_features)
        self.faiss_index.add(combined_features)
        
        # Save case_database
        self.case_database = case_database
        
        print(f"✅ FAISS index built successfully with {self.faiss_index.ntotal} cases")
        print(f"Feature dimension: {dimension}")
        
        # Build BM25 index for hybrid retrieval
        print(f"🔍 Building BM25 index for hybrid retrieval...")
        self._build_bm25_index(case_database)
        
        # Save index
        self._save_index(save_path, case_database, combined_features)
        
        return False  # Indicates newly built index
    
    def _build_bm25_index(self, case_database: pd.DataFrame):
        """Build BM25 index for keyword-based retrieval"""
        
        if not self.hybrid_available or not BM25Okapi:
            print(f"⚠️ BM25 index skipped: dependencies not available")
            self.bm25_index = None
            return
        
        try:
            # Tokenize case descriptions for BM25
            descriptions = case_database['case_description'].tolist()
            
            # Simple tokenization and preprocessing
            tokenized_corpus = []
            for desc in descriptions:
                tokens = self._tokenize_query(desc)
                tokenized_corpus.append(tokens)
            
            # Build BM25 index
            self.bm25_index = BM25Okapi(tokenized_corpus)
            
            print(f"✅ BM25 index built successfully")
            print(f"   Tokenized {len(tokenized_corpus)} documents")
            
        except Exception as e:
            print(f"⚠️ BM25 index building failed: {e}")
            self.bm25_index = None
    
    def _tokenize_query(self, query_text: str) -> List[str]:
        """Tokenize query text for BM25 search"""
        
        if not query_text:
            return []
        
        try:
            query_lower = query_text.lower()
            
            if NLTK_AVAILABLE and word_tokenize:
                tokens = word_tokenize(query_lower)
            else:
                # Fallback tokenization
                tokens = query_lower.split()
            
            # Filter tokens
            filtered_tokens = []
            for token in tokens:
                if (token.isalnum() and len(token) > 2 and 
                    (not self.stop_words or token not in self.stop_words)):
                    filtered_tokens.append(token)
            
            return filtered_tokens
            
        except Exception:
            # Ultimate fallback
            return query_text.lower().split()
    
    def _save_index(self, save_path: str, case_database: pd.DataFrame, features: np.ndarray):
        """Save FAISS index and related data"""
        
        index_file = f"{save_path}_faiss.index"
        metadata_file = f"{save_path}_metadata.pkl"
        
        faiss.write_index(self.faiss_index, index_file)
        
        # Save other necessary data
        save_data = {
            'case_database': case_database.to_dict('records'),
            'scaler': self.scaler,
            'feature_shape': features.shape
        }
        
        with open(metadata_file, 'wb') as f:
            pickle.dump(save_data, f)
            
        print(f"💾 Index saved to:")
        print(f"   Index: {index_file}")
        print(f"   Metadata: {metadata_file}")
    
    def load_index(self, save_path: str):
        """Load FAISS index"""
        
        index_file = f"{save_path}_faiss.index"
        metadata_file = f"{save_path}_metadata.pkl"
        
        self.faiss_index = faiss.read_index(index_file)
        
        with open(metadata_file, 'rb') as f:
            save_data = pickle.load(f)
            
        self.case_database = pd.DataFrame(save_data['case_database'])
        self.scaler = save_data['scaler']
        
        print(f"📖 Index loaded from:")
        print(f"   Index: {index_file}")
        print(f"   Cases: {self.faiss_index.ntotal}")
    
    def _load_and_validate_index(self, save_path: str, current_case_database: pd.DataFrame) -> bool:
        """Load and validate whether index matches current data"""
        
        try:
            # Load index
            self.load_index(save_path)
            
            # Validate data consistency
            if self.case_database is None:
                print("⚠️ Validation failed: loaded case_database is empty")
                return False
            
            # Check if case count matches (allow reasonable difference)
            cached_count = len(self.case_database)
            current_count = len(current_case_database)
            
            # Allow 1% difference to avoid rebuilding due to minor data processing differences
            tolerance = max(1000, int(cached_count * 0.01))  # At least 1000 cases tolerance
            diff = abs(cached_count - current_count)
            
            if diff > tolerance:
                print(f"⚠️ Validation failed: case count difference too large (cached: {cached_count}, current: {current_count}, diff: {diff}, tolerance: {tolerance})")
                return False
            elif diff > 0:
                print(f"💡 Minor count difference within tolerance (cached: {cached_count}, current: {current_count}, diff: {diff})")
                # Use cached data to avoid rebuilding
            
            # Check data integrity (loose overlap check)
            if 'subject_id' in current_case_database.columns and diff <= tolerance:
                sample_size = min(100, len(current_case_database), len(self.case_database))
                cached_ids = set(self.case_database.head(sample_size)['subject_id'].astype(str))
                current_ids = set(current_case_database.head(sample_size)['subject_id'].astype(str))
                
                # Calculate overlap rate, consider data mismatch only if overlap rate is too low
                overlap = len(cached_ids & current_ids)
                overlap_rate = overlap / sample_size if sample_size > 0 else 0
                
                if overlap_rate < 0.8:  # Require at least 80% overlap
                    print(f"⚠️ Validation failed: data content overlap rate too low ({overlap_rate:.1%}, expected >=80%)")
                    return False
                else:
                    print(f"✅ Data content validation passed: overlap rate {overlap_rate:.1%}")
            elif 'subject_id' not in current_case_database.columns:
                print("💡 Skipping data content validation: no subject_id field")
            
            # Check FAISS index status
            if self.faiss_index is None or self.faiss_index.ntotal == 0:
                print("⚠️ Validation failed: invalid FAISS index")
                return False
            
            # Check FAISS index count (use same tolerance)
            index_diff = abs(self.faiss_index.ntotal - cached_count)
            if index_diff > tolerance:
                print(f"⚠️ Validation failed: FAISS index count mismatch (index: {self.faiss_index.ntotal}, cached: {cached_count})")
                return False
            
            print(f"✅ Index validation passed: {cached_count} cases, FAISS index normal")
            return True
            
        except Exception as e:
            print(f"❌ Error during validation: {e}")
            return False
    
    def retrieve_similar_cases(self, query_case: pd.Series, k: int = 5, 
                              use_hybrid: bool = True, 
                              use_reranking: bool = True) -> Tuple[List[Dict], List[float]]:
        """
        Hybrid retrieval with BM25 + Vector Search + Cross-encoder reranking
        
        Args:
            query_case: Query case
            k: Number of final results to return
            use_hybrid: Whether to use hybrid BM25+Vector retrieval
            use_reranking: Whether to use cross-encoder reranking
            
        Returns:
            Tuple of similar cases and similarity scores
        """
        
        if self.faiss_index is None:
            raise ValueError("FAISS index not built, please call build_or_load_faiss_index method first")
            
        # Adaptive k: retrieve more candidates for hybrid retrieval
        retrieval_k = k * 2 if use_hybrid else k
        
        if use_hybrid and self.bm25_index is not None:
            return self._hybrid_retrieve(query_case, k, retrieval_k, use_reranking)
        else:
            return self._vector_retrieve(query_case, k, retrieval_k, use_reranking)
    
    def _process_query_features(self, query_case: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Process query case into embeddings and numerical features"""
        
        # Create query description
        query_description = self._create_case_description(query_case)
        query_embedding = self.embedding_model.encode([query_description])
        query_embedding = query_embedding.astype('float32')
        
        # Numerical features
        numerical_features = ['age_at_visit', 'heartrate', 'sbp', 'dbp', 'temperature', 'o2sat', 'pain', 'time_period']
        
        query_numerical_list = []
        query_missing_indicators = []
        
        for feat in numerical_features:
            value = query_case.get(feat, None)
            is_missing = False
            converted_value = None
            
            if value is None or value == '' or pd.isna(value):
                is_missing = True
            else:
                converted_value = safe_numeric_convert(value, default=None)
                if converted_value is None:
                    is_missing = True
            
            # Use imputation
            if is_missing:
                if hasattr(self, 'imputation_params') and feat in self.imputation_params['medians']:
                    converted_value = self.imputation_params['medians'][feat]
                else:
                    defaults = {
                        'age_at_visit': 50.0, 'heartrate': 80.0, 'sbp': 120.0, 'dbp': 80.0,
                        'temperature': 98.6, 'o2sat': 98.0, 'pain': 5.0, 'time_period': 1.0
                    }
                    converted_value = defaults.get(feat, 0.0)
            
            query_numerical_list.append(converted_value)
            query_missing_indicators.append(1 if is_missing else 0)
        
        # Combine features
        all_query_features = query_numerical_list + query_missing_indicators
        query_numerical = np.array(all_query_features, dtype=np.float64).reshape(1, -1)
        query_numerical_scaled = self.scaler.transform(query_numerical).astype('float32')
        
        return query_embedding, query_numerical_scaled
    
    def _vector_retrieve(self, query_case: pd.Series, k: int, retrieval_k: int, 
                        use_reranking: bool) -> Tuple[List[Dict], List[float]]:
        """Pure vector-based retrieval"""
        
        query_embedding, query_numerical_scaled = self._process_query_features(query_case)
        
        # Weighted combination of text and numerical similarity
        query_combined = np.hstack([
            query_embedding * self.text_weight,
            query_numerical_scaled * self.numeric_weight
        ])
        query_combined = query_combined.astype('float32')
        faiss.normalize_L2(query_combined)
        
        # FAISS retrieval
        similarities, indices = self.faiss_index.search(query_combined, retrieval_k)
        
        # Get candidates
        candidates = []
        for i, idx in enumerate(indices[0]):
            case = self.case_database.iloc[idx].to_dict()
            candidates.append((case, similarities[0][i]))
        
        # Reranking if enabled
        if use_reranking and self.cross_encoder is not None:
            candidates = self._rerank_candidates(query_case, candidates, k)
        else:
            candidates = candidates[:k]
        
        similar_cases = [case for case, score in candidates]
        scores = [float(score) for case, score in candidates]
        
        return similar_cases, scores
    
    def _hybrid_retrieve(self, query_case: pd.Series, k: int, retrieval_k: int, 
                        use_reranking: bool) -> Tuple[List[Dict], List[float]]:
        """Hybrid BM25 + Vector retrieval"""
        
        query_description = self._create_case_description(query_case)
        
        # BM25 retrieval
        bm25_candidates = self._bm25_retrieve(query_description, retrieval_k)
        
        # Vector retrieval - preserve real similarity scores
        vec_cases, vec_scores = self._vector_retrieve(query_case, retrieval_k, retrieval_k, False)
        vector_candidates = list(zip(vec_cases, vec_scores))
        
        # Merge and score candidates
        merged_candidates = self._merge_candidates(bm25_candidates, vector_candidates)
        
        # No reranking, just take top-k from merged candidates
        merged_candidates = merged_candidates[:k]
        
        similar_cases = [case for case, score in merged_candidates]
        scores = [float(score) for case, score in merged_candidates]
        
        return similar_cases, scores
    
    def _bm25_retrieve(self, query_text: str, k: int) -> List[Tuple[Dict, float]]:
        """BM25-based keyword retrieval"""
        
        if self.bm25_index is None:
            return []
        
        try:
            # Tokenize query
            query_tokens = self._tokenize_query(query_text)
            if not query_tokens:
                return []
            
            # BM25 search
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:k]
            
            candidates = []
            for idx in top_indices:
                if idx < len(self.case_database):
                    case = self.case_database.iloc[idx].to_dict()
                    score = float(scores[idx])
                    candidates.append((case, score))
            
            return candidates
            
        except Exception as e:
            print(f"⚠️ BM25 retrieval failed: {e}")
            return []
    
    def _minmax_normalize(self, scores: List[float]) -> List[float]:
        """Min-max normalization to [0,1] range, handling edge cases"""
        
        if not scores:
            return []
        
        scores = np.array(scores, dtype=np.float32)
        
        if len(scores) == 0:
            return scores.tolist()
        
        mn, mx = scores.min(), scores.max()
        
        if mx - mn < 1e-9:
            # All scores are almost the same: give 1 if >0, otherwise 0, avoid fake advantage of all 0.3/0.7
            return (np.ones_like(scores) if mx > 0 else np.zeros_like(scores)).tolist()
        
        # Min-max normalization
        normalized = (scores - mn) / (mx - mn)
        return normalized.astype(np.float32).tolist()
    
    def _merge_candidates(self, bm25_candidates: List[Tuple[Dict, float]], 
                         vector_candidates: List[Tuple[Dict, float]]) -> List[Tuple[Dict, float]]:
        """Merge BM25 and vector candidates with weighted scores using min-max normalization"""
        
        # Create unified candidate pool
        candidate_dict = {}
        
        # Add BM25 candidates
        for case, score in bm25_candidates:
            case_id = case.get('subject_id', str(hash(str(case))))
            if case_id not in candidate_dict:
                candidate_dict[case_id] = {'case': case, 'bm25_score': 0.0, 'vector_score': 0.0}
            candidate_dict[case_id]['bm25_score'] = score
        
        # Add vector candidates
        for case, score in vector_candidates:
            case_id = case.get('subject_id', str(hash(str(case))))
            if case_id not in candidate_dict:
                candidate_dict[case_id] = {'case': case, 'bm25_score': 0.0, 'vector_score': 0.0}
            candidate_dict[case_id]['vector_score'] = score
        
        # Min-max normalize scores to [0,1] range
        bm25_scores = [data['bm25_score'] for data in candidate_dict.values()]
        vector_scores = [data['vector_score'] for data in candidate_dict.values()]
        
        bm25_norm = self._minmax_normalize(bm25_scores)
        vec_norm = self._minmax_normalize(vector_scores)
        
        # Update normalized scores
        for (case_id, data), b, v in zip(candidate_dict.items(), bm25_norm, vec_norm):
            data['bm25_score'] = float(b)
            data['vector_score'] = float(v)
        
        # Weighted combination
        merged_candidates = []
        for data in candidate_dict.values():
            combined_score = (self.bm25_weight * data['bm25_score'] + 
                            self.vector_weight * data['vector_score'])
            merged_candidates.append((data['case'], combined_score))
        
        # Sort by combined score
        merged_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return merged_candidates
    
    def _rerank_candidates(self, query_case: pd.Series, candidates: List[Tuple[Dict, float]], 
                          k: int) -> List[Tuple[Dict, float]]:
        """Rerank candidates using cross-encoder"""
        
        if not candidates or self.cross_encoder is None:
            return candidates[:k]
        
        try:
            query_description = self._create_case_description(query_case)
            
            # Prepare pairs for cross-encoder
            pairs = []
            for case, _ in candidates:
                case_description = self._create_case_description(pd.Series(case))
                pairs.append([query_description, case_description])
            
            # Cross-encoder scoring
            cross_scores = self.cross_encoder.predict(pairs)
            
            # Rerank by cross-encoder scores
            reranked = list(zip([case for case, _ in candidates], cross_scores))
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            return reranked[:k]
            
        except Exception as e:
            print(f"⚠️ Cross-encoder reranking failed: {e}")
            return candidates[:k]
    
    def set_retrieval_weights(self, text_weight: float = 0.7, numeric_weight: float = 0.3,
                             bm25_weight: float = 0.3, vector_weight: float = 0.7):
        """
        Set retrieval weights for tuning performance
        
        Args:
            text_weight: Weight for text similarity (α)
            numeric_weight: Weight for numerical similarity (β)  
            bm25_weight: Weight for BM25 in hybrid retrieval
            vector_weight: Weight for vector search in hybrid retrieval
        """
        self.text_weight = text_weight
        self.numeric_weight = numeric_weight
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        
        print(f"🔧 Retrieval weights updated:")
        print(f"   Text weight (α): {text_weight}")
        print(f"   Numeric weight (β): {numeric_weight}")
        print(f"   BM25 weight: {bm25_weight}")
        print(f"   Vector weight: {vector_weight}")
    
    def adaptive_k_selection(self, query_case: pd.Series, base_k: int = 5) -> int:
        """
        Adaptive k selection based on query characteristics
        
        Args:
            query_case: Query case
            base_k: Base k value
            
        Returns:
            Adaptive k value
        """
        adaptive_k = base_k
        
        # Increase k for complex cases
        pain_score = query_case.get('pain', 0)
        if isinstance(pain_score, (int, float)) and pain_score > 7:
            adaptive_k += 2  # High pain cases need more examples
        
        # Increase k for edge cases (very young/old patients)
        age = query_case.get('age_at_visit', 50)
        if isinstance(age, (int, float)):
            if age < 18 or age > 80:
                adaptive_k += 1  # Pediatric or geriatric cases
        
        # Increase k for abnormal vitals
        hr = query_case.get('heartrate', 80)
        sbp = query_case.get('sbp', 120)
        
        if isinstance(hr, (int, float)) and (hr < 50 or hr > 120):
            adaptive_k += 1
        
        if isinstance(sbp, (int, float)) and (sbp < 90 or sbp > 160):
            adaptive_k += 1
        
        # Night time cases might need more examples
        time_period = query_case.get('time_period', 1)
        if time_period == 0:  # Night time
            adaptive_k += 1
        
        return min(adaptive_k, base_k * 2)  # Cap at 2x base_k