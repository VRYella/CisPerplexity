import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import re
from collections import Counter
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="CisPerplexity: Promoter Prediction Tool",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class PerplexityCalculator:
    """Calculate dinucleotide perplexity for DNA sequences"""
    
    @staticmethod
    def calculate_perplexity(sequence: str, window_size: int = 10) -> np.ndarray:
        """
        Calculate dinucleotide perplexity for a DNA sequence using sliding windows
        
        Args:
            sequence: DNA sequence string
            window_size: Size of sliding window
            
        Returns:
            Array of perplexity values for each window position
        """
        seq_len = len(sequence)
        if seq_len < window_size:
            return np.array([])
        
        num_windows = seq_len - window_size + 1
        perplexities = np.zeros(num_windows)
        
        # Generate all dinucleotides in the sequence
        dinucleotides = [sequence[i:i + 2] for i in range(seq_len - 1)]
        
        for i in range(num_windows):
            # Get dinucleotides in current window
            window_dinucleotides = dinucleotides[i:i + window_size - 1]
            
            # Count dinucleotide frequencies
            dinucleotide_counts = Counter(window_dinucleotides)
            total_dinucleotides = sum(dinucleotide_counts.values())
            
            if total_dinucleotides == 0:
                perplexities[i] = 0
                continue
            
            # Calculate probabilities
            probabilities = np.array(list(dinucleotide_counts.values())) / total_dinucleotides
            
            # Calculate entropy
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Add small epsilon to avoid log(0)
            
            # Calculate perplexity
            perplexities[i] = 2 ** entropy
        
        return perplexities
    
    @staticmethod
    def calculate_gc_content(sequence: str, window_size: int = 10) -> np.ndarray:
        """Calculate GC content using sliding windows"""
        seq_len = len(sequence)
        if seq_len < window_size:
            return np.array([])
        
        num_windows = seq_len - window_size + 1
        gc_percentages = np.zeros(num_windows)
        
        for i in range(num_windows):
            window = sequence[i:i + window_size]
            gc_count = window.count('G') + window.count('C')
            gc_percentages[i] = (gc_count / window_size) * 100
        
        return gc_percentages

class StructuralFeatureEncoder:
    """Encode DNA sequences using structural features"""
    
    def __init__(self, encoding_dict_path: str = None):
        """Initialize with encoding dictionary"""
        if encoding_dict_path:
            try:
                with open(encoding_dict_path, 'r') as f:
                    self.encoding_dict = json.load(f)
            except:
                # Fallback to default if file not found
                self.encoding_dict = self._get_default_encoding_dict()
        else:
            # Try to load the full dictionary first
            try:
                with open('/home/runner/work/CisPerplexity/CisPerplexity/selected_encoding_dict.json', 'r') as f:
                    self.encoding_dict = json.load(f)
            except:
                # Default encoding dictionary (subset for demo)
                self.encoding_dict = self._get_default_encoding_dict()
    
    def _get_default_encoding_dict(self) -> Dict:
        """Return default encoding dictionary"""
        return {
            "conformational:Twist:1": {
                "AA": 38.9, "AC": 31.12, "AG": 32.15, "AT": 33.81,
                "CA": 41.41, "CC": 34.96, "CG": 32.91, "CT": 32.15,
                "GA": 41.31, "GC": 38.5, "GG": 34.96, "GT": 31.12,
                "TA": 33.28, "TC": 41.31, "TG": 41.41, "TT": 38.9
            },
            "letter_based:GC_content:76": {
                "AA": 0.0, "AC": 1.0, "AG": 1.0, "AT": 0.0,
                "CA": 1.0, "CC": 2.0, "CG": 2.0, "CT": 1.0,
                "GA": 1.0, "GC": 2.0, "GG": 2.0, "GT": 1.0,
                "TA": 0.0, "TC": 1.0, "TG": 1.0, "TT": 0.0
            },
            "physicochemical:Free_energy:125": {
                "AA": -1.0, "AC": -1.44, "AG": -1.28, "AT": -0.88,
                "CA": -1.45, "CC": -1.84, "CG": -2.17, "CT": -1.28,
                "GA": -1.3, "GC": -2.24, "GG": -1.84, "GT": -1.44,
                "TA": -0.58, "TC": -1.3, "TG": -1.45, "TT": -1.0
            }
        }
    
    def encode_sequence(self, sequence: str, window_size: int = 10) -> Dict[str, np.ndarray]:
        """
        Encode sequence using structural features
        
        Args:
            sequence: DNA sequence
            window_size: Size of sliding window
            
        Returns:
            Dictionary with feature arrays for each encoding type
        """
        seq_len = len(sequence)
        if seq_len < window_size:
            return {}
        
        num_windows = seq_len - window_size + 1
        features = {}
        
        # Generate dinucleotides
        dinucleotides = [sequence[i:i + 2] for i in range(seq_len - 1)]
        
        # Encode using each feature type
        for feature_name, encoding in self.encoding_dict.items():
            feature_values = np.zeros(num_windows)
            
            for i in range(num_windows):
                window_dinucleotides = dinucleotides[i:i + window_size - 1]
                
                # Calculate mean feature value for window
                values = []
                for dinuc in window_dinucleotides:
                    if dinuc in encoding:
                        values.append(encoding[dinuc])
                
                if values:
                    feature_values[i] = np.mean(values)
            
            features[feature_name] = feature_values
        
        return features

class KadaneMaxSubarray:
    """Implementation of Kadane's Algorithm for finding maximum/minimum subarray sums"""
    
    @staticmethod
    def find_min_subarray(arr: np.ndarray, min_length: int = 1) -> Tuple[int, int, float]:
        """
        Find contiguous subarray with minimum sum using modified Kadane's algorithm
        
        Args:
            arr: Array of values (e.g., perplexity values)
            min_length: Minimum length of subarray to consider
            
        Returns:
            Tuple of (start_index, end_index, min_sum)
        """
        if len(arr) < min_length:
            return 0, 0, float('inf')
        
        n = len(arr)
        min_sum = float('inf')
        current_sum = 0
        start = 0
        end = 0
        temp_start = 0
        
        for i in range(n):
            current_sum += arr[i]
            
            if current_sum < min_sum and (i - temp_start + 1) >= min_length:
                min_sum = current_sum
                start = temp_start
                end = i
            
            if current_sum > 0:
                current_sum = 0
                temp_start = i + 1
        
        # If no negative sum found, find minimum single element
        if min_sum == float('inf'):
            min_idx = np.argmin(arr)
            return min_idx, min_idx, arr[min_idx]
        
        return start, end, min_sum
    
    @staticmethod
    def find_min_average_subarray(arr: np.ndarray, min_length: int = 1) -> Tuple[int, int, float]:
        """
        Find contiguous subarray with minimum average value
        
        Args:
            arr: Array of values
            min_length: Minimum length of subarray
            
        Returns:
            Tuple of (start_index, end_index, min_average)
        """
        if len(arr) < min_length:
            return 0, 0, float('inf')
        
        n = len(arr)
        min_avg = float('inf')
        best_start = 0
        best_end = 0
        
        # Convert to float to handle large values
        work_arr = arr.astype(np.float64)
        
        # Try all possible subarrays of length >= min_length
        for length in range(min_length, n + 1):
            for start in range(n - length + 1):
                end = start + length - 1
                subarray_sum = np.sum(work_arr[start:end+1])
                avg = subarray_sum / length
                
                if avg < min_avg:
                    min_avg = avg
                    best_start = start
                    best_end = end
        
        return best_start, best_end, min_avg
    
    @staticmethod
    def find_multiple_min_regions(arr: np.ndarray, min_length: int = 1, 
                                 num_regions: int = 3, overlap_threshold: float = 0.5) -> List[Tuple[int, int, float]]:
        """
        Find multiple non-overlapping minimum regions using Kadane's algorithm
        
        Args:
            arr: Array of values
            min_length: Minimum length of each region
            num_regions: Maximum number of regions to find
            overlap_threshold: Maximum allowed overlap between regions (0-1)
            
        Returns:
            List of tuples (start_index, end_index, average_value)
        """
        if len(arr) < min_length:
            return []
        
        regions = []
        excluded_indices = set()
        
        # Convert to float to handle infinity values
        work_arr = arr.astype(np.float64)
        max_val = np.max(work_arr) * 10  # Use large value instead of infinity
        
        for _ in range(num_regions):
            # Create modified array excluding already found regions
            modified_arr = work_arr.copy()
            for idx in excluded_indices:
                modified_arr[idx] = max_val
            
            # Find next minimum region
            start, end, avg = KadaneMaxSubarray.find_min_average_subarray(modified_arr, min_length)
            
            if avg >= max_val:  # No more valid regions
                break
            
            # Check for significant overlap with existing regions
            has_overlap = False
            region_length = end - start + 1
            
            for existing_start, existing_end, _ in regions:
                overlap_start = max(start, existing_start)
                overlap_end = min(end, existing_end)
                
                if overlap_start <= overlap_end:
                    overlap_length = overlap_end - overlap_start + 1
                    if overlap_length / region_length > overlap_threshold:
                        has_overlap = True
                        break
            
            if not has_overlap:
                regions.append((start, end, avg))
                # Mark indices as excluded
                for i in range(start, end + 1):
                    excluded_indices.add(i)
            
            # If we can't find more non-overlapping regions, break
            if len(excluded_indices) >= len(arr) * 0.8:  # If 80% of array is excluded
                break
        
        return regions

class MotifDetector:
    """Detect promoter motifs in DNA sequences"""
    
    def __init__(self):
        """Initialize with promoter motif patterns"""
        self.motif_patterns = {
            "TATA-box": r'TATA[AT]A[ATG]',
            "Inr-Human": r'[CGT][CGT]CA[CGT][AT]',
            "Inr-fly": r'TCAGT[CT]',
            "TCT": r'[CT][CT]CTTT[CT][CT]',
            "BREu": r'[GC][GC][AG]CGCC',
            "BREd": r'[AG]T[AGT][GT][GT][GT][GT]',
            "XCPE1": r'[AGT][GC]G[CT]GG[AG]A[GC][AC]',
            "XCPE2": r'[ACG]C[CT]C[AG]TT[AG]C[AC][CT]',
            "Pause Button": r'[GT]CG[AG][AT]CG',
            "Sp1 element": r'(GGCGGG|GGGCGG|CCGCCC|CCCGCC)',
            "DCEI": r'CTTC',
            "DCEII": r'CTGT',
            "DCEIII": r'AGC',
            "i-motif": r'C{3,5}[ACTG]{1,7}C{3,5}[ACTG]{1,7}C{3,5}[ACTG]{1,7}C{3,5}',
            "G-quadruplex": r'G{3,5}[ACGT]{1,7}G{3,5}[ACGT]{1,7}G{3,5}[ACGT]{1,7}G{3,5}',
            "G-tract": r'(CCCCCCC|GGGGGGG)',
            "A-tract": r'(AAAAAAA|TTTTTTT)'
        }
    
    def detect_motifs(self, sequence: str) -> Dict[str, List[Tuple[int, int, str]]]:
        """
        Detect all motifs in a sequence
        
        Args:
            sequence: DNA sequence
            
        Returns:
            Dictionary with motif matches: {motif_name: [(start, end, match_sequence), ...]}
        """
        motif_matches = {}
        
        for motif_name, pattern in self.motif_patterns.items():
            matches = []
            for match in re.finditer(pattern, sequence, re.IGNORECASE):
                matches.append((match.start(), match.end(), match.group()))
            motif_matches[motif_name] = matches
        
        return motif_matches
    
    def calculate_motif_density(self, sequence: str, window_size: int = 100) -> np.ndarray:
        """Calculate motif density in sliding windows"""
        seq_len = len(sequence)
        if seq_len < window_size:
            return np.array([])
        
        num_windows = seq_len - window_size + 1
        motif_densities = np.zeros(num_windows)
        
        # Get all motif matches
        all_motifs = self.detect_motifs(sequence)
        
        # Flatten all matches
        all_matches = []
        for motif_matches in all_motifs.values():
            all_matches.extend([match[0] for match in motif_matches])
        
        # Calculate density for each window
        for i in range(num_windows):
            window_start = i
            window_end = i + window_size
            
            # Count motifs in window
            motif_count = sum(1 for pos in all_matches if window_start <= pos < window_end)
            motif_densities[i] = motif_count / window_size * 1000  # Per 1000 bp
        
        return motif_densities

class PromoterPredictor:
    """Integrated promoter prediction using perplexity, structural features, and motifs with Kadane's Algorithm"""
    
    def __init__(self, encoding_dict_path: str = None):
        """Initialize predictor components"""
        self.perplexity_calc = PerplexityCalculator()
        self.feature_encoder = StructuralFeatureEncoder(encoding_dict_path)
        self.motif_detector = MotifDetector()
        self.kadane = KadaneMaxSubarray()
    
    def predict_promoters(self, sequence: str, window_size: int = 100, 
                         perplexity_window: int = 10,
                         perplexity_threshold: float = None) -> Dict:
        """
        Predict promoter regions using Kadane's Algorithm for maximum subarray sum on low perplexity regions
        
        Args:
            sequence: DNA sequence
            window_size: Window size for Kadane's algorithm analysis (default 100)
            perplexity_window: Window size for perplexity calculation (default 10)
            perplexity_threshold: Threshold for low perplexity (auto-calculated if None)
            
        Returns:
            Dictionary with prediction results including Kadane's algorithm results
        """
        results = {
            'sequence_length': len(sequence),
            'window_size': window_size,
            'perplexity_window': perplexity_window,
            'perplexity': None,
            'gc_content': None,
            'structural_features': None,
            'motif_density': None,
            'motif_matches': None,
            'predicted_promoters': [],
            'kadane_regions': [],
            'perplexity_threshold': None
        }
        
        if len(sequence) < perplexity_window:
            return results
        
        # Calculate perplexity using the specified perplexity window (default 10)
        perplexity = self.perplexity_calc.calculate_perplexity(sequence, perplexity_window)
        results['perplexity'] = perplexity
        
        # Calculate GC content
        gc_content = self.perplexity_calc.calculate_gc_content(sequence, perplexity_window)
        results['gc_content'] = gc_content
        
        # Encode structural features
        structural_features = self.feature_encoder.encode_sequence(sequence, perplexity_window)
        results['structural_features'] = structural_features
        
        # Calculate motif density
        motif_density = self.motif_detector.calculate_motif_density(sequence, window_size)
        results['motif_density'] = motif_density
        
        # Detect motifs
        motif_matches = self.motif_detector.detect_motifs(sequence)
        results['motif_matches'] = motif_matches
        
        # Determine perplexity threshold
        if perplexity_threshold is None:
            perplexity_threshold = np.percentile(perplexity, 25)  # Lower quartile
        results['perplexity_threshold'] = perplexity_threshold
        
        # Apply Kadane's Algorithm to find low perplexity regions
        if len(perplexity) >= window_size:
            # Use Kadane's algorithm to find minimum perplexity regions
            # We want regions with minimum average perplexity (promoter-like regions)
            min_region_length = max(window_size // 2, 20)  # Minimum meaningful region size
            
            # Find multiple low perplexity regions using Kadane's algorithm
            kadane_regions = self.kadane.find_multiple_min_regions(
                perplexity, 
                min_length=min_region_length, 
                num_regions=5,  # Find up to 5 regions
                overlap_threshold=0.3  # Allow 30% overlap
            )
            
            results['kadane_regions'] = kadane_regions
            
            # Convert Kadane's regions to promoter predictions
            for start_window, end_window, avg_perplexity in kadane_regions:
                # Convert window indices to sequence positions
                start_pos = start_window
                end_pos = min(end_window + perplexity_window - 1, len(sequence) - 1)
                region_length = end_pos - start_pos + 1
                
                # Only consider regions that meet the perplexity threshold
                if avg_perplexity <= perplexity_threshold:
                    # Extract region sequence for motif analysis
                    region_sequence = sequence[start_pos:end_pos+1]
                    region_motifs = self.motif_detector.detect_motifs(region_sequence)
                    total_motifs = sum(len(matches) for matches in region_motifs.values())
                    
                    # Calculate confidence based on:
                    # 1. How much lower the perplexity is compared to the maximum
                    # 2. Number of motifs found in the region
                    # 3. Length of the region (longer regions are more confident)
                    perplexity_score = 1 - (avg_perplexity / np.max(perplexity)) if len(perplexity) > 0 else 0
                    motif_score = min(total_motifs / region_length * 1000, 1.0)  # Normalize by length
                    length_score = min(region_length / 200, 1.0)  # Regions around 200bp get max score
                    
                    confidence = (perplexity_score * 0.5 + motif_score * 0.3 + length_score * 0.2)
                    confidence = min(confidence, 1.0)
                    
                    # Apply structural features analysis to the region
                    region_structural_features = {}
                    if structural_features:
                        for feature_name, feature_values in structural_features.items():
                            if start_window < len(feature_values) and end_window < len(feature_values):
                                region_feature_values = feature_values[start_window:end_window+1]
                                region_structural_features[feature_name] = {
                                    'mean': np.mean(region_feature_values),
                                    'std': np.std(region_feature_values),
                                    'values': region_feature_values.tolist()
                                }
                    
                    results['predicted_promoters'].append({
                        'start': start_pos,
                        'end': end_pos,
                        'length': region_length,
                        'avg_perplexity': avg_perplexity,
                        'motif_count': total_motifs,
                        'confidence': confidence,
                        'method': 'kadane_algorithm',
                        'window_start': start_window,
                        'window_end': end_window,
                        'structural_features': region_structural_features,
                        'region_motifs': {k: v for k, v in region_motifs.items() if v}  # Only non-empty motifs
                    })
        
        # Sort predicted promoters by confidence (descending)
        results['predicted_promoters'].sort(key=lambda x: x['confidence'], reverse=True)
        
        return results

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.markdown('<h1 class="main-header">🧬 CisPerplexity: Promoter Prediction Tool</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This tool predicts promoter regions in DNA sequences using an integrated approach that combines:
    - **Dinucleotide perplexity analysis** with Kadane's Algorithm for optimal low-perplexity region detection
    - **Structural features** (conformational and physicochemical properties)
    - **Promoter motif detection** (TATA-box, Initiator elements, etc.)
    
    **New Feature**: Uses Kadane's Algorithm (Maximum Subarray Sum) adapted for finding minimum perplexity regions 
    with configurable window sizes for perplexity calculation (default 10bp) and region analysis (default 100bp).
    """)
    
    # Sidebar for parameters
    st.sidebar.header("⚙️ Analysis Parameters")
    
    window_size = st.sidebar.slider(
        "Analysis Window Size (bp)",
        min_value=50,
        max_value=500,
        value=100,
        step=10,
        help="Size of window for Kadane's algorithm analysis"
    )
    
    perplexity_window = st.sidebar.slider(
        "Perplexity Window Size (bp)",
        min_value=5,
        max_value=50,
        value=10,
        step=1,
        help="Size of sliding window for perplexity calculation"
    )
    
    perplexity_threshold = st.sidebar.slider(
        "Perplexity Threshold (% percentile)",
        min_value=10,
        max_value=50,
        value=25,
        step=5,
        help="Percentile threshold for low perplexity regions"
    )
    
    # Input section
    st.markdown('<h2 class="subheader">📝 Input DNA Sequence</h2>', unsafe_allow_html=True)
    
    # Input methods
    input_method = st.radio(
        "Select input method:",
        ["Paste sequence", "Upload FASTA file", "Use example sequence"]
    )
    
    sequence = ""
    
    if input_method == "Paste sequence":
        sequence = st.text_area(
            "Enter DNA sequence (A, T, G, C only):",
            height=150,
            placeholder="ATGCGTACGTAGC..."
        )
    
    elif input_method == "Upload FASTA file":
        uploaded_file = st.file_uploader("Choose a FASTA file", type=["fasta", "fa", "txt"])
        if uploaded_file is not None:
            # Read file content
            content = uploaded_file.read().decode("utf-8")
            lines = content.strip().split('\n')
            
            # Extract sequence (skip header lines starting with '>')
            seq_lines = [line for line in lines if not line.startswith('>')]
            sequence = ''.join(seq_lines).upper()
    
    elif input_method == "Use example sequence":
        # Example promoter sequence with TATA box
        sequence = """GCGCGCGCATATAAGCGTAGCGTAGCGTAGCGTAGCGTAGCGTAGCGTAGCG
        TAGCGTAGCGTAGCGTAGCGTAGCGTAGCGTATAGCGTAGCGTAGCGTAGCG
        TAGCGTAGCGTAGCGTAGCGTAGCGTAGCGTAGCGTAGCGTAGCGTAGCGTA
        GCGTAGCGTAGCGTAGCGTAGCGTAGCGTAGCGTAGCGTAGCGTAGCGTAG""".replace('\n', '').replace(' ', '')
    
    # Clean sequence
    if sequence:
        sequence = re.sub(r'[^ATGC]', '', sequence.upper())
        st.write(f"**Sequence length:** {len(sequence)} bp")
        
        # Clean sequence
        if len(sequence) < perplexity_window:
            st.error(f"Sequence too short! Minimum length required: {perplexity_window} bp")
            return
        
        # Analysis button
        if st.button("🔬 Analyze Sequence", type="primary"):
            
            with st.spinner("Analyzing sequence..."):
                # Initialize predictor
                try:
                    # Try to load the existing encoding dictionary
                    predictor = PromoterPredictor("/home/runner/work/CisPerplexity/CisPerplexity/selected_encoding_dict.json")
                except:
                    # Use default dictionary if file not found
                    predictor = PromoterPredictor()
                
                # Run prediction
                results = predictor.predict_promoters(
                    sequence, 
                    window_size=window_size,
                    perplexity_window=perplexity_window,
                    perplexity_threshold=np.percentile(
                        predictor.perplexity_calc.calculate_perplexity(sequence, perplexity_window),
                        perplexity_threshold
                    ) if len(sequence) >= perplexity_window else None
                )
            
            # Display results
            display_results(results, sequence)

def display_results(results: Dict, sequence: str):
    """Display analysis results"""
    
    # Summary metrics
    st.markdown('<h2 class="subheader">📊 Analysis Summary</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sequence Length", f"{results['sequence_length']:,} bp")
    
    with col2:
        if 'perplexity_window' in results:
            st.metric("Perplexity Window", f"{results['perplexity_window']} bp")
        else:
            st.metric("Window Size", f"{results['window_size']} bp")
    
    with col3:
        if results['perplexity'] is not None:
            avg_perplexity = np.mean(results['perplexity'])
            st.metric("Avg Perplexity", f"{avg_perplexity:.2f}")
    
    with col4:
        st.metric("Predicted Promoters", len(results['predicted_promoters']))
    
    # Predicted promoters
    if results['predicted_promoters']:
        st.markdown('<h2 class="subheader">🎯 Predicted Promoter Regions</h2>', unsafe_allow_html=True)
        
        for i, promoter in enumerate(results['predicted_promoters']):
            with st.expander(f"Promoter Region {i+1} (Position {promoter['start']}-{promoter['end']}) - Method: {promoter.get('method', 'threshold')}"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Length", f"{promoter['length']} bp")
                with col2:
                    st.metric("Avg Perplexity", f"{promoter['avg_perplexity']:.2f}")
                with col3:
                    st.metric("Motif Count", promoter['motif_count'])
                with col4:
                    st.metric("Confidence", f"{promoter['confidence']:.2%}")
                
                # Show additional Kadane's algorithm info if available
                if 'window_start' in promoter and 'window_end' in promoter:
                    st.write(f"**Kadane's Window:** {promoter['window_start']} - {promoter['window_end']}")
                
                # Show structural features if available
                if 'structural_features' in promoter and promoter['structural_features']:
                    st.write("**Structural Features:**")
                    feature_data = []
                    for feature_name, feature_info in promoter['structural_features'].items():
                        feature_data.append({
                            'Feature': feature_name,
                            'Mean': f"{feature_info['mean']:.3f}",
                            'Std Dev': f"{feature_info['std']:.3f}"
                        })
                    if feature_data:
                        st.dataframe(pd.DataFrame(feature_data), use_container_width=True)
                
                # Show region-specific motifs if available
                if 'region_motifs' in promoter and promoter['region_motifs']:
                    st.write("**Motifs in this region:**")
                    region_motif_data = []
                    for motif_name, matches in promoter['region_motifs'].items():
                        for start, end, seq in matches:
                            region_motif_data.append({
                                'Motif': motif_name,
                                'Position': f"{start}-{end}",
                                'Sequence': seq
                            })
                    if region_motif_data:
                        st.dataframe(pd.DataFrame(region_motif_data), use_container_width=True)
                
                # Show sequence
                promoter_seq = sequence[promoter['start']:promoter['end']+1]
                st.text_area(f"Sequence:", promoter_seq, height=100)
    
    # Visualizations
    if results['perplexity'] is not None and len(results['perplexity']) > 0:
        st.markdown('<h2 class="subheader">📈 Perplexity Analysis</h2>', unsafe_allow_html=True)
        
        # Create perplexity plot
        positions = np.arange(len(results['perplexity']))
        
        fig = go.Figure()
        
        # Add perplexity trace
        fig.add_trace(go.Scatter(
            x=positions,
            y=results['perplexity'],
            mode='lines',
            name='Perplexity',
            line=dict(color='blue', width=2)
        ))
        
        # Add threshold line
        if results['perplexity_threshold']:
            fig.add_hline(
                y=results['perplexity_threshold'],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold: {results['perplexity_threshold']:.2f}"
            )
        
        # Highlight predicted promoters
        for promoter in results['predicted_promoters']:
            fig.add_vrect(
                x0=promoter['start'],
                x1=promoter['end'],
                fillcolor="red",
                opacity=0.2,
                layer="below",
                line_width=0,
            )
        
        fig.update_layout(
            title="Dinucleotide Perplexity Analysis",
            xaxis_title="Position (bp)",
            yaxis_title="Perplexity",
            hovermode='x'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # GC content plot
    if results['gc_content'] is not None and len(results['gc_content']) > 0:
        st.markdown('<h2 class="subheader">🧪 GC Content Analysis</h2>', unsafe_allow_html=True)
        
        positions = np.arange(len(results['gc_content']))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=positions,
            y=results['gc_content'],
            mode='lines',
            name='GC Content (%)',
            line=dict(color='green', width=2)
        ))
        
        # Highlight predicted promoters
        for promoter in results['predicted_promoters']:
            fig.add_vrect(
                x0=promoter['start'],
                x1=promoter['end'],
                fillcolor="red",
                opacity=0.2,
                layer="below",
                line_width=0,
            )
        
        fig.update_layout(
            title="GC Content Analysis",
            xaxis_title="Position (bp)",
            yaxis_title="GC Content (%)",
            hovermode='x'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Motif detection results
    if results['motif_matches']:
        st.markdown('<h2 class="subheader">🔍 Detected Motifs</h2>', unsafe_allow_html=True)
        
        # Count total motifs
        total_motifs = sum(len(matches) for matches in results['motif_matches'].values())
        
        if total_motifs > 0:
            # Create motif summary table
            motif_data = []
            for motif_name, matches in results['motif_matches'].items():
                if matches:
                    for start, end, seq in matches:
                        motif_data.append({
                            'Motif': motif_name,
                            'Position': f"{start}-{end}",
                            'Sequence': seq,
                            'Length': len(seq)
                        })
            
            if motif_data:
                df_motifs = pd.DataFrame(motif_data)
                st.dataframe(df_motifs, use_container_width=True)
                
                # Motif count chart
                motif_counts = df_motifs['Motif'].value_counts()
                fig = px.bar(
                    x=motif_counts.index,
                    y=motif_counts.values,
                    title="Motif Frequency",
                    labels={'x': 'Motif Type', 'y': 'Count'}
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No known promoter motifs detected in the sequence.")
    
    # Download results
    st.markdown('<h2 class="subheader">💾 Download Results</h2>', unsafe_allow_html=True)
    
    # Prepare results for download
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    download_data = {
        'sequence_info': {
            'length': int(results['sequence_length']),
            'window_size': int(results['window_size'])
        },
        'predicted_promoters': convert_numpy_types(results['predicted_promoters']),
        'motif_matches': {k: v for k, v in results['motif_matches'].items() if v}
    }
    
    # Convert to JSON
    json_data = json.dumps(download_data, indent=2)
    
    st.download_button(
        label="📄 Download Results (JSON)",
        data=json_data,
        file_name="promoter_prediction_results.json",
        mime="application/json"
    )

if __name__ == "__main__":
    main()