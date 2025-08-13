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
    """Integrated promoter prediction using perplexity, structural features, and motifs"""
    
    def __init__(self, encoding_dict_path: str = None):
        """Initialize predictor components"""
        self.perplexity_calc = PerplexityCalculator()
        self.feature_encoder = StructuralFeatureEncoder(encoding_dict_path)
        self.motif_detector = MotifDetector()
    
    def predict_promoters(self, sequence: str, window_size: int = 100, 
                         perplexity_threshold: float = None) -> Dict:
        """
        Predict promoter regions using integrated approach
        
        Args:
            sequence: DNA sequence
            window_size: Window size for analysis
            perplexity_threshold: Threshold for low perplexity (auto-calculated if None)
            
        Returns:
            Dictionary with prediction results
        """
        results = {
            'sequence_length': len(sequence),
            'window_size': window_size,
            'perplexity': None,
            'gc_content': None,
            'structural_features': None,
            'motif_density': None,
            'motif_matches': None,
            'predicted_promoters': [],
            'perplexity_threshold': None
        }
        
        if len(sequence) < window_size:
            return results
        
        # Calculate perplexity
        perplexity = self.perplexity_calc.calculate_perplexity(sequence, window_size)
        results['perplexity'] = perplexity
        
        # Calculate GC content
        gc_content = self.perplexity_calc.calculate_gc_content(sequence, window_size)
        results['gc_content'] = gc_content
        
        # Encode structural features
        structural_features = self.feature_encoder.encode_sequence(sequence, window_size)
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
        
        # Predict promoter regions (low perplexity regions)
        low_perplexity_indices = np.where(perplexity < perplexity_threshold)[0]
        
        # Group consecutive indices into regions
        if len(low_perplexity_indices) > 0:
            regions = []
            current_start = low_perplexity_indices[0]
            current_end = low_perplexity_indices[0]
            
            for i in range(1, len(low_perplexity_indices)):
                if low_perplexity_indices[i] == current_end + 1:
                    current_end = low_perplexity_indices[i]
                else:
                    regions.append((current_start, current_end + window_size - 1))
                    current_start = low_perplexity_indices[i]
                    current_end = low_perplexity_indices[i]
            
            # Add the last region
            regions.append((current_start, current_end + window_size - 1))
            
            # Calculate confidence for each region
            for start, end in regions:
                region_length = end - start + 1
                region_perplexity = np.mean(perplexity[start:min(start + region_length - window_size + 1, len(perplexity))])
                
                # Count motifs in region
                region_sequence = sequence[start:end+1]
                region_motifs = self.motif_detector.detect_motifs(region_sequence)
                total_motifs = sum(len(matches) for matches in region_motifs.values())
                
                confidence = (1 - region_perplexity / np.max(perplexity)) * 0.7 + (total_motifs / region_length * 1000) * 0.3
                confidence = min(confidence, 1.0)
                
                results['predicted_promoters'].append({
                    'start': start,
                    'end': end,
                    'length': region_length,
                    'avg_perplexity': region_perplexity,
                    'motif_count': total_motifs,
                    'confidence': confidence
                })
        
        return results

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.markdown('<h1 class="main-header">🧬 CisPerplexity: Promoter Prediction Tool</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This tool predicts promoter regions in DNA sequences using an integrated approach that combines:
    - **Dinucleotide perplexity analysis** (promoters typically have lower perplexity)
    - **Structural features** (conformational and physicochemical properties)
    - **Promoter motif detection** (TATA-box, Initiator elements, etc.)
    """)
    
    # Sidebar for parameters
    st.sidebar.header("⚙️ Analysis Parameters")
    
    window_size = st.sidebar.slider(
        "Window Size (bp)",
        min_value=50,
        max_value=500,
        value=100,
        step=10,
        help="Size of sliding window for analysis"
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
        
        if len(sequence) < window_size:
            st.error(f"Sequence too short! Minimum length required: {window_size} bp")
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
                    perplexity_threshold=np.percentile(
                        predictor.perplexity_calc.calculate_perplexity(sequence, window_size),
                        perplexity_threshold
                    ) if len(sequence) >= window_size else None
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
            with st.expander(f"Promoter Region {i+1} (Position {promoter['start']}-{promoter['end']})"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Length", f"{promoter['length']} bp")
                with col2:
                    st.metric("Avg Perplexity", f"{promoter['avg_perplexity']:.2f}")
                with col3:
                    st.metric("Motif Count", promoter['motif_count'])
                with col4:
                    st.metric("Confidence", f"{promoter['confidence']:.2%}")
                
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