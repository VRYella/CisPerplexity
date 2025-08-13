# CisPerplexity: Promoter Prediction Tool

A Streamlit web application for predicting promoter regions in DNA sequences using an integrated approach that combines dinucleotide perplexity analysis, structural features, and promoter motif detection.

## Features

### 🧬 Integrated Algorithm
- **Dinucleotide Perplexity Analysis**: Calculates perplexity using sliding windows to identify low-complexity regions (promoters typically have lower perplexity)
- **Structural Features**: Encodes sequences using conformational and physicochemical properties from the structural dictionary
- **Motif Detection**: Identifies known promoter motifs including TATA-box, Initiator elements, and other regulatory sequences

### 📊 Interactive Web Interface
- **Multiple Input Methods**: Paste sequences, upload FASTA files, or use example sequences
- **Real-time Analysis**: Live perplexity calculation and promoter prediction
- **Interactive Visualizations**: Plotly charts showing perplexity, GC content, and motif frequency
- **Detailed Results**: Expandable promoter region details with confidence scores
- **Data Export**: Download results in JSON format

### 🔬 Analysis Components
1. **Perplexity Calculator**: Sliding window dinucleotide perplexity using entropy calculations
2. **Structural Feature Encoder**: DNA property encoding using the comprehensive dictionary
3. **Motif Detector**: Pattern matching for 17+ known promoter motifs
4. **Integrated Predictor**: Combines all features for confident promoter prediction

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

## Usage

1. **Select Input Method**: Choose between pasting sequence, uploading FASTA, or using example
2. **Adjust Parameters**: Set window size and perplexity threshold in the sidebar
3. **Analyze Sequence**: Click the analyze button to run prediction
4. **Review Results**: Examine predicted promoter regions, visualizations, and detected motifs
5. **Download Data**: Export results for further analysis

## Algorithm Details

### Perplexity Calculation
- Uses sliding windows to calculate dinucleotide frequencies
- Computes entropy: H = -Σ(p * log2(p))
- Calculates perplexity: 2^H
- Lower perplexity indicates lower complexity (potential promoter regions)

### Structural Features
The encoding dictionary includes:
- **Conformational**: Twist, Roll, Slide, Tilt, Wedge, Major Groove Depth, etc.
- **Physicochemical**: Free energy, Melting temperature, Stiffness
- **Letter-based**: GC content, Purine content, Keto content

### Promoter Motifs
Detects patterns for:
- TATA-box, Initiator elements (Human/Fly)
- TCT, BREu, BREd elements
- Structural motifs (i-motif, G-quadruplex)
- Core promoter elements (Sp1, DPE, etc.)

## Files Structure

- `streamlit_app.py`: Main Streamlit application
- `selected_encoding_dict.json`: Structural feature dictionary
- `requirements.txt`: Python dependencies
- Jupyter notebooks: Original research and analysis code

## Research Background

This tool implements algorithms from the CisPerplexity research project, which explores the relationship between DNA dinucleotide perplexity and promoter regions. The hypothesis is that promoter regions exhibit lower perplexity compared to neighboring genomic regions due to their regulatory constraints and sequence composition.