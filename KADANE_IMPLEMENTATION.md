# Kadane's Algorithm Implementation for CisPerplexity

## Overview

This implementation adds Kadane's Algorithm (Maximum Subarray Sum) to the CisPerplexity promoter prediction tool for optimal detection of low perplexity regions that may indicate promoter sequences.

## Key Features

### 1. KadaneMaxSubarray Class
- **find_min_subarray()**: Finds contiguous subarray with minimum sum using modified Kadane's algorithm
- **find_min_average_subarray()**: Finds subarray with minimum average value  
- **find_multiple_min_regions()**: Identifies multiple non-overlapping low perplexity regions

### 2. Enhanced PromoterPredictor
- Uses Kadane's algorithm instead of simple threshold-based detection
- Configurable window sizes:
  - **Analysis Window** (default 100bp): Size for Kadane's algorithm region detection
  - **Perplexity Window** (default 10bp): Size for dinucleotide perplexity calculation
- Enhanced confidence scoring based on perplexity, motifs, and region length
- Integration with structural features analysis
- Region-specific motif detection

### 3. Algorithm Flow
1. Calculate dinucleotide perplexity using 10bp sliding windows
2. Apply Kadane's algorithm on 100bp regions to find optimal low-perplexity subarrays
3. Filter regions by perplexity threshold
4. Analyze structural features for each identified region
5. Detect motifs specifically within each region
6. Calculate confidence scores combining multiple factors

## Configuration Parameters

- **Analysis Window Size**: 50-500bp (default 100bp) - Controls Kadane's algorithm region size
- **Perplexity Window Size**: 5-50bp (default 10bp) - Controls perplexity calculation granularity  
- **Perplexity Threshold**: 10-50% percentile (default 25%) - Threshold for low perplexity regions

## Results Format

Each predicted promoter region includes:
- **Position**: Start and end coordinates in sequence
- **Method**: "kadane_algorithm" to distinguish from threshold-based predictions
- **Kadane's Window**: The specific window indices identified by the algorithm
- **Structural Features**: Mean and standard deviation for each structural property
- **Region Motifs**: Motifs detected specifically within the identified region
- **Enhanced Confidence**: Multi-factor confidence score (0-100%)

## Performance

- Handles sequences up to 1000+ bp efficiently
- Processing time scales approximately O(n²) for region finding
- Memory efficient with numpy array operations
- Finds optimal regions rather than arbitrary threshold crossings

## Validation

The implementation has been thoroughly tested with:
- Basic Kadane's algorithm correctness
- Multiple window size configurations  
- Structural features integration
- Region-specific motif detection
- Edge cases and boundary conditions
- Performance with large sequences

## Usage

```python
predictor = PromoterPredictor()
results = predictor.predict_promoters(
    sequence,
    window_size=100,        # Analysis window for Kadane's
    perplexity_window=10,   # Perplexity calculation window
    perplexity_threshold=None  # Auto-calculated if None
)
```

The Streamlit UI provides interactive controls for all parameters and visualizes the results with highlighted regions and detailed analysis.