#!/usr/bin/env python3
"""
Prediction Processor for Anatomagram Visualization
Handles conversion of model predictions to anatomagram-compatible format
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class PredictionRecord:
    """Single tissue prediction record"""
    tissue_id: str
    tissue_name: str
    prediction_value: float
    uberon_id: str
    svg_uberon_id: str
    is_direct_match: bool
    can_visualize: bool


class AnatomagramPredictionProcessor:
    """Process model predictions for anatomagram visualization"""
    
    def __init__(self, anatomagram_dir: str):
        self.anatomagram_dir = Path(anatomagram_dir)
        self.data_dir = self.anatomagram_dir / "data"

        # Load enhanced mapping and anatomical region names
        self.enhanced_mapping = self._load_enhanced_mapping()
        self.uberon_names = self._load_uberon_names()
        self.uberon_descriptions = self._load_uberon_descriptions()
        
    def _load_enhanced_mapping(self) -> Dict:
        """Load the enhanced mapping file"""
        mapping_file = self.data_dir / "tissue_mapping_enhanced.json"
        with open(mapping_file, 'r') as f:
            return json.load(f)
    
    def _load_uberon_names(self) -> Dict[str, str]:
        """Load UBERON ID to anatomical region name mapping from descriptions file"""
        uberon_file = self.data_dir / "uberon_descriptions.json"
        with open(uberon_file, 'r') as f:
            descriptions_data = json.load(f)

        # Extract just the labels (names) from the descriptions data
        return {uberon_id: data['label'] for uberon_id, data in descriptions_data.items()}

    def _load_uberon_descriptions(self) -> Dict[str, str]:
        """Load UBERON ID to full description mapping"""
        uberon_file = self.data_dir / "uberon_descriptions.json"
        with open(uberon_file, 'r') as f:
            descriptions_data = json.load(f)

        # Extract full descriptions from the descriptions data
        return {uberon_id: data['description'] for uberon_id, data in descriptions_data.items()}

    def process_predictions(self, predictions: pd.DataFrame, 
                          tissue_id_col: str = 'tissue_id',
                          prediction_col: str = 'prediction',
                          aggregation_strategy: str = 'mean') -> Dict[str, Any]:
        """
        Process model predictions for anatomagram visualization
        
        Args:
            predictions: DataFrame with tissue_id and prediction columns
            tissue_id_col: Name of tissue ID column
            prediction_col: Name of prediction value column
            aggregation_strategy: How to aggregate multiple predictions per SVG element
                                'mean', 'max', 'min', 'weighted_mean'
        
        Returns:
            Dictionary with processed data ready for anatomagram visualization
        """
        
        # Convert predictions to records
        prediction_records = []
        
        for _, row in predictions.iterrows():
            tissue_id = str(row[tissue_id_col])
            prediction_value = float(row[prediction_col])
            
            # Look up tissue mapping
            if tissue_id in self.enhanced_mapping['tissue_mappings']:
                tissue_data = self.enhanced_mapping['tissue_mappings'][tissue_id]
                
                record = PredictionRecord(
                    tissue_id=tissue_id,
                    tissue_name=tissue_data['tissue_name'],
                    prediction_value=prediction_value,
                    uberon_id=tissue_data['uberon_id'],
                    svg_uberon_id=tissue_data['svg_uberon_id'],
                    is_direct_match=tissue_data['is_direct_match'],
                    can_visualize=tissue_data['can_visualize']
                )
                
                prediction_records.append(record)
            
            elif tissue_id in self.enhanced_mapping['cell_line_mappings']:
                # Skip cell lines for anatomogram visualization
                print(f"Skipping cell line: {self.enhanced_mapping['cell_line_mappings'][tissue_id]['tissue_name']}")
                continue
            
            else:
                print(f"Warning: Unknown tissue_id {tissue_id} - skipping")
                continue
        
        # Group by SVG UBERON ID for aggregation
        svg_groups = {}
        for record in prediction_records:
            if record.can_visualize:
                svg_id = record.svg_uberon_id
                if svg_id not in svg_groups:
                    svg_groups[svg_id] = []
                svg_groups[svg_id].append(record)
        
        # Aggregate predictions for each SVG element
        svg_predictions = {}
        aggregation_details = {}
        
        for svg_id, records in svg_groups.items():
            values = [r.prediction_value for r in records]
            
            # Calculate aggregated value
            if aggregation_strategy == 'mean':
                aggregated_value = np.mean(values)
            elif aggregation_strategy == 'max':
                aggregated_value = np.max(values)
            elif aggregation_strategy == 'min':
                aggregated_value = np.min(values)
            elif aggregation_strategy == 'weighted_mean':
                # Weight by confidence (direct matches get higher weight)
                weights = [1.0 if r.is_direct_match else 0.7 for r in records]
                aggregated_value = np.average(values, weights=weights)
            else:
                aggregated_value = np.mean(values)  # Default fallback
            
            svg_predictions[svg_id] = aggregated_value
            
            # Store details for tooltip enhancement
            aggregation_details[svg_id] = {
                'aggregated_value': aggregated_value,
                'contributing_tissues': [
                    {
                        'tissue_name': r.tissue_name,
                        'individual_value': r.prediction_value,
                        'is_direct_match': r.is_direct_match,
                        'original_uberon': r.uberon_id
                    }
                    for r in records
                ],
                'aggregation_strategy': aggregation_strategy,
                'tissue_count': len(records)
            }
        
        # Create visualization data
        visualization_data = {
            'svg_element_values': svg_predictions,
            'aggregation_details': aggregation_details,
            'metadata': {
                'total_predictions': len(predictions),
                'visualizable_predictions': len(prediction_records),
                'cell_lines_excluded': max(0, len(predictions) - len(prediction_records)),
                'svg_elements_colored': len(svg_predictions),
                'aggregation_strategy': aggregation_strategy
            },
            'coverage_report': self._generate_coverage_report(prediction_records)
        }
        
        return visualization_data
    
    def _generate_coverage_report(self, prediction_records: List[PredictionRecord]) -> Dict:
        """Generate coverage analysis report"""
        total = len(prediction_records)
        direct_matches = sum(1 for r in prediction_records if r.is_direct_match)
        hierarchy_fallbacks = sum(1 for r in prediction_records if not r.is_direct_match and r.can_visualize)
        cannot_visualize = sum(1 for r in prediction_records if not r.can_visualize)
        
        return {
            'total_tissues': total,
            'direct_svg_matches': direct_matches,
            'hierarchy_fallbacks': hierarchy_fallbacks,
            'cannot_visualize': cannot_visualize,
            'visualization_coverage': (direct_matches + hierarchy_fallbacks) / total * 100 if total > 0 else 0
        }
    
    def create_enhanced_tooltips(self, visualization_data: Dict) -> Dict[str, str]:
        """Create enhanced tooltips with anatomical region names and contributing tissues"""
        tooltips = {}
        
        for svg_id, details in visualization_data['aggregation_details'].items():
            tissues = details['contributing_tissues']
            
            # Get anatomical region name from UBERON ID (fallback to ID if not found)
            region_name = self.uberon_names.get(svg_id, svg_id.replace('_', ':'))
            
            # Standardize all tooltips to use region name + contributing tissues format
            tooltip = f"{region_name} ({details['aggregation_strategy']}): {details['aggregated_value']:.3f}\n"
            tooltip += f"From {len(tissues)} tissues:\n"
            
            for tissue in tissues:
                marker = "•" if tissue['is_direct_match'] else "◦"
                tooltip += f"{marker} {tissue['tissue_name']}: {tissue['individual_value']:.3f}\n"
            
            tooltips[svg_id] = tooltip.strip()
        
        return tooltips
    
    def generate_debug_report(self, predictions: pd.DataFrame, visualization_data: Dict) -> None:
        """Generate comprehensive debug report"""
        print("=== PREDICTION PROCESSING DEBUG REPORT ===\n")
        
        metadata = visualization_data['metadata']
        coverage = visualization_data['coverage_report']
        
        print(f"Input predictions: {metadata['total_predictions']}")
        print(f"Visualizable predictions: {metadata['visualizable_predictions']}")
        print(f"Cell lines excluded: {metadata['cell_lines_excluded']}")
        print(f"SVG elements to color: {metadata['svg_elements_colored']}")
        print(f"Aggregation strategy: {metadata['aggregation_strategy']}")
        
        print(f"\nCOVERAGE ANALYSIS:")
        print(f"Direct SVG matches: {coverage['direct_svg_matches']} ({coverage['direct_svg_matches']/coverage['total_tissues']*100:.1f}%)")
        print(f"Hierarchy fallbacks: {coverage['hierarchy_fallbacks']} ({coverage['hierarchy_fallbacks']/coverage['total_tissues']*100:.1f}%)")
        print(f"Total visualization coverage: {coverage['visualization_coverage']:.1f}%")
        
        if coverage['cannot_visualize'] > 0:
            print(f"Cannot visualize: {coverage['cannot_visualize']} tissues")
        
        print(f"\nSVG ELEMENT AGGREGATION:")
        for svg_id, details in visualization_data['aggregation_details'].items():
            tissue_count = details['tissue_count']
            if tissue_count > 1:
                print(f"{svg_id}: {tissue_count} tissues → {details['aggregated_value']:.3f}")
                for tissue in details['contributing_tissues']:
                    print(f"  - {tissue['tissue_name']}: {tissue['individual_value']:.3f}")


def create_sample_predictions() -> pd.DataFrame:
    """Create sample AD risk predictions for testing"""
    sample_data = [
        {'tissue_id': '7', 'prediction': 0.594},   # adipose - subcutaneous
        {'tissue_id': '14', 'prediction': 0.524},  # blood
        {'tissue_id': '15', 'prediction': 0.435},  # brain - amygdala
        {'tissue_id': '16', 'prediction': 0.521},  # brain - anterior cingulate
        {'tissue_id': '17', 'prediction': 0.445},  # brain - caudate
        {'tissue_id': '20', 'prediction': 0.512},  # brain - cortex
        {'tissue_id': '21', 'prediction': 0.498},  # brain - frontal cortex
        {'tissue_id': '22', 'prediction': 0.556},  # brain - hippocampus
        {'tissue_id': '43', 'prediction': 0.387},  # liver
        {'tissue_id': '44', 'prediction': 0.612},  # lung
        {'tissue_id': '56', 'prediction': 0.434},  # spleen
    ]
    
    return pd.DataFrame(sample_data)


if __name__ == "__main__":
    # Test the prediction processor
    anatomagram_dir = Path(__file__).parent.parent
    processor = AnatomagramPredictionProcessor(str(anatomagram_dir))
    
    # Create sample predictions
    sample_predictions = create_sample_predictions()
    print("Sample AD Risk Predictions:")
    print(sample_predictions)
    print()
    
    # Process predictions
    viz_data = processor.process_predictions(sample_predictions, aggregation_strategy='mean')
    
    # Generate debug report
    processor.generate_debug_report(sample_predictions, viz_data)
    
    # Create enhanced tooltips
    tooltips = processor.create_enhanced_tooltips(viz_data)
    print(f"\nENHANCED TOOLTIPS:")
    for svg_id, tooltip in list(tooltips.items())[:3]:  # Show first 3
        print(f"{svg_id}:")
        print(f"  {tooltip}")
        print()