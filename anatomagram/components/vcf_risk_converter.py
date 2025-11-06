"""Enhanced VCF2Risk and VCF2Expression to Anatomagram converters using unified backend.

This file provides backward-compatible wrapper classes around the unified PredictionConverter.
All existing imports and APIs work unchanged.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Import unified backend
from .prediction_converter import PredictionConverter


class EnhancedVCFRiskConverter:
    """Enhanced VCF2Risk to anatomagram converter - WRAPPER around PredictionConverter."""

    def __init__(self, anatomagram_dir: Optional[str] = None, aggregation_strategy: str = 'mean'):
        """Initialize enhanced risk converter.

        Args:
            anatomagram_dir: Path to anatomagram directory (defaults to parent of this file)
            aggregation_strategy: How to aggregate multiple predictions per SVG element
        """
        # Use unified backend
        self._backend = PredictionConverter(anatomagram_dir, aggregation_strategy)

        # Expose backend properties for compatibility
        self.anatomagram_dir = self._backend.anatomagram_dir
        self.aggregation_strategy = self._backend.aggregation_strategy
        self.processor = self._backend.processor
        self.enhanced_mapping = self._backend.enhanced_mapping

    def convert_predictions_to_anatomagram(
        self,
        predictions_df: pd.DataFrame,
        risk_item_name: str = "AD_RISK",
        aggregation_strategy: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Convert VCF2Risk predictions using enhanced processor.

        Args:
            predictions_df: DataFrame with columns: tissue_id, tissue_name, ad_risk
            risk_item_name: Name to use for the risk "gene" in anatomagram format
            aggregation_strategy: Override default aggregation strategy

        Returns:
            Tuple of (anatomagram_data, visualization_metadata)
        """
        return self._backend.convert(
            predictions_df,
            item_name=risk_item_name,
            prediction_type="risk",
            aggregation_strategy=aggregation_strategy
        )

    def get_uberon_map(self) -> Dict[str, str]:
        """Get UBERON map with RISK-SPECIFIC formatting.

        CRITICAL: Risk format adds "(aggregated)" suffix for hierarchy fallbacks.

        Returns:
            Dictionary mapping SVG UBERON IDs to enhanced display names
        """
        if not self.enhanced_mapping:
            return {}

        uberon_map = {}
        for tissue_data in self.enhanced_mapping['tissue_mappings'].values():
            svg_uberon_id = tissue_data['svg_uberon_id']
            display_name = tissue_data['display_name']

            if svg_uberon_id and display_name:
                # Risk-specific: add (aggregated) suffix for hierarchy fallbacks
                if not tissue_data['is_direct_match']:
                    display_name = f"{display_name} (aggregated)"
                uberon_map[svg_uberon_id] = display_name

        return uberon_map

    def get_enhanced_summary(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """Get risk-specific summary with coverage analysis.

        Args:
            predictions_df: The predictions DataFrame

        Returns:
            Comprehensive summary of prediction processing
        """
        if not self.processor:
            return {"error": "Enhanced processor not available"}

        # Process predictions to get coverage data
        viz_data = self.processor.process_predictions(
            predictions_df,
            tissue_id_col='tissue_id',
            prediction_col='ad_risk',
            aggregation_strategy=self.aggregation_strategy
        )

        coverage = viz_data['coverage_report']
        metadata = viz_data['metadata']

        return {
            "enhanced_system": True,
            "input_predictions": len(predictions_df),
            "visualizable_tissues": coverage['total_tissues'],
            "direct_svg_matches": coverage['direct_svg_matches'],
            "hierarchy_fallbacks": coverage['hierarchy_fallbacks'],
            "visualization_coverage_percent": coverage['visualization_coverage'],
            "svg_elements_colored": metadata['svg_elements_colored'],
            "aggregation_strategy": metadata['aggregation_strategy'],
            "cannot_visualize": coverage['cannot_visualize'],
            "cell_lines_excluded": metadata.get('cell_lines_excluded', 0),
            "hierarchy_fallback_count": len(self.enhanced_mapping['hierarchy_fallbacks'])
        }


class EnhancedVCFExpressionConverter:
    """Enhanced VCF2Expression to anatomagram converter - WRAPPER around PredictionConverter."""

    def __init__(self, anatomagram_dir: Optional[str] = None, aggregation_strategy: str = 'mean'):
        """Initialize enhanced expression converter.

        Args:
            anatomagram_dir: Path to anatomagram directory (defaults to parent of this file)
            aggregation_strategy: How to aggregate multiple predictions per SVG element
        """
        # Use unified backend
        self._backend = PredictionConverter(anatomagram_dir, aggregation_strategy)

        # Expose backend properties for compatibility
        self.anatomagram_dir = self._backend.anatomagram_dir
        self.aggregation_strategy = self._backend.aggregation_strategy
        self.processor = self._backend.processor
        self.enhanced_mapping = self._backend.enhanced_mapping

    def convert_predictions_to_anatomagram(
        self,
        predictions_df: pd.DataFrame,
        gene_name: str = "SELECTED_GENE",
        aggregation_strategy: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Convert VCF2Expression predictions using enhanced processor.

        Args:
            predictions_df: DataFrame with VCF expression predictions
                           Expected columns: tissue_names, predicted_expression
            gene_name: Name to use for the gene in anatomagram format
            aggregation_strategy: Override default aggregation strategy

        Returns:
            Tuple of (anatomagram_data, visualization_metadata)
        """
        return self._backend.convert(
            predictions_df,
            item_name=gene_name,
            prediction_type="expression",
            aggregation_strategy=aggregation_strategy
        )

    def get_uberon_map(self) -> Dict[str, str]:
        """Get UBERON map with EXPRESSION-SPECIFIC formatting.

        CRITICAL: Expression format does NOT add suffix (different from risk).

        Returns:
            Dictionary mapping SVG UBERON IDs to display names
        """
        if not self.enhanced_mapping:
            return {}

        uberon_map = {}
        for tissue_data in self.enhanced_mapping['tissue_mappings'].values():
            svg_uberon_id = tissue_data['svg_uberon_id']
            display_name = tissue_data['display_name']

            if svg_uberon_id and display_name:
                # Expression-specific: NO suffix added (different from risk)
                uberon_map[svg_uberon_id] = display_name

        return uberon_map

    def get_enhanced_summary(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """Get expression-specific summary with range statistics.

        Args:
            predictions_df: The predictions DataFrame

        Returns:
            Comprehensive summary including expression range
        """
        if not self.processor:
            return {'error': 'Enhanced processor not available'}

        expression_df = self._backend._prepare_expression_data(predictions_df)
        viz_data = self.processor.process_predictions(
            expression_df,
            tissue_id_col='tissue_id',
            prediction_col='expression_value',
            aggregation_strategy=self.aggregation_strategy
        )

        return {
            'input_predictions': len(predictions_df),
            'flattened_predictions': len(expression_df),
            'unique_tissues': len(expression_df['tissue_name'].unique()) if len(expression_df) > 0 else 0,
            'visualization_coverage_percent': viz_data['coverage_report']['visualization_coverage'],
            'direct_svg_matches': viz_data['coverage_report']['direct_svg_matches'],
            'hierarchy_fallbacks': viz_data['coverage_report']['hierarchy_fallbacks'],
            'svg_elements_colored': len(viz_data['svg_element_values']),
            'aggregation_strategy': self.aggregation_strategy,
            'expression_range': {
                'min': float(expression_df['expression_value'].min()) if len(expression_df) > 0 else 0,
                'max': float(expression_df['expression_value'].max()) if len(expression_df) > 0 else 0,
                'mean': float(expression_df['expression_value'].mean()) if len(expression_df) > 0 else 0
            }
        }

    def _fuzzy_match_tissue_name(self, target_name: str, tissue_name_to_id: Dict[str, str]) -> Optional[str]:
        """Delegate to backend for fuzzy matching."""
        return self._backend._fuzzy_match_tissue_name(target_name, tissue_name_to_id)


# Legacy converter class (kept for compatibility)
class VCFRiskToAnatomagramConverter:
    """Legacy converter - redirects to enhanced system."""

    def __init__(self, tissue_mapping_path: Optional[str] = None):
        """Initialize legacy converter (redirects to enhanced)."""
        print("Note: VCFRiskToAnatomagramConverter is deprecated. Use EnhancedVCFRiskConverter instead.")
        self.converter = EnhancedVCFRiskConverter()

    def convert_predictions_to_anatomagram(self, predictions_df: pd.DataFrame):
        """Convert using enhanced system."""
        return self.converter.convert_predictions_to_anatomagram(predictions_df)

    def get_uberon_map(self):
        """Get UBERON map."""
        return self.converter.get_uberon_map()


# Public API function for VCF2Risk
def convert_vcf_risk_predictions(
    predictions_df: pd.DataFrame,
    tissue_mapping_path: Optional[str] = None,
    aggregation_strategy: str = 'mean'
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Convert VCF2Risk predictions using enhanced anatomagram system.

    Args:
        predictions_df: DataFrame with VCF2Risk predictions
        tissue_mapping_path: Optional path (ignored - uses enhanced system)
        aggregation_strategy: How to aggregate multiple predictions per SVG element

    Returns:
        Tuple of (anatomagram_data, uberon_map)
    """
    # Use enhanced converter
    converter = EnhancedVCFRiskConverter(aggregation_strategy=aggregation_strategy)

    # Convert predictions
    anatomagram_data, enhanced_metadata = converter.convert_predictions_to_anatomagram(predictions_df)
    uberon_map = converter.get_uberon_map()

    # Print enhanced summary for user feedback
    summary = converter.get_enhanced_summary(predictions_df)
    print(f"\n=== Enhanced Anatomagram Conversion Results ===")
    print(f"Input predictions: {summary['input_predictions']}")
    print(f"Visualization coverage: {summary['visualization_coverage_percent']:.1f}%")
    print(f"Direct SVG matches: {summary['direct_svg_matches']}")
    print(f"Hierarchy fallbacks: {summary['hierarchy_fallbacks']}")
    print(f"SVG elements colored: {summary['svg_elements_colored']}")
    print(f"Aggregation strategy: {summary['aggregation_strategy']}")

    if summary['hierarchy_fallbacks'] > 0:
        print(f"Note: {summary['hierarchy_fallbacks']} tissues use intelligent hierarchy mapping")

    return anatomagram_data, uberon_map


# Public API function for VCF2Expression
def convert_vcf_expression_predictions(
    predictions_df: pd.DataFrame,
    gene_name: str = "SELECTED_GENE",
    aggregation_strategy: str = 'mean'
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Convert VCF2Expression predictions using enhanced anatomagram system.

    Args:
        predictions_df: DataFrame with VCF2Expression predictions
        gene_name: Name for the gene in visualization
        aggregation_strategy: How to aggregate multiple predictions per SVG element

    Returns:
        Tuple of (anatomagram_data, uberon_map)
    """
    # Use enhanced expression converter
    converter = EnhancedVCFExpressionConverter(aggregation_strategy=aggregation_strategy)

    # Convert predictions
    anatomagram_data, enhanced_metadata = converter.convert_predictions_to_anatomagram(predictions_df, gene_name)
    uberon_map = converter.get_uberon_map()

    # Print enhanced summary for user feedback
    summary = converter.get_enhanced_summary(predictions_df)
    print(f"\n=== Enhanced Expression Anatomagram Conversion Results ===")
    print(f"Input predictions: {summary['input_predictions']}")
    print(f"Flattened predictions: {summary['flattened_predictions']} (from {summary['unique_tissues']} tissues)")
    print(f"Visualization coverage: {summary['visualization_coverage_percent']:.1f}%")
    print(f"Direct SVG matches: {summary['direct_svg_matches']}")
    print(f"Hierarchy fallbacks: {summary['hierarchy_fallbacks']}")
    print(f"SVG elements colored: {summary['svg_elements_colored']}")
    print(f"Aggregation strategy: {summary['aggregation_strategy']}")
    print(f"Expression range: {summary['expression_range']['min']:.3f} to {summary['expression_range']['max']:.3f} (mean: {summary['expression_range']['mean']:.3f})")

    if summary['hierarchy_fallbacks'] > 0:
        print(f"Note: {summary['hierarchy_fallbacks']} tissues use intelligent hierarchy mapping")

    return anatomagram_data, uberon_map
