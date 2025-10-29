"""Unified prediction converter backend for VCF2Risk and VCF2Expression predictions."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re

# Use relative import instead of sys.path.append
import sys
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from prediction_processor import AnatomagramPredictionProcessor


class PredictionConverter:
    """Unified backend for all prediction types (risk, expression, etc)."""

    def __init__(self, anatomagram_dir: Optional[str] = None, aggregation_strategy: str = 'mean'):
        """Initialize unified converter.

        Args:
            anatomagram_dir: Path to anatomagram directory (defaults to parent of this file)
            aggregation_strategy: How to aggregate multiple predictions per SVG element
                                 Options: 'mean', 'max', 'min', 'weighted_mean'
        """
        if anatomagram_dir is None:
            anatomagram_dir = str(Path(__file__).parent.parent)

        self.anatomagram_dir = anatomagram_dir
        self.aggregation_strategy = aggregation_strategy

        # Initialize enhanced prediction processor
        try:
            self.processor = AnatomagramPredictionProcessor(self.anatomagram_dir)
            self.enhanced_mapping = self.processor.enhanced_mapping
            self._print_init_summary()
        except Exception as e:
            print(f"❌ Error loading enhanced system: {e}")
            self.processor = None
            self.enhanced_mapping = None

    def _print_init_summary(self):
        """Print initialization summary."""
        print(f"✅ Enhanced anatomagram system loaded successfully")
        print(f"   - Aggregation strategy: {self.aggregation_strategy}")
        print(f"   - Enhanced mapping with {len(self.enhanced_mapping['tissue_mappings'])} tissues")
        print(f"   - Hierarchy fallbacks: {len(self.enhanced_mapping['hierarchy_fallbacks'])}")

    def convert(
        self,
        predictions_df: pd.DataFrame,
        item_name: str,
        prediction_type: str,
        aggregation_strategy: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Universal conversion method.

        Args:
            predictions_df: DataFrame with predictions (format depends on type)
            item_name: Name for the item in visualization
            prediction_type: Type of prediction ("risk" or "expression")
            aggregation_strategy: Override default aggregation strategy

        Returns:
            Tuple of (anatomagram_data, enhanced_metadata)
        """
        if not self.processor:
            return self._fallback_conversion(predictions_df, item_name)

        # Preprocess based on type
        if prediction_type == "expression":
            processed_df = self._prepare_expression_data(predictions_df)
            prediction_col = 'expression_value'

            # Check if mapping succeeded
            if len(processed_df) == 0:
                print("❌ ERROR: No tissues could be mapped to anatomagram visualization!")
                return self._empty_result(item_name)
        else:  # risk
            processed_df = predictions_df.copy()
            prediction_col = 'ad_risk'

        strategy = aggregation_strategy or self.aggregation_strategy

        # Common processing pipeline (identical for both types!)
        viz_data = self.processor.process_predictions(
            processed_df,
            tissue_id_col='tissue_id',
            prediction_col=prediction_col,
            aggregation_strategy=strategy
        )

        # Convert to anatomagram format (identical for both types!)
        anatomagram_data = {
            "genes": {
                item_name: viz_data['svg_element_values']
            }
        }

        # Enhanced metadata for widget (identical for both types!)
        enhanced_metadata = {
            'coverage_report': viz_data['coverage_report'],
            'aggregation_details': viz_data['aggregation_details'],
            'enhanced_tooltips': self.processor.create_enhanced_tooltips(viz_data),
            'uberon_names': self.processor.uberon_names,
            'uberon_descriptions': self.processor.uberon_descriptions,  # Full anatomical descriptions
            'hierarchy_fallbacks': self.enhanced_mapping['hierarchy_fallbacks'],
            'enhanced_mapping': self.enhanced_mapping,
            'metadata': viz_data['metadata'],
            'prediction_type': prediction_type  # Track type for summary generation
        }

        return anatomagram_data, enhanced_metadata

    def _prepare_expression_data(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare expression data for processing.

        Supports both VCF2Expression (scalar per row) and VCF2Risk (nested lists) formats.
        Handles multiple column name variations and both scalar/list tissue data.
        Creates reverse mapping from tissue names to tissue IDs with normalization.

        This is the ONLY major difference between risk and expression converters.
        """

        def to_list(obj):
            """Convert various input formats to list."""
            if isinstance(obj, (list, tuple, np.ndarray)):
                return list(obj)
            elif isinstance(obj, str) and ',' in obj:
                return [s.strip() for s in obj.split(',')]
            elif pd.isna(obj):
                return []
            else:
                return [obj]

        def canonicalize(tissue_name):
            """Canonicalize tissue name: lowercase, normalize spacing and dashes."""
            return re.sub(r'\s*-\s*', ' - ',
                         re.sub(r'\s+', ' ',
                               str(tissue_name).strip().lower().replace('_', ' ')))

        rows = []

        # Create reverse mappings: tissue_name -> tissue_id and tissue_id -> tissue_info
        tissue_name_to_id = {}
        id_to_tissue_info = {}
        for tissue_id, tissue_info in self.enhanced_mapping['tissue_mappings'].items():
            tissue_name_to_id[tissue_info['tissue_name']] = tissue_id
            id_to_tissue_info[tissue_id] = tissue_info

        # Non-destructive tissue aliases (preserve "whole blood" vs "blood" distinction)
        tissue_aliases = {
            'aorta': 'artery - aorta',
            'aortic artery': 'artery - aorta',
            'thyroid_gland': 'thyroid',
            'whole_blood': 'whole blood',  # Keep distinct from 'blood'
            'blood_whole': 'whole blood'
        }

        # Build canonicalized mappings
        canonical_tissue_map = {}
        for tissue_name, tissue_id in tissue_name_to_id.items():
            canonical_tissue_map[canonicalize(tissue_name)] = tissue_id

        # Add alias mappings
        alias_tissue_map = {}
        for alias, canonical in tissue_aliases.items():
            if canonical in tissue_name_to_id:
                alias_tissue_map[canonicalize(alias)] = tissue_name_to_id[canonical]

        # Determine tissue column: first present in order of preference
        tissue_col = None
        for col in ['tissue_names', 'tissues', 'tissue_name', 'tissue']:
            if col in predictions_df.columns:
                tissue_col = col
                break

        if not tissue_col:
            print("❌ ERROR: No tissue column found in predictions DataFrame!")
            print(f"Available columns: {list(predictions_df.columns)}")
            return pd.DataFrame(rows)

        unmapped_tissues = set()

        for idx, row in predictions_df.iterrows():
            tissues = to_list(row[tissue_col])
            expressions = to_list(row.get('predicted_expression', []))

            # Handle broadcasting: single expression to multiple tissues
            if len(expressions) == 1 and len(tissues) > 1:
                expressions = expressions * len(tissues)

            # Process tissue-expression pairs
            for i, tissue in enumerate(tissues):
                # Get expression value with bounds checking
                if i < len(expressions):
                    expr_val = expressions[i]
                else:
                    expr_val = expressions[0] if expressions else 0.0

                # Unwrap nested expression lists and convert to float
                if isinstance(expr_val, (list, tuple, np.ndarray)):
                    expression_value = float(expr_val[0]) if expr_val else 0.0
                else:
                    expression_value = float(expr_val)

                # Tissue ID lookup strategy
                tissue_id = None
                original_tissue = tissue

                # Strategy 1: Direct tissue_id lookup (for numeric inputs)
                tissue_str = str(tissue).strip()
                if tissue_str.isdigit() and tissue_str in id_to_tissue_info:
                    tissue_id = tissue_str

                # Strategy 2: Exact name match
                if not tissue_id:
                    tissue_id = tissue_name_to_id.get(tissue_str)

                # Strategy 3: Canonicalized name match
                if not tissue_id:
                    canonical_tissue = canonicalize(tissue_str)
                    tissue_id = canonical_tissue_map.get(canonical_tissue)

                # Strategy 4: Alias lookup
                if not tissue_id:
                    tissue_id = alias_tissue_map.get(canonical_tissue)

                # Strategy 5: Fuzzy matching (existing method)
                if not tissue_id:
                    tissue_id = self._fuzzy_match_tissue_name(tissue_str, tissue_name_to_id)

                # Process successful mapping
                if tissue_id:
                    canonical_name = id_to_tissue_info[tissue_id]['tissue_name']
                    rows.append({
                        'tissue_id': tissue_id,
                        'tissue_name': canonical_name,
                        'expression_value': expression_value
                    })

                    # Log mapping info if transformation occurred
                    if str(original_tissue) != canonical_name:
                        print(f"Info: Mapped '{original_tissue}' → '{canonical_name}' (ID: {tissue_id})")
                else:
                    unmapped_tissues.add(str(tissue))

        # Enhanced error reporting
        if len(rows) == 0:
            print("❌ ERROR: No tissue names could be mapped to known tissues!")
            print("\n📋 Debug Information:")
            print(f"   - Input format: Column '{tissue_col}' with {len(predictions_df)} rows")
            print(f"   - Available mappings: {len(tissue_name_to_id)} tissues")

            if unmapped_tissues:
                print(f"\n❌ Unmapped inputs (first 5): {list(unmapped_tissues)[:5]}")

            print(f"\n✅ Available tissue names (first 10):")
            for name in sorted(tissue_name_to_id.keys())[:10]:
                print(f"     - {name}")
            if len(tissue_name_to_id) > 10:
                print(f"     ... and {len(tissue_name_to_id) - 10} more")
        else:
            print(f"✅ Mapped {len(rows)} tissue-expression pairs from {len(predictions_df)} input rows")
            if unmapped_tissues:
                print(f"⚠️  Could not map {len(unmapped_tissues)} tissue names: {list(unmapped_tissues)[:3]}...")

        return pd.DataFrame(rows)

    def _fuzzy_match_tissue_name(self, target_name: str, tissue_name_to_id: Dict[str, str]) -> Optional[str]:
        """Attempt fuzzy matching for tissue names."""
        target_lower = target_name.lower().strip()

        # Try partial matches
        for mapped_name, tissue_id in tissue_name_to_id.items():
            mapped_lower = mapped_name.lower().strip()

            if target_lower in mapped_lower or mapped_lower in target_lower:
                # Only match if strings are reasonably long to avoid false positives
                if len(target_lower) >= 4 and len(mapped_lower) >= 4:
                    return tissue_id

        return None

    def _empty_result(self, item_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return empty but valid result structure."""
        return {
            "genes": {item_name: {}}
        }, {
            'coverage_report': {'visualization_coverage': 0, 'direct_svg_matches': 0, 'hierarchy_fallbacks': 0},
            'aggregation_details': {},
            'enhanced_tooltips': {},
            'hierarchy_fallbacks': self.enhanced_mapping['hierarchy_fallbacks'] if self.enhanced_mapping else {},
            'metadata': {}
        }

    def _fallback_conversion(self, predictions_df: pd.DataFrame, item_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Fallback conversion if enhanced system fails."""
        print("⚠️  Using basic fallback conversion - enhanced features not available")
        return {"genes": {item_name: {}}}, {'fallback_mode': True}
