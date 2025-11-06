"""Anatomagram visualization components for DNA2Cell."""

# Import data_processor
from .data_processor import ExpressionDataProcessor

# Import converter classes and functions directly (no longer need importlib workaround)
from .vcf_risk_converter import (
    EnhancedVCFRiskConverter,
    EnhancedVCFExpressionConverter,
    VCFRiskToAnatomagramConverter,
    convert_vcf_risk_predictions,
    convert_vcf_expression_predictions
)

# Import unified backend for advanced users
from .prediction_converter import PredictionConverter

# Lazy import for AnatomagramMultiViewWidget to avoid anywidget dependency issues
def AnatomagramMultiViewWidget(*args, **kwargs):
    """Lazy import wrapper for AnatomagramMultiViewWidget.

    Supports both multi-view and single-view modes:
    - Multi-view: available_views=["male", "female", "brain"]
    - Single-view: sex="brain" (auto-converts to single-element available_views)
    """
    from .anatomagram_widget import AnatomagramMultiViewWidget as _AnatomagramMultiViewWidget
    return _AnatomagramMultiViewWidget(*args, **kwargs)

__all__ = [
    'AnatomagramMultiViewWidget',  # Single unified widget (Step 7 cleanup)
    'ExpressionDataProcessor',
    'EnhancedVCFRiskConverter',
    'EnhancedVCFExpressionConverter',
    'VCFRiskToAnatomagramConverter',
    'PredictionConverter',
    'convert_vcf_risk_predictions',
    'convert_vcf_expression_predictions'
]
