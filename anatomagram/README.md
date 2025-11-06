# Anatomagram Visualization System

Interactive anatomical visualizations for DNA2Cell genomics workflows. Display tissue-specific gene expression and disease risk predictions on human anatomogram diagrams.

## Overview

The anatomagram visualization system provides interactive anatomical diagrams that show genomic data mapped to specific tissues and organs. Data is colored on SVG-based anatomical diagrams with hover tooltips showing detailed information.

### Key Features

- **Multi-view support**: Male, female, and brain anatomagrams
- **Enhanced tooltips**: Tissue descriptions, hierarchical mappings, and aggregation details
- **Responsive scaling**: Automatic sizing based on container dimensions
- **Theme-aware styling**: Adapts to light/dark themes
- **Performance optimization**: SVG caching and cached dimensions for sub-200ms tab switching
- **100% tissue coverage**: All model predictions can be visualized

## Widget Types

### AnatomagramWidget
Single-view anatomagram widget for displaying one anatomical view at a time.

```python
from anatomagram.components import AnatomagramWidget

widget = AnatomagramWidget(
    visualization_data={"genes": {"APOE": {"UBERON_0000955": 0.85, ...}}},
    selected_item="APOE",
    sex="male",  # "male", "female", or "brain"
    color_palette="viridis",
    scale_type="linear",
    debug=False
)
```

### AnatomagramMultiViewWidget
Multi-view anatomagram with tabbed interface and shared caching across views for optimal performance.

```python
from anatomagram.components import AnatomagramMultiViewWidget

widget = AnatomagramMultiViewWidget(
    visualization_data={"genes": {"APOE": {...}}},
    selected_item="APOE",
    available_views=["male", "female", "brain"],
    current_view="male",
    color_palette="viridis",
    scale_type="linear",
    debug=False
)

# Create marimo tabs for view switching
tabs = widget.create_view_tabs(mo)
```

## Usage Examples

### VCF2Expression Workflow

```python
from processors.vcfprocessor import VCFProcessor
from anatomagram.components import convert_vcf_expression_predictions, AnatomagramMultiViewWidget

# Run VCF2Expression predictions
vcf_processor = VCFProcessor(model_class='v4_pcg')
model, checkpoint_path, trainer = vcf_processor.load_model()
predictions_df = vcf_processor.predict(...)

# Convert to anatomagram format
viz_data, uberon_map, enhanced_tooltips = convert_vcf_expression_predictions(
    predictions_df,
    gene_name="APOE",
    aggregation_strategy='mean'
)

# Create widget
widget = AnatomagramMultiViewWidget(
    visualization_data=viz_data,
    selected_item="APOE",
    available_views=["male", "female", "brain"],
    uberon_names=uberon_map,
    enhanced_tooltips=enhanced_tooltips
)
```

### VCF2Risk Workflow

```python
from processors.ad_risk import ADRiskPredictor
from anatomagram.components import convert_vcf_risk_predictions, AnatomagramWidget

# Run VCF2Risk predictions
risk_predictor = ADRiskPredictor()
predictions_df = risk_predictor.predict(...)

# Convert to anatomagram format
viz_data, uberon_map, enhanced_tooltips = convert_vcf_risk_predictions(
    predictions_df,
    risk_item_name="AD_RISK",
    aggregation_strategy='mean'
)

# Create widget
widget = AnatomagramWidget(
    visualization_data=viz_data,
    selected_item="AD_RISK",
    sex="brain",
    data_type="risk",
    uberon_names=uberon_map,
    enhanced_tooltips=enhanced_tooltips
)
```

## Widget Parameters

### Common Parameters

- **visualization_data** (Dict): Nested dict `{"genes": {gene_id: {uberon_id: value}}}`
- **selected_item** (str): Gene ID or risk item to display
- **color_palette** (str): D3 color scheme (`"viridis"`, `"magma"`, `"inferno"`, `"plasma"`, `"turbo"`, `"cividis"`, `"warm"`, `"cool"`)
- **scale_type** (str): Color scale type (`"linear"` or `"log"`)
- **data_type** (str): Data type for labels (`"expression"` or `"risk"`)
- **threshold** (float): Minimum value to color (default: 0.0)
- **size_scale** (float): Widget size multiplier (default: 1.0)
- **no_data_opacity** (float): Opacity for tissues without data (default: 0.5)
- **uberon_names** (Dict): UBERON ID to tissue name mapping for tooltips
- **enhanced_tooltips** (Dict): SVG element ID to enhanced tooltip HTML mapping
- **debug** (bool): Enable debug logging in browser console

### Single-View Parameters

- **sex** (str): Anatomagram view (`"male"`, `"female"`, or `"brain"`)

### Multi-View Parameters

- **available_views** (List[str]): Views to load and cache (e.g., `["male", "female", "brain"]`)
- **current_view** (str): Initially displayed view

## Tissue Mapping System

The anatomagram system uses a two-layer architecture:

```
Layer 1: vocabs/tissue_vocab.yaml
  ↓ (core model tissue vocabulary)

Layer 2: anatomagram/data/tissue_mapping_enhanced.json
  ↓ (visualization mappings + hierarchy fallbacks)

Output: Colored anatomagram SVGs
```

### Direct Matches vs Hierarchy Fallbacks

- **Direct Match**: Tissue has exact SVG element (e.g., `brain - cortex` → cerebral cortex SVG)
- **Hierarchy Fallback**: Tissue maps to parent structure (e.g., `adipose - subcutaneous` → generic adipose tissue SVG)

Hierarchy fallbacks ensure 100% visualization coverage even when the SVG anatomogram doesn't have fine-grained anatomical detail.

**For details**: See [data/README.md](data/README.md)

## Test Notebooks

### Comprehensive Testing
- **test_all_anatomagrams.py**: Tests all three anatomagram views with sample data
- **test_female_anatomagram.py**: Validates female-specific tissue coloring
- **test_brain_anatomagram.py**: Validates brain region coloring
- **test_real_vcf2exp_predictions.py**: Tests with real VCF2Expression predictions

```bash
marimo edit notebooks/test_all_anatomagrams.py
```

### Production Notebooks
- **vcf2exp.py**: Full VCF2Expression workflow with anatomagram visualization
- **vcf2risk.py**: Full VCF2Risk workflow with anatomagram visualization

## Validation

### Validate Tissue Mappings
```bash
python anatomagram/utils/validate_mappings.py
```

Checks:
- Vocab ↔ mapping correspondence
- Mapping internal consistency
- Hierarchy fallback integrity
- Visualization coverage statistics

### Validate SVG Element Coverage
```bash
python anatomagram/utils/validate_svg_mappings.py
```

Checks:
- All UBERON IDs in SVG files have mappings
- Reports unmapped elements
- Provides coverage statistics per anatomagram

## Troubleshooting

### Tissue Not Coloring?

1. **Enable debug mode**:
   ```python
   widget = AnatomagramWidget(..., debug=True)
   ```
   Check browser console (F12) for debug output.

2. **Run validation**:
   ```bash
   python anatomagram/utils/validate_svg_mappings.py
   ```

3. **Check data format**:
   ```python
   # Correct format
   viz_data = {
       "genes": {
           "GENE_ID": {
               "UBERON_0000955": 0.85,  # UBERON ID → value
               "UBERON_0002107": 0.72
           }
       }
   }
   ```

### Female/Brain Anatomagram Not Showing Tissues?

- Verify `sex="female"` or `sex="brain"` parameter
- Check that UBERON IDs in data match female/brain SVG elements
- Run validation: `python anatomagram/utils/validate_svg_mappings.py`

### Tooltips Not Showing?

- Ensure `uberon_names` parameter is provided
- Check `uberon_descriptions.json` has entries for UBERON IDs
- Verify `enhanced_tooltips` parameter if using hierarchical tooltips

### Performance Issues?

- Use `AnatomagramMultiViewWidget` for tabbed views (shared caching)
- Enable SVG caching with `available_views` parameter
- Reduce `size_scale` if rendering is slow

### Theme Not Applying?

- Widget uses CSS custom properties for theming
- Ensure `czi-sds-theme.css` is loaded if using custom styles
- Check browser console for CSS errors

## Architecture

### Components

- **base_widget.py**: `BaseAnatomagramWidget` with shared traitlets and methods
- **anatomagram_widget.py**: `AnatomagramWidget` (single-view) and `AnatomagramMultiViewWidget` (multi-view)
- **prediction_converter.py**: Base converter with shared prediction processing
- **vcf_risk_converter.py**: VCF2Risk-specific conversion
- **data_processor.py**: AnatomagramDataProcessor for predictions

### Data

- **tissue_mapping_enhanced.json**: Primary tissue → UBERON → SVG mapping
- **uberon_descriptions.json**: UBERON IDs with labels and descriptions
- **tissue_descendants.json**: UBERON ontology hierarchy
- **svg_analysis_results.json**: SVG metadata and coverage statistics

### Assets

- **homo_sapiens.male.svg**: Male anatomagram (150 UBERON elements)
- **homo_sapiens.female.svg**: Female anatomagram (156 UBERON elements)
- **homo_sapiens.brain.svg**: Brain anatomagram (~50 UBERON elements)

## Performance

- **Initial render**: <280ms (cached dimensions)
- **Tab switching**: <200ms (shared SVG caching)
- **Tooltip rendering**: <10ms (pre-computed HTML)
- **Memory usage**: Minimal (shared cache across views)

## Development

### Adding New Tissues

1. Add to `vocabs/tissue_vocab.yaml`
2. Add mapping to `anatomagram/data/tissue_mapping_enhanced.json`
3. Run validation: `python anatomagram/utils/validate_mappings.py`
4. Test visualization: `marimo edit notebooks/test_all_anatomagrams.py`

See [data/README.md](data/README.md) for detailed instructions.

### Running Tests

```bash
# Run validation
python anatomagram/utils/validate_mappings.py
python anatomagram/utils/validate_svg_mappings.py

# Visual testing
marimo edit notebooks/test_all_anatomagrams.py
marimo edit notebooks/test_female_anatomagram.py
marimo edit notebooks/test_brain_anatomagram.py
```

## References

- **UBERON Ontology**: http://uberon.github.io/
- **Expression Atlas**: https://www.ebi.ac.uk/gxa/home
- **GTEx Portal**: https://gtexportal.org/
- **AnyWidget**: https://anywidget.dev/

## Version History

- **v0.4.0** (Oct 2024): BaseAnatomagramWidget refactoring, enhanced tooltips, 100% coverage
- **v0.3.0** (Oct 2024): AnatomagramMultiViewWidget with shared caching
- **v0.2.0** (Oct 2024): Responsive scaling and theme-aware styling
- **v0.1.0** (Sep 2024): Initial release with single-view widget
