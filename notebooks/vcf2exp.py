# /// script
# [tool.marimo.display]
# theme = "dark"
# ///

# TODO: Remove forced dark mode script above before production deployment
# This forces dark theme for development - remove to restore automatic light/dark switching

import marimo

__generated_with = "0.17.7"
app = marimo.App(
    app_title="VCF2Expression with Anatomagram",
    css_file="czi-sds-theme.css",
)


@app.cell(hide_code=True)
def _():

    import logging

    # Suppress verbose debug logs from markdown and matplotlib
    logging.getLogger('MARKDOWN').setLevel(logging.INFO)  # or WARNING
    logging.getLogger('matplotlib').setLevel(logging.INFO)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    logging.getLogger('matplotlib.pyplot').setLevel(logging.INFO)

    import marimo as mo
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Standard library imports
    import os
    import pandas as pd
    import tempfile
    from textwrap import dedent

    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, leaves_list

    # Project imports
    from processors.vcfprocessor import VCFProcessor
    from anatomagram.components.anatomagram_widget import AnatomagramMultiViewWidget

    # Import anatomagram components
    from anatomagram.components import convert_vcf_risk_predictions, convert_vcf_expression_predictions
    from anatomagram.components.vcf_risk_converter import EnhancedVCFRiskConverter, EnhancedVCFExpressionConverter
    return (
        AnatomagramMultiViewWidget,
        EnhancedVCFExpressionConverter,
        Path,
        VCFProcessor,
        go,
        leaves_list,
        linkage,
        mo,
        np,
        os,
        pd,
        squareform,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # VCF2Expression: Individual-Level Gene Expression Prediction with VariantFormer

    **Estimated time to complete:** 15-20 minutes | **Model:** VariantFormer (1.2B parameters)

    ## Learning Goals
    - Predict tissue-specific gene expression from individual genetic variants
    - Understand how mutations affect gene regulation across 63 human tissues and cell lines
    - Visualize sample-specific expression patterns on interactive anatomograms
    - Interpret the biological impact of genetic variation using a state-of-the-art foundation model

    ## Prerequisites
    - **Hardware**: GPU with 40GB+ VRAM (NVIDIA H100 recommended)
    - **Input Data**: VCF file with genetic variants (GRCh38 reference genome)
    - **Model**: Pre-trained VariantFormer checkpoint (14GB)
    - **Software**: VariantFormer package with anatomogram visualization components
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Understanding VariantFormer

    VariantFormer is a 1.2-billion-parameter transformer foundation model that predicts how an individual's unique combination of genetic variants affects gene expression across all major tissues of the human body. This tutorial demonstrates how VariantFormer can be used to analyze variant-to-expression relationships at the individual level—the first model capable of cross-gene, cross-tissue expression prediction from individual whole genomes.

    ### Key Innovations

    **Mutation-Aware Architecture**
    - Two-stage hierarchical design processes both regulatory regions and gene sequences
    - Pre-trained encoders capture variant effects on cis-regulatory elements (cCREs) and gene bodies
    - 25-layer CRE modulator and 25-layer gene modulator transformer stacks model complex regulatory interactions
    - Cross-attention mechanisms link distal regulatory elements to target genes

    **Tissue Specificity**
    - Tissue context embeddings enable predictions across 63 GTEx tissues and cell lines
    - Captures tissue-specific regulatory effects (e.g., brain vs. liver expression differences)
    - Trained on paired whole-genome sequencing and RNA-seq data from GTEx v8

    **Scientific Validation**
    - Model attention patterns correlate with experimental chromatin accessibility data
    - Validated against independent eQTL effect sizes across diverse populations
    - Predictions align with known genotype-phenotype associations in disease cohorts

    ### Analysis Workflow

    1. **Load VCF** → Input genetic variants (SNPs, indels) in standard VCF format
    2. **Select Genes & Tissues** → Choose genes and tissues to analyze across organ systems
    3. **VariantFormer Processing** → Model predicts individual-level expression values
    4. **Interactive Visualization** → Explore results using anatomogram visualizations

    Let's begin by loading a VCF file for analysis.
    """)
    return


@app.cell(hide_code=True)
def _(Path):
    # Constants for VCF analysis
    DEFAULT_VCF_PATH = str(Path(__file__).parent.parent / '_artifacts' / 'HG00096.vcf.gz')
    return (DEFAULT_VCF_PATH,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Load a VCF File
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    vcf_file_browser = mo.ui.file_browser(
        initial_path="./_artifacts/",
        filetypes=[".vcf", ".vcf.gz", ".gz"],
        selection_mode="file",
        multiple=False,
        label="Select VCF file for expression analysis"
    )

    mo.vstack([
        mo.md("**VCF File Selection:**"),
        vcf_file_browser,
        mo.md("""
        💡 **Browse and select VCF file.** Default location shows `HG00096.vcf.gz` sample.
        Navigate to select different VCF files from the cluster.
        """)
    ])
    return (vcf_file_browser,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configure Analysis Parameters

    ### Selecting Genes and Tissues

    VariantFormer can analyze expression for any protein-coding gene across 63 GTEx tissues (55 tissues) and cell lines (8 lines). This comprehensive analysis reveals how genetic variants affect gene regulation differently across organ systems.

    **Why tissue-specific analysis matters:**
    - The same genetic variant can increase expression in one tissue while decreasing it in another
    - Tissue-specific regulatory effects are critical for understanding genotype-phenotype relationships
    - VariantFormer's tissue embeddings capture context-dependent gene regulation mechanisms

    **Example Gene Categories:**
    - **Neurological**: APOE, APP, MAPT, PSEN1, PSEN2 (neurodegenerative disease associations)
    - **Metabolic**: LDLR, PCSK9 (lipid metabolism), CYP2D6, CYP2C19 (drug metabolism)
    - **Developmental**: FOXP2 (language), MC1R (pigmentation)
    - **General interest**: TP53, BRCA1/2, EGFR (widely studied genes)

    **Tissue Coverage:**
    Analysis includes all major organ systems: 16 brain regions, cardiovascular (heart, blood, arteries), digestive (liver, pancreas, stomach), respiratory (lung), urinary (kidney), musculoskeletal, endocrine, and immune tissues, plus 8 cancer cell lines commonly used in genomics research.
    """)
    return


@app.cell(hide_code=True)
def _(VCFProcessor, mo, pd):
    # Initialize VCF processor for expression analysis
    vcf_processor = VCFProcessor(model_class='v4_pcg')

    # Get available genes and tissues
    genes_df = vcf_processor.get_genes()
    tissues = vcf_processor.get_tissues()

    # Create gene selection with ALL genes (no limit)
    # Dict format: {label: gene_id} so dropdown displays labels
    # Marimo multiselect shows keys and returns values
    gene_options = {
        f"{row['gene_name']} | {row['gene_id']}": row['gene_id']
        for _, row in genes_df.iterrows()
    }

    # Create reverse mapping for label lookups
    gene_id_to_label = {v: k for k, v in gene_options.items()}

    # Define brain/neurological disease genes for default selection
    BRAIN_GENE_SYMBOLS = ['APOE', 'APP', 'MAPT', 'PSEN1', 'PSEN2', 'SNCA', 'BDNF', 'HTT']

    # Find corresponding formatted labels (dict keys)
    default_brain_gene_labels = []
    for symbol in BRAIN_GENE_SYMBOLS:
        matches = genes_df[genes_df['gene_name'] == symbol]
        if len(matches) > 0:
            _gene_id = matches.iloc[0]['gene_id']  # Local variable
            label = f"{symbol} | {_gene_id}"
            if label in gene_options:
                default_brain_gene_labels.append(label)
        else:
            print(f"⚠️  Brain gene {symbol} not found in dataset")

    # Multi-select with max_selections=1 for searchable single gene selection
    selected_genes = mo.ui.multiselect(
        options=gene_options,
        value=default_brain_gene_labels[:1] if len(default_brain_gene_labels) > 0 else [],  # Start with APOE only
        label="Select Gene to Analyze",
        max_selections=1
    )

    # Create comprehensive tissue selection for full anatomagram coverage
    comprehensive_tissue_names = [
        'adipose - subcutaneous', 'adipose - visceral (omentum)', 'blood',
        'brain - amygdala', 'brain - anterior cingulate cortex (ba24)', 'brain - caudate (basal ganglia)',
        'brain - cerebellar hemisphere', 'brain - cerebellum', 'brain - cortex',
        'brain - frontal cortex (ba9)', 'brain - hippocampus', 'brain - hypothalamus',
        'brain - nucleus accumbens (basal ganglia)', 'brain - putamen (basal ganglia)',
        'brain - spinal cord (cervical c-1)', 'brain - substantia nigra',
        'breast - mammary tissue', 'heart - left ventricle', 'kidney - cortex',
        'kidney - medulla', 'liver', 'lung', 'muscle - skeletal', 'nerve - tibial',
        'pancreas', 'spleen', 'stomach', 'thyroid'
    ]

    # Filter to only include tissues that exist in the system
    available_tissue_names = set(tissues)
    comprehensive_tissues = list(tissues)

    # Create tissue selection options
    tissue_options = list(comprehensive_tissues)

    # Create tissue metadata DataFrame
    _cell_lines = ['A549', 'Caki2', 'GM23248', 'HepG2', 'K562', 'NCI-H460', 'Panc1', 'SK-N-SH']

    def _categorize_tissue(tissue_name):
        """Categorize tissue by organ system."""
        name_lower = tissue_name.lower()
        if 'brain' in name_lower or 'nerve' in name_lower or 'spinal' in name_lower:
            return 'Nervous System'
        elif 'heart' in name_lower or 'artery' in name_lower or 'blood' in name_lower:
            return 'Cardiovascular'
        elif 'liver' in name_lower or 'pancreas' in name_lower or 'stomach' in name_lower:
            return 'Digestive'
        elif 'lung' in name_lower:
            return 'Respiratory'
        elif 'kidney' in name_lower:
            return 'Urinary'
        elif 'muscle' in name_lower:
            return 'Musculoskeletal'
        elif 'adipose' in name_lower or 'breast' in name_lower:
            return 'Integumentary'
        elif 'spleen' in name_lower or 'thyroid' in name_lower:
            return 'Endocrine/Immune'
        else:
            return 'Other'

    tissues_df = pd.DataFrame({
        'tissue_name': list(comprehensive_tissues),
        'system': [_categorize_tissue(t) for t in comprehensive_tissues],
        'source': ['Cell Line' if t in _cell_lines else 'GTEx Tissue' for t in comprehensive_tissues],
    })

    selected_tissues = mo.ui.multiselect(
        options=tissue_options,
        value=list(comprehensive_tissues),  # Default to all comprehensive tissues
        label="Select Tissues for Expression Analysis"
    )

    globals().update(dict(
        vcf_processor=vcf_processor,
        genes_df=genes_df,
        tissues=tissues,
        selected_genes=selected_genes,
        selected_tissues=selected_tissues,
        tissues_df=tissues_df,
        gene_options=gene_options,
        gene_id_to_label=gene_id_to_label,
        tissue_options=tissue_options,
    ))
    return (
        gene_id_to_label,
        genes_df,
        selected_genes,
        selected_tissues,
        tissues_df,
        vcf_processor,
    )


@app.cell
def _(genes_df, mo, selected_genes, selected_tissues, tissues_df):
    # Display selection summary (separate cell to avoid accessing .value in creation cell)
    _gene_count = len(selected_genes.value)  # List with 0 or 1 element
    mo.md(f"""
    **Selection Summary**:
    - **Gene**: {_gene_count} selected
    - **Tissues**: {len(selected_tissues.value)} selected

    💡 **Available**: {len(genes_df):,} total genes across {len(tissues_df)} tissues
    """)
    return


@app.cell
def _(genes_df, mo, pd, selected_genes):
    # Filter genes table based on multiselect with max_selections=1
    # selected_genes.value is a list with 0 or 1 gene IDs
    _selected_gene_ids = selected_genes.value

    if len(_selected_gene_ids) > 0:
        # Show only selected gene
        _filtered_genes_df = genes_df[genes_df['gene_id'] == _selected_gene_ids[0]]
    else:
        # Show nothing when nothing selected
        _filtered_genes_df = pd.DataFrame(columns=genes_df.columns)

    # Create filtered table without checkboxes
    genes_table_filtered = mo.ui.table(
        _filtered_genes_df,
        selection=None,
        show_column_summaries=False,
        label=f"Showing {len(_filtered_genes_df)} of {len(genes_df)} genes"
    )
    return (genes_table_filtered,)


@app.cell
def _(genes_table_filtered, mo, selected_genes):
    # Display gene selection UI
    mo.vstack([
        mo.md("### Gene Selection"),
        mo.md("**Select a gene using dropdown** (table shows selected gene):"),
        selected_genes,
        genes_table_filtered
    ])
    return


@app.cell
def _(mo, pd, selected_tissues, tissues_df):
    # Filter tissues table based on dropdown selection
    _selected_tissue_names = selected_tissues.value

    if len(_selected_tissue_names) > 0:
        # Show only selected tissues
        _filtered_tissues_df = tissues_df[tissues_df['tissue_name'].isin(_selected_tissue_names)]
    else:
        # Show nothing when nothing selected
        _filtered_tissues_df = pd.DataFrame(columns=tissues_df.columns)

    # Create filtered table without checkboxes
    tissues_table_filtered = mo.ui.table(
        _filtered_tissues_df,
        selection=None,
        show_column_summaries=False,
        label=f"Showing {len(_filtered_tissues_df)} of {len(tissues_df)} tissues"
    )
    return (tissues_table_filtered,)


@app.cell
def _(mo, selected_tissues, tissues_table_filtered):
    # Display tissue selection UI
    mo.vstack([
        mo.md("### Tissue Selection"),
        mo.md("**Select tissues using dropdown** (table shows selected tissues):"),
        selected_tissues,
        tissues_table_filtered
    ])
    return


@app.cell(hide_code=True)
def _(
    DEFAULT_VCF_PATH,
    gene_id_to_label,
    mo,
    os,
    selected_genes,
    selected_tissues,
    vcf_file_browser,
):
    # Format gene names for display
    # With multiselect (max_selections=1), selected_genes.value returns a list with 0 or 1 gene IDs
    if len(selected_genes.value) == 0:
        _gene_display = "No gene selected"
    else:
        _gene_id = selected_genes.value[0]
        _gene_label = gene_id_to_label.get(_gene_id, _gene_id)
        _gene_display = f"`{_gene_label}`"

    # Get VCF file path from file browser (use local variable - not returned)
    if vcf_file_browser.value and len(vcf_file_browser.value) > 0:
        _vcf_display_path = vcf_file_browser.value[0]
        _vcf_display = os.path.basename(_vcf_display_path.id)
    else:
        _vcf_display_path = DEFAULT_VCF_PATH
        _vcf_display = "HG00096.vcf.gz (default)"

    mo.md(f"""
    ## Running Expression Analysis

    **Configuration:**
    - VCF File: `{_vcf_display}`
    - Selected Gene: {_gene_display}
    - Tissues: {len(selected_tissues.value)} selected

    Analysis will begin automatically...
    """)
    pass
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Running VariantFormer Predictions

    ### Model Execution Pipeline

    VariantFormer processes the individual's genetic variants through its neural architecture:

    **1. Model Loading** (~2-3 minutes)
    - Loading 1.2 billion parameters including tissue-specific embedding modules
    - Initializing mutation-aware encoders and 50-layer transformer architecture (25 CRE + 25 gene modulators)
    - Setting up GPU acceleration for efficient inference

    **2. Variant Processing** (~30 seconds)
    - Parsing genetic variants from the VCF file (SNPs, indels, structural variants)
    - Mapping variants to cis-regulatory elements (cCREs) and gene body coordinates
    - Tokenizing sequences with BPE vocabulary and encoding variant effects

    **3. Expression Prediction** (~1-2 minutes)
    - CRE modulator layers (25 layers) analyze regulatory element perturbations
    - Gene modulator layers (25 layers) integrate gene body sequence context
    - Cross-attention mechanisms model distal enhancer-promoter interactions
    - Tissue-specific context embeddings produce tissue-resolved predictions

    **4. Output Generation**
    - Log-scale RNA abundance predictions for each gene-tissue pair
    - Individual-level expression values conditioned on the sample's variant profile
    - Results formatted for downstream visualization and analysis

    Progress indicators below track the analysis pipeline.
    """)
    return


@app.cell(hide_code=True)
def _(
    DEFAULT_VCF_PATH,
    gene_id_to_label,
    pd,
    selected_genes,
    selected_tissues,
    vcf_file_browser,
    vcf_processor,
):
    # Analysis runs automatically when selections change
    if len(selected_genes.value) == 0:
        print("⚠️  No gene selected. Please select a gene.")
        expression_predictions = None
        query_df = None
        all_gene_ids = []
        vcf_path = None
        gene_id = None
    else:
        # Get VCF path from file browser
        if vcf_file_browser.value and len(vcf_file_browser.value) > 0:
            vcf_path = vcf_file_browser.value[0].id  # Get selected file path (string)
            print(f"📁 Using selected VCF: {vcf_path}")
        else:
            vcf_path = DEFAULT_VCF_PATH
            print(f"📁 Using default VCF: {vcf_path}")

        try:
            # Process the selected gene
            # With multiselect (max_selections=1), selected_genes.value returns a list with 1 gene ID
            gene_id = selected_genes.value[0]  # Already the gene_id
            gene_label = gene_id_to_label.get(gene_id, gene_id)  # Look up formatted label

            # Store gene ID (single value, but keep as list for consistency)
            all_gene_ids = [gene_id]

            # Prepare comprehensive query for expression analysis
            selected_tissue_list = selected_tissues.value
            query_df = pd.DataFrame({
                'gene_id': [gene_id],
                'tissues': [','.join(selected_tissue_list)]
            })

            print(f"🔄 Creating dataset for expression analysis...")
            print(f"   VCF: {vcf_path}")
            print(f"   Gene: {gene_label}")
            print(f"   Gene ID: {gene_id}")
            print(f"   Tissues: {len(selected_tissue_list)} tissues selected")
            print(f"   First 5 tissues: {selected_tissue_list[:5]}")
            print(f"   Query shape: {query_df.shape}")

            # Create dataset and dataloader
            vcf_dataset, dataloader = vcf_processor.create_data(vcf_path, query_df)

            print(f"✅ Dataset created: {len(vcf_dataset)} samples, {len(dataloader)} batches")

            # Load model
            print(f"🔄 Loading VariantFormer model...")
            model, checkpoint_path, trainer = vcf_processor.load_model()
            print(f"✅ Model loaded successfully")

            # Run predictions
            print(f"🔄 Running expression predictions...")
            expression_predictions = vcf_processor.predict(model, checkpoint_path, trainer, dataloader, vcf_dataset)
            print(f"✅ Expression predictions complete: {expression_predictions.shape}")

        except Exception as e:
            print(f"❌ Error in expression analysis: {e}")
            expression_predictions = None
            all_gene_ids = []
            gene_id = None

    globals().update(dict(expression_predictions=expression_predictions, query_df=query_df, all_gene_ids=all_gene_ids))
    return (expression_predictions,)


@app.cell
def _(expression_predictions, mo):
    if expression_predictions is not None:
        print("📊 EXPRESSION PREDICTION RESULTS")
        print("=" * 50)
        print(f"Results shape: {expression_predictions.shape}")
        print(f"Columns: {list(expression_predictions.columns)}")

        # Display results
        mo.as_html(expression_predictions)
    else:
        mo.md("⏳ **Waiting for expression analysis to complete...**")
    return


@app.cell
def _(expression_predictions, mo):
    mo.ui.table(expression_predictions)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive Expression Visualization

    ### Understanding Individual-Level Expression Predictions

    The anatomogram displays predicted gene expression levels across human tissues conditioned on the individual's variant profile. This visualization shows how variant-specific regulatory effects manifest across different tissue contexts.

    **How to Interpret the Results:**

    **Color Intensity**
    - Warmer colors (red/yellow) = Higher predicted expression
    - Cooler colors (blue/purple) = Lower predicted expression
    - Gray = No data or very low expression
    - Values represent log2-transformed RNA abundance predictions

    **Interactive Features**
    - **Hover** over tissues to see expression values and tissue annotations
    - **Click** to view UBERON tissue ontology details
    - **Switch tabs** for male, female, and brain-specific anatomical views
    - Enhanced tooltips provide tissue system classifications

    **Scientific Interpretation**
    - Expression levels reflect the combined regulatory effects of all variants affecting the gene
    - Tissue-specific patterns reveal differential regulatory logic across cell types
    - Cross-tissue comparison identifies genes with ubiquitous vs. restricted expression
    - Results can be compared to GTEx population distributions to assess variant rarity

    **Note**: These are model predictions conditioned on the individual's variant genotype—they represent the model's learned mapping from variant profiles to expression phenotypes based on GTEx training data.
    """)
    return


@app.cell
def _(
    EnhancedVCFExpressionConverter,
    aggregation_strategy,
    expression_predictions,
    selected_genes,
):
    enhanced_converter = EnhancedVCFExpressionConverter(aggregation_strategy=aggregation_strategy.value)

    # Use the selected gene for anatomagram
    # With multiselect (max_selections=1), selected_genes.value is a list with 0 or 1 gene IDs
    if expression_predictions is not None and len(selected_genes.value) > 0:
        first_gene_id = selected_genes.value[0]  # Already the gene_id

        anatomagram_data, enhanced_metadata = enhanced_converter.convert_predictions_to_anatomagram(
              expression_predictions, gene_name=first_gene_id)

        # Extract all necessary assets
        uberon_map = enhanced_converter.get_uberon_map()
        enhanced_tooltips = enhanced_metadata['enhanced_tooltips']
        uberon_names = enhanced_metadata['uberon_names']
        uberon_descriptions = enhanced_metadata['uberon_descriptions']  # Full anatomical descriptions
    else:
        first_gene_id = None
        anatomagram_data = None
        uberon_map = {}
        enhanced_tooltips = {}
        uberon_names = {}
        uberon_descriptions = {}
    return (
        anatomagram_data,
        enhanced_tooltips,
        first_gene_id,
        uberon_descriptions,
        uberon_map,
        uberon_names,
    )


@app.cell
def _(
    AnatomagramMultiViewWidget,
    anatomagram_data,
    color_palette,
    enhanced_tooltips,
    first_gene_id,
    mo,
    scale_type,
    uberon_names,
):
    # Create multi-view anatomagram widget with reactive controls
    if anatomagram_data is not None and first_gene_id is not None:
        multi_widget = AnatomagramMultiViewWidget(
            visualization_data=anatomagram_data,
            selected_item=first_gene_id,
            available_views=["male", "female", "brain"],
            current_view="male",
            color_palette=color_palette.value,
            scale_type=scale_type.value,
            debug=False,  # Set to True for debugging on cluster
            uberon_names=uberon_names,
            enhanced_tooltips=enhanced_tooltips,
        )

        # Let the widget create its own tabs
        tabs = multi_widget.create_view_tabs(mo)
    else:
        multi_widget = None
        tabs = None
    return multi_widget, tabs


@app.cell
def _(
    anatomagram_data,
    enhanced_tooltips,
    first_gene_id,
    pd,
    uberon_descriptions,
    uberon_map,
):
    # Generate table summary from existing data
    def create_table_summary():
        """Create table summary from anatomagram data and tooltips."""
        if anatomagram_data is None or first_gene_id is None:
            return pd.DataFrame()

        _gene_id = first_gene_id  # Use local variable
        gene_data = anatomagram_data["genes"][_gene_id]

        table_rows = []
        for uberon_id, expression_value in gene_data.items():
            tissue_name = uberon_map.get(uberon_id, uberon_id)
            tooltip_info = enhanced_tooltips.get(uberon_id, "No details available")

            # Get UBERON description
            uberon_key = uberon_id.replace(':', '_')  # UBERON:123 -> UBERON_123
            description = uberon_descriptions.get(uberon_key, 'N/A')

            table_rows.append({
                'UBERON_Tissue_Name': tissue_name,
                'UBERON_ID': uberon_id,
                'Predicted_Expression': float(expression_value),
                'UBERON_Tissue_Description': description,
                'Details': tooltip_info
            })

        return pd.DataFrame(table_rows).sort_values('Predicted_Expression', ascending=False)

    table_summary = create_table_summary()
    return (table_summary,)


@app.cell
def _(multi_widget, tabs):
    # Reactive: update widget view when tab changes
    multi_widget.current_view = tabs.value
    return


@app.cell(hide_code=True)
def _(mo, tabs):
    mo.hstack([tabs], justify='center')
    return


@app.cell(hide_code=True)
def _(mo):
    # Analysis configuration
    aggregation_strategy = mo.ui.dropdown(
        options=[
            "mean",
            "max",
            "min",
            "weighted_mean"
        ],
        value="mean",
        label=""
    )

    color_palette = mo.ui.dropdown(
        options=[
            "viridis",
            "plasma",
            "inferno", 
            "turbo",
            "cividis"
        ],
        value="viridis",
        label=""
    )

    scale_type = mo.ui.dropdown(
        options=[
            "linear",
            "log"
        ],
        value="linear",
        label=""
    )

    mo.md(f"""
    **Visualization Controls:**
    - Aggregation: {aggregation_strategy}
    - Color: {color_palette} palette with {scale_type} scaling
    """)

    mo.hstack([aggregation_strategy, color_palette, scale_type], justify="center")
    return aggregation_strategy, color_palette, scale_type


@app.cell
def _(multi_widget):
    multi_widget
    return


@app.cell
def _(table_summary):
    table_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Exploratory Analysis of Pre-computed Results

    The following sections use **pre-computed predictions** from the sample VCF file (HG00096.vcf.gz) to demonstrate genome-wide expression patterns and tissue relationships. This data was generated by running VariantFormer on all ~18,000 protein-coding genes.

    ## Interactive Heatmap Visualization

    ### Genome-Wide Expression Heatmap

    This section loads the full expression predictions from a pre-computed dataset and creates an interactive clustered heatmap showing expression patterns across ~18,000 genes and 63 tissues. The heatmap uses hierarchical clustering to reveal biological relationships between genes and tissues.

    **Key Features:**
    - **Full Matrix**: ~18k genes × 63 tissues (~1.1M cells)
    - **WebGL Rendering**: Efficient pan/zoom for large matrices
    - **Hierarchical Clustering**: Correlation distance + Ward linkage
    - **Transformations**: log1p, z-score normalization, outlier clipping
    - **Drill-Down**: Function to explore specific gene subsets

    **How to Interpret:**
    - Color intensity shows relative expression (z-scored)
    - Clustered ordering groups similar genes/tissues together
    - Use pan/zoom to explore regions of interest
    - Hover for detailed gene/tissue/value information
    """)
    return


@app.cell(hide_code=True)
def _(Path, mo, np):
    # Configuration
    PARQUET_PATH = Path(__file__).parent / "example_data" / "HG00096_vcf2exp_predictions.parquet"
    APPLY_LOG1P = True
    ZSCORE_PER_GENE = True
    CLIP_SD = 2.5
    FLOAT_DTYPE = np.float32
    COLORSCALE = "RdBu"
    REVERSE_COLORSCALE = True

    mo.md(f"""
    **Loading Pre-Computed Expression Matrix**
    - Data: `{PARQUET_PATH}`
    - Transforms: log1p={APPLY_LOG1P}, z-score={ZSCORE_PER_GENE}, clip=±{CLIP_SD}σ
    """)
    return (
        APPLY_LOG1P,
        CLIP_SD,
        COLORSCALE,
        FLOAT_DTYPE,
        PARQUET_PATH,
        REVERSE_COLORSCALE,
        ZSCORE_PER_GENE,
    )


@app.cell
def _(
    APPLY_LOG1P,
    CLIP_SD,
    FLOAT_DTYPE,
    PARQUET_PATH,
    ZSCORE_PER_GENE,
    np,
    pd,
):
    # Load and process expression data
    print("🔄 Loading expression data...")
    df_raw = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
    print(f"   Loaded {df_raw.shape[0]:,} genes")

    def unwrap_nested_array(arr):
        """Unwrap deeply nested numpy arrays."""
        if isinstance(arr, np.ndarray):
            if len(arr) == 1:
                return unwrap_nested_array(arr[0])
            if all(isinstance(x, np.ndarray) and len(x) == 1 for x in arr):
                return np.array([float(x[0]) for x in arr])
        return arr

    # Extract tissue names
    tissues_nested = df_raw['tissues'].iloc[0]
    tissues_hm = unwrap_nested_array(tissues_nested)
    n_tissues = len(tissues_hm)
    print(f"   Found {n_tissues} tissues")

    # Build gene × tissue matrix
    genes_hm = []
    gene_ids_hm = []
    expr_matrix = []

    for idx, row in df_raw.iterrows():
        _gene_name_loop = row['gene_name']
        _gene_id_loop = row['gene_id']
        expr_nested = row['predicted_expression']
        expr = unwrap_nested_array(expr_nested)

        if len(expr) == n_tissues:
            genes_hm.append(_gene_name_loop)
            gene_ids_hm.append(_gene_id_loop)
            expr_matrix.append(expr)

    # Convert to numpy array
    A_hm = np.array(expr_matrix, dtype=FLOAT_DTYPE)
    print(f"✅ Matrix: {A_hm.shape[0]:,} genes × {A_hm.shape[1]} tissues = {A_hm.size:,} cells")

    # Apply transformations
    print("🔄 Applying transformations...")
    if APPLY_LOG1P:
        np.log1p(A_hm, out=A_hm)
        print("   ✓ log1p")

    if ZSCORE_PER_GENE:
        m = A_hm.mean(axis=1, keepdims=True)
        s = A_hm.std(axis=1, ddof=1, keepdims=True)
        s[s == 0] = 1.0
        A_hm = (A_hm - m) / s
        print("   ✓ z-score per gene")

    if CLIP_SD is not None:
        lo, hi = -float(CLIP_SD), float(CLIP_SD)
        np.clip(A_hm, lo, hi, out=A_hm)
        print(f"   ✓ clipped to [{lo:.1f}, {hi:.1f}]")

    print(f"   Data range: [{A_hm.min():.3f}, {A_hm.max():.3f}]")
    return A_hm, genes_hm, tissues_hm


@app.cell
def _(A_hm, genes_hm, leaves_list, linkage, np, squareform, tissues_hm):
    # Hierarchical clustering
    print("🔄 Performing hierarchical clustering...")

    def compute_ordered_leaves(X: np.ndarray, axis: int) -> np.ndarray:
        """Compute optimal leaf ordering using correlation distance and Ward linkage."""
        M = X if axis == 0 else X.T
        M = np.nan_to_num(M, nan=0.0)
        D = 1.0 - np.corrcoef(M)
        np.fill_diagonal(D, 0.0)
        Z = linkage(squareform(D, checks=False), method="ward")
        return leaves_list(Z)

    row_leaves_hm = compute_ordered_leaves(A_hm, axis=0)
    col_leaves_hm = compute_ordered_leaves(A_hm, axis=1)

    A_ord_hm = A_hm[row_leaves_hm][:, col_leaves_hm]
    genes_ord_hm = [genes_hm[i] for i in row_leaves_hm]
    tissues_ord_hm = [tissues_hm[j] for j in col_leaves_hm]

    print(f"✅ Clustering complete: {A_ord_hm.shape[0]:,} × {A_ord_hm.shape[1]}")
    return A_ord_hm, genes_ord_hm, tissues_ord_hm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tissue Hierarchical Clustering Dendrogram

    ### Understanding Tissue Relationships

    The dendrogram below shows hierarchical clustering of tissues based on their gene expression correlation patterns. Tissues with similar regulatory profiles cluster together, revealing biological relationships across organ systems.

    **Interpretation:**
    - **Vertical distance**: Height indicates dissimilarity between clusters (larger = more different)
    - **Branch structure**: Tissues that merge at lower heights have more similar expression profiles
    - **Biological insight**: Clustering often groups tissues by developmental origin or physiological function

    **Expected patterns:**
    - Brain regions cluster together (shared neural regulatory programs)
    - Metabolically active tissues group by function (liver, muscle, adipose)
    - Hormone-responsive tissues may cluster (breast, ovary, prostate)

    This dendrogram uses the same correlation distance metric and Ward linkage as the heatmap column ordering.
    """)
    return


@app.cell
def _(A_hm, go, linkage, np, squareform, tissues_hm):
    # Create tissue dendrogram with pre-computed linkage
    import scipy.cluster.hierarchy as sch

    print("🌳 Creating tissue dendrogram...")

    # Pre-compute linkage using correlation distance (same as heatmap)
    M_tissues = A_hm.T  # Transpose: tissues as rows, genes as columns
    M_tissues = np.nan_to_num(M_tissues, nan=0.0)

    # Compute correlation-based distance matrix
    corr_matrix = np.corrcoef(M_tissues)
    dist_matrix = 1.0 - corr_matrix
    np.fill_diagonal(dist_matrix, 0.0)

    # Compute Ward linkage from distance matrix
    Z_tissues = linkage(squareform(dist_matrix, checks=False), method='ward')

    # Use scipy dendrogram to get the structure
    dendro_data = sch.dendrogram(
        Z_tissues,
        labels=tissues_hm,
        orientation='top',
        no_plot=True,  # Don't plot with matplotlib
    )

    # Extract dendrogram coordinates for plotly
    icoord = np.array(dendro_data['icoord'])
    dcoord = np.array(dendro_data['dcoord'])
    ordered_labels = dendro_data['ivl']

    # Build plotly traces for dendrogram branches
    traces = []
    for i in range(len(icoord)):
        traces.append(
            go.Scatter(
                x=icoord[i],
                y=dcoord[i],
                mode='lines',
                line=dict(color='rgb(100,100,100)', width=1.5),
                hoverinfo='skip',
            )
        )

    # Create figure
    dendro_fig = go.Figure(data=traces)

    # Update layout
    dendro_fig.update_layout(
        title="Tissue Hierarchical Clustering Dendrogram (Correlation Distance)",
        height=300,
        width=1000,
        showlegend=False,
        xaxis=dict(
            tickvals=[5 + _i * 10 for _i in range(len(ordered_labels))],
            ticktext=ordered_labels,
            tickfont=dict(size=9),
            tickangle=45,
        ),
        yaxis=dict(
            title="Distance",
            showticklabels=True,
        ),
        margin=dict(l=100, r=20, t=40, b=120),
    )

    print(f"✅ Tissue dendrogram created ({len(tissues_hm)} tissues)")
    return (dendro_fig,)


@app.cell
def _(dendro_fig):
    dendro_fig
    return


@app.cell
def _(
    A_ord_hm,
    COLORSCALE,
    REVERSE_COLORSCALE,
    genes_ord_hm,
    go,
    tissues_ord_hm,
):
    # Create standalone heatmap figure
    print("🎨 Creating gene expression heatmap...")

    fig_hm = go.Figure(
        data=go.Heatmap(
            z=A_ord_hm,
            x=tissues_ord_hm,
            y=genes_ord_hm,
            colorscale=COLORSCALE,
            reversescale=REVERSE_COLORSCALE,
            colorbar=dict(title="Expression<br>(z-score)"),
            hovertemplate="<b>%{y}</b><br>%{x}<br>value=%{z:.3f}<extra></extra>",
            connectgaps=False,
        )
    )

    fig_hm.update_layout(
        title=f"Predicted Gene Expression across 55 tissues and 8 cell lines ({A_ord_hm.shape[0]:,} genes)",
        xaxis=dict(
            title="Tissues",
            tickfont=dict(size=9),
            tickangle=45,
        ),
        yaxis=dict(
            title="Genes",
            showticklabels=False,  # Hide 18k gene labels
        ),
        width=1000,
        height=800,
        dragmode="pan",
        margin=dict(l=100, r=20, t=60, b=120),
    )

    print("✅ Heatmap ready")
    print("💡 Use pan/zoom to explore")
    return (fig_hm,)


@app.cell
def _(fig_hm):
    fig_hm
    return


@app.cell
def _(
    A_ord_hm,
    COLORSCALE,
    REVERSE_COLORSCALE,
    genes_ord_hm,
    go,
    mo,
    np,
    tissues_ord_hm,
):
    # Drill-down helper function
    def plot_subset(genes_subset):
        """Plot an interactive subset of genes with richer hover information.

        Args:
            genes_subset: List of gene names to plot

        Returns:
            Plotly figure object

        Note:
            Plotly automatically uses WebGL rendering for large matrices.
        """
        gset = set(genes_subset)
        mask = [g in gset for g in genes_ord_hm]
        mask_idx = np.nonzero(mask)[0]

        if len(mask_idx) == 0:
            print(f"❌ No matching genes found in: {genes_subset}")
            return None

        subA = A_ord_hm[mask_idx, :]
        subG = [g for g in genes_ord_hm if g in gset]

        f = go.Figure(
            data=go.Heatmap(
                z=subA,
                x=tissues_ord_hm,
                y=subG,
                colorscale=COLORSCALE,
                reversescale=REVERSE_COLORSCALE,
                colorbar=dict(title="Expression"),
                hovertemplate="<b>%{y}</b><br>%{x}<br>value=%{z:.3f}<extra></extra>",
                connectgaps=False,
            )
        )
        f.update_layout(
            title=f"Subset Heatmap ({len(subG)} genes × {len(tissues_ord_hm)} tissues)",
            xaxis_title="Tissues",
            yaxis_title="Genes",
            width=1000,
            height=max(400, len(subG) * 15),
            margin=dict(l=100, r=20, t=60, b=80),
        )
        return f

    mo.md("""
    **Drill-Down Function Available:**
    Use `plot_subset(['GENE1', 'GENE2', ...])` to visualize specific gene subsets with enhanced hover details.

    Example:
    ```python
    subset_fig = plot_subset(['TP53', 'BRCA1', 'EGFR'])
    subset_fig
    ```

    **Note:** Plotly automatically optimizes rendering (WebGL for large matrices, SVG for small subsets).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Understanding Model Predictions

    ### Scientific Interpretation of Expression Outputs

    Expression predictions represent the model's learned associations between variant profiles and tissue-specific gene regulation. Key interpretation principles:

    **What the Values Represent**
    - Expression values are log2-scale RNA abundance predictions trained on GTEx data
    - Higher values indicate stronger predicted transcriptional activity in that tissue context
    - Predictions are individual-level (conditioned on variant genotype), not population averages
    - Values should be interpreted relative to the gene's typical expression range across tissues

    **Scientific Context**

    VariantFormer learns variant-to-expression mappings from examples:
    - **Regulatory variants** can increase or decrease expression depending on their effect on TF binding, chromatin accessibility, or RNA processing
    - **Tissue specificity** arises from tissue-specific enhancers, epigenetic states, and regulatory factor expression
    - **Compound effects**: Multiple variants across a gene's regulatory landscape contribute additively or non-additively
    - **Model validation**: Predictions correlate with eQTL effect sizes in held-out populations

    **Example Genes for Exploration:**
    - **APOE**: Tissue-specific isoform expression with known e2/e3/e4 variant effects
    - **BRCA1/2**: Differential expression in hormone-responsive tissues
    - **CYP genes**: Pharmacogenomic variants affecting hepatic drug metabolism
    - **FOXP2**: Brain-specific regulatory regions controlling language-related expression

    ### Next Steps

    **Research Applications:**
    1. **Multi-gene analysis** - Analyze co-regulated gene sets or pathways
    2. **Cohort comparison** - Compare predictions across individuals with different variant profiles
    3. **eQTL validation** - Compare predictions to experimental eQTL effect estimates
    4. **Downstream modeling** - Use predictions as input for disease risk models or functional scoring

    **Exporting Results:**
    - Save anatomogram visualizations for presentations or publications
    - Export expression tables for statistical analysis in R/Python
    - Share data with collaborators for multi-omic integration studies

    ---

    ## Acknowledgments & Citations

    ### VariantFormer Model

    **Citation:**
    Ghosal, S., et al. (2025). VariantFormer: A hierarchical transformer integrating DNA sequences with genetic variation and regulatory landscapes for personalized gene expression prediction. *bioRxiv* 2025.10.31.685862. DOI: [10.1101/2025.10.31.685862](https://doi.org/10.1101/2025.10.31.685862)

    **Training Data:** GTEx v8 paired whole-genome sequencing and RNA-seq data

    ### Anatomogram Visualizations

    **Citation:**
    Moreno, P., et al. (2022). Expression Atlas update: gene and protein expression in multiple species. *Nucleic Acids Research*. 50(D1):D129-D140. DOI: [10.1093/nar/gkab1030](https://doi.org/10.1093/nar/gkab1030)

    **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

    **Source:** [Expression Atlas, EMBL-EBI](https://www.ebi.ac.uk/gxa/home)

    The anatomogram SVG assets have been integrated into the VariantFormer visualization framework to provide interactive tissue-specific expression mapping.

    ### Additional Resources
    - [VariantFormer GitHub](https://github.com/czi-ai/variantformer)
    - [GTEx Portal](https://gtexportal.org/) - Population expression data
    - [gnomAD](https://gnomad.broadinstitute.org/) - Population variant frequencies

    ---

    ## Responsible Use Statement

    This tool is provided **exclusively for research and educational purposes**. Important considerations:

    **Research Tool Disclaimer**
    - VariantFormer is a research model, **not a clinical diagnostic tool**
    - Predictions should not be used for medical decision-making without appropriate validation
    - This tool does not provide medical advice, diagnosis, or treatment recommendations
    - Consult qualified healthcare professionals for any health-related questions

    **Scientific Limitations**
    - Predictions are based on GTEx training data and may not generalize to all populations
    - Rare or novel variants may have uncertain predicted effects
    - Model does not account for environmental factors, epigenetic variation, or post-transcriptional regulation
    - Expression predictions are probabilistic and should be validated experimentally when possible

    **Data Privacy**
    - VCF data is processed locally and is not uploaded to external servers
    - Users are responsible for ensuring compliance with relevant data governance policies
    - Handle genetic data according to institutional IRB protocols and privacy regulations

    **Acceptable Use**

    Please follow the [CZI Acceptable Use Policy](https://virtualcellmodels.cziscience.com/acceptable-use-policy) when using this tool. This tool is intended for:
    - Academic research and genomics education
    - Exploratory analysis of variant-to-expression relationships
    - Hypothesis generation for experimental validation

    **Not intended for:**
    - Clinical diagnosis or treatment decisions
    - Direct-to-consumer genetic interpretation
    - Insurance or employment decisions

    ---

    *Thank you for using VCF2Expression with VariantFormer. We hope this research tool advances scientific understanding of variant regulatory effects and gene expression biology.*
    """)
    return


if __name__ == "__main__":
    app.run()
