import marimo

__generated_with = "0.13.9"
app = marimo.App(app_title="VCF2Risk Analysis", css_file="czi-sds-theme.css")


@app.cell
def _():
    import logging

    # Suppress verbose debug logs from markdown and matplotlib
    logging.getLogger('MARKDOWN').setLevel(logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.INFO)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    logging.getLogger('matplotlib.pyplot').setLevel(logging.INFO)

    import marimo as mo
    import sys
    from pathlib import Path
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from processors import ad_risk
    from anatomagram.components.anatomagram_widget import AnatomagramMultiViewWidget
    from anatomagram.components.vcf_risk_converter import EnhancedVCFRiskConverter
    return AnatomagramMultiViewWidget, EnhancedVCFRiskConverter, Path, ad_risk, mo, pd


@app.cell
def _(mo):
    mo.md(
        """
    # VCF2Risk Tutorial: Alzheimer's Disease Risk Prediction

    **Estimated time to complete:** ~10 minutes (on NVIDIA H100 GPU)

    ## Learning Goals

    * Learn how to predict tissue-specific Alzheimer's disease risk from genetic variants
    * Understand the VCF2Risk pipeline: variants → expression → embeddings → disease risk
    * Explore gene-specific AD risk patterns across different tissues using interactive visualizations
    * Interpret AD risk scores and expression predictions in biological context

    ## Prerequisites

    **Hardware:**
    - GPU with 40GB+ VRAM (NVIDIA H100 recommended for optimal performance)
    - 32GB+ system RAM for processing all 45 tissues

    **Software:**
    - Python 3.12+
    - PyTorch with CUDA support
    - VariantFormer repository with dependencies installed

    **Input Data:**
    - VCF file with genetic variants (standard VCF v4.2+)
    - Reference genome: GRCh38/hg38
    - Demo: Uses HG00096 sample from 1000 Genomes Project

    ## Introduction

    **VCF2Risk** predicts how genetic variants in a specific gene contribute to Alzheimer's disease risk across different tissues.

    ### Model Architecture

    The pipeline combines two AI components:

    **1. VariantFormer Model** (Seq2Gene + Seq2Reg transformers):
    - **Input**: DNA sequence with variants from VCF file
    - **Output**: Tissue-specific gene expression predictions + 1536-dimensional embeddings
    - **Purpose**: Captures how genetic variants affect gene regulation in each tissue
    - **Size**: 14GB checkpoint, ~1.2B parameters

    **2. AD Risk Predictors** (Random forests for each gene-tissue pair):
    - **Input**: Gene-tissue embeddings from VariantFormer model
    - **Output**: Alzheimer's disease risk probability (0-1 scale)
    - **Training**: Separate models for each gene-tissue pair (~16,400 genes × 45 tissues)
    - **Format**: Treelite `.tl` model files stored in S3

    ### Pipeline Flow

    ```
    VCF Variants → VariantFormer Model → [Expression + Embedding] → AD Predictor → Risk Score
                                          ↑ intermediate             ↑ primary output
    ```

    ### Input Data Requirements

    **VCF File:**
    - Standard VCF format (v4.2 or later)
    - Reference genome: **GRCh38/hg38** (critical - must match training data)
    - Can be bgzipped (.vcf.gz) or uncompressed
    - Must contain variants for the selected gene region

    **Gene Selection:**
    - Choose one gene per analysis
    - Only genes with trained AD predictors available (~16,400 genes)
    - Dropdown auto-filters to available genes

    **Tissue Selection:**
    - 45 out of 63 GTEx tissues have AD risk models
    - Can analyze all tissues or focus on specific organ systems
    - Default: All 45 tissues for comprehensive analysis

    ### Expected Outputs

    **For each gene-tissue combination:**

    1. **Predicted Expression** (intermediate output):
       - How variants alter gene expression in that tissue
       - Log-scale expression values
       - Provides biological context for risk scores

    2. **AD Risk Score** (primary output):
       - Probability (0-1) that gene contributes to AD in this tissue
       - Trained from AD case-control gene expression datasets
       - Higher scores = greater predicted disease contribution
       - Tissue-specific: same gene can have different risk across tissues
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Setup

    This tutorial uses pre-loaded models and sample data:
    - **VariantFormer model**: 14GB checkpoint for expression prediction
    - **AD risk predictors**: Downloaded from S3 as needed (gene+tissue-specific)
    - **Sample VCF**: HG00096 from 1000 Genomes Project

    The following cell initializes the model and verifies environment setup.

    **Expected initialization time:** ~15-20 seconds
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""## 🧠 Model Initialization""")
    return


@app.cell
def _(ad_risk, mo):
    mo.md("Loading VariantFormer AD risk prediction model...")

    adrisk = ad_risk.ADriskFromVCF()

    mo.md("✅ **Model loaded successfully!**").callout(kind="success")
    return (adrisk,)


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
        label="Select VCF file for AD risk analysis"
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


@app.cell
def _(mo):
    mo.md(
        """
    ## Select Gene for Analysis

    Choose one gene to analyze for AD risk contribution. The dropdown shows only genes
    that have trained AD risk predictors available.

    **Recommended genes for Alzheimer's disease analysis:**
    - **APOE** (Apolipoprotein E): Strongest genetic risk factor for late-onset AD
    - **APP** (Amyloid Precursor Protein): Mutations cause early-onset familial AD
    - **PSEN1** (Presenilin 1): Familial AD gene, affects amyloid processing
    - **PSEN2** (Presenilin 2): Another familial AD gene
    - **MAPT** (Microtubule Associated Protein Tau): Associated with tauopathies
    - **TREM2** (Triggering Receptor on Myeloid Cells 2): Immune gene linked to AD

    **Why gene-specific predictors?**

    Each AD risk predictor is trained for a specific gene-tissue combination, learning
    how that gene's regulatory patterns (captured in the embedding) relate to AD pathology
    in that particular tissue context.
    """
    )
    return


@app.cell
def _(adrisk, mo):
    # Get all available genes
    genes_df = adrisk.genes_map.reset_index()

    # Get genes that have AD predictors available
    available_ad_genes = adrisk.ad_preds.get_unique('gene_id')
    genes_with_ad = genes_df[genes_df['gene_id'].isin(available_ad_genes)]

    # Create dropdown options {label: gene_id}
    gene_options = {
        f"{row['gene_name']} | {row['gene_id']}": row['gene_id']
        for _, row in genes_with_ad.iterrows()
    }

    # Find APOE for default (or first available AD gene)
    apoe_matches = genes_with_ad[genes_with_ad['gene_name'] == 'APOE']
    if len(apoe_matches) > 0:
        default_gene_id = apoe_matches.iloc[0]['gene_id']
    else:
        # Fallback to first gene with AD predictor
        default_gene_id = list(gene_options.values())[0]

    # Find the label for the default gene
    default_label = [k for k, v in gene_options.items() if v == default_gene_id][0]

    # Multi-select with max_selections=1 for searchable single gene selection
    gene_selector = mo.ui.multiselect(
        options=gene_options,
        value=[default_label],
        label="Select Gene for AD Risk Analysis",
        max_selections=1
    )

    return gene_selector, genes_with_ad, gene_options


@app.cell
def _(gene_selector, genes_with_ad, mo, pd):
    # Filter genes table based on multiselect with max_selections=1
    # gene_selector.value is a list with 0 or 1 gene IDs
    _selected_gene_ids = gene_selector.value

    if len(_selected_gene_ids) > 0:
        # Show only selected gene
        _filtered_genes_df = genes_with_ad[genes_with_ad['gene_id'] == _selected_gene_ids[0]]
    else:
        # Show nothing when nothing selected
        _filtered_genes_df = pd.DataFrame(columns=genes_with_ad.columns)

    # Create filtered table without checkboxes
    genes_table_filtered = mo.ui.table(
        _filtered_genes_df,
        selection=None,
        show_column_summaries=False,
        label=f"Showing {len(_filtered_genes_df)} of {len(genes_with_ad)} genes"
    )
    return (genes_table_filtered,)


@app.cell
def _(gene_selector, genes_table_filtered, mo):
    # Display gene selection UI
    mo.vstack([
        mo.md("**Select a gene using dropdown** (table shows selected gene):"),
        gene_selector,
        genes_table_filtered
    ])
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Select Tissues for Analysis

    Choose which tissues to analyze for AD risk. By default, all 45 tissues with
    trained AD risk predictors are selected for comprehensive analysis.

    **Tissue Coverage:**
    - **45 out of 63 GTEx tissues** have AD risk models trained
    - Includes major organ systems: nervous, cardiovascular, digestive, respiratory, etc.
    - **13 brain regions** available for CNS-focused analysis

    **Analysis Strategies:**
    - **Comprehensive** (default): All 45 tissues to see complete risk landscape
    - **Brain-focused**: Select only CNS tissues for neurological analysis
    - **Comparative**: Choose a few key tissues for targeted comparison
    - **System-specific**: Focus on one organ system (e.g., cardiovascular)

    **Note:** Processing all 45 tissues takes ~3-4 minutes. Brain-only subset (~13 tissues) completes faster (~1-2 minutes).
    """
    )
    return


@app.cell
def _(adrisk, mo, pd):
    # Get tissues that have AD predictors available (from manifest)
    available_tissue_ids_in_manifest = adrisk.ad_preds.get_unique('tissue_id')

    # Filter adrisk.tissue_map to only tissues with AD predictors
    tissues_with_ad = adrisk.tissue_map[adrisk.tissue_map.index.isin(available_tissue_ids_in_manifest)]
    ad_tissue_names = list(tissues_with_ad['tissue'])

    # Create tissue dataframe for display
    tissues_df = pd.DataFrame({
        'tissue_name': ad_tissue_names,
        'tissue_id': [adrisk.tissue_map[adrisk.tissue_map['tissue'] == name].index[0] for name in ad_tissue_names]
    })

    # Multi-select with all AD-available tissues selected by default
    tissue_selector = mo.ui.multiselect(
        options=ad_tissue_names,  # Human-readable tissue names
        value=ad_tissue_names,    # Default: ALL tissues with AD predictors
        label="Select Tissues for Analysis"
    )

    return tissue_selector, tissues_df, tissues_with_ad


@app.cell
def _(mo, pd, tissue_selector, tissues_df):
    # Filter tissues table based on multiselect selection
    _selected_tissue_names = tissue_selector.value

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
def _(mo, tissue_selector, tissues_table_filtered):
    # Display tissue selection UI
    mo.vstack([
        mo.md("**Select tissues using dropdown** (table shows selected tissues):"),
        tissue_selector,
        tissues_table_filtered
    ])
    return


@app.cell
def _(gene_selector, genes_with_ad, mo, tissue_selector, tissues_df):
    # Display selection summary
    _gene_count = len(gene_selector.value)  # List with 0 or 1 element
    mo.md(f"""
    **Selection Summary**:
    - **Gene**: {_gene_count} selected
    - **Tissues**: {len(tissue_selector.value)} selected

    💡 **Available**: {len(genes_with_ad):,} genes with AD predictors across {len(tissues_df)} tissues
    """)
    return


@app.cell
def _(mo):
    mo.md("""## ⚙️ Analysis Configuration""")
    return


@app.cell
def _(adrisk, gene_selector, tissue_selector):
    # Get gene_id from multiselect (returns list with 0 or 1 gene IDs)
    selected_gene_id = gene_selector.value[0] if len(gene_selector.value) > 0 else None

    # Convert selected tissue names to tissue IDs
    selected_tissue_names = tissue_selector.value
    tissue_ids = [
        int(adrisk.tissue_map[adrisk.tissue_map['tissue'] == name].index[0])
        for name in selected_tissue_names
    ]

    return selected_gene_id, tissue_ids


@app.cell
def _(DEFAULT_VCF_PATH, gene_options, gene_selector, mo, tissue_selector, vcf_file_browser):
    # Display configuration summary
    # With multiselect (max_selections=1), gene_selector.value returns a list with 0 or 1 labels
    if len(gene_selector.value) == 0:
        gene_label = "No gene selected"
    else:
        gene_label = gene_selector.value[0]  # The full label (e.g., "APOE | ENSG...")

    # Determine VCF display name
    if vcf_file_browser.value and len(vcf_file_browser.value) > 0:
        _vcf_display = vcf_file_browser.value[0].id.split('/')[-1]
    else:
        _vcf_display = DEFAULT_VCF_PATH.split('/')[-1]

    mo.md(
        f"""
    **Analysis Settings:**
    - VCF File: `{_vcf_display}`
    - Selected Gene: `{gene_label.split(' | ')[0] if ' | ' in gene_label else gene_label}`
    - Tissues: {len(tissue_selector.value)} selected
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Running AD Risk Prediction

    The prediction pipeline executes the following steps:

    1. **Load VCF variants** for the selected gene region
    2. **Predict gene expression** across selected tissues using VariantFormer model
    3. **Generate embeddings** (1536-dim regulatory state representations)
    4. **Download AD predictors** from S3 (one `.tl` model per tissue)
    5. **Compute AD risk scores** for each gene-tissue combination

    **Processing time:** ~3-4 minutes for 45 tissues on H100 GPU

    The prediction runs automatically when you change gene or tissue selections (reactive execution).
    """
    )
    return


@app.cell
def _(DEFAULT_VCF_PATH, adrisk, genes_with_ad, mo, selected_gene_id, tissue_ids, vcf_file_browser):
    # Get VCF path from file browser or use default
    if vcf_file_browser.value and len(vcf_file_browser.value) > 0:
        vcf_path = vcf_file_browser.value[0].id  # Get selected file path
        print(f"📁 Using selected VCF: {vcf_path}")
    else:
        vcf_path = DEFAULT_VCF_PATH  # Use default from _artifacts
        print(f"📁 Using default VCF: {vcf_path}")

    # Get gene name for display messaging
    gene_name = genes_with_ad[genes_with_ad['gene_id'] == selected_gene_id].iloc[0]['gene_name']

    mo.md(f"Predicting AD risk for **{gene_name}** across **{len(tissue_ids)} tissues**...")

    # Run AD risk prediction pipeline
    # This executes:
    #   1. VCF parsing and variant extraction
    #   2. VariantFormer model inference (expression + embeddings)
    #   3. S3 download of gene+tissue-specific AD predictor models
    #   4. AD risk score computation from embeddings
    predictions_df = adrisk(vcf_path, [selected_gene_id] * len(tissue_ids), tissue_ids)

    mo.md(f"""
    ✅ **Prediction completed!**
    - Gene: {predictions_df.iloc[0]['gene_name']} ({predictions_df.iloc[0]['gene_id']})
    - Tissues analyzed: {len(predictions_df)}
    - Mean AD risk: {predictions_df['ad_risk'].mean():.6f}
    - Std AD risk: {predictions_df['ad_risk'].std():.6f}
    - Risk range: [{predictions_df['ad_risk'].min():.6f}, {predictions_df['ad_risk'].max():.6f}]
    """).callout(kind="success")
    return (predictions_df,)


@app.cell
def _(mo):
    mo.md(
        """
    ## Model Outputs

    ### Understanding the Results

    Each row represents predictions for one tissue. The table shows both intermediate
    and final outputs from the pipeline.

    **Columns explained:**

    | Column | Description | Interpretation |
    |--------|-------------|----------------|
    | **gene_name** | Gene symbol (e.g., APOE) | The analyzed gene |
    | **tissue_name** | GTEx tissue identifier | Specific tissue analyzed |
    | **predicted_expression** | Gene expression value (log-scale) | How variants affect gene activity - provides biological context |
    | **ad_risk** | AD risk score (0-1 probability) | **Primary output** - probability gene contributes to AD in this tissue |

    **How to interpret:**

    - **AD Risk Score (0-1)**:
      - **0.0**: Low predicted contribution to AD
      - **1.0**: High predicted contribution to AD

    - **Predicted Expression** (context):
      - Shows whether variants increase or decrease gene activity
      - Helps explain *why* risk might be high (e.g., overexpression of risk gene)
      - Intermediate output from VariantFormer model

    - **Tissue Specificity**:
      - Same gene can have different risk scores across tissues
      - Reflects tissue-specific biology and disease mechanisms
      - Brain tissues often show distinct patterns for neurological disease genes
    """
    )
    return


@app.cell
def _(mo, predictions_df):
    # Display results table
    results_table = predictions_df[['gene_name', 'tissue_name', 'predicted_expression', 'ad_risk']].copy()
    results_table['predicted_expression'] = results_table['predicted_expression'].round(4)
    results_table['ad_risk'] = results_table['ad_risk'].round(6)

    mo.ui.table(results_table)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Risk Distribution Across Tissues

    This bar chart displays AD risk scores for all analyzed tissues, sorted and color-coded
    by risk level.

    **What to look for:**
    - **High-risk tissues**: Darker colors (yellow), taller bars
    - **Tissue patterns**: Do certain organ systems cluster together in risk?
    - **Outliers**: Tissues with unusually high or low risk compared to others
    - **Brain regions**: For AD genes, often show elevated risk in CNS tissues

    **Interactivity:** Hover over bars to see exact risk values and tissue names.
    """
    )
    return


@app.cell
def _(mo, predictions_df):
    import plotly.express as px

    fig = px.bar(
        predictions_df,
        x='tissue_name',
        y='ad_risk',
        title=f'AD Risk Predictions: {predictions_df.iloc[0]["gene_name"]} across Tissues',
        color='ad_risk',
        color_continuous_scale='viridis',
        labels={'ad_risk': 'AD Risk Score', 'tissue_name': 'Tissue'}
    )
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(height=500)

    mo.ui.plotly(fig)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Anatomical Risk Mapping

    The anatomagram displays AD risk scores spatially mapped onto human body diagrams,
    providing intuitive visualization of tissue-specific disease risk patterns.

    **Features:**
    - **Three anatomical views**: Male, female, and brain-focused anatomies
    - **Color-coded risk levels**: Viridis palette (purple = low risk, yellow = high risk)
    - **Interactive tooltips**: Hover over colored regions for detailed information
    - **Hierarchical mapping**: Related tissues intelligently aggregated to anatomical structures

    **How to use:**
    - **Switch between tabs** to see different anatomical perspectives
    - **Hover over tissues** to see exact risk values and tissue names
    - **Compare patterns** across different body systems visually
    - **Identify risk hotspots** where disease contribution is concentrated

    **Interpretation tips:**
    - For AD genes (APOE, APP, etc.), look for elevated risk in brain regions
    - Peripheral tissues may show lower risk for CNS-focused disease genes
    - Uniform risk across tissues suggests gene-wide regulatory effects
    - Clustered risk in specific systems hints at tissue-specific mechanisms
    """
    )
    return


@app.cell
def _(EnhancedVCFRiskConverter, mo, predictions_df):
    # Use enhanced converter for full metadata (matches vcf2exp pattern)
    enhanced_converter = EnhancedVCFRiskConverter(aggregation_strategy='mean')

    anatomagram_data, enhanced_metadata = enhanced_converter.convert_predictions_to_anatomagram(predictions_df)

    # Extract all necessary components
    uberon_map = enhanced_converter.get_uberon_map()
    enhanced_tooltips = enhanced_metadata['enhanced_tooltips']  # Dict with tooltip info
    uberon_names = enhanced_metadata['uberon_names']

    mo.md(f"""
    ✅ **Data prepared for anatomagram:**
    - Risk predictions: {len(predictions_df)} tissues
    - UBERON mappings: {len(uberon_map)} tissues
    - Enhanced tooltips: {len(enhanced_tooltips)} tooltips
    """).callout(kind="info")
    return anatomagram_data, enhanced_tooltips, uberon_names


@app.cell
def _(
    AnatomagramMultiViewWidget,
    anatomagram_data,
    enhanced_tooltips,
    mo,
    uberon_names,
):
    # Create multi-view anatomagram widget
    multi_widget = AnatomagramMultiViewWidget(
        visualization_data=anatomagram_data,
        selected_item="AD_RISK",
        available_views=["male", "female", "brain"],
        current_view="male",
        color_palette="viridis",
        scale_type="linear",
        debug=False,
        uberon_names=uberon_names,
        enhanced_tooltips=enhanced_tooltips  # Pass dict instead of None
    )

    # Create tabbed interface
    tabs = multi_widget.create_view_tabs(mo)
    return multi_widget, tabs


@app.cell
def _(multi_widget, tabs):
    multi_widget.current_view = tabs.value
    return


@app.cell
def _(tabs):
    # Display the anatomagram
    tabs
    return


@app.cell
def _(multi_widget):
    multi_widget
    return


@app.cell
def _(mo):
    mo.md("""## 📋 Summary Statistics""")
    return


@app.cell
def _(mo, predictions_df):
    # Summary statistics
    stats_df = predictions_df[['tissue_name', 'ad_risk']].copy()
    stats_df = stats_df.sort_values('ad_risk', ascending=False)

    mo.vstack([
        mo.md("### Top 5 Risk Tissues"),
        mo.ui.table(stats_df.head(5)),
        mo.md("### Bottom 5 Risk Tissues"),
        mo.ui.table(stats_df.tail(5))
    ])
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Interpreting Your Results

    ### What do AD risk scores mean?

    The risk scores (0-1) represent the **predicted probability** that this gene's regulatory
    state contributes to Alzheimer's disease in each tissue, based on:

    - Expression patterns learned from AD case-control cohorts
    - Regulatory signatures captured in gene-tissue embeddings
    - Variant effects on gene expression in your VCF file

    ### Clinical Context

    **These are research predictions, not clinical diagnoses.** They indicate:

    - Genes and tissues where variants may influence AD biology
    - Tissue-specific mechanisms of genetic risk
    - Hypotheses for follow-up experimental validation
    - Potential therapeutic targets for further investigation

    ### Limitations

    - Predictions based on population-level training data
    - Individual AD risk depends on many factors beyond single genes
    - Some tissues may lack sufficient AD training data
    - Scores reflect correlation, not necessarily causation
    - Model does not account for environmental factors, epigenetics, or post-transcriptional regulation
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Next Steps

    ### Analyze Your Own Data

    To run VCF2Risk on your own genetic data:

    1. **Prepare VCF file**: Ensure it uses GRCh38 reference genome
    2. **Update VCF path**: Edit the `vcf_path` variable in the configuration cell
    3. **Select gene**: Choose gene(s) of interest from the dropdown
    4. **Select tissues**: Choose relevant tissues for your research question
    5. **Export results**: Save predictions with `predictions_df.to_csv('my_results.csv')`

    ### Further Exploration

    **Comparative analysis:**
    - Run notebook multiple times with different AD-associated genes (APOE, APP, PSEN1, etc.)
    - Compare risk patterns across genes to identify common vs. gene-specific tissue effects

    **Focused analysis:**
    - Select only brain tissues for CNS-specific AD mechanisms
    - Focus on peripheral tissues to explore systemic disease contributions

    **Multi-omic integration:**
    - Correlate with VCF2Expression results (gene expression predictions)
    - Integrate with proteomics or metabolomics data
    - Validate high-risk predictions with functional genomics experiments

    **Data export:**
    - Save results table: `predictions_df.to_csv('ad_risk_results.csv', index=False)`
    - Export for R/Python statistical analysis
    - Share with collaborators for further investigation
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
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

    ### AD Risk Predictors

    **Training Data:**
    - **GTEx v8**: Tissue-specific gene expression reference data
    - **AD cohort datasets**: Case-control data for risk predictor training

    ### Additional Resources

    - [VariantFormer GitHub](https://github.com/czi-ai/variantformer)
    - [GTEx Portal](https://gtexportal.org/) - Population gene expression data
    - [gnomAD](https://gnomad.broadinstitute.org/) - Population variant frequencies
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Responsible Use

    **This tool is for research purposes only.**

    ### Research Tool Disclaimer

    - VCF2Risk is a research model, **not a clinical diagnostic tool**
    - Predictions should not be used for medical decision-making without appropriate validation
    - This tool does not provide medical advice, diagnosis, or treatment recommendations
    - Consult qualified healthcare professionals for any health-related questions

    ### Scientific Limitations

    - Predictions are based on GTEx and AD cohort training data - may not generalize to all populations
    - Rare or novel variants may have uncertain predicted effects
    - Model does not account for environmental factors, epigenetic variation, or post-transcriptional regulation
    - AD risk scores are probabilistic and should be validated experimentally when possible

    ### Data Privacy

    - VCF data is processed locally and not uploaded to external servers
    - Users are responsible for ensuring compliance with relevant data governance policies
    - Handle genetic data according to institutional IRB protocols and privacy regulations

    ### Acceptable Use

    Follow the [CZI Acceptable Use Policy](https://virtualcellmodels.cziscience.com/acceptable-use-policy).

    **This tool is intended for:**
    - Academic research and genomics education
    - Exploratory analysis of variant-to-disease relationships
    - Hypothesis generation for experimental validation
    - Understanding tissue-specific AD mechanisms

    **Not intended for:**
    - Clinical diagnosis or treatment decisions
    - Direct-to-consumer genetic interpretation
    - Insurance or employment decisions
    - Medical advice or health recommendations
    """
    )
    return


if __name__ == "__main__":
    app.run()
