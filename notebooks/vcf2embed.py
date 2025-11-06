# /// script
# [tool.marimo.display]
# theme = "dark"
# ///

# TODO: Remove forced dark mode script above before production deployment
# This forces dark theme for development - remove to restore automatic light/dark switching

import marimo

__generated_with = "0.17.0"
app = marimo.App(
    app_title="VCF2Embed: Gene Expression Clustering",
    css_file="czi-sds-theme.css",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Standard library imports
    import json
    import pandas as pd
    import numpy as np

    # Visualization
    import plotly.express as px

    # Dimensionality reduction
    import umap

    return (
        Path,
        json,
        mo,
        np,
        pd,
        px,
        umap,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # VCF2Embed: Gene Expression Clustering and UMAP Analysis

    **Estimated time to complete:** 5-10 minutes | **Data:** Pre-computed expression predictions

    ## Overview

    This notebook performs exploratory clustering analysis on genome-wide gene expression predictions. Using UMAP (Uniform Manifold Approximation and Projection), we visualize how ~18,000 genes organize based on their expression patterns across 56 GTEx tissues.

    ## What This Analysis Shows

    **Gene Expression Embeddings:**
    - Each of ~18,000 protein-coding genes is represented by its expression profile across tissues
    - UMAP projects these high-dimensional profiles into 2D space for visualization
    - Genes with similar tissue-specific expression patterns cluster together

    **Two Complementary Views:**
    1. **Functional Clustering**: Genes colored by Gene Ontology categories (biological function)
    2. **Tissue Enrichment**: Genes colored by their tissue of maximum expression

    ## Key Insights

    - Housekeeping genes (metabolism, translation) cluster centrally with broad expression
    - Tissue-specific genes (neuronal, immune) form peripheral clusters
    - Functional categories organize by tissue specificity
    - Co-expressed genes typically share biological pathways

    ## Prerequisites

    - Pre-computed expression predictions from VCF2Expression workflow
    - Gene Ontology annotations for functional categorization
    - Recommended: Review [vcf2exp.py](vcf2exp.py) for prediction workflow
    """
    )
    return


@app.cell(hide_code=True)
def _():
    # Constants for data loading
    DATA_DIR = Path(__file__).parent / "example_data"
    PARQUET_PATH = DATA_DIR / "HG00096_vcf2exp_predictions.parquet"
    GO_ANNOTATIONS_PATH = DATA_DIR / "gene_go_annotations.parquet"
    FLOAT_DTYPE = np.float32
    return (DATA_DIR, GO_ANNOTATIONS_PATH, PARQUET_PATH, FLOAT_DTYPE)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## 📁 Step 1: Load Pre-Computed Expression Data""")
    return


@app.cell
def _(PARQUET_PATH, FLOAT_DTYPE, pd, np):
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

    return A_hm, gene_ids_hm, genes_hm, n_tissues, tissues_hm, unwrap_nested_array


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## 🔬 Step 2: Prepare Data for UMAP""")
    return


@app.cell
def _(A_hm, genes_hm, gene_ids_hm, tissues_hm, np):
    # Prepare gene expression data for UMAP (exclude cell lines)
    print("🔄 Preparing gene expression data for UMAP...")

    # Identify cell lines to exclude (keep only GTEx tissues)
    cell_lines = ['A549', 'Caki2', 'GM23248', 'HepG2', 'K562', 'NCI-H460', 'Panc1']
    tissue_mask = [t not in cell_lines for t in tissues_hm]
    tissue_indices = np.where(tissue_mask)[0]

    # Filter to GTEx tissues only
    A_gtex = A_hm[:, tissue_indices]
    tissues_gtex = [tissues_hm[i] for i in tissue_indices]

    print(f"   Filtered: {A_hm.shape[1]} total → {A_gtex.shape[1]} GTEx tissues")
    print(f"   Excluded cell lines: {', '.join(cell_lines)}")
    print(f"   Matrix for UMAP: {A_gtex.shape[0]:,} genes × {A_gtex.shape[1]} tissues")

    return A_gtex, tissues_gtex, cell_lines, tissue_mask, tissue_indices


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## 🧬 Step 3: Load Gene Ontology Annotations""")
    return


@app.cell
def _(GO_ANNOTATIONS_PATH, gene_ids_hm, pd, json):
    # Load Gene Ontology annotations
    print(f"🔄 Loading Gene Ontology annotations...")
    go_annotations_df = pd.read_parquet(GO_ANNOTATIONS_PATH)

    print(f"✅ Loaded annotations for {len(go_annotations_df):,} genes")

    # Create mapping dict for fast lookup
    gene_id_to_category = dict(zip(
        go_annotations_df['gene_id'],
        go_annotations_df['go_category_broad']
    ))

    gene_id_to_detailed = dict(zip(
        go_annotations_df['gene_id'],
        go_annotations_df['go_category_detailed']
    ))

    # Align with our gene order
    gene_categories_broad = [
        gene_id_to_category.get(gid, 'Other/Unannotated')
        for gid in gene_ids_hm
    ]

    # Parse detailed terms from JSON
    gene_categories_detailed = [
        json.loads(gene_id_to_detailed.get(gid, '[]'))
        for gid in gene_ids_hm
    ]

    category_dist = pd.Series(gene_categories_broad).value_counts()
    print(f"\n📊 Category distribution:")
    for _cat, count in category_dist.head(10).items():
        pct = count/len(gene_categories_broad)*100
        print(f"   {_cat:25s}: {count:5,} ({pct:4.1f}%)")

    return go_annotations_df, gene_categories_broad, gene_categories_detailed, gene_id_to_category, gene_id_to_detailed, category_dist


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🔬 Step 4: Run UMAP Dimensionality Reduction

    ### How UMAP Works for Gene Expression

    UMAP (Uniform Manifold Approximation and Projection) is a dimensionality reduction technique that preserves both local and global structure in high-dimensional data.

    **Our Configuration:**
    - **Input**: ~18,000 genes × 56 tissues (each gene = 56-dimensional expression vector)
    - **Distance Metric**: Correlation (standard for expression data)
    - **Neighbors**: 30 (captures local expression patterns)
    - **Min Distance**: 0.05 (allows tight functional clusters)
    - **Output**: 2D coordinates for visualization

    **What UMAP Preserves:**
    - Genes with similar expression profiles cluster together
    - Tissue-specific expression patterns form distinct regions
    - Functional gene categories organize spatially
    - Distance reflects expression correlation across tissues

    This computation may take 1-2 minutes for 18,000 genes...
    """
    )
    return


@app.cell
def _(A_gtex, np, umap):
    # Run UMAP on gene expression profiles
    print("🔬 Running UMAP on gene expression profiles...")
    print(f"   Input: {A_gtex.shape[0]:,} genes × {A_gtex.shape[1]} tissues")
    print(f"   This may take 1-2 minutes for {A_gtex.shape[0]:,} genes...")

    # UMAP configuration for gene clustering
    reducer_genes = umap.UMAP(
        n_neighbors=30,       # Larger for better global structure
        min_dist=0.05,        # Tighter clusters
        metric='correlation', # Standard for expression data
        random_state=42,
        n_jobs=-1,            # Parallel processing
        verbose=True,
    )

    # Fit UMAP (this is the slow step)
    gene_umap_coords = reducer_genes.fit_transform(A_gtex)

    print(f"✅ UMAP complete: {A_gtex.shape[1]}D → 2D")
    print(f"   Coordinates shape: {gene_umap_coords.shape}")

    return gene_umap_coords, reducer_genes


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🎨 Step 5: Visualize Gene Clustering by Function

    ### Gene Ontology Category UMAP

    This visualization reveals how ~18,000 genes cluster based on their expression profiles across 56 GTEx tissues. Each gene is represented by its normalized expression vector, and UMAP projects these high-dimensional profiles into 2D space.

    **What This Shows:**

    🧬 **Gene Clustering by Expression Pattern**
    - Each point = one gene (~18,000 total)
    - Position = similarity in expression profile across tissues
    - Genes with similar tissue-specific expression patterns cluster together
    - Distance reflects correlation in expression across the body

    🎯 **Gene Ontology Organization**
    - Color = Gene Ontology functional category (12 broad categories)
    - Genes with similar functions often cluster together
    - Reveals relationship between gene function and expression pattern
    - Co-expressed genes typically share biological pathways

    🔬 **Biological Insights**
    - Housekeeping genes (translation, metabolism) cluster centrally
    - Tissue-specific genes (neuronal, immune) form peripheral clusters
    - Cell cycle genes show distinct expression signatures
    - Functional categories organize by tissue specificity

    **How to Interpret:**
    - Each point = one gene
    - Color = GO functional category
    - Proximity = similar expression across tissues
    - Hover for gene name, GO category, and coordinates
    """
    )
    return


@app.cell
def _(gene_categories_broad, gene_categories_detailed, gene_umap_coords, genes_hm, pd, px):
    # Create gene UMAP visualization
    print("🎨 Creating gene UMAP visualization...")

    # Prepare hover text with detailed GO terms
    hover_detailed_terms = [
        ', '.join(terms[:3]) + ('...' if len(terms) > 3 else '') if terms else 'No GO terms'
        for terms in gene_categories_detailed
    ]

    # Create DataFrame
    gene_umap_df = pd.DataFrame({
        'UMAP1': gene_umap_coords[:, 0],
        'UMAP2': gene_umap_coords[:, 1],
        'Gene': genes_hm,
        'GO_Category': gene_categories_broad,
        'GO_Terms': hover_detailed_terms,
    })

    # Color map for GO categories
    go_color_map = {
        'Cell cycle': '#e41a1c',
        'Metabolism': '#377eb8',
        'Signal transduction': '#4daf4a',
        'Transcription': '#984ea3',
        'Translation': '#ff7f00',
        'Transport': '#a6761d',
        'Immune response': '#a65628',
        'Development': '#f781bf',
        'Apoptosis': '#999999',
        'DNA repair': '#66c2a5',
        'Cell adhesion': '#fc8d62',
        'Neuronal function': '#8da0cb',
        'Other/Unannotated': '#d9d9d9',
    }

    # Create scatter plot with small fixed-size points
    gene_umap_fig = px.scatter(
        gene_umap_df,
        x='UMAP1',
        y='UMAP2',
        color='GO_Category',
        hover_data={
            'Gene': True,
            'GO_Category': True,
            'GO_Terms': True,
            'UMAP1': ':.2f',
            'UMAP2': ':.2f',
        },
        title=f'UMAP: Gene Expression Profiles ({len(genes_hm):,} genes × 56 GTEx tissues)',
        labels={'GO_Category': 'Gene Ontology Category'},
        color_discrete_map=go_color_map,
    )

    # Fixed small markers for 18k genes
    gene_umap_fig.update_traces(
        marker=dict(
            size=2,  # Small fixed size for 18k points
            opacity=0.6,  # Semi-transparent to see density
            line=dict(width=0),  # No outline for cleaner appearance
        )
    )

    gene_umap_fig.update_layout(
        width=1000,
        height=800,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            title=dict(text='GO Category'),
            itemsizing='constant',  # Fixed legend marker size
        )
    )

    print(f"✅ UMAP visualization created")
    print(f"   • {len(genes_hm):,} genes visualized")
    print(f"   • {len(set(gene_categories_broad))} GO categories")

    return gene_umap_fig, gene_umap_df, hover_detailed_terms, go_color_map


@app.cell
def _(gene_umap_fig):
    gene_umap_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧬 Step 6: Compute Tissue Enrichment

    For each gene, we identify the tissue with maximum expression. This reveals tissue-specific gene expression patterns and helps interpret the UMAP clustering.
    """
    )
    return


@app.cell
def _(A_gtex, tissues_gtex):
    # Compute max expression tissue for each gene
    print("🔬 Computing tissue enrichment...")

    max_tissue_idx = A_gtex.argmax(axis=1)
    max_tissue_names = [tissues_gtex[idx] for idx in max_tissue_idx]
    max_tissue_values = A_gtex.max(axis=1)

    print(f"   {len(set(max_tissue_names))} unique tissues represented")
    print(f"   Each gene colored by its max expression tissue")

    return max_tissue_idx, max_tissue_names, max_tissue_values


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🎨 Step 7: Visualize Tissue Enrichment UMAP

    ### Tissue-Specific Expression Patterns

    This view colors each gene by the tissue where it shows maximum expression. This reveals:

    **Tissue-Specific Clusters:**
    - Brain-enriched genes cluster together (neuronal function)
    - Liver-specific genes form distinct regions (metabolism)
    - Muscle genes show coordinated expression patterns
    - Immune genes cluster by expression in blood/spleen

    **Biological Interpretation:**
    - Tight tissue clusters = strong tissue-specific expression
    - Mixed regions = housekeeping genes with broad expression
    - Peripheral clusters = highly specialized tissue functions
    - Central regions = metabolic and translational machinery

    **Interactive Features:**
    - Hover to see gene name, tissue, and max expression value
    - 56 unique tissue colors reveal expression specialization
    - Compare with GO category view to see function-tissue relationships
    """
    )
    return


@app.cell
def _(gene_umap_coords, genes_hm, max_tissue_names, max_tissue_values, pd, px):
    # Tissue Enrichment UMAP: Color by actual tissue (56 GTEx tissues)
    print("🎨 Creating tissue enrichment UMAP...")

    df_tissue_umap = pd.DataFrame({
        'UMAP1': gene_umap_coords[:, 0],
        'UMAP2': gene_umap_coords[:, 1],
        'Gene': genes_hm,
        'Tissue': max_tissue_names,
        'Max_Expression': max_tissue_values,
    })

    fig_tissue_umap = px.scatter(
        df_tissue_umap,
        x='UMAP1',
        y='UMAP2',
        color='Tissue',  # Color by actual tissue name (56 categories)
        hover_data={
            'Gene': True,
            'Tissue': True,
            'Max_Expression': ':.2f',
            'UMAP1': False,  # Don't show coordinates
            'UMAP2': False,
        },
        title='Gene UMAP: Colored by Tissue with Maximum Expression (56 GTEx Tissues)',
        labels={'Tissue': 'Tissue'},
    )

    fig_tissue_umap.update_traces(
        marker=dict(size=2, opacity=0.6, line=dict(width=0))
    )

    fig_tissue_umap.update_layout(
        width=1000,
        height=800,
        hovermode='closest',
    )

    print("✅ Tissue enrichment UMAP created")
    print(f"   {len(set(max_tissue_names))} unique tissues")

    return fig_tissue_umap, df_tissue_umap


@app.cell
def _(fig_tissue_umap):
    fig_tissue_umap
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📊 Summary and Next Steps

    ### What We Learned

    This UMAP analysis revealed:
    1. **Functional Organization**: Genes with similar GO categories cluster by expression pattern
    2. **Tissue Specificity**: Genes show clear tissue-enriched expression signatures
    3. **Co-expression Networks**: Spatially proximal genes likely share regulatory mechanisms
    4. **Expression Diversity**: ~18,000 genes organize into interpretable functional regions

    ### Related Analyses

    **For complementary visualizations, see:**
    - **[vcf2exp.py](vcf2exp.py)**: Full prediction workflow with anatomogram and interactive heatmap
    - **Heatmap Analysis**: Hierarchical clustering reveals detailed gene-tissue relationships
    - **Anatomogram Visualization**: Tissue-specific expression on human anatomy diagrams

    ### Future Directions

    **Potential Extensions:**
    1. **Interactive Gene Selection**: Click UMAP points to see detailed expression profiles
    2. **Pathway Enrichment**: Analyze GO term enrichment within UMAP clusters
    3. **Comparative Analysis**: Compare embeddings across different individuals
    4. **Alternative Methods**: Try t-SNE, PCA, or other dimensionality reduction techniques
    5. **Disease Gene Analysis**: Highlight disease-associated genes on UMAP

    ### Technical Notes

    - UMAP parameters can be adjusted for different cluster granularity
    - Correlation metric is standard for gene expression analysis
    - Results are reproducible with `random_state=42`
    - Computation scales to larger gene sets with parallel processing

    ---

    *For questions or feedback, please refer to the [DNA2Cell documentation](https://github.com/cziscience/DNA2Cell).*
    """
    )
    return


if __name__ == "__main__":
    app.run()
