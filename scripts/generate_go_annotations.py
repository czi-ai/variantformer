#!/usr/bin/env python3
"""
Generate Gene Ontology annotations for all genes in the prediction dataset.
This creates a static file to avoid API calls during notebook execution.

Usage:
    python scripts/generate_go_annotations.py

Output:
    notebooks/example_data/gene_go_annotations.parquet
    notebooks/example_data/gene_go_annotations.csv
"""

import pandas as pd
import numpy as np
import mygene
from pathlib import Path
from tqdm import tqdm
import json
import sys

# Add parent to path for imports if needed
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_gene_ids(parquet_path):
    """Load gene IDs from prediction parquet file."""
    print(f"📁 Loading gene IDs from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"   Found {len(df):,} genes")
    return df['gene_id'].tolist(), df['gene_name'].tolist()


def fetch_go_annotations(gene_ids, gene_names, batch_size=1000):
    """Fetch GO annotations with two-pass strategy: Ensembl ID then symbol fallback."""
    mg = mygene.MyGeneInfo()

    print(f"\n🔄 TWO-PASS Query Strategy for {len(gene_ids):,} genes...")
    print(f"   Batch size: {batch_size}")

    # Remove version numbers from Ensembl IDs (.12, .13, etc.)
    gene_ids_clean = [gid.split('.')[0] for gid in gene_ids]

    # PASS 1: Query by Ensembl ID
    print(f"\n🔄 PASS 1: Querying by Ensembl ID...")
    results_pass1 = []

    for i in tqdm(range(0, len(gene_ids_clean), batch_size), desc="Pass 1 (Ensembl ID)"):
        batch = gene_ids_clean[i:i+batch_size]

        results = mg.querymany(
            batch,
            scopes='ensembl.gene',
            fields='go.BP,symbol,name',
            species='human',
            returnall=True,
        )

        results_pass1.extend(results['out'])

    # Build lookup from pass 1 - ONLY include genes WITH GO annotations
    ensembl_to_result = {}
    ensembl_matched_no_go = {}

    for r in results_pass1:
        query_id = r.get('query')
        if query_id and 'notfound' not in r:
            if 'go' in r and r.get('go', {}).get('BP'):
                # Has GO annotations - use it
                ensembl_to_result[query_id] = r
            else:
                # Matched but no GO - save separately, might retry
                ensembl_matched_no_go[query_id] = r

    print(f"   Pass 1: {len(ensembl_to_result):,} genes with GO annotations")
    print(f"   Pass 1: {len(ensembl_matched_no_go):,} genes matched but no GO (will retry)")

    # Identify genes for pass 2: completely failed OR matched without GO
    failed_indices = []
    failed_symbols = []

    for i, (gid_clean, gname) in enumerate(zip(gene_ids_clean, gene_names)):
        # Retry if: (1) didn't match at all, OR (2) matched but has no GO
        if gid_clean not in ensembl_to_result:
            failed_indices.append(i)
            failed_symbols.append(gname)

    print(f"   Genes to retry in Pass 2: {len(failed_indices):,}")

    # PASS 2: Query failed genes by symbol
    if failed_symbols:
        print(f"\n🔄 PASS 2: Querying {len(failed_symbols):,} failed genes by symbol...")

        results_pass2 = []

        for i in tqdm(range(0, len(failed_symbols), batch_size), desc="Pass 2 (Symbol)"):
            batch = failed_symbols[i:i+batch_size]

            results = mg.querymany(
                batch,
                scopes='symbol',  # Query by gene symbol instead!
                fields='go.BP,symbol,name',
                species='human',
                returnall=True,
            )

            results_pass2.extend(results['out'])

        # Build lookup from pass 2 - include any match even without GO
        symbol_to_result = {}
        for r in results_pass2:
            query_symbol = r.get('query')
            if query_symbol and 'notfound' not in r:
                symbol_to_result[query_symbol] = r

        genes_with_go_pass2 = sum(1 for r in symbol_to_result.values() if 'go' in r)
        print(f"   Pass 2: Matched {len(symbol_to_result):,} genes")
        print(f"   Pass 2: {genes_with_go_pass2:,} genes have GO annotations ({genes_with_go_pass2/len(failed_symbols)*100:.1f}% recovery)")

    else:
        symbol_to_result = {}
        recovered = 0

    # Combine results from both passes
    print(f"\n🔄 Combining results from both passes...")
    final_results = []

    for i, (gid_clean, gname) in enumerate(zip(gene_ids_clean, gene_names)):
        if gid_clean in ensembl_to_result:
            final_results.append(ensembl_to_result[gid_clean])
        elif gname in symbol_to_result:
            final_results.append(symbol_to_result[gname])
        else:
            final_results.append({'query': gid_clean, 'notfound': True})

    # Count final statistics
    genes_with_go_total = sum(1 for r in final_results if 'go' in r and r.get('go', {}).get('BP'))

    print(f"\n📊 Final statistics:")
    print(f"   Total genes: {len(gene_ids):,}")
    print(f"   Pass 1 (Ensembl): {len(ensembl_to_result):,} matched")
    print(f"   Pass 2 (Symbol): {len(symbol_to_result):,} recovered")
    print(f"   Total with GO terms: {genes_with_go_total:,} ({genes_with_go_total/len(gene_ids)*100:.1f}%)")
    print(f"   Missing/no GO: {len(gene_ids) - genes_with_go_total:,}")

    return final_results


def map_to_go_slim_categories(bp_terms_list):
    """
    Map detailed GO BP terms to simplified categories for visualization.

    Returns:
        tuple: (broad_category, representative_term, all_term_names)
    """

    if not bp_terms_list or not isinstance(bp_terms_list, list):
        return 'Other/Unannotated', None, []

    # Extract term names
    term_names = [t.get('term', '') for t in bp_terms_list if isinstance(t, dict) and 'term' in t]

    if not term_names:
        return 'Other/Unannotated', None, []

    all_terms_str = ' '.join(term_names).lower()

    # Category mapping with priority order (check most specific first)
    # Expanded keywords to capture more GO terms
    category_rules = [
        ('Cell cycle', ['cell cycle', 'mitosis', 'cell division', 'mitotic', 'chromosome segregation', 'g1/s transition', 'g2/m transition', 'cytokinesis']),
        ('DNA repair', ['dna repair', 'dna damage', 'dna recombination', 'base excision', 'double-strand break', 'nucleotide excision']),
        ('Transcription', ['transcription', 'rna polymerase', 'gene expression', 'chromatin', 'histone modification', 'chromatin remodeling']),
        ('Translation', ['translation', 'ribosom', 'protein synthesis', 'peptide biosynthetic', 'translational']),
        ('Signal transduction', ['signal transduction', 'signaling', 'receptor', 'kinase cascade', 'mapk', 'phosphorylation', 'g protein', 'gpcr', 'receptor signaling', 'ion channel', 'calcium signaling', 'potassium channel', 'sodium channel', 'channel activity']),
        ('Apoptosis', ['apoptosis', 'programmed cell death', 'necrosis', 'caspase', 'cell death']),
        ('Immune response', ['immune', 'defense response', 'inflammation', 'innate immune', 'adaptive immune', 'antigen', 'leukocyte']),
        ('Metabolism', ['metabolic', 'biosynthetic', 'catabolic', 'synthesis', 'degradation', 'oxidation', 'glycolysis', 'citric acid', 'lipid', 'carbohydrate', 'amino acid', 'protein modification', 'ubiquitin', 'sumoylation', 'acetylation', 'methylation']),
        ('Transport', ['transport', 'localization', 'vesicle', 'secretion', 'endocytosis', 'exocytosis', 'protein localization', 'intracellular transport', 'ion transport', 'transmembrane', 'import', 'export']),
        ('Development', ['development', 'differentiation', 'morphogenesis', 'organogenesis', 'embryonic', 'pattern specification', 'cell fate', 'polarity', 'fertilization', 'reproduction', 'gametogenesis']),
        ('Cell adhesion', ['cell adhesion', 'extracellular matrix', 'cell-cell adhesion', 'focal adhesion', 'cell junction', 'integrin']),
        ('Neuronal function', ['neuron', 'synaptic', 'neurotransmitter', 'axon', 'dendrite', 'synapse', 'neurogenesis', 'neural', 'nervous system']),
    ]

    # Find first matching category
    for category, keywords in category_rules:
        for keyword in keywords:
            if keyword in all_terms_str:
                # Get the most specific matching term
                matching_term = next((t for t in term_names if keyword in t.lower()), term_names[0])
                return category, matching_term, term_names

    # Default: Genes WITH GO terms but no keyword match → "Other Biological Process"
    # These genes DO have GO annotations, just not in our 12 main categories
    return 'Other Biological Process', term_names[0] if term_names else None, term_names


def process_results(results, original_gene_ids, original_gene_names):
    """Process mygene results, retaining both broad and detailed categories."""

    print(f"\n🔄 Processing {len(original_gene_ids):,} gene annotations...")

    # Build efficient lookup dictionaries for both passes
    # Pass 1 results have Ensembl IDs as query
    # Pass 2 results have gene symbols as query
    ensembl_lookup = {}
    symbol_lookup = {}

    for r in results:
        query = r.get('query')
        if not query or 'notfound' in r:
            continue

        # Check if query looks like Ensembl ID or gene symbol
        if query.startswith('ENSG'):
            ensembl_lookup[query] = r
        else:
            symbol_lookup[query] = r

    print(f"   Built lookups: {len(ensembl_lookup):,} Ensembl, {len(symbol_lookup):,} Symbol")

    rows = []

    for orig_id, orig_name in tqdm(zip(original_gene_ids, original_gene_names), total=len(original_gene_ids), desc="Processing"):
        clean_id = orig_id.split('.')[0]

        # Try to find result: first by Ensembl ID, then by symbol
        result = ensembl_lookup.get(clean_id) or symbol_lookup.get(orig_name)

        if result and 'go' in result:
            go_bp = result.get('go', {}).get('BP', [])
            symbol = result.get('symbol', orig_name)
            full_name = result.get('name', '')

            # Map to categories
            if isinstance(go_bp, list) and go_bp:
                broad_category, rep_term, detailed_terms = map_to_go_slim_categories(go_bp)

                # Store full GO term objects (limit to 100 for file size)
                go_terms_limited = go_bp[:100]
                go_terms_json = json.dumps(go_terms_limited)
            else:
                broad_category = 'Other/Unannotated'
                rep_term = None
                detailed_terms = []
                go_terms_json = '[]'

            rows.append({
                'gene_id': orig_id,
                'gene_name': orig_name,
                'symbol': symbol,
                'full_name': full_name,
                'go_category_broad': broad_category,
                'go_category_detailed': json.dumps(detailed_terms),  # Store as JSON list
                'go_terms_full': go_terms_json,
                'representative_term': rep_term,
                'num_go_terms': len(detailed_terms),
            })
        else:
            # No GO annotations found
            rows.append({
                'gene_id': orig_id,
                'gene_name': orig_name,
                'symbol': orig_name,
                'full_name': '',
                'go_category_broad': 'Other/Unannotated',
                'go_category_detailed': '[]',
                'go_terms_full': '[]',
                'representative_term': None,
                'num_go_terms': 0,
            })

    return pd.DataFrame(rows)


def main():
    """Main execution function."""

    # Paths
    repo_root = Path(__file__).parent.parent
    parquet_path = repo_root / 'notebooks/example_data/HG00096_vcf2exp_predictions.parquet'
    output_parquet = repo_root / 'notebooks/example_data/gene_go_annotations.parquet'
    output_csv = repo_root / 'notebooks/example_data/gene_go_annotations.csv'

    print("🧬 Gene Ontology Annotation Generator")
    print("="*70)

    # Load gene IDs
    gene_ids, gene_names = load_gene_ids(parquet_path)

    # Fetch GO annotations (two-pass strategy)
    results = fetch_go_annotations(gene_ids, gene_names, batch_size=1000)

    # Process results
    go_df = process_results(results, gene_ids, gene_names)

    # Statistics
    print(f"\n📊 Annotation Statistics:")
    print(f"   Total genes: {len(go_df):,}")
    annotated = (go_df['go_category_broad'] != 'Other/Unannotated').sum()
    print(f"   Annotated: {annotated:,} ({annotated/len(go_df)*100:.1f}%)")
    print(f"   Unannotated: {(go_df['go_category_broad'] == 'Other/Unannotated').sum():,}")

    print(f"\n   Broad category distribution:")
    cat_dist = go_df['go_category_broad'].value_counts()
    for cat, count in cat_dist.items():
        pct = count/len(go_df)*100
        print(f"     {cat:25s}: {count:5,} ({pct:5.1f}%)")

    # Save to parquet
    print(f"\n💾 Saving annotations...")
    go_df.to_parquet(output_parquet, index=False)
    print(f"   ✅ Parquet: {output_parquet}")
    print(f"      Size: {output_parquet.stat().st_size / (1024*1024):.2f} MB")

    # Also save as CSV for easy inspection
    go_df.to_csv(output_csv, index=False)
    print(f"   ✅ CSV: {output_csv}")
    print(f"      Size: {output_csv.stat().st_size / (1024*1024):.2f} MB")

    # Sample output
    print(f"\n📋 Sample annotations:")
    print(go_df[['gene_name', 'go_category_broad', 'representative_term', 'num_go_terms']].head(10).to_string(index=False))

    print("\n" + "="*70)
    print("✅ GO annotation generation complete!")
    print("="*70)

    return go_df


if __name__ == '__main__':
    go_df = main()
