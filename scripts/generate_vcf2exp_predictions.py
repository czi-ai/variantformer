#!/usr/bin/env python3
"""
Generate real VCF2Expression predictions using gene-level batching.

This script processes genes in batches to make the full prediction tractable,
with resume capability and intermediate result saving.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import json
from datetime import datetime, timedelta
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from processors.vcfprocessor import VCFProcessor


def create_gene_batches(genes_df, batch_size=100):
    """Split genes into batches for processing."""
    total_genes = len(genes_df)
    batches = []
    
    for i in range(0, total_genes, batch_size):
        end_idx = min(i + batch_size, total_genes)
        batch_genes = genes_df.iloc[i:end_idx].copy()
        batch_info = {
            'batch_id': i // batch_size,
            'start_idx': i,
            'end_idx': end_idx,
            'genes_df': batch_genes,
            'gene_count': len(batch_genes)
        }
        batches.append(batch_info)
    
    return batches


def load_progress(progress_file):
    """Load progress from previous run."""
    if Path(progress_file).exists():
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Could not load progress file: {e}")
            return {}
    return {}


def save_progress(progress_file, progress_data):
    """Save current progress."""
    try:
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2, default=str)
    except Exception as e:
        print(f"⚠️  Could not save progress: {e}")


def process_gene_batch(batch_info, all_tissues_list, vcf_processor, model_components, vcf_path, output_dir, dataloader_batch_size=4):
    """Process a single batch of genes with configurable DataLoader batch size for memory management."""
    batch_id = batch_info['batch_id']
    genes_df = batch_info['genes_df']
    
    print(f"\n🔄 Processing batch {batch_id}: genes {batch_info['start_idx']}-{batch_info['end_idx']} ({batch_info['gene_count']} genes)")
    
    # Create query_df for this batch: one row per gene with comma-separated tissues
    batch_query_rows = []
    for _, gene_row in genes_df.iterrows():
        batch_query_rows.append({
            'gene_id': gene_row['gene_id'],
            'tissues': ','.join(all_tissues_list)
        })
    
    batch_query_df = pd.DataFrame(batch_query_rows)
    print(f"   Query size: {len(batch_query_df):,} rows ({batch_info['gene_count']} genes, {len(all_tissues_list)} tissues per gene)")
    print(f"   DataLoader batch size: {dataloader_batch_size} (memory optimization)")
    
    try:
        start_time = time.time()
        
        # Create dataset and dataloader with custom batch_size for memory management
        batch_dataset, batch_dataloader = vcf_processor.create_data(vcf_path, batch_query_df, batch_size=dataloader_batch_size)
        print(f"   ✅ Dataset created: {len(batch_dataset)} samples, {len(batch_dataloader)} batches")
        
        # Run predictions using pre-loaded model
        model, checkpoint_path, trainer = model_components
        batch_predictions = vcf_processor.predict(model, checkpoint_path, trainer, batch_dataloader, batch_dataset)
        
        end_time = time.time()
        elapsed_minutes = (end_time - start_time) / 60
        
        print(f"   ✅ Predictions complete: {batch_predictions.shape} ({elapsed_minutes:.1f} min)")
        
        # Convert numpy arrays to lists for parquet compatibility
        if 'predicted_expression' in batch_predictions.columns:
            batch_predictions['predicted_expression'] = batch_predictions['predicted_expression'].apply(
                lambda x: x.tolist() if hasattr(x, 'tolist') else x
            )
        if 'embeddings' in batch_predictions.columns:
            batch_predictions['embeddings'] = batch_predictions['embeddings'].apply(
                lambda x: x.tolist() if hasattr(x, 'tolist') else x
            )
        
        # Save intermediate results
        batch_output_file = Path(output_dir) / f"batch_{batch_id:03d}_predictions.parquet"
        batch_predictions.to_parquet(batch_output_file, index=False)
        print(f"   💾 Saved to: {batch_output_file}")
        
        return {
            'batch_id': batch_id,
            'success': True,
            'predictions_shape': batch_predictions.shape,
            'elapsed_minutes': elapsed_minutes,
            'output_file': str(batch_output_file),
            'timestamp': datetime.now()
        }
        
    except Exception as e:
        print(f"   ❌ Error processing batch {batch_id}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'batch_id': batch_id,
            'success': False,
            'error': str(e),
            'timestamp': datetime.now()
        }


def combine_batch_results(output_dir, all_genes_df, all_tissues_list):
    """Combine all batch prediction files into final datasets."""
    print(f"\n📊 Combining batch results...")
    
    batch_files = sorted(Path(output_dir).glob("batch_*_predictions.parquet"))
    if not batch_files:
        print(f"❌ No batch files found in {output_dir}")
        return None, None
    
    print(f"   Found {len(batch_files)} batch files")
    
    # Load and combine all batch predictions
    all_predictions = []
    for batch_file in batch_files:
        try:
            batch_df = pd.read_parquet(batch_file)
            all_predictions.append(batch_df)
            print(f"   ✅ Loaded {batch_file.name}: {batch_df.shape}")
        except Exception as e:
            print(f"   ❌ Error loading {batch_file.name}: {e}")
    
    if not all_predictions:
        print(f"❌ No valid batch files could be loaded")
        return None, None
    
    # Combine all predictions
    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    print(f"✅ Combined predictions: {combined_predictions.shape}")
    
    # Group by gene to create final dataset structure
    print(f"📋 Creating final dataset structure...")
    
    if 'gene_id' not in combined_predictions.columns:
        print(f"❌ No gene_id column found in predictions")
        return None, None
    
    unique_gene_ids = combined_predictions['gene_id'].unique()
    print(f"   Unique genes in predictions: {len(unique_gene_ids):,}")
    
    final_dataset_rows = []
    
    for gene_id in unique_gene_ids:
        gene_predictions = combined_predictions[combined_predictions['gene_id'] == gene_id]
        
        # Get gene name
        matching_genes = all_genes_df[all_genes_df['gene_id'] == gene_id]
        gene_name = matching_genes['gene_name'].iloc[0] if len(matching_genes) > 0 else gene_id
        
        # Extract tissue names and predictions  
        if 'tissue_names' in gene_predictions.columns:
            tissue_names = gene_predictions['tissue_names'].iloc[0] if isinstance(gene_predictions['tissue_names'].iloc[0], list) else list(gene_predictions['tissue_names'])
        elif 'tissues' in gene_predictions.columns:
            tissue_names = list(gene_predictions['tissues']) if not isinstance(gene_predictions['tissues'].iloc[0], list) else gene_predictions['tissues'].iloc[0]
        else:
            tissue_names = all_tissues_list
        
        if 'predicted_expression' in gene_predictions.columns:
            if isinstance(gene_predictions['predicted_expression'].iloc[0], list):
                expressions = gene_predictions['predicted_expression'].iloc[0]
            else:
                expressions = list(gene_predictions['predicted_expression'])
        else:
            expressions = [0.0] * len(tissue_names)
        
        final_dataset_rows.append({
            'gene_id': gene_id,
            'gene_name': gene_name,
            'tissues': tissue_names,
            'predicted_expression': expressions
        })
    
    final_dataset = pd.DataFrame(final_dataset_rows)
    print(f"✅ Final dataset created: {len(final_dataset):,} genes")
    
    # Create only the full dataset (no more subsets)
    dataset_subsets = {
        'full': final_dataset
    }
    
    return final_dataset, dataset_subsets


def create_query_index(full_dataset_df, all_tissues_list):
    """Create query-optimized index format for fast lookups."""
    print(f"\n📊 Creating query-optimized index...")
    
    # Load tissue vocabulary for tissue ID mapping
    import yaml
    import numpy as np
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    tissue_vocab_path = project_root / 'vocabs' / 'tissue_vocab.yaml'
    
    try:
        with open(tissue_vocab_path, 'r') as f:
            tissue_vocab = yaml.safe_load(f)
    except Exception as e:
        print(f"⚠️  Could not load tissue vocabulary, using index mapping: {e}")
        # Fallback: create mapping from tissue names to indices
        tissue_vocab = {tissue: idx for idx, tissue in enumerate(all_tissues_list)}
    
    query_rows = []
    
    for _, row in full_dataset_df.iterrows():
        gene_id = row['gene_id']
        tissue_names = row['tissues']
        expressions = row['predicted_expression']
        
        # Handle nested numpy arrays - tissues might be array of arrays
        if isinstance(tissue_names, np.ndarray):
            if len(tissue_names) == 1 and isinstance(tissue_names[0], np.ndarray):
                # tissues is array containing one array of all tissue names
                tissue_names = tissue_names[0].tolist()
            else:
                tissue_names = tissue_names.tolist()
        elif hasattr(tissue_names, 'tolist'):
            tissue_names = tissue_names.tolist()
        elif not isinstance(tissue_names, list):
            tissue_names = list(tissue_names)
            
        # Handle nested numpy arrays - expressions might be array of arrays  
        if isinstance(expressions, np.ndarray):
            if len(expressions) == 1 and isinstance(expressions[0], np.ndarray):
                # expressions is array containing one array of all expression values
                expressions = expressions[0].tolist()
            else:
                expressions = expressions.tolist()
        elif hasattr(expressions, 'tolist'):
            expressions = expressions.tolist()
        elif not isinstance(expressions, list):
            expressions = list(expressions)
        
        # Create one row per gene-tissue combination
        for tissue_name, expression in zip(tissue_names, expressions):
            tissue_id = tissue_vocab.get(tissue_name)
            if tissue_id is not None:
                query_rows.append({
                    'gene_id': gene_id,
                    'tissue_id': tissue_id,
                    'expression': float(expression)  # Ensure single float value
                })
    
    query_index_df = pd.DataFrame(query_rows)
    print(f"   ✅ Query index created: {len(query_index_df):,} gene-tissue combinations")
    
    return query_index_df


def save_final_datasets(dataset_subsets, output_dir, all_tissues_list):
    """Save final datasets - full model data only."""
    if not dataset_subsets:
        print(f"❌ No datasets to save")
        return []
    
    print(f"\n💾 Saving final datasets...")
    
    saved_files = []
    
    # Only save the full dataset (no more subsets)
    if 'full' in dataset_subsets:
        full_dataset_df = dataset_subsets['full']
        
        # Save full model output (complete data)
        full_filename = "vcf2exp_full_predictions.parquet"
        full_filepath = Path(output_dir) / full_filename
        full_dataset_df.to_parquet(full_filepath, index=False)
        
        if full_filepath.exists():
            file_size_mb = full_filepath.stat().st_size / (1024 * 1024)
            saved_files.append({
                'name': 'full_model_output',
                'genes': len(full_dataset_df),
                'filename': full_filename,
                'path': str(full_filepath),
                'size_mb': file_size_mb
            })
            print(f"   ✅ Full model output: {full_filename} ({file_size_mb:.2f} MB)")
    
    return saved_files


def print_transfer_commands(saved_files, output_dir):
    """Print SSH transfer commands."""
    if not saved_files:
        return
        
    print(f"\n🚚 SSH Transfer Commands:")
    print("=" * 60)
    
    for file_info in saved_files:
        print(f"scp cluster:{file_info['path']} ./{file_info['filename']}")
    
    print(f"\n# Or transfer entire directory:")
    print(f"scp -r cluster:{output_dir} ./vcf2exp_predictions/")
    print("=" * 60)


def estimate_time_remaining(completed_batches, total_batches, avg_time_per_batch):
    """Estimate time remaining for completion."""
    remaining_batches = total_batches - completed_batches
    remaining_minutes = remaining_batches * avg_time_per_batch
    remaining_hours = remaining_minutes / 60
    
    if remaining_hours > 24:
        return f"{remaining_hours/24:.1f} days"
    elif remaining_hours > 1:
        return f"{remaining_hours:.1f} hours"
    else:
        return f"{remaining_minutes:.0f} minutes"


def main():
    parser = argparse.ArgumentParser(description="Generate real VCF2Expression predictions with batching")
    parser.add_argument("--vcf-path", 
                       default="/mnt/czi-sci-ai/intrinsic-variation-gene-ex-2/project_gene_regulation/dna2cell_training/v2_pcg_flash2/sample_vcf/HG00096.vcf.gz",
                       help="Path to VCF file")
    parser.add_argument("--output-dir", default="/tmp/vcf2exp_predictions_real", 
                       help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Number of genes per batch")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from previous run")
    parser.add_argument("--max-genes", type=int, default=None,
                       help="Limit number of genes (for testing)")
    parser.add_argument("--combine-only", action="store_true",
                       help="Only combine existing batch results (skip prediction)")
    parser.add_argument("--dataloader-batch-size", type=int, default=4,
                       help="DataLoader batch size (reduce for memory management, default: 4)")
    
    args = parser.parse_args()
    
    print("🚀 Real VCF2Expression Prediction Generator")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print(f"VCF path: {args.vcf_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Batch size: {args.batch_size} genes")
    print(f"DataLoader batch size: {args.dataloader_batch_size}")
    print(f"Resume: {args.resume}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    progress_file = output_dir / "progress.json"
    
    try:
        if not args.combine_only:
            # Initialize VCF processor
            print(f"\n📋 Loading processor and data...")
            vcf_processor = VCFProcessor(model_class='D2C_PCG')
            all_genes_df = vcf_processor.get_genes()
            all_tissues_list = list(vcf_processor.get_tissues())
            
            # Limit genes if requested
            if args.max_genes:
                all_genes_df = all_genes_df.head(args.max_genes)
            
            print(f"✅ Loaded {len(all_genes_df):,} genes")
            print(f"✅ Loaded {len(all_tissues_list)} tissues/cell lines")
            print(f"📊 Total predictions: {len(all_genes_df):,} × {len(all_tissues_list)} = {len(all_genes_df) * len(all_tissues_list):,}")
            
            # Create gene batches
            gene_batches = create_gene_batches(all_genes_df, args.batch_size)
            print(f"📦 Created {len(gene_batches)} batches ({args.batch_size} genes each)")
            
            # Load progress
            progress = load_progress(progress_file) if args.resume else {}
            completed_batches = set(progress.get('completed_batches', []))
            
            print(f"📈 Progress: {len(completed_batches)}/{len(gene_batches)} batches completed")
            
            # Load model once (reuse for all batches)
            print(f"\n🔄 Loading DNA2Cell model...")
            model, checkpoint_path, trainer = vcf_processor.load_model()
            model_components = (model, checkpoint_path, trainer)
            print(f"✅ Model loaded successfully")
            
            # Process batches
            batch_times = []
            
            for batch_info in gene_batches:
                batch_id = batch_info['batch_id']
                
                if batch_id in completed_batches:
                    print(f"\n⏭️  Skipping batch {batch_id} (already completed)")
                    continue
                
                # Process batch
                result = process_gene_batch(
                    batch_info, all_tissues_list, vcf_processor, 
                    model_components, args.vcf_path, output_dir, args.dataloader_batch_size
                )
                
                # Update progress
                if result['success']:
                    completed_batches.add(batch_id)
                    batch_times.append(result['elapsed_minutes'])
                    
                    # Calculate ETA
                    if len(batch_times) > 0:
                        avg_time = np.mean(batch_times[-10:])  # Average of last 10 batches
                        eta = estimate_time_remaining(len(completed_batches), len(gene_batches), avg_time)
                        print(f"   📊 Progress: {len(completed_batches)}/{len(gene_batches)} batches, ETA: {eta}")
                
                # Save progress
                progress_data = {
                    'completed_batches': list(completed_batches),
                    'total_batches': len(gene_batches),
                    'last_update': datetime.now(),
                    'batch_times': batch_times[-50:]  # Keep last 50 times
                }
                save_progress(progress_file, progress_data)
            
            print(f"\n✅ All batches completed!")
        
        # Combine results
        print(f"\n📊 Loading data for combination...")
        if args.combine_only:
            vcf_processor = VCFProcessor(model_class='D2C_PCG')
            all_genes_df = vcf_processor.get_genes()
            all_tissues_list = list(vcf_processor.get_tissues())
        
        final_dataset, dataset_subsets = combine_batch_results(output_dir, all_genes_df, all_tissues_list)
        
        if dataset_subsets:
            # Save final datasets
            saved_files = save_final_datasets(dataset_subsets, output_dir, all_tissues_list)
            
            # Print transfer commands
            print_transfer_commands(saved_files, output_dir)
            
            print(f"\n✅ Real prediction generation completed!")
            print(f"📁 Results saved to: {output_dir}")
            print(f"⏱️  Completed at: {datetime.now()}")
        
    except Exception as e:
        print(f"\n❌ Error in prediction generation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())