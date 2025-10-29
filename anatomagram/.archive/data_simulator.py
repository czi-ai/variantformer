"""Data simulator for anatomagram development and testing.

This module generates realistic simulated data for both VCF2Risk and VCF2Expression
workflows without requiring actual model predictions. Uses real tissue mappings
and biologically plausible distributions.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class AnatomagramDataSimulator:
    """Simulator for generating realistic anatomagram data for development."""
    
    def __init__(self, tissue_mapping_path: Optional[str] = None):
        """Initialize simulator with tissue mappings.
        
        Args:
            tissue_mapping_path: Path to tissue mapping JSON file
        """
        if tissue_mapping_path is None:
            # Default to the manual mapping in the anatomagram data directory
            current_dir = Path(__file__).parent.parent
            tissue_mapping_path = current_dir / "data" / "tissue_uberon_mapping_manual.json"
        
        self.tissue_mapping_path = Path(tissue_mapping_path)
        self.tissue_mappings = self._load_tissue_mappings()
        self.tissues_only = self._filter_tissues_only()
        
        print(f"✅ AnatomagramDataSimulator initialized")
        print(f"   - Loaded {len(self.tissue_mappings)} total entries")
        print(f"   - Available tissues (excluding cell lines): {len(self.tissues_only)}")
        
    def _load_tissue_mappings(self) -> Dict:
        """Load tissue mappings from JSON file."""
        try:
            with open(self.tissue_mapping_path, 'r') as f:
                data = json.load(f)
            return data['tissue_mappings']
        except Exception as e:
            print(f"❌ Error loading tissue mappings from {self.tissue_mapping_path}: {e}")
            return {}
    
    def _filter_tissues_only(self) -> Dict:
        """Filter to only include actual tissues (exclude cell lines)."""
        return {
            tissue_id: info for tissue_id, info in self.tissue_mappings.items()
            if info.get('tissue_type') == 'tissue'
        }
    
    def simulate_risk_data(
        self,
        num_tissues: Optional[int] = None,
        seed: int = 42,
        risk_profile: str = 'mixed'
    ) -> pd.DataFrame:
        """Simulate VCF2Risk prediction data.
        
        Args:
            num_tissues: Number of tissues to simulate (default: all available tissues)
            seed: Random seed for reproducibility
            risk_profile: Risk distribution profile ('low', 'high', 'mixed', 'brain_focused')
            
        Returns:
            DataFrame with columns: tissue_id, tissue_name, ad_risk
        """
        np.random.seed(seed)
        
        if num_tissues is None:
            selected_tissues = self.tissues_only
        else:
            tissue_ids = list(self.tissues_only.keys())
            selected_ids = np.random.choice(tissue_ids, size=min(num_tissues, len(tissue_ids)), replace=False)
            selected_tissues = {tid: self.tissues_only[tid] for tid in selected_ids}
        
        rows = []
        
        for tissue_id, tissue_info in selected_tissues.items():
            tissue_name = tissue_info['tissue_name']
            
            # Generate risk based on profile and tissue type
            risk_value = self._generate_tissue_risk(tissue_name, risk_profile)
            
            rows.append({
                'tissue_id': int(tissue_id),
                'tissue_name': tissue_name,
                'ad_risk': risk_value
            })
        
        df = pd.DataFrame(rows)
        print(f"✅ Simulated AD risk data for {len(df)} tissues (profile: {risk_profile})")
        print(f"   - Risk range: {df['ad_risk'].min():.3f} to {df['ad_risk'].max():.3f}")
        print(f"   - Mean risk: {df['ad_risk'].mean():.3f}")
        
        return df
    
    def simulate_expression_data(
        self,
        genes: List[str] = None,
        num_tissues: Optional[int] = None,
        seed: int = 42,
        expression_profile: str = 'realistic'
    ) -> pd.DataFrame:
        """Simulate VCF2Expression prediction data.
        
        Args:
            genes: List of gene names to simulate (default: APOE, PSEN1, PSEN2)
            num_tissues: Number of tissues per gene (default: all available tissues)
            seed: Random seed for reproducibility
            expression_profile: Expression pattern ('low', 'high', 'realistic', 'brain_specific')
            
        Returns:
            DataFrame with columns: tissues, predicted_expression (compatible with converter)
        """
        np.random.seed(seed)
        
        if genes is None:
            genes = ['APOE', 'PSEN1', 'PSEN2']
        
        if num_tissues is None:
            selected_tissues = list(self.tissues_only.values())
        else:
            tissue_list = list(self.tissues_only.values())
            selected_tissues = np.random.choice(tissue_list, size=min(num_tissues, len(tissue_list)), replace=False)
        
        rows = []
        
        for gene in genes:
            # Create tissue lists for this gene
            tissue_names = [t['tissue_name'] for t in selected_tissues]
            expressions = []
            
            for tissue_info in selected_tissues:
                tissue_name = tissue_info['tissue_name']
                expr_value = self._generate_tissue_expression(gene, tissue_name, expression_profile)
                expressions.append(expr_value)
            
            # Create row in format expected by converter
            rows.append({
                'gene': gene,
                'tissues': tissue_names,  # List of tissue names
                'predicted_expression': expressions  # List of expression values
            })
        
        df = pd.DataFrame(rows)
        print(f"✅ Simulated expression data for {len(genes)} genes across {len(selected_tissues)} tissues")
        print(f"   - Genes: {genes}")
        print(f"   - Expression profile: {expression_profile}")
        
        # Print expression ranges per gene
        for _, row in df.iterrows():
            gene = row['gene']
            expr_values = row['predicted_expression']
            print(f"   - {gene}: {min(expr_values):.2f} to {max(expr_values):.2f} (mean: {np.mean(expr_values):.2f})")
        
        return df
    
    def _generate_tissue_risk(self, tissue_name: str, profile: str) -> float:
        """Generate realistic AD risk value for a tissue."""
        tissue_lower = tissue_name.lower()
        
        # Base risk parameters by profile
        if profile == 'low':
            base_alpha, base_beta = 2, 8  # Low risk bias
        elif profile == 'high':
            base_alpha, base_beta = 6, 4  # High risk bias
        elif profile == 'brain_focused':
            base_alpha, base_beta = 3, 7  # Low overall, high for brain
        else:  # mixed
            base_alpha, base_beta = 4, 6  # Moderate
        
        # Tissue-specific modulation
        alpha, beta = base_alpha, base_beta
        
        # Brain tissues have higher risk in brain-focused profile
        if 'brain' in tissue_lower:
            if profile == 'brain_focused':
                alpha, beta = 8, 3  # Much higher for brain tissues
            else:
                alpha += 1  # Slightly higher for brain in other profiles
        
        # Blood-related tissues
        elif 'blood' in tissue_lower:
            alpha += 0.5
        
        # Liver and kidney (metabolic organs)
        elif tissue_lower in ['liver', 'kidney - cortex', 'kidney - medulla']:
            alpha += 0.3
        
        # Generate from beta distribution and clip to valid range
        risk = np.random.beta(alpha, beta)
        return np.clip(risk, 0.001, 0.999)  # Avoid exact 0 or 1
    
    def _generate_tissue_expression(self, gene: str, tissue_name: str, profile: str) -> float:
        """Generate realistic gene expression value for a tissue."""
        tissue_lower = tissue_name.lower()
        gene_upper = gene.upper()
        
        # Base expression parameters (log-normal: mu=log(median), sigma=log-scale variation)
        if profile == 'low':
            base_mu, base_sigma = -1.0, 0.8  # Low expression
        elif profile == 'high':
            base_mu, base_sigma = 1.5, 0.6   # High expression
        elif profile == 'brain_specific':
            base_mu, base_sigma = -0.5, 1.0  # Low default, high for brain
        else:  # realistic
            base_mu, base_sigma = 0.5, 1.2   # Moderate with high variation
        
        mu, sigma = base_mu, base_sigma
        
        # Gene-specific patterns
        if gene_upper == 'APOE':
            # APOE highly expressed in brain and liver
            if 'brain' in tissue_lower or 'liver' in tissue_lower:
                mu += 1.5
            elif 'adrenal' in tissue_lower or 'kidney' in tissue_lower:
                mu += 0.8
        
        elif gene_upper == 'PSEN1':
            # PSEN1 broadly expressed, higher in brain
            if 'brain' in tissue_lower:
                mu += 1.0
            elif 'heart' in tissue_lower or 'muscle' in tissue_lower:
                mu += 0.5
        
        elif gene_upper == 'PSEN2':
            # PSEN2 more restricted, brain and some other tissues
            if 'brain' in tissue_lower:
                mu += 1.2
            elif tissue_lower in ['pancreas', 'thyroid', 'adrenal gland']:
                mu += 0.6
            else:
                mu -= 0.3  # Lower in most other tissues
        
        # Tissue-specific modulation (applies to all genes)
        if profile == 'brain_specific' and 'brain' in tissue_lower:
            mu += 2.0  # Much higher in brain for brain-specific profile
        
        # Generate from log-normal and ensure positive
        expression = np.random.lognormal(mu, sigma)
        return max(expression, 0.01)  # Ensure minimum expression
    
    def get_tissue_summary(self) -> Dict:
        """Get summary of available tissues."""
        brain_tissues = [t for t in self.tissues_only.values() if 'brain' in t['tissue_name'].lower()]
        organ_tissues = [t for t in self.tissues_only.values() if 'brain' not in t['tissue_name'].lower()]
        
        return {
            'total_tissues': len(self.tissues_only),
            'brain_tissues': len(brain_tissues),
            'organ_tissues': len(organ_tissues),
            'tissue_names': [t['tissue_name'] for t in self.tissues_only.values()],
            'brain_tissue_names': [t['tissue_name'] for t in brain_tissues],
            'organ_tissue_names': [t['tissue_name'] for t in organ_tissues]
        }
    
    def simulate_mixed_scenario(self, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create a mixed scenario with both risk and expression data for testing.
        
        Returns:
            Tuple of (risk_df, expression_df) for comprehensive testing
        """
        np.random.seed(seed)
        
        # Use subset of tissues for focused testing
        num_tissues = min(25, len(self.tissues_only))
        
        # Generate risk data with brain-focused profile
        risk_df = self.simulate_risk_data(
            num_tissues=num_tissues,
            seed=seed,
            risk_profile='brain_focused'
        )
        
        # Generate expression data for AD-related genes
        expression_df = self.simulate_expression_data(
            genes=['APOE', 'PSEN1', 'PSEN2'],
            num_tissues=num_tissues,
            seed=seed + 1,
            expression_profile='realistic'
        )
        
        print(f"\n✅ Mixed scenario generated:")
        print(f"   - Risk data: {len(risk_df)} tissues")
        print(f"   - Expression data: {len(expression_df)} genes × {num_tissues} tissues each")
        
        return risk_df, expression_df


def create_demo_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience function to create demo data for quick testing."""
    simulator = AnatomagramDataSimulator()
    return simulator.simulate_mixed_scenario(seed=123)


if __name__ == "__main__":
    # Demo usage
    print("=== AnatomagramDataSimulator Demo ===\n")
    
    simulator = AnatomagramDataSimulator()
    
    # Show tissue summary
    summary = simulator.get_tissue_summary()
    print(f"📊 Tissue Summary:")
    print(f"   - Total tissues: {summary['total_tissues']}")
    print(f"   - Brain tissues: {summary['brain_tissues']}")
    print(f"   - Other organs: {summary['organ_tissues']}")
    
    print(f"\n🧠 Brain tissues: {summary['brain_tissue_names'][:5]}...")
    print(f"🫁 Organ tissues: {summary['organ_tissue_names'][:5]}...")
    
    # Generate sample data
    print(f"\n" + "="*50)
    risk_df, expression_df = simulator.simulate_mixed_scenario()
    
    print(f"\n📋 Sample Risk Data:")
    print(risk_df.head())
    
    print(f"\n📋 Sample Expression Data:")
    print(expression_df.head())