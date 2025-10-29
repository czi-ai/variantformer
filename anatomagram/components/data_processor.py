"""Data processing utilities for anatomogram expression data."""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Any, Union
from io import StringIO, BytesIO


class ExpressionDataProcessor:
    """Process and validate gene expression data for anatomogram visualization."""
    
    def __init__(self):
        self.supported_formats = ['.json', '.csv', '.tsv']
    
    def load_json(self, file_content: bytes) -> Dict[str, Any]:
        """Load expression data from JSON content.
        
        Args:
            file_content: Raw file content as bytes
            
        Returns:
            Dictionary with expression data
            
        Raises:
            ValueError: If JSON is invalid or structure is incorrect
        """
        try:
            data = json.loads(file_content.decode('utf-8'))
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise ValueError(f"Error loading JSON: {e}")
    
    def load_csv(self, file_content: bytes, sep: str = ',') -> Dict[str, Any]:
        """Load expression data from CSV/TSV content.
        
        Expected format:
        - First column: Gene names
        - Subsequent columns: UBERON IDs as headers
        
        Args:
            file_content: Raw file content as bytes
            sep: Separator character (',' for CSV, '\t' for TSV)
            
        Returns:
            Dictionary in standard format with 'genes' key
        """
        try:
            # Read CSV/TSV into DataFrame
            df = pd.read_csv(BytesIO(file_content), sep=sep)
            
            # Check if first column contains gene names
            if df.empty:
                raise ValueError("CSV file is empty")
            
            # Get gene column (first column)
            gene_col = df.columns[0]
            genes_dict = {}
            
            # Convert each row to gene expression dict
            for _, row in df.iterrows():
                gene_name = str(row[gene_col])
                # Get expression values for all tissues (skip gene column)
                tissue_values = {}
                for col in df.columns[1:]:
                    try:
                        # Ensure numeric value
                        value = float(row[col])
                        tissue_values[col] = value
                    except (ValueError, TypeError):
                        # Skip non-numeric values
                        continue
                
                if tissue_values:  # Only add if there are valid values
                    genes_dict[gene_name] = tissue_values
            
            return {"genes": genes_dict}
            
        except Exception as e:
            raise ValueError(f"Error loading CSV: {e}")
    
    def load_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Load expression data from file content based on filename extension.
        
        Args:
            file_content: Raw file content
            filename: Original filename to determine format
            
        Returns:
            Processed expression data dictionary
        """
        filename_lower = filename.lower()
        
        if filename_lower.endswith('.json'):
            return self.load_json(file_content)
        elif filename_lower.endswith('.csv'):
            return self.load_csv(file_content, sep=',')
        elif filename_lower.endswith('.tsv'):
            return self.load_csv(file_content, sep='\t')
        else:
            raise ValueError(f"Unsupported file format. Supported: {', '.join(self.supported_formats)}")
    
    def validate_format(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate the expression data format.
        
        Args:
            data: Expression data dictionary
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Check top-level structure
        if not isinstance(data, dict):
            return False, "Data must be a dictionary"
        
        if 'genes' not in data:
            return False, "Data must contain 'genes' key"
        
        if not isinstance(data['genes'], dict):
            return False, "'genes' must be a dictionary"
        
        if not data['genes']:
            return False, "No genes found in data"
        
        # Validate gene entries
        for gene, tissues in data['genes'].items():
            if not isinstance(tissues, dict):
                return False, f"Expression values for gene '{gene}' must be a dictionary"
            
            # Check tissue values
            for tissue, value in tissues.items():
                if not isinstance(value, (int, float)):
                    return False, f"Expression value for {gene}/{tissue} must be numeric, got {type(value).__name__}"
                
                if not isinstance(tissue, str):
                    return False, f"Tissue ID must be string, got {type(tissue).__name__}"
        
        return True, "Data is valid"
    
    def normalize_values(self, data: Dict[str, Any], method: str = 'minmax') -> Dict[str, Any]:
        """Normalize expression values to 0-1 range.
        
        Args:
            data: Expression data dictionary
            method: Normalization method ('minmax' or 'zscore')
            
        Returns:
            Dictionary with normalized values
        """
        # Collect all values
        all_values = []
        for gene_data in data['genes'].values():
            all_values.extend(gene_data.values())
        
        if not all_values:
            return data
        
        # Calculate normalization parameters
        values_array = np.array(all_values)
        
        if method == 'minmax':
            min_val = values_array.min()
            max_val = values_array.max()
            
            if max_val == min_val:
                # All values are the same
                scale = 1.0
                offset = 0.5  # Map to middle of range
            else:
                scale = 1.0 / (max_val - min_val)
                offset = -min_val * scale
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        
        # Apply normalization
        normalized_data = {"genes": {}}
        for gene, tissues in data['genes'].items():
            normalized_tissues = {}
            for tissue, value in tissues.items():
                if method == 'minmax':
                    if max_val == min_val:
                        normalized_value = 0.5
                    else:
                        normalized_value = value * scale + offset
                    normalized_tissues[tissue] = float(np.clip(normalized_value, 0, 1))
            
            normalized_data['genes'][gene] = normalized_tissues
        
        return normalized_data
    
    def get_gene_list(self, data: Dict[str, Any]) -> List[str]:
        """Extract sorted list of gene names.
        
        Args:
            data: Expression data dictionary
            
        Returns:
            Sorted list of gene names
        """
        if 'genes' not in data:
            return []
        
        return sorted(data['genes'].keys())
    
    def get_tissue_list(self, data: Dict[str, Any]) -> Set[str]:
        """Extract all unique tissue IDs.
        
        Args:
            data: Expression data dictionary
            
        Returns:
            Set of unique tissue IDs
        """
        tissues = set()
        
        if 'genes' in data:
            for gene_data in data['genes'].values():
                tissues.update(gene_data.keys())
        
        return tissues
    
    def get_summary_statistics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics for the expression data.
        
        Args:
            data: Expression data dictionary
            
        Returns:
            Dictionary with summary statistics
        """
        if 'genes' not in data or not data['genes']:
            return {
                'num_genes': 0,
                'num_tissues': 0,
                'mean_expression': 0,
                'std_expression': 0,
                'min_expression': 0,
                'max_expression': 0
            }
        
        # Collect statistics
        all_values = []
        tissues = set()
        
        for gene_data in data['genes'].values():
            all_values.extend(gene_data.values())
            tissues.update(gene_data.keys())
        
        if all_values:
            values_array = np.array(all_values)
            stats = {
                'num_genes': len(data['genes']),
                'num_tissues': len(tissues),
                'mean_expression': float(values_array.mean()),
                'std_expression': float(values_array.std()),
                'min_expression': float(values_array.min()),
                'max_expression': float(values_array.max()),
                'total_data_points': len(all_values)
            }
        else:
            stats = {
                'num_genes': len(data['genes']),
                'num_tissues': 0,
                'mean_expression': 0,
                'std_expression': 0,
                'min_expression': 0,
                'max_expression': 0,
                'total_data_points': 0
            }
        
        return stats
    
    def filter_by_threshold(self, data: Dict[str, Any], threshold: float) -> Dict[str, Any]:
        """Filter expression data by minimum threshold.
        
        Args:
            data: Expression data dictionary
            threshold: Minimum expression value to include
            
        Returns:
            Filtered expression data
        """
        filtered_data = {"genes": {}}
        
        for gene, tissues in data.get('genes', {}).items():
            filtered_tissues = {
                tissue: value 
                for tissue, value in tissues.items() 
                if value >= threshold
            }
            
            # Only include gene if it has tissues above threshold
            if filtered_tissues:
                filtered_data['genes'][gene] = filtered_tissues
        
        return filtered_data