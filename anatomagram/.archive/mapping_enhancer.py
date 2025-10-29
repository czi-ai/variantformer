#!/usr/bin/env python3
"""
Anatomagram Mapping Enhancer
Creates enhanced unified mapping system that handles SVG-tissue mismatches
"""

import json
import re
from pathlib import Path
from typing import Dict, Set, List, Optional, Tuple


class AnatomagramMappingEnhancer:
    """Enhanced mapping system for anatomagram visualization"""
    
    def __init__(self, anatomagram_dir: str):
        self.anatomagram_dir = Path(anatomagram_dir)
        self.data_dir = self.anatomagram_dir / "data"
        self.svg_dir = self.anatomagram_dir / "assets" / "svg"
        
        # Load existing data
        self.manual_mapping = self._load_manual_mapping()
        self.colleague_csv = self._load_colleague_csv()
        self.svg_uberons = self._extract_svg_uberons()
        self.cellxgene_descendants = self._load_cellxgene_descendants()
        
    def _load_manual_mapping(self) -> Dict:
        """Load the manual tissue mapping (authoritative source)"""
        with open(self.data_dir / "tissue_uberon_mapping_manual.json", 'r') as f:
            return json.load(f)
    
    def _load_colleague_csv(self) -> Dict[str, str]:
        """Load colleague's CSV and extract CL terms"""
        cl_terms = {}
        with open(self.data_dir / "tissue_to_uberon.csv", 'r') as f:
            lines = f.readlines()
            
        for line in lines[1:]:  # Skip header
            if line.strip():
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    tissue_name = parts[0]
                    ontology_id = parts[1]
                    if ontology_id.startswith('CL:'):
                        cl_terms[tissue_name] = ontology_id
        
        return cl_terms
    
    def _extract_svg_uberons(self) -> Set[str]:
        """Extract all UBERON IDs from both SVG files"""
        svg_uberons = set()
        
        for svg_file in ['homo_sapiens.male.svg', 'homo_sapiens.female.svg']:
            svg_path = self.svg_dir / svg_file
            if svg_path.exists():
                with open(svg_path, 'r') as f:
                    content = f.read()
                    ids = re.findall(r'id="(UBERON_\d+)"', content)
                    svg_uberons.update(ids)
        
        return svg_uberons
    
    def _load_cellxgene_descendants(self) -> Dict:
        """Load CellxGene tissue descendants for hierarchy"""
        descendants_file = self.data_dir / "tissue_descendants.json"
        if descendants_file.exists():
            with open(descendants_file, 'r') as f:
                return json.load(f)
        return {}
    
    def create_hierarchy_mapping(self) -> Dict[str, str]:
        """Create hierarchy fallback mapping for tissues without direct SVG matches"""
        
        # Manual hierarchy mapping based on anatomical knowledge
        # These are educated guesses for missing UBERON IDs → available SVG IDs
        hierarchy_fallbacks = {
            # Adipose tissues → generic adipose
            'UBERON_0002190': 'UBERON_0001013',  # subcutaneous adipose → adipose tissue
            'UBERON_0010414': 'UBERON_0001013',  # omental fat pad → adipose tissue
            
            # Brain substructures → brain
            'UBERON_0009835': 'UBERON_0000956',  # anterior cingulate → cerebral cortex
            'UBERON_0001873': 'UBERON_0000955',  # caudate → brain
            'UBERON_0013540': 'UBERON_0001870',  # BA9 → frontal cortex
            'UBERON_0001898': 'UBERON_0000955',  # hypothalamus → brain
            'UBERON_0001882': 'UBERON_0000955',  # nucleus accumbens → brain
            'UBERON_0001874': 'UBERON_0000955',  # putamen → brain
            'UBERON_0002038': 'UBERON_0000955',  # substantia nigra → brain
            
            # Spinal cord → nervous system
            'UBERON_0002726': 'UBERON_0002240',  # cervical spinal cord → spinal cord (if exists)
            
            # Colon parts → colon/intestine
            'UBERON_0001159': 'UBERON_0001155',  # sigmoid colon → colon
            'UBERON_0001157': 'UBERON_0001155',  # transverse colon → colon
            
            # Esophagus parts → esophagus
            'UBERON_0002469': 'UBERON_0001043',  # esophageal mucosa → esophagus
            'UBERON_0004648': 'UBERON_0001043',  # esophageal muscularis → esophagus
            
            # Cervix parts → reproductive
            'UBERON_0000458': 'UBERON_0000002',  # endocervix → uterine cervix (if exists)
            
            # Kidney parts → kidney
            'UBERON_0000362': 'UBERON_0002113',  # kidney medulla → kidney
            
            # Skin regions → generic skin
            'UBERON_0036149': 'UBERON_0000014',  # skin suprapubic → zone of skin
            'UBERON_0004264': 'UBERON_0000014',  # skin lower leg → zone of skin
            
            # Nerves and arteries - may need broader categories
            'UBERON_0001323': 'UBERON_0001021',  # tibial nerve → nerve
            'UBERON_0007610': 'UBERON_0001637',  # tibial artery → artery
            
            # Salivary gland
            'UBERON_0001830': 'UBERON_0001044',  # minor salivary → salivary gland
        }
        
        # Verify fallbacks point to SVG-available IDs
        validated_fallbacks = {}
        for missing_id, fallback_id in hierarchy_fallbacks.items():
            if fallback_id in self.svg_uberons:
                validated_fallbacks[missing_id] = fallback_id
            else:
                print(f"Warning: Fallback {fallback_id} for {missing_id} not in SVG either")
        
        return validated_fallbacks
    
    def create_enhanced_mapping(self) -> Dict:
        """Create enhanced unified mapping with hierarchy resolution"""
        
        # Start with manual mapping as base
        enhanced_mapping = {
            'tissue_mappings': {},
            'cell_line_mappings': {},
            'hierarchy_fallbacks': self.create_hierarchy_mapping(),
            'metadata': {
                'description': 'Enhanced unified mapping with hierarchy resolution',
                'svg_coverage': f'{len(self.svg_uberons)} SVG elements available',
                'hierarchy_fallbacks': len(self.create_hierarchy_mapping()),
                'created': '2024-09-11',
                'source': 'Manual corrections + CL terms + hierarchy resolution'
            }
        }
        
        # Process tissue mappings
        for tissue_id, tissue_data in self.manual_mapping['tissue_mappings'].items():
            if tissue_data['tissue_type'] == 'tissue':
                # Check if UBERON ID exists in SVG
                uberon_id = tissue_data['uberon_id']
                svg_compatible_id = uberon_id
                is_direct_match = uberon_id in self.svg_uberons
                
                # If not in SVG, try hierarchy fallback
                if not is_direct_match and uberon_id in enhanced_mapping['hierarchy_fallbacks']:
                    svg_compatible_id = enhanced_mapping['hierarchy_fallbacks'][uberon_id]
                    is_direct_match = False
                
                enhanced_mapping['tissue_mappings'][tissue_id] = {
                    **tissue_data,
                    'svg_uberon_id': svg_compatible_id,
                    'is_direct_match': is_direct_match,
                    'can_visualize': is_direct_match or (uberon_id in enhanced_mapping['hierarchy_fallbacks'])
                }
            
            elif tissue_data['tissue_type'] == 'cell_line':
                # Add CL ontology terms for cell lines
                tissue_name = tissue_data['tissue_name']
                cl_term = self.colleague_csv.get(tissue_name)
                
                enhanced_mapping['cell_line_mappings'][tissue_id] = {
                    **tissue_data,
                    'cl_ontology_id': cl_term,
                    'exclude_from_anatomagram': True
                }
        
        return enhanced_mapping
    
    def generate_visualization_report(self, enhanced_mapping: Dict) -> None:
        """Generate comprehensive visualization capability report"""
        
        tissue_mappings = enhanced_mapping['tissue_mappings']
        
        # Count visualization capabilities
        direct_matches = sum(1 for t in tissue_mappings.values() if t['is_direct_match'])
        hierarchy_matches = sum(1 for t in tissue_mappings.values() 
                               if not t['is_direct_match'] and t['can_visualize'])
        cannot_visualize = sum(1 for t in tissue_mappings.values() if not t['can_visualize'])
        
        print("=== ENHANCED MAPPING VISUALIZATION REPORT ===\n")
        print(f"Total tissues: {len(tissue_mappings)}")
        print(f"Direct SVG matches: {direct_matches} ({direct_matches/len(tissue_mappings)*100:.1f}%)")
        print(f"Hierarchy fallbacks: {hierarchy_matches} ({hierarchy_matches/len(tissue_mappings)*100:.1f}%)")
        print(f"Cannot visualize: {cannot_visualize} ({cannot_visualize/len(tissue_mappings)*100:.1f}%)")
        print(f"Total visualizable: {direct_matches + hierarchy_matches} ({(direct_matches + hierarchy_matches)/len(tissue_mappings)*100:.1f}%)")
        
        if cannot_visualize > 0:
            print(f"\nTISSUES STILL CANNOT BE VISUALIZED:")
            for tissue_id, tissue_data in tissue_mappings.items():
                if not tissue_data['can_visualize']:
                    print(f"  {tissue_data['tissue_name']:35} → {tissue_data['uberon_id']}")
        
        print(f"\nHIERARCHY FALLBACK MAPPINGS:")
        for missing_id, fallback_id in enhanced_mapping['hierarchy_fallbacks'].items():
            print(f"  {missing_id} → {fallback_id}")
    
    def save_enhanced_mapping(self, output_file: str = None):
        """Save enhanced mapping to file"""
        if output_file is None:
            output_file = str(self.data_dir / "tissue_mapping_enhanced.json")
        
        enhanced_mapping = self.create_enhanced_mapping()
        
        with open(output_file, 'w') as f:
            json.dump(enhanced_mapping, f, indent=2)
        
        print(f"Enhanced mapping saved to: {output_file}")
        
        # Generate and display report
        self.generate_visualization_report(enhanced_mapping)
        
        return enhanced_mapping


if __name__ == "__main__":
    # Run enhancement
    anatomagram_dir = Path(__file__).parent.parent
    enhancer = AnatomagramMappingEnhancer(str(anatomagram_dir))
    
    enhanced_mapping = enhancer.save_enhanced_mapping()