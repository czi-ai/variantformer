#!/usr/bin/env python3
"""
SVG Anatomagram Analyzer
Extracts UBERON IDs and metadata from anatomagram SVG files
"""

import re
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from typing import Dict, Set, List, Tuple


class SVGAnatomagramAnalyzer:
    """Analyze anatomagram SVG files for UBERON IDs and structure"""
    
    def __init__(self, svg_directory: str):
        self.svg_dir = Path(svg_directory)
        self.male_svg = self.svg_dir / "homo_sapiens.male.svg"
        self.female_svg = self.svg_dir / "homo_sapiens.female.svg"
    
    def extract_uberon_ids(self, svg_file: Path) -> Dict[str, Dict]:
        """Extract all UBERON IDs and their metadata from an SVG file"""
        uberon_data = {}
        
        # Read SVG content
        with open(svg_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse XML
        try:
            root = ET.fromstring(content)
            namespaces = {
                'svg': 'http://www.w3.org/2000/svg',
                'inkscape': 'http://www.inkscape.org/namespaces/inkscape'
            }
            
            # Find all elements with UBERON IDs
            for elem in root.iter():
                elem_id = elem.get('id', '')
                if elem_id.startswith('UBERON_'):
                    uberon_id = elem_id
                    
                    # Extract metadata
                    metadata = {
                        'uberon_id': uberon_id,
                        'tag': elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag,
                        'inkscape_label': elem.get(f'{{{namespaces["inkscape"]}}}label', ''),
                        'has_title': False,
                        'title_text': '',
                        'children_count': len(list(elem)),
                        'attributes': dict(elem.attrib)
                    }
                    
                    # Look for title element
                    for child in elem:
                        if child.tag.endswith('title'):
                            metadata['has_title'] = True
                            metadata['title_text'] = child.text or ''
                            break
                    
                    uberon_data[uberon_id] = metadata
            
        except ET.ParseError as e:
            print(f"Error parsing {svg_file}: {e}")
            
        return uberon_data
    
    def analyze_all_svgs(self) -> Dict[str, Dict]:
        """Analyze both male and female SVG files"""
        results = {}
        
        if self.male_svg.exists():
            print(f"Analyzing {self.male_svg}...")
            results['male'] = self.extract_uberon_ids(self.male_svg)
            print(f"Found {len(results['male'])} UBERON IDs in male SVG")
        
        if self.female_svg.exists():
            print(f"Analyzing {self.female_svg}...")
            results['female'] = self.extract_uberon_ids(self.female_svg)
            print(f"Found {len(results['female'])} UBERON IDs in female SVG")
        
        return results
    
    def get_combined_uberon_set(self, results: Dict) -> Set[str]:
        """Get unique set of all UBERON IDs across both SVG files"""
        all_ids = set()
        for svg_type, data in results.items():
            all_ids.update(data.keys())
        return all_ids
    
    def compare_with_mappings(self, tissue_mapping_file: str, results: Dict) -> Dict:
        """Compare SVG UBERON IDs with tissue mapping file"""
        # Load tissue mappings
        with open(tissue_mapping_file, 'r') as f:
            mapping_data = json.load(f)
        
        # Extract UBERON IDs from mappings (convert format)
        mapping_uberons = set()
        tissue_to_uberon = {}
        
        for tissue_id, tissue_data in mapping_data['tissue_mappings'].items():
            if tissue_data['tissue_type'] == 'tissue' and tissue_data['uberon_id']:
                # Convert UBERON_XXXXXXX to UBERON_XXXXXXX format (already correct)
                uberon_id = tissue_data['uberon_id']
                mapping_uberons.add(uberon_id)
                tissue_to_uberon[tissue_data['tissue_name']] = uberon_id
        
        # Get SVG UBERON IDs
        svg_uberons = self.get_combined_uberon_set(results)
        
        # Compare
        in_both = mapping_uberons.intersection(svg_uberons)
        mapping_only = mapping_uberons.difference(svg_uberons)
        svg_only = svg_uberons.difference(mapping_uberons)
        
        comparison = {
            'total_mapping_uberons': len(mapping_uberons),
            'total_svg_uberons': len(svg_uberons),
            'in_both': sorted(list(in_both)),
            'mapping_only': sorted(list(mapping_only)),
            'svg_only': sorted(list(svg_only)),
            'coverage_percentage': len(in_both) / len(mapping_uberons) * 100 if mapping_uberons else 0
        }
        
        return comparison
    
    def generate_report(self, output_file: str = None):
        """Generate comprehensive analysis report"""
        print("=== SVG Anatomagram Analysis Report ===\n")
        
        # Analyze SVG files
        results = self.analyze_all_svgs()
        
        # Get combined UBERON set
        all_uberons = self.get_combined_uberon_set(results)
        print(f"Total unique UBERON IDs across both SVGs: {len(all_uberons)}\n")
        
        # Show breakdown by SVG
        for svg_type, data in results.items():
            print(f"{svg_type.upper()} SVG:")
            print(f"  UBERON IDs: {len(data)}")
            print(f"  With titles: {sum(1 for d in data.values() if d['has_title'])}")
            print()
        
        # Compare with tissue mapping
        mapping_file = self.svg_dir.parent / "data" / "tissue_uberon_mapping_manual.json"
        if mapping_file.exists():
            print("=== Comparison with Tissue Mapping ===")
            comparison = self.compare_with_mappings(str(mapping_file), results)
            
            print(f"Tissue mapping UBERON IDs: {comparison['total_mapping_uberons']}")
            print(f"SVG UBERON IDs: {comparison['total_svg_uberons']}")
            print(f"Coverage: {comparison['coverage_percentage']:.1f}%")
            print(f"IDs in both: {len(comparison['in_both'])}")
            print(f"Mapping only: {len(comparison['mapping_only'])}")
            print(f"SVG only: {len(comparison['svg_only'])}")
            print()
            
            if comparison['mapping_only']:
                print("UBERON IDs in mapping but NOT in SVG (won't visualize):")
                for uid in comparison['mapping_only']:
                    print(f"  {uid}")
                print()
            
            if comparison['svg_only'][:10]:  # Show first 10
                print("UBERON IDs in SVG but not in mapping (available for use):")
                for uid in comparison['svg_only'][:10]:
                    print(f"  {uid}")
                if len(comparison['svg_only']) > 10:
                    print(f"  ... and {len(comparison['svg_only']) - 10} more")
                print()
        
        # Save detailed results if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump({
                    'svg_analysis': results,
                    'combined_uberons': sorted(list(all_uberons)),
                    'comparison': comparison if mapping_file.exists() else None
                }, f, indent=2)
            print(f"Detailed results saved to: {output_file}")


if __name__ == "__main__":
    # Run analysis
    svg_dir = Path(__file__).parent.parent / "assets" / "svg"
    analyzer = SVGAnatomagramAnalyzer(str(svg_dir))
    
    output_file = Path(__file__).parent.parent / "data" / "svg_analysis_results.json"
    analyzer.generate_report(str(output_file))