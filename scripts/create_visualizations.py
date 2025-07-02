#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from nilearn import datasets, plotting, image, maskers
import json

def create_seed_visualization():
    """Create ACC seed mask overlay visualization"""
    
    print("Creating ACC seed visualization...")
    
    # Load reference brain
    ref_img = datasets.load_mni152_template(resolution=2)
    
    # Create ACC seed mask
    acc_coords = [(0, 24, 26)]
    seed_masker = maskers.NiftiSpheresMasker(
        acc_coords,
        radius=6,
        allow_overlap=True
    )
    
    # Generate seed mask
    seed_mask = seed_masker.fit().mask_img_
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Sagittal view
    plotting.plot_roi(
        seed_mask,
        bg_img=ref_img,
        display_mode='x',
        cut_coords=[0],
        axes=axes[0],
        title='ACC Seed (Sagittal)',
        colorbar=False
    )
    
    # Coronal view
    plotting.plot_roi(
        seed_mask,
        bg_img=ref_img,
        display_mode='y',
        cut_coords=[24],
        axes=axes[1],
        title='ACC Seed (Coronal)',
        colorbar=False
    )
    
    # Axial view
    plotting.plot_roi(
        seed_mask,
        bg_img=ref_img,
        display_mode='z',
        cut_coords=[26],
        axes=axes[2],
        title='ACC Seed (Axial)',
        colorbar=False
    )
    
    plt.tight_layout()
    plt.savefig('results/acc_seed_mask.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Seed visualization saved: results/acc_seed_mask.png")

def create_connectivity_plot():
    """Create connectivity results plot"""
    
    print("Creating connectivity results plot...")
    
    # Load results
    results_df = pd.read_csv('results/acc_connectivity_rigorous_results.csv')
    
    # Get top 15 regions by absolute effect size
    results_df['abs_cohens_d'] = results_df['cohens_d'].abs()
    top_regions = results_df.nlargest(15, 'abs_cohens_d')
    
    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color bars by significance
    colors = ['red' if sig else 'lightblue' for sig in top_regions['significant_fdr']]
    
    bars = ax.barh(range(len(top_regions)), top_regions['cohens_d'], color=colors)
    
    # Customize plot
    ax.set_yticks(range(len(top_regions)))
    ax.set_yticklabels([r.replace(', ', ',\n') for r in top_regions['region']], fontsize=10)
    ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12)
    ax.set_title("ACC Connectivity: Strongest Connections\n(Red = FDR significant, Blue = Non-significant)", 
                fontsize=14)
    ax.axvline(0, color='black', linestyle='-', alpha=0.3)
    ax.grid(axis='x', alpha=0.3)
    
    # Add significance markers
    for i, (_, row) in enumerate(top_regions.iterrows()):
        if row['significant_fdr']:
            ax.text(row['cohens_d'] + 0.01, i, f"p={row['p_fdr_corrected']:.3f}",
                   va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/acc_connectivity_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Connectivity plot saved: results/acc_connectivity_results.png")

def create_summary_stats():
    """Create summary statistics visualization"""
    
    print("Creating summary statistics...")
    
    # Load analysis info
    with open('results/rigorous_connectivity_analysis.json', 'r') as f:
        analysis_info = json.load(f)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Sample size info
    ax1.bar(['Subjects'], [analysis_info['n_subjects']], color='skyblue')
    ax1.set_ylabel('Count')
    ax1.set_title('Sample Size')
    ax1.set_ylim(0, 35)
    for i, v in enumerate([analysis_info['n_subjects']]):
        ax1.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
    
    # Statistical results
    stats = analysis_info['statistics']
    categories = ['Total\nTests', 'Uncorrected\nSignificant', 'FDR\nSignificant']
    values = [stats['n_tests'], stats['n_significant_uncorrected'], stats['n_significant_fdr']]
    colors = ['lightgray', 'orange', 'red']
    
    bars = ax2.bar(categories, values, color=colors)
    ax2.set_ylabel('Number of Regions')
    ax2.set_title('Statistical Results')
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', fontweight='bold')
    
    # Atlas coverage
    atlases = ['Cortical', 'Subcortical']
    regions = [49, 22]  # From our analysis
    ax3.pie(regions, labels=atlases, autopct='%1.0f%%', colors=['lightcoral', 'lightblue'])
    ax3.set_title('Atlas Coverage\n(71 total regions)')
    
    # Preprocessing summary
    preprocessing = ['Global Signal\nRegression', 'Temporal\nFiltering', 'Detrending', 'Standardization']
    ax4.bar(preprocessing, [1, 1, 1, 1], color='lightgreen')
    ax4.set_ylabel('Applied')
    ax4.set_title('Preprocessing Steps')
    ax4.set_ylim(0, 1.2)
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['No', 'Yes'])
    
    plt.tight_layout()
    plt.savefig('results/analysis_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Summary statistics saved: results/analysis_summary.png")

if __name__ == "__main__":
    create_seed_visualization()
    create_connectivity_plot()
    create_summary_stats()
    print("\n✓ All visualizations complete!")
    print("Generated files:")
    print("  - results/acc_seed_mask.png")
    print("  - results/acc_connectivity_results.png") 
    print("  - results/analysis_summary.png")
