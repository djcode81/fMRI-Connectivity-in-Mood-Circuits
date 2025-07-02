#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path
import json
from nilearn import datasets, image, maskers, signal
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import warnings

def load_dataset_info():
    data_dir = Path("data/raw/depression")
    
    task_file = data_dir / "task-rest_bold.json"
    if task_file.exists():
        with open(task_file, 'r') as f:
            task_info = json.load(f)
        tr = task_info.get('RepetitionTime', 2.5)
    else:
        tr = 2.5
        warnings.warn("Using default TR=2.5s")
    
    participants_file = data_dir / "participants.tsv"
    if participants_file.exists():
        participants_df = pd.read_csv(participants_file, sep='\t')
    else:
        participants_df = None
    
    return tr, participants_df

def extract_manual_confounds(func_file, tr):
    """Extract basic confounds from functional data when fMRIPrep confounds unavailable"""
    func_img = image.load_img(func_file)
    
    # Extract global signal using NiftiMasker
    global_masker = maskers.NiftiMasker(
        detrend=False,
        standardize=False,
        t_r=tr
    )
    
    # Get time series and compute global signal (mean across all voxels)
    global_time_series = global_masker.fit_transform(func_img)
    global_signal = np.mean(global_time_series, axis=1)
    
    # Create basic confounds DataFrame
    n_volumes = func_img.shape[-1]
    confounds_df = pd.DataFrame({
        'global_signal': global_signal,
        'linear_trend': np.arange(n_volumes),
        'quadratic_trend': np.arange(n_volumes) ** 2
    })
    
    return confounds_df

def find_valid_subjects(data_dir, min_subjects=10):
    subject_dirs = list(data_dir.glob("sub-*"))
    valid_subjects = []
    
    for subj_dir in subject_dirs:
        subj_id = subj_dir.name
        func_file = subj_dir / "ses-pre" / "func" / f"{subj_id}_ses-pre_task-rest_bold.nii.gz"
        
        if func_file.exists():
            try:
                func_img = image.load_img(func_file)
                if func_img.shape[-1] >= 50:  # At least 50 timepoints
                    valid_subjects.append(subj_id)
            except Exception as e:
                print(f"Warning: Could not load {subj_id}: {e}")
    
    valid_subjects = sorted(valid_subjects)
    
    if len(valid_subjects) < min_subjects:
        print(f"Warning: Only {len(valid_subjects)} subjects found, minimum {min_subjects} recommended")
    
    return valid_subjects

def compute_subject_connectivity(subject_id, data_dir, tr, acc_coords, atlas_cortical, atlas_subcortical):
    func_file = data_dir / subject_id / "ses-pre" / "func" / f"{subject_id}_ses-pre_task-rest_bold.nii.gz"
    
    if not func_file.exists():
        return None
    
    try:
        func_img = image.load_img(func_file)
        
        # Extract manual confounds (global signal, trends)
        confounds_df = extract_manual_confounds(func_file, tr)
        print(f"  Extracted {len(confounds_df.columns)} confound regressors")
        
        # Clean functional data with confounds
        func_clean = image.clean_img(
            func_img,
            detrend=True,
            standardize="zscore_sample",
            low_pass=0.1,
            high_pass=0.01,
            t_r=tr,
            confounds=confounds_df.values,
            ensure_finite=True
        )
        
        # Extract ACC time series
        acc_masker = maskers.NiftiSpheresMasker(
            acc_coords,
            radius=6,
            detrend=False,  # Already done in clean_img
            standardize=False,  # Already done in clean_img
            t_r=tr,
            ensure_finite=True
        )
        
        acc_time_series = acc_masker.fit_transform(func_clean)
        
        # Extract cortical regions
        cortical_masker = maskers.NiftiLabelsMasker(
            atlas_cortical.maps,
            detrend=False,
            standardize=False,
            t_r=tr,
            ensure_finite=True
        )
        
        cortical_time_series = cortical_masker.fit_transform(func_clean)
        
        # Extract subcortical regions
        subcortical_masker = maskers.NiftiLabelsMasker(
            atlas_subcortical.maps,
            detrend=False,
            standardize=False,
            t_r=tr,
            ensure_finite=True
        )
        
        subcortical_time_series = subcortical_masker.fit_transform(func_clean)
        
        # Combine cortical and subcortical
        all_time_series = np.hstack([cortical_time_series, subcortical_time_series])
        all_labels = list(atlas_cortical.labels) + list(atlas_subcortical.labels)
        
        # Compute correlations
        correlations = np.corrcoef(acc_time_series.T, all_time_series.T)[0, 1:]
        
        # Fisher z-transform with clipping to avoid infinite values
        correlations_clipped = np.clip(correlations, -0.999, 0.999)
        z_scores = np.arctanh(correlations_clipped)
        
        return {
            'subject': subject_id,
            'correlations': correlations,
            'z_scores': z_scores,
            'region_labels': all_labels,
            'n_timepoints': func_clean.shape[-1],
            'n_cortical': len(atlas_cortical.labels),
            'n_subcortical': len(atlas_subcortical.labels)
        }
        
    except Exception as e:
        print(f"  Error processing {subject_id}: {e}")
        return None

def analyze_depression_acc_connectivity():
    data_dir = Path("data/raw/depression")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    tr, participants_df = load_dataset_info()
    print(f"Using TR: {tr}s")
    
    acc_coords = [(0, 24, 26)]
    print(f"ACC seed coordinates: {acc_coords[0]}")
    
    # Load both cortical and subcortical atlases
    atlas_cortical = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_subcortical = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    
    print(f"Cortical regions: {len(atlas_cortical.labels)}")
    print(f"Subcortical regions: {len(atlas_subcortical.labels)}")
    print(f"Total regions: {len(atlas_cortical.labels) + len(atlas_subcortical.labels)}")
    
    valid_subjects = find_valid_subjects(data_dir)
    print(f"\nValid subjects found: {len(valid_subjects)}")
    
    connectivity_results = []
    failed_subjects = []
    
    for subject_id in valid_subjects:
        print(f"Processing {subject_id}...")
        result = compute_subject_connectivity(
            subject_id, data_dir, tr, acc_coords, atlas_cortical, atlas_subcortical
        )
        
        if result is not None:
            connectivity_results.append(result)
            print(f"  ✓ {subject_id}: {result['n_timepoints']} timepoints, {len(result['z_scores'])} regions")
        else:
            failed_subjects.append(subject_id)
            print(f"  ✗ {subject_id}: failed")
    
    print(f"\nSuccessful: {len(connectivity_results)} subjects")
    print(f"Failed: {len(failed_subjects)} subjects")
    
    if len(connectivity_results) < 5:
        raise ValueError("Insufficient subjects for reliable analysis")
    
    return connectivity_results, atlas_cortical, atlas_subcortical, tr, participants_df

def statistical_analysis_with_fdr(connectivity_results):
    """Statistical analysis with proper multiple comparisons correction"""
    
    print("\nPerforming statistical analysis...")
    
    # Find minimum number of regions across subjects
    min_regions = min(len(subj['z_scores']) for subj in connectivity_results)
    print(f"Using {min_regions} regions common to all subjects")
    
    # Collect z-scores for subjects with consistent regions
    valid_z_scores = []
    valid_subjects = []
    
    for subj in connectivity_results:
        if len(subj['z_scores']) >= min_regions:
            valid_z_scores.append(subj['z_scores'][:min_regions])
            valid_subjects.append(subj['subject'])
    
    all_z_scores = np.array(valid_z_scores)
    n_subjects = len(all_z_scores)
    
    print(f"Final analysis: {n_subjects} subjects × {min_regions} regions")
    
    # Compute group statistics
    mean_z_scores = np.mean(all_z_scores, axis=0)
    sem_z_scores = stats.sem(all_z_scores, axis=0)
    std_z_scores = np.std(all_z_scores, axis=0, ddof=1)
    
    # One-sample t-tests against zero
    t_stats, p_values = stats.ttest_1samp(all_z_scores, 0, axis=0)
    
    # Multiple comparisons correction using FDR
    rejected, p_corrected = fdrcorrection(p_values, alpha=0.05, method='indep')
    
    # Effect sizes (Cohen's d)
    cohens_d = mean_z_scores / std_z_scores
    
    print(f"Uncorrected significant: {np.sum(p_values < 0.05)}")
    print(f"FDR-corrected significant: {np.sum(rejected)}")
    
    # Get region labels from first subject, excluding background
    region_labels = connectivity_results[0]['region_labels'][:min_regions]
    
    # Remove background region (typically first region, label 0)
    if 'Background' in region_labels:
        bg_idx = region_labels.index('Background')
        # Remove background from all data
        all_z_scores = np.delete(all_z_scores, bg_idx, axis=1)
        region_labels = [label for i, label in enumerate(region_labels) if i != bg_idx]
        min_regions -= 1
        print(f"Removed background region, using {min_regions} regions")
    
    results = []
    for i, (region, mean_z, sem_z, std_z, t_stat, p_val, p_corr, significant, cohens_d_val) in enumerate(
        zip(region_labels, mean_z_scores, sem_z_scores, std_z_scores, 
            t_stats, p_values, p_corrected, rejected, cohens_d)
    ):
        results.append({
            'region': region,
            'mean_z_score': float(mean_z),
            'sem_z_score': float(sem_z),
            'std_z_score': float(std_z),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'p_fdr_corrected': float(p_corr),
            'significant_fdr': bool(significant),
            'cohens_d': float(cohens_d_val),
            'n_subjects': n_subjects
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('p_fdr_corrected')
    
    return results_df

def save_rigorous_results(connectivity_results, statistical_results, tr):
    results_dir = Path("results")
    
    analysis_summary = {
        'dataset': 'ds003007 - Depression resting-state connectivity',
        'n_subjects': len(connectivity_results),
        'tr': tr,
        'seed_region': 'Dorsal Anterior Cingulate Cortex',
        'seed_coords': [0, 24, 26],
        'seed_radius_mm': 6,
        'atlases': ['Harvard-Oxford cortical', 'Harvard-Oxford subcortical'],
        'preprocessing': {
            'confound_regressors': ['global_signal', 'linear_trend', 'quadratic_trend'],
            'standardization': 'zscore_sample',
            'temporal_filter': {'low_pass': 0.1, 'high_pass': 0.01},
            'detrending': True
        },
        'statistics': {
            'test': 'one-sample t-test vs zero',
            'multiple_comparisons': 'FDR correction (Benjamini-Hochberg)',
            'alpha': 0.05,
            'n_tests': len(statistical_results),
            'n_significant_uncorrected': int((statistical_results['p_value'] < 0.05).sum()),
            'n_significant_fdr': int(statistical_results['significant_fdr'].sum())
        }
    }
    
    with open(results_dir / "rigorous_connectivity_analysis.json", 'w') as f:
        json.dump(analysis_summary, f, indent=2)
    
    statistical_results.to_csv(results_dir / "acc_connectivity_rigorous_results.csv", index=False)
    
    # Save only significant results
    significant_results = statistical_results[statistical_results['significant_fdr']]
    if len(significant_results) > 0:
        significant_results.to_csv(results_dir / "acc_connectivity_significant_fdr.csv", index=False)
        print(f"FDR-significant results saved: {len(significant_results)} regions")
    else:
        print("No regions survived FDR correction")
    
    return results_dir

if __name__ == "__main__":
    connectivity_results, atlas_cortical, atlas_subcortical, tr, participants_df = analyze_depression_acc_connectivity()
    
    if len(connectivity_results) >= 5:
        statistical_results = statistical_analysis_with_fdr(connectivity_results)
        results_dir = save_rigorous_results(connectivity_results, statistical_results, tr)
        
        print(f"\nRigorous analysis complete!")
        print(f"Results saved to: {results_dir}")
        print(f"Sample size: {len(connectivity_results)} subjects")
        print(f"FDR-corrected significant connections: {statistical_results['significant_fdr'].sum()}")
        
        # Show top results
        top_results = statistical_results.head(10)
        print(f"\nTop 10 strongest connections:")
        for _, row in top_results.iterrows():
            sig_marker = "***" if row['significant_fdr'] else ""
            print(f"  {row['region']}: Z={row['mean_z_score']:.3f}, p_FDR={row['p_fdr_corrected']:.4f} {sig_marker}")
    else:
        print("Insufficient data for rigorous analysis")
