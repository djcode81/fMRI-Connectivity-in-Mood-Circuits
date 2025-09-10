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
    func_img = image.load_img(func_file)
    
    global_masker = maskers.NiftiMasker(
        detrend=False,
        standardize=False,
        t_r=tr
    )
    
    global_time_series = global_masker.fit_transform(func_img)
    global_signal = np.mean(global_time_series, axis=1)
    
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
                if func_img.shape[-1] >= 50:
                    valid_subjects.append(subj_id)
            except Exception as e:
                pass
    
    valid_subjects = sorted(valid_subjects)
    
    if len(valid_subjects) < min_subjects:
        warnings.warn(f"Only {len(valid_subjects)} subjects found, minimum {min_subjects} recommended")
    
    return valid_subjects

def create_common_roi_mask(connectivity_results, atlas_cortical, atlas_subcortical):
    total_atlas_regions = len(atlas_cortical.labels) + len(atlas_subcortical.labels)
    all_labels = list(atlas_cortical.labels) + list(atlas_subcortical.labels)
    
    roi_counts = np.zeros(total_atlas_regions)
    
    for result in connectivity_results:
        n_regions = len(result['z_scores'])
        roi_counts[:n_regions] += 1
    
    n_subjects = len(connectivity_results)
    
    common_mask = roi_counts == n_subjects
    
    if 'Background' in all_labels:
        bg_idx = all_labels.index('Background')
        common_mask[bg_idx] = False
    
    for i, label in enumerate(all_labels):
        if 'White Matter' in label or 'white matter' in label:
            common_mask[i] = False
    
    common_indices = np.where(common_mask)[0]
    common_labels = [all_labels[i] for i in common_indices]
    
    return common_indices, common_labels

def compute_subject_connectivity(subject_id, data_dir, tr, acc_coords, atlas_cortical, atlas_subcortical):
    func_file = data_dir / subject_id / "ses-pre" / "func" / f"{subject_id}_ses-pre_task-rest_bold.nii.gz"
    
    if not func_file.exists():
        return None
    
    try:
        func_img = image.load_img(func_file)
        
        confounds_df = extract_manual_confounds(func_file, tr)
        
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
        
        acc_masker = maskers.NiftiSpheresMasker(
            acc_coords,
            radius=6,
            detrend=False,
            standardize=False,
            t_r=tr,
            ensure_finite=True
        )
        
        acc_time_series = acc_masker.fit_transform(func_clean)
        
        cortical_masker = maskers.NiftiLabelsMasker(
            atlas_cortical.maps,
            detrend=False,
            standardize=False,
            t_r=tr,
            ensure_finite=True
        )
        
        cortical_time_series = cortical_masker.fit_transform(func_clean)
        
        subcortical_masker = maskers.NiftiLabelsMasker(
            atlas_subcortical.maps,
            detrend=False,
            standardize=False,
            t_r=tr,
            ensure_finite=True
        )
        
        subcortical_time_series = subcortical_masker.fit_transform(func_clean)
        
        all_time_series = np.hstack([cortical_time_series, subcortical_time_series])
        
        correlations = np.corrcoef(acc_time_series.T, all_time_series.T)[0, 1:]
        
        correlations_clipped = np.clip(correlations, -0.999, 0.999)
        z_scores = np.arctanh(correlations_clipped)
        
        return {
            'subject': subject_id,
            'correlations': correlations,
            'z_scores': z_scores,
            'n_timepoints': func_clean.shape[-1],
            'n_cortical': cortical_time_series.shape[1],
            'n_subcortical': subcortical_time_series.shape[1]
        }
        
    except Exception as e:
        return None

def analyze_depression_acc_connectivity():
    data_dir = Path("data/raw/depression")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    tr, participants_df = load_dataset_info()
    
    acc_coords = [(0, 24, 26)]
    
    atlas_cortical = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_subcortical = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    
    valid_subjects = find_valid_subjects(data_dir)
    
    connectivity_results = []
    failed_subjects = []
    
    for subject_id in valid_subjects:
        result = compute_subject_connectivity(
            subject_id, data_dir, tr, acc_coords, atlas_cortical, atlas_subcortical
        )
        
        if result is not None:
            connectivity_results.append(result)
        else:
            failed_subjects.append(subject_id)
    
    if len(connectivity_results) < 5:
        raise ValueError("Insufficient subjects for reliable analysis")
    
    return connectivity_results, atlas_cortical, atlas_subcortical, tr, participants_df

def statistical_analysis_with_proper_masking(connectivity_results, atlas_cortical, atlas_subcortical):
    common_indices, common_labels = create_common_roi_mask(
        connectivity_results, atlas_cortical, atlas_subcortical
    )
    
    valid_z_scores = []
    valid_subjects = []
    
    for subj in connectivity_results:
        subj_z_scores = np.array(subj['z_scores'])[common_indices]
        valid_z_scores.append(subj_z_scores)
        valid_subjects.append(subj['subject'])
    
    all_z_scores = np.array(valid_z_scores)
    n_subjects, n_regions = all_z_scores.shape
    
    mean_z_scores = np.mean(all_z_scores, axis=0)
    sem_z_scores = stats.sem(all_z_scores, axis=0)
    std_z_scores = np.std(all_z_scores, axis=0, ddof=1)
    
    t_stats, p_values = stats.ttest_1samp(all_z_scores, 0, axis=0)
    
    rejected, p_corrected = fdrcorrection(p_values, alpha=0.05, method='indep')
    
    cohens_d = mean_z_scores / std_z_scores
    
    results = []
    for i, (region, mean_z, sem_z, std_z, t_stat, p_val, p_corr, significant, cohens_d_val) in enumerate(
        zip(common_labels, mean_z_scores, sem_z_scores, std_z_scores, 
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
    
    subjects_used = pd.DataFrame({'subject_id': valid_subjects})
    subjects_used.to_csv('results/participants_used.tsv', sep='\t', index=False)
    
    return results_df, n_regions

def save_rigorous_results(connectivity_results, statistical_results, n_regions_analyzed, tr):
    results_dir = Path("results")
    
    n_subjects = len(connectivity_results)
    example_result = connectivity_results[0]
    n_cortical_actual = example_result['n_cortical'] 
    n_subcortical_actual = example_result['n_subcortical']
    
    analysis_summary = {
        'dataset': 'ds003007 - Depression resting-state connectivity',
        'n_subjects': n_subjects,
        'tr': tr,
        'seed_region': 'Dorsal Anterior Cingulate Cortex',
        'seed_coords': [0, 24, 26],
        'seed_radius_mm': 6,
        'atlases': ['Harvard-Oxford cortical', 'Harvard-Oxford subcortical'],
        'atlas_coverage': {
            'cortical_regions': n_cortical_actual,
            'subcortical_regions': n_subcortical_actual,
            'total_available': n_cortical_actual + n_subcortical_actual,
            'analyzed_after_masking': n_regions_analyzed
        },
        'preprocessing': {
            'confound_regressors': ['global_signal', 'linear_trend', 'quadratic_trend'],
            'standardization': 'zscore_sample',
            'temporal_filter': {'low_pass': 0.1, 'high_pass': 0.01},
            'detrending': True,
            'note': 'Motion parameters, WM, CSF not available (raw data, not fMRIPrep derivatives)'
        },
        'statistics': {
            'test': 'one-sample t-test vs zero',
            'multiple_comparisons': 'FDR correction (Benjamini-Hochberg)',
            'alpha': 0.05,
            'n_tests': n_regions_analyzed,
            'n_significant_uncorrected': int((statistical_results['p_value'] < 0.05).sum()),
            'n_significant_fdr': int(statistical_results['significant_fdr'].sum()),
            'masking': 'Only ROIs present in all subjects included'
        }
    }
    
    with open(results_dir / "rigorous_connectivity_analysis.json", 'w') as f:
        json.dump(analysis_summary, f, indent=2)
    
    statistical_results.to_csv(results_dir / "acc_connectivity_rigorous_results.csv", index=False)
    
    significant_results = statistical_results[statistical_results['significant_fdr']]
    if len(significant_results) > 0:
        significant_results.to_csv(results_dir / "acc_connectivity_significant_fdr.csv", index=False)
    
    return results_dir

if __name__ == "__main__":
    connectivity_results, atlas_cortical, atlas_subcortical, tr, participants_df = analyze_depression_acc_connectivity()
    
    if len(connectivity_results) >= 5:
        statistical_results, n_regions_analyzed = statistical_analysis_with_proper_masking(
            connectivity_results, atlas_cortical, atlas_subcortical
        )
        results_dir = save_rigorous_results(connectivity_results, statistical_results, n_regions_analyzed, tr)
    else:
        raise ValueError("Insufficient data for rigorous analysis")
