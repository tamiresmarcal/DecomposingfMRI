#!/usr/bin/env bash
# Reference maps for the brain-state x maps pilot.
#
#   pilots/state_maps/fetch_reference_maps.sh [DATA_DIR]     (default: pilots/state_maps/data)
#
# Every file comes from a host that serves plain HTTPS -- TemplateFlow's S3
# bucket and raw.githubusercontent.com -- so this runs where OSF, NITRC and
# atlas.brainnetome.org are unreachable (a cluster login node, a sandbox).
# ~280 MB, almost all of it the PET volumes.
#
#   molecular   19 PET receptor/transporter maps, one tracer per target: the
#               set of Hansen et al. 2022 (Nat Neurosci), which is what
#               neuromaps serves as its MNI152 annotations. Pulled from the
#               paper's repo because neuromaps' own fetcher needs OSF.
#   canonical   Yeo 2011 7-network, fsaverage5, from TemplateFlow
#   behavioural Brainnetome cortical parcellation (210), fsaverage, from
#               TemplateFlow. The BrainMap behavioural-domain profiles per
#               subregion are NOT here -- see README.md.
#   state atlas Harvard-Oxford cortical, lateralised, thr25, 2 mm (the cortical
#               half of the pipeline's `harvardoxford` atlas), from TemplateFlow
#   surfaces    fsaverage5 pial / white / sphere, from TemplateFlow
set -euo pipefail

DATA="${1:-$(dirname "$0")/data}"
TF=https://templateflow.s3.amazonaws.com
HR=https://raw.githubusercontent.com/netneurolab/hansen_receptors/main/data/PET_nifti_images

PET=(
  5HT1a_way_hc36_savli.nii
  5HT1b_p943_hc65_gallezot.nii.gz
  5HT2a_cimbi_hc29_beliveau.nii
  5HT4_sb20_hc59_beliveau.nii
  5HT6_gsk_hc30_radhakrishnan.nii.gz
  5HTT_dasb_hc100_beliveau.nii
  A4B2_flubatine_hc30_hillmer.nii.gz
  CB1_omar_hc77_normandin.nii.gz
  D1_SCH23390_hc13_kaller.nii
  D2_flb457_hc55_sandiego.nii.gz
  DAT_fpcit_hc174_dukart_spect.nii
  GABAa-bz_flumazenil_hc16_norgaard.nii
  H3_cban_hc8_gallezot.nii.gz
  M1_lsn_hc24_naganawa.nii.gz
  mGluR5_abp_hc73_smart.nii
  MU_carfentanil_hc204_kantonen.nii
  NAT_MRB_hc77_ding.nii.gz
  NMDA_ge179_hc29_galovic.nii.gz
  VAChT_feobv_hc18_aghourian_sum.nii
)

TEMPLATEFLOW=(
  tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_atlas-HOCPAL_desc-th25_dseg.nii.gz
  tpl-fsaverage/tpl-fsaverage_atlas-brainnetome_dseg.tsv
  tpl-fsaverage/tpl-fsaverage_hemi-L_den-164k_atlas-brainnetome_dseg.label.gii
  tpl-fsaverage/tpl-fsaverage_hemi-R_den-164k_atlas-brainnetome_dseg.label.gii
  tpl-fsaverage/tpl-fsaverage_hemi-L_den-10k_atlas-Yeo2011_seg-7n_dseg.label.gii
  tpl-fsaverage/tpl-fsaverage_hemi-R_den-10k_atlas-Yeo2011_seg-7n_dseg.label.gii
)
for h in L R; do
  for s in pial white sphere; do
    TEMPLATEFLOW+=("tpl-fsaverage/tpl-fsaverage_hemi-${h}_den-10k_${s}.surf.gii")
  done
done

get() {  # url dest -- skip what is already there, never leave a partial file
  [[ -s "$2" ]] && return 0
  curl -sSf --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
}

mkdir -p "$DATA/pet" "$DATA/templateflow"
for f in "${PET[@]}"; do get "$HR/$f" "$DATA/pet/$f" & done
for f in "${TEMPLATEFLOW[@]}"; do get "$TF/$f" "$DATA/templateflow/$(basename "$f")" & done
wait
echo "reference maps in $DATA: $(find "$DATA" -type f | wc -l) files, $(du -sh "$DATA" | cut -f1)"
