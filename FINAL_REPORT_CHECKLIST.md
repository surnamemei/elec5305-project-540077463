# Final Report Checklist

This checklist maps every number, figure and reference in `FINAL_REPORT.md` / `FINAL_REPORT.pdf` to its source, and lists files that must **not** be cited.

Source of the report: `report/FINAL_REPORT.pandoc.md` (built with `bash report/build_report.sh`).

---

## 1. Rebuild order (all numbers in the report come from these outputs)

```bash
python src/run_all_experiments.py        # ASR, per subset (DATASET_NAME inside the script); not re-run for the report
python src/bootstrap_analysis.py         # ΔWER bootstrap CIs
python src/error_analysis.py             # S/D/I per condition (see §5, known discrepancy)
python src/signal_distortion_analysis.py # LSD, retained bandwidth, D(f); 2 × 500 utterances
python src/representation_analysis.py    # standardised + raw drift, conv + 12 layers; writes standardisation stats
python src/predictor_analysis.py         # across-/pooled/within-condition correlations
python src/integrated_analysis.py        # merged condition-level table
python src/encodec_extension.py          # EnCodec extension (needs representation_standardisation.pt)
python src/anisotropy_check.py           # unrelated-utterance cosine per layer
python src/opus_mode_check.py            # Opus TOC coding mode / bandwidth
python src/final_report_tables.py        # authoritative tables + 83 validation checks
python src/final_case_studies.py         # case selection, quartile table, Figure 4
python src/final_report_figures.py       # Figures 1–3
bash report/build_report.sh              # FINAL_REPORT.md + FINAL_REPORT.pdf (PANDOC=/path/to/pandoc)
```

Validation status: `results/final_report/validation_report.txt` — **105 PASS, 0 FAIL, 4 NOTE**. Every number quoted in the report was also re-derived from the tables below and string-matched against `report/FINAL_REPORT.pandoc.md` and `FINAL_REPORT.md` (115 automated checks, 0 failures).

The checks verify:

- corpus WER for all 24 ASR conditions, recomputed with JiWER;
- identical 500-utterance sets for ASR, signal and representation analyses (both subsets), and for EnCodec (test-clean);
- total errors S + D + I;
- all 22 utterance-level bootstrap CIs in `bootstrap_results.csv`, reproduced independently to within 0.15 pp;
- speaker-level block-bootstrap CIs (the primary CIs in the report): 22 checks that they give the same zero-exclusion conclusion as the utterance-level CIs;
- the EnCodec WAV baseline equals the main WAV baseline.

---

## 2. Reported numbers and their source files

All paths are relative to `results/`.

| Report location | Number(s) | Source |
|---|---|---|
| Abstract, §4.1, Table 1 | WAV WER 3.17% (test-clean), 8.26% (test-other) | `final_report/table_main_results.csv` (from `test-*_experiment_details.csv`, recomputed with JiWER) |
| Table 1 | measured bitrate, WER, ΔWER, 95% CI for 11 conditions × 2 subsets | `final_report/table_main_results.csv` (`ci_lower_pp`/`ci_upper_pp` = speaker-level block bootstrap; utterance-level in `ci_utterance_*`) |
| Table 1 | LSD, retained bandwidth (test-clean) | `signal_distortion_summary.csv` → `final_report/table_main_results.csv` |
| Table 1 caption | test-other differs by ≤ 1.4 dB LSD, ≤ 0.06 kHz bandwidth, ≤ 0.09 drift | `final_report/table_main_results.csv` (max abs difference: MP3 16k LSD 1.38 dB; Opus 8k bandwidth 0.054 kHz; Opus 6k drift 0.084) |
| Table 1 | layer-12 standardised drift | `representation_similarity_summary.csv` (`sdrift_mean`) |
| §3.1 | 2,620 / 2,939 utterances; 40 / 33 speakers; 10,168 / 8,775 reference words | torchaudio LibriSpeech; `final_report/table_main_results.csv` (`num_speakers`, `reference_words`) |
| §4.1 | published no-LM WER for the same model: 3.4 / 8.5 (full test sets) | Baevski et al. 2020, Table 10 (verified by reference audit) |
| §3.2 | Opus mode: CELT-WB at 64k; SILK-WB at 32k (99.97%), 16k, 12k; SILK-NB at 8k, 6k (100%) | `final_report/table_opus_modes.csv` (50 utterances) |
| §3.2 | FFmpeg 6.1.1, LAME 3.100, libopus 1.4; MP3 CBR MPEG-2 L3 16 kHz mono; Opus VBR, application=audio, 20 ms | `ffmpeg -version`, `dpkg -l`, `ffmpeg -h encoder=...`, `ffprobe` (recorded 2026-09-26) |
| §3.4 | alignment lag MP3 0 samples, Opus ≤ 2 samples | `signal_distortion_summary.csv` (`estimated_delay_mean_samples`) |
| §3.5 | unrelated-utterance cosine: 0.04 conv, 0.30 layer 10, 0.96 layer 11 (raw); 0.02–0.08 after standardisation | `final_report/table_anisotropy.csv` (50 pairs) |
| Abstract, §4.1 | MP3 24k test-clean ΔWER +0.02 [−0.26, +0.31] | `final_report/table_main_results.csv` |
| §4.1 | test-clean ≥ MP3 24k / ≥ Opus 16k: ΔWER −0.07 … +0.11, all CIs include 0 | `final_report/table_main_results.csv` |
| §4.1 | substitution share of net extra errors 84–91% (MP3 16k, Opus 8k, Opus 6k, both subsets); test-other Opus 6k +858 S, +116 D, +38 I | `final_report/table_main_results.csv` (`delta_*`, JiWER) |
| §4.2 | cutoffs: MP3 ~7.3 / 5.8 / 5.6 kHz; Opus 8.0 kHz to 12k, 4.6–4.7 kHz at 8k/6k | `signal_distortion_summary.csv` (`retained_bandwidth_mean_hz`) |
| §4.3 | conv → layer-12 attenuation 2.7–4.4× (test-clean), 1.5–2.7× (test-other); peak-based 2.7–4.4 / 1.5–2.6; examples Opus 16k 0.095 → 0.022, Opus 6k 0.425 → 0.214 | `final_report/table_layer_attenuation.csv` |
| §4.3 | raw layer 10 → 11 drop 5–64× | `final_report/table_layer_attenuation.csv` (`raw_layer10_to_layer11_ratio`: 5.4–63.8) |
| §4.3 | drift increases monotonically with compression at every layer | checked from `representation_similarity_summary.csv` (0 violations over 2 subsets × 2 codecs × 13 levels) |
| Abstract, §4.4 | condition Pearson: L12 0.96 [0.91, 0.98] / 0.97 [0.96, 0.98]; LSD 0.75 [0.66, 0.81] / 0.79 [0.76, 0.81]; difference +0.21 [+0.16, +0.25] / +0.18 [+0.17, +0.20] | `final_report/table_predictors.csv` (speaker-level bootstrap) |
| §4.4 | condition Spearman: LSD 0.77 / 0.99; L12 0.78 / 0.97 | `final_report/table_predictors.csv` |
| §4.4 | pooled Spearman: L12 0.21 [0.17, 0.25] / 0.42 [0.37, 0.46]; LSD 0.15 / 0.32 | `final_report/table_predictors.csv` |
| Abstract, §4.4 | within-condition mean Spearman: LSD −0.02 [−0.06, +0.02] / +0.03 [−0.03, +0.10]; L12 0.09 [0.05, 0.13] / 0.23 [0.18, 0.27] | `final_report/table_predictors.csv` |
| §4.4 | per-condition within ρ (L12): test-other Opus 8k 0.53, 6k 0.64; test-clean Opus 8k 0.24, 6k 0.40 | `final_report/table_within_condition.csv` |
| §4.4 | quartiles, test-other Opus 8k: 86% vs 23% (L12), 53% vs 45% (LSD); test-clean Opus 8k: 30% vs 7% (L12), 15% vs 25% (LSD) | `final_report/table_drift_quartiles.csv` |
| §3.7, §4.5, Fig. 4 | case pools 43 / 10; robust 7729-102255-0025 (LSD 10.8, L12 0.042); failure 672-122797-0073 (LSD 10.5, L12 0.129, conv 0.32 vs 0.29); condition quartiles 0.062 / 0.100; "FLAMED UP → FLAME OT" | `final_report/case_study_selection.csv` |
| §4.6, Table 2 | EnCodec WER / ΔWER / LSD / drift | `final_report/table_encodec.csv` (from `encodec_extension_results.csv`, recomputed) |
| §4.6 | EnCodec lag 0 and retained bandwidth 8 kHz (six utterances) | spot check run on 2026-09-26 (not saved as a file; see §5) |

---

## 3. Figures

| Figure | File | Generated by | Data |
|---|---|---|---|
| Fig. 1 | `results/final_report/fig1_integrated.png` | `src/final_report_figures.py` | `integrated_analysis_summary.csv` |
| Fig. 2 | `results/final_report/fig2_drift_by_layer.png` | `src/representation_analysis.py` (copied by `final_report_figures.py`) | `representation_similarity_summary.csv` (`sdrift_mean`) |
| Fig. 3 | `results/final_report/fig3_predictors.png` | `src/final_report_figures.py` | `predictor_analysis/condition_level_predictors.csv`, `predictor_analysis/predictor_correlations.csv` |
| Fig. 4 | `results/final_report/fig4_case_study.png` | `src/final_case_studies.py` | `case_study_selection.csv`; audio re-encoded with the same FFmpeg command |

Supporting figure (repository only): `results/figures/frequency_distortion_comparison.png` (D(f), 80 dB floor), generated by `src/signal_distortion_analysis.py`.

All figures use standardised drift. ΔWER is in percentage points. No figure uses the 95%-energy bandwidth or raw-cosine drift.

---

## 4. References

All 29 cited references were verified on 2026-09-26 against primary or authoritative sources: publisher or proceedings pages, Crossref DOIs, ACL Anthology, PMLR, the ISCA archive, arXiv and the RFC Editor. Claims were checked against the paper text where it was accessible. BibTeX: `FINAL_REPORT_REFERENCES.bib`.

| Key | Verified source | Claim used in report | Status |
|---|---|---|---|
| baevski2020wav2vec | NeurIPS 33, pp. 12449–12460 | CNN encoder + transformer, CTC fine-tuning; Base LS-960 no-LM WER 3.4/8.5 (Table 10) | VERIFIED; author corrected to "Yuhao Zhou" (proceedings) |
| hsu2021robust | Interspeech 2021, doi:10.21437/Interspeech.2021-236 | Wav2Vec2 degrades under domain mismatch (data domain, not codecs) | VERIFIED |
| pasad2021layerwise | ASRU 2021, doi:10.1109/ASRU51503.2021.9688093 | acoustic → phonetic → word hierarchy; top layers change most in ASR fine-tuning; autoencoder-style reversion | VERIFIED |
| pasad2023comparative | ICASSP 2023, doi:10.1109/ICASSP49357.2023.10096149 | layer trends depend on pre-training objective | VERIFIED |
| panayotov2015librispeech | ICASSP 2015, doi:10.1109/ICASSP.2015.7178964 | 16 kHz; clean/other split by speaker difficulty; **source audio MP3-compressed** | VERIFIED |
| rfc6716 | RFC 6716, doi:10.17487/RFC6716 | SILK/CELT; NB 4 kHz, WB 8 kHz; §2.1.1 sweet spots 8–12 kbit/s NB, 16–20 kbit/s WB (20 ms frames); TOC byte (§3.1) | VERIFIED (exact wording) |
| brandenburg1999mp3 | AES 17th Int. Conf., 1999, paper 17-009 | masking-threshold quantisation; high-frequency bandwidth lost at low bitrates (§5.1.1) | VERIFIED; wording changed from "low-pass" to "bandwidth sacrificed" |
| besacier2001effect | MMSP 2001, pp. 301–306, doi:10.1109/MMSP.2001.962750 | MPEG coding degrades HMM ASR at low bitrates; GSM/G.711 acceptable | VERIFIED |
| hirsch2000aurora | ASR2000 (ISCA ITRW), pp. 181–188 | noise-robustness benchmark | VERIFIED |
| narayanan2018domain | SLT 2018, doi:10.1109/SLT.2018.8639610 | MP3 64k / Opus 24k no loss; MP3 23k 10.5 → 13.6% WER; codec augmentation recovers | VERIFIED |
| drude2021opus | Interspeech 2021, doi:10.21437/Interspeech.2021-1214 | Opus bitrate vs far-field WER trade-off | VERIFIED |
| radford2023whisper | ICML 2023, PMLR 202 | large-scale weak supervision improves robustness | VERIFIED |
| huang2022distortion | Interspeech 2022, doi:10.21437/Interspeech.2022-519 | SSL models degrade under noise/reverb; no codecs tested | VERIFIED |
| zhu2022noiserobust | ICASSP 2022, doi:10.1109/ICASSP43922.2022.9747379 | Wav2Vec2 output cosine similarity (noisy vs clean) rises with SNR | VERIFIED |
| wang2022wav2vecswitch | ICASSP 2022, doi:10.1109/ICASSP43922.2022.9746929 | clean/noisy representation consistency improves robustness | VERIFIED |
| chai2021cegm | IEEE/ACM TASLP 29, doi:10.1109/TASLP.2020.3036783 | internal-posterior distance correlates with WER better than PESQ/STOI | VERIFIED |
| iwamoto2022artifacts | Interspeech 2022, doi:10.21437/Interspeech.2022-318 | artefacts, not residual noise, drive ASR errors after enhancement | VERIFIED |
| defossez2023encodec | TMLR 2023, arXiv:2210.13438 | 24 kHz mono model at 1.5/3/6/12/24 kbps; RVQ | VERIFIED |
| zeghidour2022soundstream | IEEE/ACM TASLP 30, doi:10.1109/TASLP.2021.3129994 | neural codec with RVQ | VERIFIED |
| wu2024codecsuperb | Findings ACL 2024, doi:10.18653/v1/2024.findings-acl.616 | Whisper WER on resynthesised LibriSpeech; WER falls as bitrate rises | VERIFIED |
| shi2024espnetcodec | SLT 2024, doi:10.1109/SLT61566.2024.10832289 | ASR WER after codec resynthesis | VERIFIED |
| wang2025audiocodecbench | arXiv:2509.02349 | codec benchmark incl. WER (repository suggested by teaching staff) | VERIFIED; **preprint, labelled as such** |
| timkey2021bark | EMNLP 2021, doi:10.18653/v1/2021.emnlp-main.372 | 1–3 rogue dimensions dominate cosine; standardisation corrects (text LMs; applied here by analogy) | VERIFIED |
| ethayarajh2019contextual | EMNLP-IJCNLP 2019, doi:10.18653/v1/D19-1006 | anisotropy; generally higher in upper layers (text models) | VERIFIED |
| kornblith2019similarity | ICML 2019, PMLR 97 | CKA | VERIFIED |
| gray1980distortion | IEEE Trans. ASSP 28(4), doi:10.1109/TASSP.1980.1163421 | general citation for spectral distortion measures only (not the LSD formula) | VERIFIED (metadata; full text paywalled) |
| bisani2004bootstrap | ICASSP 2004, doi:10.1109/ICASSP.2004.1326009 | paired bootstrap; **recommends speaker-level resampling**, now adopted | VERIFIED |
| efron1993bootstrap | Chapman & Hall 1993, doi:10.1007/978-1-4899-4541-9 | bootstrap | VERIFIED (series number removed as unverified) |
| graves2006ctc | ICML 2006, doi:10.1145/1143844.1143891 | CTC | VERIFIED |

Verified but not cited (reserve): DASB (arXiv:2406.14294; evaluates discrete tokens without resynthesis, so it is not suitable for the WER-after-resynthesis claim) and Prescott et al. (arXiv:2603.09034, adversarial focus). Not to be cited as ASR evidence: Siegert et al., ESSV 2016 (spectral analysis only).

---

## 5. Known issues and contradictions between files

1. **S/D/I split: `error_analysis.py` vs JiWER.** For four test-other conditions, the custom alignment back-trace in `src/error_analysis.py` breaks ties differently from JiWER. The totals are identical; for example, test-other Opus 6k is 1466/172/99 in `error_summary.csv` and 1454/178/105 in JiWER, both 1,737 errors. The report uses JiWER, as stated in the Methods. `results/error_analysis/error_summary.csv` should not be quoted for S/D/I. The README table was updated to the JiWER split.
2. **Condition-level Spearman does not favour late-layer drift.** Only Pearson (linearity) and the pooled and within-condition analyses separate layer-12 drift from LSD. The report states this explicitly (§4.4).
3. **EnCodec spot checks** (zero lag, 8 kHz retained bandwidth) were run on six utterances only and are reported as such. The EnCodec LSD uses length matching without cross-correlation, which is justified by the zero measured lag.
4. **Condition-level CIs** reflect utterance resampling only, not the choice of the 11 codec settings (12 points per subset). This is stated in §3.6 and §6.
5. **Bootstrap unit.** An earlier version of the report used utterance-level bootstrap CIs. Following the verified recommendation of Bisani & Ney (2004), the report now uses speaker-level block-bootstrap CIs for ΔWER and for all correlations. No zero-exclusion conclusion changed. The intervals widen mainly for severe test-other conditions: Opus 6k goes from [+10.30, +12.83] to [+9.07, +14.18]. `results/bootstrap_results.csv` and older README tables still contain the utterance-level CIs.
6. **LibriSpeech source audio was MP3-compressed** (Panayotov et al. 2015), so every condition is tandem coding. This is stated in §3.1 and §6.
7. **The standardisation calibration set** (first 100 selected test-clean utterances) overlaps with the evaluated test-clean utterances. It is used only for unsupervised per-dimension scaling with no labels; test-other uses the same statistics.

---

## 6. Stale files and results that must NOT be cited

| File / source | Why it is stale |
|---|---|
| `ELEC5305 Project Proposal v1.pdf`, `v2.pdf` | Pre-correction results: 95%-energy bandwidth, unfloored D_spec, raw-cosine drift at layers 1/6/12 on 100 utterances, and EnCodec on a different 100-utterance sample (WAV WER 4.22%) |
| Git history before the corrections (`signal_distortion_summary.csv` etc. at commit `2541e60`) | `D_spec` with ε = 1e-8 and no floor (e.g. MP3 128k = 317 dB², Opus 16k = 72 dB²); 95%-energy "bandwidth" (~2.9 kHz everywhere); 100 utterances |
| Raw-cosine drift columns (`drift_*`, `drift_mean`) | Not comparable across layers (anisotropy). They are kept for reference only and must not be used for cross-layer claims. There is no "~100× deep-layer attenuation" |
| `results/error_analysis/error_summary.csv` S/D/I split | See §5.1 (totals are fine) |
| `results/error_analysis/selected_failure_cases.*`, `results/figures/case_studies/`, `results/figures/local_case_studies/` | Older failure cases selected without the drift-based rule. Superseded by `results/final_report/case_study_selection.csv` and Figure 4 |
| `results/figures/integrated_analysis.png`, `results/figures/predictor_condition_scatter.png`, `results/figures/predictor_correlation_by_layer.png` | Correct data, but superseded by `results/final_report/fig1`/`fig3` for the report |
| `results/mp3_32k_results.csv`, `results/baseline_results.csv` | 10-utterance pilot outputs |
| `PROJECT_RATIONALE.md` §18 (Layer 1/6/12 description) | Development history; describes the pre-correction representation analysis |

---

## 7. Limitations (as stated in the report, §6)

1. One recogniser (Wav2Vec2 Base, CTC, greedy decoding).
2. Layer 12 is directly upstream of the CTC output, so its correlation with WER is partly structural.
3. Bandwidth is confounded with quantisation precision.
4. LibriSpeech is read speech.
5. Encoder defaults were used (LAME CBR; libopus VBR, application=audio).
6. Bitrate is not perceptual quality; no perceptual metric was used.
7. Within-condition correlations are modest. Condition-level correlations rest on 12 points per subset.
8. EnCodec uses a 16 → 24 → 16 kHz path (controlled) and covers test-clean only.
9. Case studies are illustrative.

---

## 8. Final cleanup (wording) and proposal

Wording revisions in `report/FINAL_REPORT.pandoc.md`. No numbers or figures changed.

- Conclusion: bandwidth statement softened. The largest degradations are concentrated where the cutoff enters the speech band, particularly on test-other; bandwidth loss alone was not sufficient on test-clean; in-band quantisation also matters.
- §5.3: case study described as two utterances with similar global distortion and early-layer drift but different late-layer drift and outcomes.
- §5.5: secondary answer phrased as "consistent with" attenuation before the upper representations; states that specific acoustic cues were not isolated.
- §5.4: "resampling control indicates … contributes negligibly"; "suggests" the drift–WER relationship is not codec-specific.
- Conclusion closing sentence: "attenuation of codec-induced change before it reaches the upper representations".
- Added "Acknowledgement of AI Assistance" before References. No official ELEC5305 AI-declaration format was found in the project files; the proposal guide contains none.

Proposal v3: `ELEC5305 Project Proposal v3.pdf`, built from `report/proposal_v3.md` with `bash report/build_proposal.sh`. It follows `ELEC5305_Project_Report_Guidelines (2).pdf` (the Project Proposal Guide):

- sections 1–9;
- body (sections 1–7) of 823 words, within 700–1000;
- 10 peer-reviewed references plus RFC 6716;
- GitHub project and Pages links;
- 3 pages.

Its Table A1 values match `results/final_report/table_main_results.csv`.

## 9. Final validation checklist

- [x] Every reported number verified against the current result files (§2)
- [x] All confidence intervals verified: 22 utterance-level ΔWER CIs reproduced independently; speaker-level CIs computed in `final_report_tables.py`; correlation CIs (speaker-level) from `predictor_analysis.py`
- [x] All reference metadata verified (§4)
- [x] No "~100× deep-layer attenuation" claim (`grep -n "100×" FINAL_REPORT.md` returns nothing)
- [x] No 95%-energy bandwidth result used (only mentioned as the superseded method in §3.4)
- [x] Raw cosine not used as the main cross-layer metric
- [x] Signal, representation, task and EnCodec analyses use the shared 500-utterance sets (validation_report.txt)
- [x] Proposal v2 not used as a result source
- [x] EnCodec marked as an extension
- [x] Limitations explicit, each with a future-work item
- [x] Length: 12 PDF pages = about 10.2 pages of body text, figures and tables + about 1.3 pages of references; no overfull pages (xelatex log checked)
