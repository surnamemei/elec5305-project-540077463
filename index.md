---
layout: default
title: ELEC5305 Project
---

<style>
:root {
  --ink: #1f2937;
  --muted: #6b7280;
  --line: #e5e7eb;
  --soft: #f8fafc;
  --card: #ffffff;
  --accent: #2563eb;
  --accent-soft: #eff6ff;
  --success-soft: #ecfdf5;
  --success: #047857;
}

.project-wrap {
  max-width: 980px;
  margin: 0 auto;
  color: var(--ink);
  line-height: 1.65;
}

.hero {
  padding: 2.2rem 2rem;
  border: 1px solid var(--line);
  border-radius: 18px;
  background: linear-gradient(135deg, #f8fbff 0%, #ffffff 55%, #f8fafc 100%);
  margin-bottom: 1.6rem;
}

.hero h1 {
  margin: 0 0 .7rem 0;
  font-size: 2.2rem;
  line-height: 1.15;
}

.hero p {
  margin: 0;
  font-size: 1.05rem;
  color: var(--muted);
}

.status-line {
  margin-top: .95rem;
  display: inline-block;
  padding: .38rem .7rem;
  border-radius: 999px;
  background: var(--success-soft);
  color: var(--success);
  font-size: .88rem;
  font-weight: 700;
}

.badges {
  margin-top: 1rem;
  display: flex;
  flex-wrap: wrap;
  gap: .55rem;
}

.badge {
  display: inline-block;
  padding: .35rem .7rem;
  border-radius: 999px;
  background: var(--accent-soft);
  color: #1d4ed8;
  font-size: .88rem;
  font-weight: 600;
}

.hero-actions {
  margin-top: 1.15rem;
  display: flex;
  flex-wrap: wrap;
  gap: .7rem;
}

.button {
  display: inline-block;
  padding: .62rem .9rem;
  border-radius: 10px;
  text-decoration: none !important;
  font-weight: 700;
  border: 1px solid var(--accent);
}

.button.primary {
  background: var(--accent);
  color: white !important;
}

.button.secondary {
  background: white;
  color: var(--accent) !important;
}

.section {
  margin-top: 2rem;
}

.callout {
  border-left: 4px solid var(--accent);
  background: var(--soft);
  padding: 1rem 1.1rem;
  border-radius: 10px;
  margin: 1rem 0 1.5rem 0;
}

.grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1rem;
  margin: 1.2rem 0 1.7rem 0;
}

.card {
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 1rem 1.1rem;
  background: var(--card);
}

.card .value {
  font-size: 1.6rem;
  font-weight: 750;
  margin-bottom: .15rem;
}

.card .label {
  font-size: .9rem;
  color: var(--muted);
}

.findings {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  margin: 1rem 0 1.5rem 0;
}

.finding {
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 1rem 1.1rem;
  background: #fff;
}

.finding strong {
  display: block;
  margin-bottom: .35rem;
}

.table-wrap {
  overflow-x: auto;
  border: 1px solid var(--line);
  border-radius: 12px;
  margin: .9rem 0 1.3rem 0;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 0;
}

th, td {
  padding: .7rem .8rem;
  border-bottom: 1px solid var(--line);
  text-align: left;
  white-space: nowrap;
}

th {
  background: var(--soft);
}

tr:last-child td {
  border-bottom: none;
}

.figure {
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: .8rem;
  background: #fff;
  margin: 1rem 0 1.5rem 0;
}

.figure img {
  width: 100%;
  border-radius: 10px;
}

.small {
  font-size: .92rem;
  color: var(--muted);
}

.references li {
  margin-bottom: .7rem;
}

.footer-note {
  margin-top: 2rem;
  padding-top: 1rem;
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-size: .9rem;
}

code {
  background: #f3f4f6;
  padding: .08rem .3rem;
  border-radius: 5px;
}

@media (max-width: 760px) {
  .grid, .findings {
    grid-template-columns: 1fr;
  }

  .hero {
    padding: 1.4rem;
  }

  .hero h1 {
    font-size: 1.8rem;
  }
}
</style>

<div class="project-wrap">

<div class="hero">

<h1>Why Is Wav2Vec2 Robust to Lossy Audio Compression?</h1>

<p>
A signal-, representation-, and task-level study of MP3 and Opus compression,
investigating why Wav2Vec2 can remain robust despite measurable codec-induced distortion.
</p>

<div class="status-line">
Core experiments complete — final analysis and reporting stage
</div>

<div class="badges">
<span class="badge">Wav2Vec2</span>
<span class="badge">LibriSpeech</span>
<span class="badge">MP3 + Opus</span>
<span class="badge">Signal Distortion</span>
<span class="badge">Representation Drift</span>
<span class="badge">Bootstrap CI</span>
<span class="badge">EnCodec Extension</span>
</div>

<div class="hero-actions">
<a class="button primary" href="ELEC5305%20Project%20Proposal%20v1.pdf">
View Proposal PDF
</a>

<a class="button secondary"
href="https://github.com/surnamemei/elec5305-project-540077463">
View GitHub Repository
</a>
</div>

</div>

<section class="section">

<h2>Research Question</h2>

<div class="callout">

<strong>
How do MP3 and Opus compression alter the acoustic signal and the internal
representations of Wav2Vec2, and which codec-induced distortions are associated
with the onset of ASR errors?
</strong>

<br><br>

Secondary question:

<em>
Is Wav2Vec2 robust because lossy codecs preserve the acoustic information
important to the recogniser even when measurable signal distortion is already substantial?
</em>

</div>

</section>

<section class="section">

<h2>Project Objective</h2>

<p>
Rather than treating Word Error Rate (WER) as the only outcome, this project
connects three levels of analysis:
</p>

<div class="grid">

<div class="card">
<div class="value">1</div>
<div class="label"><strong>Signal level</strong><br>spectral distortion and retained bandwidth</div>
</div>

<div class="card">
<div class="value">2</div>
<div class="label"><strong>Representation level</strong><br>Wav2Vec2 hidden-layer drift</div>
</div>

<div class="card">
<div class="value">3</div>
<div class="label"><strong>Task level</strong><br>WER and recognition errors</div>
</div>

</div>

</section>

<section class="section">

<h2>Experimental Design</h2>

<p>
The main experiment uses the fixed pretrained
<code>WAV2VEC2_ASR_BASE_960H</code> model and two LibriSpeech subsets:
<code>test-clean</code> and <code>test-other</code>.
</p>

<p>
For each subset, 500 utterances are selected using the fixed random seed
<code>5305</code>. The same utterances are used across all codec conditions.
</p>

<div class="table-wrap">

<table>

<thead>
<tr>
<th>Codec</th>
<th>Bitrates</th>
</tr>
</thead>

<tbody>
<tr>
<td>WAV</td>
<td>Uncompressed baseline</td>
</tr>

<tr>
<td>MP3</td>
<td>128, 64, 32, 24, 16 kbps</td>
</tr>

<tr>
<td>Opus</td>
<td>64, 32, 16, 12, 8, 6 kbps</td>
</tr>
</tbody>

</table>

</div>

<p>
All MP3 and Opus files are encoded using FFmpeg with
<code>libmp3lame</code> and <code>libopus</code>.
Actual effective bitrate is measured from encoded file size and duration.
</p>

</section>

<section class="section">

<h2>Main Recognition Results</h2>

<div class="grid">

<div class="card">
<div class="value">3.17%</div>
<div class="label">test-clean WAV baseline WER</div>
</div>

<div class="card">
<div class="value">5.45%</div>
<div class="label">test-clean Opus 6 kbps WER</div>
</div>

<div class="card">
<div class="value">19.79%</div>
<div class="label">test-other Opus 6 kbps WER</div>
</div>

</div>

<div class="table-wrap">

<table>

<thead>
<tr>
<th>Dataset</th>
<th>Condition</th>
<th>WER</th>
<th>ΔWER</th>
<th>Compression Ratio</th>
</tr>
</thead>

<tbody>

<tr>
<td>test-clean</td>
<td>WAV</td>
<td>3.17%</td>
<td>0.00 pp</td>
<td>1.00×</td>
</tr>

<tr>
<td>test-clean</td>
<td>MP3 16k</td>
<td>4.08%</td>
<td>+0.91 pp</td>
<td>15.40×</td>
</tr>

<tr>
<td>test-clean</td>
<td>Opus 8k</td>
<td>4.32%</td>
<td>+1.15 pp</td>
<td>31.41×</td>
</tr>

<tr>
<td>test-clean</td>
<td><strong>Opus 6k</strong></td>
<td><strong>5.45%</strong></td>
<td><strong>+2.28 pp</strong></td>
<td><strong>40.28×</strong></td>
</tr>

<tr>
<td>test-other</td>
<td>WAV</td>
<td>8.26%</td>
<td>0.00 pp</td>
<td>1.00×</td>
</tr>

<tr>
<td>test-other</td>
<td>MP3 16k</td>
<td>12.00%</td>
<td>+3.74 pp</td>
<td>15.33×</td>
</tr>

<tr>
<td>test-other</td>
<td>Opus 8k</td>
<td>14.89%</td>
<td>+6.63 pp</td>
<td>31.52×</td>
</tr>

<tr>
<td>test-other</td>
<td><strong>Opus 6k</strong></td>
<td><strong>19.79%</strong></td>
<td><strong>+11.53 pp</strong></td>
<td><strong>40.23×</strong></td>
</tr>

</tbody>

</table>

</div>

<p class="small">
Selected representative conditions are shown here. Full bitrate sweeps are available
in the repository results.
</p>

</section>

<section class="section">

<h2>Three-Level Analysis</h2>

<div class="figure">
<img src="results/figures/integrated_analysis.png"
alt="Integrated signal representation and recognition analysis">

<p class="small">
Integrated analysis linking log-spectral distortion, Wav2Vec2 representation drift,
and WER across representative compression conditions.
</p>
</div>

<div class="findings">

<div class="finding">
<strong>Signal distortion can appear before large WER changes</strong>
Moderate compression can alter the acoustic signal while recognition remains close
to the WAV baseline.
</div>

<div class="finding">
<strong>Early representations are more sensitive</strong>
Layer 1 generally shows substantially larger codec-induced drift than Layers 6 and 12.
</div>

<div class="finding">
<strong>Deeper representations remain more stable</strong>
Under moderate compression, much of the perturbation visible in early layers is reduced
in deeper Wav2Vec2 representations.
</div>

<div class="finding">
<strong>Severe compression eventually affects deeper layers</strong>
At aggressive low bitrates, deeper-layer drift increases together with larger WER changes.
</div>

</div>

</section>

<section class="section">

<h2>Representation-Level Analysis</h2>

<div class="figure">
<img src="results/figures/representation_drift_by_layer.png"
alt="Wav2Vec2 representation drift across layers">

<p class="small">
Mean representation drift for Layers 1, 6 and 12 across 100 fixed
test-clean utterances.
</p>
</div>

<p>
The results suggest that Wav2Vec2 progressively reduces codec-induced perturbations
across its hidden layers. However, this robustness becomes weaker when compression
is sufficiently severe.
</p>

</section>

<section class="section">

<h2>Signal-Level Analysis</h2>

<div class="figure">
<img src="results/figures/frequency_distortion_comparison.png"
alt="Frequency dependent distortion">

<p class="small">
Frequency-dependent distortion under MP3 and Opus compression.
</p>
</div>

<p>
Very low bitrate conditions show substantial modification of the acoustic signal
and reduced retained bandwidth. However, the magnitude of signal distortion alone
does not fully predict ASR failure.
</p>

</section>

<section class="section">

<h2>Statistical and Error Analysis</h2>

<p>
Paired bootstrap analysis is used to estimate confidence intervals for ΔWER.
The strongest main-experiment degradation occurs at Opus 6 kbps.
</p>

<div class="table-wrap">

<table>

<thead>
<tr>
<th>Dataset</th>
<th>Condition</th>
<th>ΔWER</th>
<th>95% CI</th>
</tr>
</thead>

<tbody>

<tr>
<td>test-clean</td>
<td>MP3 16k</td>
<td>+0.91 pp</td>
<td>[+0.59, +1.29]</td>
</tr>

<tr>
<td>test-clean</td>
<td>Opus 6k</td>
<td>+2.28 pp</td>
<td>[+1.86, +2.76]</td>
</tr>

<tr>
<td>test-other</td>
<td>MP3 16k</td>
<td>+3.74 pp</td>
<td>[+3.07, +4.43]</td>
</tr>

<tr>
<td>test-other</td>
<td>Opus 6k</td>
<td>+11.53 pp</td>
<td>[+10.30, +12.83]</td>
</tr>

</tbody>

</table>

</div>

<p>
Recognition failures under severe compression are dominated by substitutions,
with smaller increases in deletions and insertions.
</p>

</section>

<section class="section">

<h2>Optional Neural Codec Extension</h2>

<p>
EnCodec was evaluated as a small extension to test whether the same
signal → representation → recognition pattern also appears with a neural codec.
</p>

<p>
Because the mono EnCodec model operates at 24 kHz, an uncompressed
<code>16 → 24 → 16 kHz</code> resampling control was included.
</p>

<div class="table-wrap">

<table>

<thead>
<tr>
<th>Condition</th>
<th>WER</th>
<th>ΔWER</th>
<th>Layer 12 Drift</th>
</tr>
</thead>

<tbody>

<tr>
<td>WAV</td>
<td>4.22%</td>
<td>0.00 pp</td>
<td>0.000</td>
</tr>

<tr>
<td>Resampling control</td>
<td>4.28%</td>
<td>+0.06 pp</td>
<td>0.001</td>
</tr>

<tr>
<td>EnCodec 24k</td>
<td>4.28%</td>
<td>+0.06 pp</td>
<td>0.016</td>
</tr>

<tr>
<td>EnCodec 6k</td>
<td>5.17%</td>
<td>+0.94 pp</td>
<td>0.029</td>
</tr>

<tr>
<td><strong>EnCodec 1.5k</strong></td>
<td><strong>13.06%</strong></td>
<td><strong>+8.83 pp</strong></td>
<td><strong>0.110</strong></td>
</tr>

</tbody>

</table>

</div>

<div class="figure">
<img src="results/figures/encodec_extension.png"
alt="EnCodec extension analysis">

<p class="small">
The resampling control produces negligible recognition change, while increasingly
severe EnCodec compression is associated with larger representation drift and WER.
</p>
</div>

</section>

<section class="section">

<h2>Current Interpretation</h2>

<div class="callout">

The results suggest that measurable codec-induced signal distortion can occur
before recognition performance degrades strongly.

<br><br>

Wav2Vec2 appears to suppress part of this perturbation through its learned
representations, particularly in deeper layers. Under sufficiently severe compression,
deeper representation drift also increases and is accompanied by substantially higher WER.

</div>

<p>
This makes the project more than a codec-ranking experiment: the main contribution
is the connection between signal degradation, learned representation stability,
and downstream recognition performance.
</p>

</section>

<section class="section">

<h2>Representative Failure Cases</h2>

<p>
Individual failure analysis confirms that severe compression can transform
previously correct WAV recognition into large substitution- and deletion-heavy errors.
Representative cases are stored in:
</p>

<p>
<code>results/error_analysis/selected_failure_cases.csv</code>
</p>

<p>
and
</p>

<p>
<code>results/error_analysis/selected_failure_cases.txt</code>
</p>

</section>

<section class="section">

<h2>References</h2>

<ol class="references">

<li>
A. Baevski, Y. Zhou, A. Mohamed, and M. Auli,
“wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations,”
2020.
</li>

<li>
W.-N. Hsu et al.,
“Robust wav2vec 2.0: Analyzing Domain Shift in Self-Supervised Pre-Training,”
2021.
</li>

<li>
J.-M. Valin, K. Vos, and T. Terriberry,
“Definition of the Opus Audio Codec,” RFC 6716, 2012.
</li>

<li>
V. Panayotov, G. Chen, D. Povey, and S. Khudanpur,
“LibriSpeech: An ASR Corpus Based on Public Domain Audio Books,”
Proc. IEEE ICASSP, 2015.
</li>

<li>
A. Défossez et al.,
“High Fidelity Neural Audio Compression,”
2022.
</li>

</ol>

</section>

<section class="section">

<h2>Next Steps</h2>

<ul>
<li>finalise literature grounding and related-work discussion</li>
<li>select the strongest figures for the final report</li>
<li>document limitations and interpretation carefully</li>
<li>prepare the final research report</li>
<li>prepare the project demonstration video</li>
</ul>

</section>

<section class="section">

<h2>Project Repository</h2>

<p>
Full source code, experiment outputs, analysis scripts, and reproducibility notes are available at:
</p>

<p>
<a href="https://github.com/surnamemei/elec5305-project-540077463">
github.com/surnamemei/elec5305-project-540077463
</a>
</p>

</section>

<div class="footer-note">
ELEC5305 project — core experiments complete; final interpretation and reporting are in progress.
</div>

</div>