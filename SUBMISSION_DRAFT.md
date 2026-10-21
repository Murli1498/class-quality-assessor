# Brief explanation (<=300 words) - DRAFT, fill the [BRACKETS] after calibrating on the real samples

**Approach.** Quality is judged on the teaching and learning experience, not on how a class sounds. The pipeline
extracts three evidence streams: a local Whisper transcript, audio signals (speech ratio, dead air, SNR) and visual
signals (screen activity, scene changes, faces). An LLM then scores seven rubric dimensions 1-5 (clarity, student
engagement, checking understanding, pacing, delivery, management, technical), each backed by timestamped quotes.
The label is computed in code from the weighted score with cut points fitted on the sample recordings, so it is
auditable and tunable, rather than being the model's free-text opinion.

**Why.** Children's classes differ mostly in interaction and responsiveness, which the transcript shows best.
Audio/video metrics are treated as context because they mislead: loud is not engaging, silence can be healthy wait
time, no faces can be a slide view. The prompt names these traps. Audio stays on the host; only text goes to the LLM
(screenshots optional, off by default) because children may be identifiable.

**What worked / failed.** [e.g. dimension X separated Poor from Excellent best (r = ...); engagement was the strongest
signal; raw audio metrics alone were near-useless (r = ...). Calibrated accuracy on N samples: exact [..], within one
band [..]. Failure: ...]

**Limitations.** Tiny labelled set, so thresholds are a fit, not a validated model. Whisper under-hears quiet child
speech and code-switching, so participation can be under-counted; no speaker diarisation. Visual understanding is
shallow unless screenshots are enabled. LLM scores vary run to run. Single-pass at inference.
