# Brief explanation (300 words)

Quality is judged on the teaching, not on how loud a class is. The app transcribes the recording on the server, measures speech, silence and screen activity, and can sample a few screenshots. Gemini then scores seven dimensions from 1 to 5, each with a timestamped observation: individual engagement, rapport and opening, instructional clarity, feedback and homework, energy, presentation, and pacing. The Poor, Average, Good or Excellent label is computed in code from the weighted score. The dimensions come from the organiser's own reviewer notes.

On the 10 labelled classes, rapport (correlation 0.73) and pacing (0.64) tracked the true grade. Loudness did not. Mean scores rose from Poor 3.12 to Average 3.32, Good 3.55 and Excellent 4.08. With the default bands, 3 of 10 labels matched exactly and 8 of 10 were within one band. Tighter cutoffs would match 7 of 10, but they sit only 0.15 apart, so they were not saved. They would memorise this set.

Open models on the free Groq tier scored every class near 2, so an Excellent class and a Poor class were indistinguishable. Gemini's free quota is about 20 requests a day per model, and some models were overloaded. One class filed as a quality issue still scored 3.80, because the teaching itself was adequate and the faults were technical.

Ten recordings is a sanity check, not a validated model, and each class is judged once. The transcript can miss quiet children and mixed languages, and there is no speaker separation unless a platform transcript is uploaded. Background, face and lighting need screenshots, which stay off unless enabled, because children may be identifiable.
