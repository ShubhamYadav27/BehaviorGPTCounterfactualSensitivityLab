# BehaviorGPT Counterfactual Sensitivity Lab
## Live Demonstration Script (5–7 Minute Walkthrough)

**Audience:** Jens & Unbox AI Engineering Team  
**Presenter:** Engineering Internship / Entry-Level Candidate  
**Goal:** Demonstrate understanding of the BehaviorGPT model's observable behavior, clean experimental design, working software integration, empirical output analysis, and realistic discussion of next technical steps.

---

### Timing Overview (Total: 6 Minutes)

| Time | Segment | Focus |
|---|---|---|
| **0:00 - 0:45** | 1. Introduction & Research Question | Motivation, problem formulation, scientific scope |
| **0:45 - 1:45** | 2. How the SDK & Model Work | Foundation model paradigm, single forward pass, client adapter |
| **1:45 - 3:00** | 3. Experimental Protocol & Invariant Design | 40 strictly paired cases, invariant validation, deterministic generation |
| **3:00 - 4:15** | 4. Empirical Results & Live Dashboard Walkthrough | Top-1 flip rate, Jaccard overlap, rank displacement |
| **4:15 - 5:15** | 5. The Noteworthy Finding (Deep-Dive Case) | Action-Verb Stability (Case 003) vs. Position Sensitivity (Case 004) |
| **5:15 - 6:00** | 6. Limitations, Engineering Lessons & Future Work | Technical humility, what we can/cannot conclude, next experiments |

---

### Step-by-Step Script & Speaking Points

#### 1. Introduction & Research Question (0:00 - 0:45)
> "Hi Jens, thanks again for inviting me to explore BehaviorGPT. When you mentioned that developers keep building things on top of the model that you didn't expect, I wanted to investigate how the model actually responds to counterfactual perturbations in behavioral sequences.
>
> My research question was:
> **'When exactly one action in a behavioral history is changed, how does BehaviorGPT's subsequent prediction change, if at all?'**
>
> I wanted to stress upfront that I approached this without the assumption that outputs *must* change, or that changing is always better. My objective was to measure the model's actual sensitivity boundaries and representational dynamics under strictly controlled sequence interventions."

#### 2. How the Official SDK & Model Work (0:45 - 1:45)
> "I started by cloning the `Unbox-AI/behaviorgpt` SDK repository, reviewing the architecture and the research publications.
>
> BehaviorGPT treats chronological human action streams the way language models treat text: each interaction—whether a view, an add-to-cart, or a search—is an event token in a behavioral sentence.
>
> Two key properties stood out from the public documentation:
> 1. **Catalog ranking in a single forward pass:** Rather than generating tokens sequentially, BehaviorGPT evaluates the entire catalog against the user's behavioral history. Observed HTTP round-trip times were 286–961 ms including network overhead.
> 2. **Probability scores:** The response includes probability values (`prob`), item metadata, and term-level attribution scores. Whether these are formally calibrated has not been independently verified.
>
> In my integration, I built a typed `BehaviorGPTClient` adapter that connects to the public showcase endpoint at `behaviorgpt.unboxai.com/api`, and also supports deterministic mock replay for our automated CI test suite. This is the public showcase, not the official SDK gateway."

#### 3. Experimental Protocol & Invariant Design (1:45 - 3:00)
> *(Open the Web Dashboard or terminal showing the 13 passing pytest tests)*
> "To maintain scientific validity, I designed an intervention protocol with a strictly enforced property:
> - I generated 40 deterministic synthetic histories across 5 consumer archetypes (`ATHLETIC_RUNNER`, `TECH_ENTHUSIAST`, `LUXURY_SHOPPER`, `HOME_DECORATOR`, `CROSS_SHOPPER`).
> - For each history, our `ControlledMutator` produced a paired variant modifying **exactly one action**, keeping all other items, timestamps, offsets, and sequence positions completely identical.
> - An automated assertion verified that $\text{diff} == 1$ for every pair before submitting to the model.
>
> We tested four intervention strategies:
> 1. Cross-category item swaps
> 2. Within-category item swaps
> 3. Action-verb swaps (`viewItem` vs `addToCart` on the same item)
> 4. Early-position mutations (mutating index 0 in length 4–6 sequences)."

#### 4. Empirical Results & Live Dashboard Walkthrough (3:00 - 4:15)
> *(Switch to the Dashboard at `http://127.0.0.1:8000/`)*
> "We executed all 40 paired experiments (80 forward passes) against the live BehaviorGPT model with a 100% success rate.
>
> Across the entire suite:
> - **65% of single-action interventions changed the Top-1 predicted next item.**
> - **Top-5 Candidate Set Overlap (Jaccard similarity) averaged 42.5%.**
> - **Mean rank displacement was 1.83 ranks.**
>
> But the most interesting insight came from breaking the results down by mutation strategy."

#### 5. The Noteworthy Finding: Action-Verb Stability vs. Position Sensitivity (4:15 - 5:15)
> *(Click on 'Investigated Cases' or select Case 003 in the Case Inspector)*
> "Look at **Case 003**:
> Here, the user browsed luxury and athletic items, and on the terminal action, we changed *only* the action verb: `viewItem` on Nike Jordan Retro 4 to `addToCart` on the exact same item.
>
> The result was **complete invariance**:
> - Top-1 flip: **False**
> - Jaccard@10: **100.0%**
> - Mean rank drift: **0.00 ranks**
> - Probability TVD: **0.000000**
>
> Across all 10 action-type swaps in the dataset, the Top-1 invariance rate was **90.0%**, with an average Jaccard@5 of **90.0%**.
>
> This observed stability suggests that, for this dataset, item identity has a stronger influence on the model's output than the action verb. We report this as an empirical observation, not a claim about internal representations.
>
> In contrast, look at **Case 004**:
> In a 6-action sequence of Timberland boot interactions, mutating the opening action at index 0 from a boot (`paB0CPSK1GVB`) to a luxury Coach bag flipped the Top-1 prediction to Coach bags and UGG boots, dropping Top-5 Jaccard to **0.00%**. The model's output was highly sensitive to changes at the opening position."

#### 6. Limitations, Engineering Lessons & Future Work (5:15 - 6:00)
> "Finally, let's talk about limitations and what this does and does not prove:
> - This measures **local sensitivity**, not causal validity or real-world conversion accuracy. A changed prediction doesn't prove the user would buy it.
> - In terms of engineering lessons: the showcase endpoint is sensitive to payload schema — including unexpected fields produced error responses — and we identified the set of accepted event discriminators.
> - As a next technical step, I would love to explore **sequence length decay curves** (measuring how sensitivity decays as a function of distance from the prediction horizon) and **dwell-time warping** (testing temporal attention by varying millisecond offsets).
>
> All code, tests, and data artifacts are fully reproducible in the repository. I'm happy to dive into any part of the architecture or code paths!"

---

### Anticipated Engineering Questions & Confident Answers

**Q1: Why did you test on synthetic data instead of a public benchmark dataset?**
> *Answer:* "Controlled counterfactual testing requires strict, synthetic paired variants where we hold every variable constant except the single targeted action. Natural datasets rarely contain identical sessions differing by exactly one event, which introduces confounding variables (different times of day, different users, varying sequence lengths). Synthetic data allowed us to test causal sensitivity cleanly."

**Q2: Did you notice any non-determinism across identical calls?**
> *Answer:* "Yes, we tested repeatability with identical histories and timestamps; the live model returned identical candidate rankings and probabilities with zero variance. However, when timestamps are altered, the model's temporal encodings can introduce slight score perturbations, which is why we enforced deterministic timestamp offsets across paired variants."

**Q3: How would you extend this into a production monitoring or evaluation tool?**
> *Answer:* "I would package this into an automated Behavioral Invariance CI pipeline. Whenever a new model version is deployed, this test harness can run counterfactual regression tests to check whether the model's sensitivity profile has changed significantly on key user archetypes."
