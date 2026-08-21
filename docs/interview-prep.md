# Interview prep

Talking points for this repo, grounded in numbers that are actually in it. Every figure
below is reproducible from a script in `scripts/`; none of it is rounded in a flattering
direction.

## The one-liner

> I reproduced SASRec from scratch to within half a point of the paper's HR@10, then found
> that the BERT4Rec-beats-SASRec result on ML-1M is decided by a dropout default nobody
> chose deliberately — and rebuilt the same codebase as a TIGER-style generative recommender
> with RQ-quantized semantic IDs, which lost, and I can tell you exactly where and why.

The original version of this line ended "...and quantified exactly when semantic IDs beat
atomic IDs (cold-start) and when they don't." **Don't use that version.** It was written
before the experiment ran, and the experiment failed in the direction it was supposed to
win. The line above is the one the data supports.

If asked to expand, the three results in decreasing order of confidence:

1. **BERT4Rec's win on ML-1M is a baseline-configuration artifact.** RecBole gives SASRec
   dropout 0.5 and BERT4Rec 0.2; they are otherwise identical on every architectural
   default. Changing that one line moves the comparison by +3.71% HR@10 / +6.33% NDCG@10 —
   more than the entire margin it was supposed to explain. Same BERT4Rec run, three
   different conclusions.
2. **Sampled metrics hide implementation differences.** Two SASRec implementations agree to
   +0.61% on sampled HR@10 and diverge by +40% / +53% on full ranking. Agreement under the
   sampled protocol is not agreement.
3. **Semantic IDs cost more than they buy on Amazon Beauty**, and the mechanism is a
   diversity collapse (14% catalogue coverage vs 76%) that is only partly a scoring-time
   artifact.

---

## L1 — concepts

### Why causal (unidirectional) attention in SASRec?

Because the task is next-item prediction and the training signal comes from every position
at once. With a causal mask, position *t* predicts item *t+1* using only items 1..*t*, so a
single forward pass over a length-*n* sequence yields *n* training examples, none of which
can see their own answer. Drop the mask and position *t* attends to item *t+1* — the model
learns to copy the target and the metric goes to ~1.0 in training and to chance at test.

This is also the substantive difference with BERT4Rec, which uses bidirectional attention
and a cloze objective: it masks a random subset of positions and predicts them from both
sides. That is a richer objective per sequence but a much weaker one per step — BERT4Rec
sees roughly `mask_ratio` (0.2) targets per sequence where SASRec sees all of them, which
is precisely why the training-budget question at the heart of the reproducibility
controversy is the right question to ask.

**Follow-up you should expect:** "so BERT4Rec just needs more epochs?" Honest answer: that
is the Petrov & Macdonald claim, and *this repo did not test it*. The training-budget curve
(4x / 10x epochs) was cut on GPU cost — ~58 GPU-hours per model. Only the 1x point exists.
Say so.

### Why tie the input and output embeddings?

Three reasons, in order of how much they mattered here:

- **Parameter count.** The item table dominates. On Beauty, 12,101 items × 64 dims is
  774k parameters out of SASRec's 828k total — 93% of the model. Tying halves what would
  otherwise be two of those tables.
- **Consistency of the item space.** The output layer scores an item by dot product against
  its embedding. If input and output tables are separate, "the vector that represents item
  *i* in a history" and "the vector that recognizes item *i* as a target" are learned
  independently, and rare items get two under-trained vectors instead of one.
- **It makes the atomic-vs-semantic comparison legible.** GenRec ties too, so the parameter
  difference between the two models is entirely about how items are represented and not
  about a bookkeeping choice.

### What do semantic IDs actually solve?

An atomic ID table gives every item a free parameter vector, so an item seen twice in
training has a vector trained on two gradient steps, and an item seen zero times has a
vector at its initialization — structurally unreachable. Semantic IDs replace the item with
a short sequence of codes derived from its content embedding, so a never-before-seen item
still has a representation, assembled from codes that thousands of other items also use.

Here: 12,101 Beauty items → 3 levels × 256 residual-KMeans codes + a 4th disambiguation
token = **782 token embeddings**, a 15.5x smaller table. GenRec runs on 113,472 parameters
against SASRec's 828,352 — **13.7%**.

**And the honest part:** on this dataset that trade lost. −28.96% sampled HR@10, −57.8% on
full ranking. The cold-start payoff appears only after popularity-debiasing and only on the
unseen bucket. Details in L2.

---

## L2 — implementation

### Sampled metrics: what's the bias, and did you see it?

Krichene & Rendle (2020) show that ranking one positive against ~100 sampled negatives is
not a monotone proxy for ranking against the full catalogue — the metric can order two
models differently than full ranking does, because sampling mostly draws easy negatives and
therefore rewards getting the target roughly right rather than exactly right.

This repo has a direct instance rather than a citation. RecBole's SASRec and this repo's
SASRec agree to **+0.61% on sampled HR@10** and differ by **+40% HR@10 / +53% NDCG@10** on
full ranking. Same protocol shape, same data, same split. The divergence grows with how much
the metric cares about *where* in the ranking the target lands, which points at the training
objective — RecBole trains cross-entropy over the full catalogue, this repo trains BCE
against one sampled negative per position, per the original paper. That hypothesis is
untested here; no loss-only ablation was run. It is the largest unexplained effect in the
project and I say so in the README.

Two practical consequences I'd defend:

- **Report both protocols, always side by side.** Every table in this repo does.
- **Freeze the negatives.** `negatives.json` is generated once at seed 42 and reused by
  every model. The RecBole runs are the exception — their evaluator draws its own — and
  rescoring one run's raw predictions both ways puts that difference at +2.28% HR@10 /
  +5.38% NDCG@10, the same order as the margins being discussed. So rows from different
  draws are never differenced against each other.

### Codebook collapse — how did you mitigate it?

I didn't have to, and that is the interesting answer. The plan had RQ-VAE as a stretch goal
specifically to fix collapse. RQ-KMeans (3 levels, 256 codes each, KMeans on the residual at
each level) produced **0 dead codes on both datasets**, so there was no collapse to fix and
the stretch goal was dropped rather than performed. KMeans initializes from the data and
assigns every centroid a non-empty cluster by construction; the collapse mode RQ-VAE exists
to solve is a gradient-descent-on-a-codebook problem, not an inherent property of residual
quantization.

What I'd watch for if I did use RQ-VAE: dead-code fraction per level as a first-class
training metric, not a post-hoc check.

**The quality checks that did matter:**

| | ML-1M | Beauty |
|---|---|---|
| collision rate on the 3-token code | 1.46% | **11.78%** |
| within-prefix cosine @ depth 3 (vs. random pair) | 0.753 (0.439) | 0.871 (0.288) |

Prefix coherence rises monotonically with depth on both — that is the coarse-to-fine
property the generative model depends on, and it's worth verifying rather than assuming.

Two things I found that nobody designed:

- **Beauty collides at 11.78%.** For ~1 item in 8, the only thing separating it from a
  catalogue neighbour is a token carrying no content signal. That caps what semantic IDs
  can do there.
- **ML-1M's codes encode release year at least as strongly as genre.** Items sharing a
  2-token prefix are 1.68 years apart on average against a 15.85-year baseline — because
  MovieLens titles have the year in the string. The text format chose a clustering axis
  that I didn't.

### Constrained decoding — how is it built, and how do you know it's right?

A Trie over the catalogue's code sequences, with a vectorized beam search that masks
illegal continuations at every level. Correctness is pinned by tests rather than by
inspection: any decode result must be a legal item; beam scores must equal direct scoring of
the same item; and at sufficient width the beam must return the exhaustive argmax.

**Is the Trie load-bearing?** Yes, measurably. Unconstrained greedy decoding is legal
**81.8%** of the time after training (32.8% after two epochs). So the model does learn the
code manifold, but nearly one in five of its unconstrained first choices is not an item that
exists. Constrained decoding is doing real work, not tidying up.

### The mistake I'd lead with

I originally ranked GenRec by beam search and argued the approximation "can only cost the
generative side" — supported by a beam-width sweep showing HR@10 flat from beam 20 to
beam 200. **That reasoning was wrong, and wrong in the flattering direction.**

On the same 1,500 users: beam-20 reports HR@10 0.0407, exhaustive scoring gives 0.0240, and
the **mean true rank of a beam-reported top-10 hit is 167**. Beam pruning discards 236 of
256 first codes, so the high-scoring items that should have outranked the target never enter
the returned list — the target's position in a 20-item list flatters it.

The sweep missed this because widening the beam does two opposing things at once: it finds
more true targets *and* more competitors that outrank them. They roughly cancel, so a flat
curve reads as convergence. **The sweep tested whether the beam finds the target; it never
tested whether the beam's ranking is faithful.** The right control was the cheap one I
skipped: compare against exhaustive scoring.

Every superseded number moved the same way — the overall loss is −57.8%, not the −44.6% the
beam reported. The correction is in the README with the old number still visible.

This is the third time in the project a control turned out to measure something adjacent to
what it was supposed to control (Week 3's epoch budget, Week 4's dropout default, then
this). The pattern is identical each time: cheap, plausible, and answering a slightly
different question than the one being asked.

### The result I'd defend hardest

Semantic IDs are supposed to help where atomic IDs are weakest. Prediction: GenRec loses on
the head, closes the gap on the tail. **As trained, the gap widens as items get rarer — the
exact opposite.**

| bucket (target's train frequency) | users | SASRec | GenRec | GenRec debiased α=1 |
|---|---|---|---|---|
| unseen (0) | 138 | 0.0000 | 0.0072 | **0.0725** |
| tail (1–4) | 4,594 | **0.0185** | 0.0026 | 0.0144 |
| head (20+) | 8,092 | **0.1033** | 0.0606 | 0.0138 |
| overall | 22,363 | **0.0594** | 0.0250 | 0.0118 |

Debiased (`score − α·log P_prior(item)`), the claim survives **in its narrowest form**: on
items never seen in training, GenRec retrieves 7.25% where SASRec retrieves 0.00% — 10 hits
in 138 against none, Fisher exact one-sided **p = 0.0008**. On tail items it becomes
statistically indistinguishable from SASRec (p = 0.059). Semantic IDs really can reach items
an atomic embedding table structurally cannot. The price is more than half the overall
accuracy, paid on the head where the traffic is.

**Why it collapses:** GenRec's entire output covers 14% of the catalogue (1,749 distinct
items across all top-10s) against SASRec's 76% (9,221), and the median item it recommends
has 63 training appearances against SASRec's 22.

But the popularity prior is only *part* of it, and the debiased run is what shows that.
Debiasing changes the *composition* of the recommendations enormously — head share 74.1% →
7.0%, median training frequency 63 → 5, and 11.6% of slots going to items never seen in
training. Yet coverage moves only **1,749 → 1,976**, a 13% gain on a number that needs 5x.
Debiasing slides along a fixed frontier — trading head accuracy for tail accuracy — without
making the model more discriminative. If the prior were the whole story, that 13% would have
been a 500%.

**The part I think is genuinely under-discussed:** swapping atomic for semantic IDs also
silently swaps an *unnormalized* scorer (SASRec's dot product) for a *normalized* one
(GenRec ranks by P(item | history), which contains the popularity prior). Nobody sets out to
change the scoring rule; it arrives with the architecture — the same way Week 4's dropout
arrived with the framework.

Per-level code accuracy, teacher-forced on the true prefix: **9.8% / 17.9% / 22.4% / 86.3%**.
Accuracy *rises* with depth as the prefix narrows the choice, and the content-free
disambiguation token is nearly free — so Beauty's 11.78% collision rate is not the
bottleneck it looked like. **The binding constraint is the first code.**

### How do you know any of these margins are real?

Five seeds of this repo's SASRec on ML-1M, evaluation negatives frozen so only training
noise varies:

| Metric | rel. std | comparison floor (2·√2·σ) |
|---|---|---|
| sampled HR@10 | 0.28% | **0.96%** |
| full HR@10 | 1.19% | **3.37%** |

**Full-ranking metrics are ~4x noisier than sampled ones** — separating 3,416 items is far
more sensitive to initialization than separating 101. `scripts/seed_variance.py` prints
every margin claimed in the repo against that floor.

It cost me two Week 3 conclusions. It also confirmed that the residual SASRec-vs-BERT4Rec
margin (+0.31% / +0.45%) is inside noise and should be read as a tie, and that the dropout
effect (+3.71% / +6.33%) is 4–6x the floor.

Caveat I volunteer: five seeds estimate σ loosely, and the floor is measured on *one* model,
*one* framework, *one* dataset. Applied to RecBole runs and to Beauty it is borrowed, not
measured. Every RecBole number is still a single seed.

---

## L3 — judgment

### Which paradigm should a company actually use, at what scale?

The honest framing is that these are not competing answers to one question; they are
answers to different questions about catalogue turnover.

**Use atomic-ID sequential models (SASRec and descendants) when your catalogue is stable
and your items accumulate interactions.** They're cheap, they're understood, the failure
modes are known, and an embedding table is the most expressive thing you can give an item
that has data. Most companies are here and should stop reading.

**Semantic IDs earn their keep when the cold-start bucket is a large fraction of the
business** — marketplaces with continuous listing churn, UGC, news, short-form video — where
a meaningful share of impressions are for items with near-zero interaction history. That is
exactly the regime my unseen-bucket result speaks to: 7.25% vs 0.00% is a categorical
difference, not a marginal one, because an atomic table cannot rank an untrained embedding
at all.

**What my numbers say about the trade:** on Amazon Beauty — a catalogue with plenty of
history per item — semantic IDs cost 58% of full-ranking HR@10 to buy a bucket that is 0.6%
of test users. That is a bad trade on that dataset, and I'd expect it to stay bad on any
catalogue with similar turnover. Flip the bucket sizes and the arithmetic flips with them.

**The parameter argument is real but shouldn't be oversold.** 13.7% of the parameters at 71%
of sampled HR@10 is a genuinely interesting point on the compression curve, and it's the
reason I'd call this result "negative but not uninteresting." But the comparison isn't
parameter-matched and can't be — matching would mean crippling SASRec's item table or
inflating GenRec's hidden dimension, and the compression *is* the method under test.

**What I'd actually build first if I owned this problem:** not a generative recommender.
A hybrid — atomic embeddings for items above an interaction threshold, content-derived
representations below it, with the threshold tuned on exactly the bucketed metric in my
cold-start table. My results say the generative model's advantage is confined to a bucket
that a much simpler mechanism also addresses, and its cost is spread across every other
bucket.

### What does the HSTU / generative-recommender scaling story mean for a mid-size company?

The scaling-law results are a claim about a regime: given enough interaction data and
enough compute, a generative sequence model keeps improving where a fixed-capacity
retrieval model plateaus. Both preconditions are load-bearing, and a mid-size company
usually has neither.

Three things I'd want a mid-size team to internalize:

1. **You are probably not compute-bound, you are protocol-bound.** My Week 4 result is the
   cautionary tale: a dropout default moved the headline more than the architecture choice
   it was attributed to. Before anyone funds a paradigm change, the existing comparison
   should be checked for the boring explanations — matched hyperparameters, matched budget,
   matched evaluation draw, a measured noise floor. That work costs days, not GPU-months,
   and it changed my conclusion twice.
2. **Serving cost is a first-class constraint, not an implementation detail.** Ranking
   12,101 items per request is not a serving-time option, so the demo in this repo uses
   beam search — and I can now quantify that the demo's list is *measurably more flattering*
   than the offline tables, because beam ranking is unfaithful. That gap between what you
   evaluate and what you serve is a real production problem, and it doesn't appear in the
   papers.
3. **The scaling argument is untested in this repo, and I won't pretend otherwise.** The
   training-budget curve was the signature figure in the plan and it was cut on cost. I
   have the 1x point. That's a limitation, and it's listed as one.

This is the "right tool for the problem" argument in a specific form: the generative
paradigm's advantages — cold-start reach, parameter compression, scaling headroom — are
real, and every one of them is conditional on a regime that a mid-size company should
verify it's actually in before paying the cost.

---

## Questions I'd want to be asked (and short answers)

**"Your Beauty SASRec is above the published band. Did you tune it to fit?"**
No — it's 0.5097 against a 0.4654–0.5054 target, +0.43pp *above*. I report it as-is and list
it as a limitation rather than tuning until it lands inside the band, because tuning toward
a published number is how you launder a bug into a reproduction.

**"You skipped RQ-VAE. Isn't that the actual TIGER method?"**
Yes, and that's a real gap between this and TIGER. I skipped it on evidence: it was in the
plan specifically to mitigate codebook collapse, and RQ-KMeans produced 0 dead codes on both
datasets, so the failure mode it exists to fix wasn't present. Whether RQ-VAE's *learned*
quantization would have produced more discriminative codes — separate question, untested,
and given that my binding constraint is first-code accuracy at 9.8%, it's the most promising
untested lever I have.

**"What's the single biggest hole in this project?"**
The training-budget curve. It was the signature figure and it's the one claim at the heart
of the BERT4Rec controversy that I can't speak to. Second: the +40%/+53% full-ranking
divergence between the two SASRecs is unexplained, and a loss-only ablation would probably
close it.

**"What would you do differently?"**
Run the expensive-but-definitive control first. Three times I ran a cheap control that
answered an adjacent question — beam-width instead of beam-faithfulness, epoch-matched
instead of hyperparameter-matched, sampled instead of full. Each time the cheap control said
"fine" and the definitive one reversed a conclusion. I'd now budget for the definitive
version of any control that a headline number rests on.
