# Three Times I Ran the Wrong Control

*What six weeks of reproducing sequential recommenders — SASRec, BERT4Rec, and a
TIGER-style generative model — taught me about the difference between a check that
passes and a check that means something.*

---

I set out to do something unglamorous: reproduce SASRec (2018) from scratch, independently
check the BERT4Rec reproducibility controversy, and then rebuild the same codebase as a
TIGER-style generative recommender with quantized semantic IDs. Seven years of sequential
recommendation in one repo, with every number reproducible from a script.

The headline results are fine. SASRec landed inside the paper's range on the first full
training run. The generative model lost. Those are both in the README.

But the thing I actually learned isn't in either result. It's that **three separate times,
I ran a control that was cheap, plausible, and answered a slightly different question than
the one I was asking** — and each time the control said "you're fine" while the real answer
was "you've reversed a conclusion." Every one of those was caught by a later, more
expensive check that I nearly didn't run.

This post is about those three.

---

## First, the boring part that went right

SASRec is a causal transformer over a user's item history: position *t* predicts item
*t+1*, so a single forward pass over a length-*n* sequence gives you *n* training examples.
The implementation is not hard. Getting it *right* has a handful of known footguns, and I
wrote five unit tests targeting them before training anything:

- **Causal mask direction.** Changing the last token of a sequence must not change the
  encoder output at any earlier position. If your mask is inverted, position *t* attends to
  item *t+1*, the model learns to copy the target, and your training metric goes to ~1.0
  while test goes to chance.
- **Padding excluded from attention**, tested against `input_seqs == 0` directly.
- **Positional embedding range**, tested with a full-length no-padding sequence so the
  indexing doesn't throw at exactly the boundary.
- **Candidate-subset scoring and full-catalogue scoring agree numerically** on overlapping
  items. If they don't, your sampled and full-ranking numbers for the *same model* aren't
  comparable, and you won't notice until you try to explain a gap between them.

Then the full run, ML-1M, 200 epochs:

| Metric | Paper (Kang & McAuley 2018) | This repo |
|---|---|---|
| sampled HR@10 | 0.80–0.83 | **0.8190** |
| sampled NDCG@10 | 0.57–0.60 | **0.5948** |

Inside the range on the first attempt, no debugging iteration. I'd budgeted two days for an
"align with the paper" slog and didn't need them. The five tests cost about an hour.

That's the last thing in this post that went according to plan.

---

## Control #1: I matched the evaluation protocol with great care and never checked the model

The BERT4Rec reproducibility controversy, roughly: BERT4Rec reports large wins over SASRec,
and a line of work (Petrov & Macdonald, 2022, most prominently) argues those wins shrink or
vanish once you control for training budget — BERT4Rec's cloze objective supervises only
~20% of positions per sequence against SASRec's 100%, so it needs far more epochs to
converge, and the original comparisons didn't give it to it.

So I set up a comparison and was *scrupulous* about the evaluation protocol. Same
leave-one-out split. Same 5-core filtering. Same frozen negatives file, generated once at
seed 42 and reused by every model. Same metric code. I ran both models through RecBole at
200 epochs and got:

**BERT4Rec beats RecBole's SASRec by +3.39% HR@10 / +5.86% NDCG@10.**

Which contradicted my own from-scratch SASRec, which BERT4Rec merely tied. Two SASRecs,
same protocol, different stories. So I went looking at the configs.

RecBole's SASRec and BERT4Rec property files are **identical on every architectural
default** — 2 layers, 2 heads, hidden 64, inner size 256, cross-entropy loss — except one
line. SASRec gets dropout 0.5. BERT4Rec gets 0.2.

I changed that one value and nothing else:

| Comparison (same protocol, same budget) | HR@10 | NDCG@10 | Winner |
|---|---|---|---|
| BERT4Rec vs. SASRec at **dropout 0.5** (RecBole's default) | +3.39% | +5.86% | BERT4Rec, on both |
| BERT4Rec vs. **my** SASRec | −1.94% | +1.48% | a tie |
| BERT4Rec vs. SASRec at **dropout 0.2** | −0.31% | −0.45% | SASRec, on both |
| **effect of the dropout default alone** | **+3.71%** | **+6.33%** | — |

**The dropout default is worth more than the entire margin it was supposed to explain.**
Same BERT4Rec run. Three different conclusions. Decided by a value nobody chose
deliberately for this comparison — it's just what shipped in the framework's config file.

I want to be precise about what went wrong here, because it's subtle. My control was
"match the evaluation protocol across models." That control *passed*. It was also
answering the wrong question. The thing that determines whether an architecture comparison
means anything isn't only whether the models are *evaluated* the same way — it's whether
they're *configured* comparably. I checked one and assumed the other, which is precisely
the failure mode the reproducibility literature describes. I reproduced it by accident and
then diagnosed it.

---

## Control #2: my ablation baseline was silently 100 epochs richer

Here's a smaller one, which I include because it's the kind of thing that never makes it
into a paper.

I ran four ablations on ML-1M — positional embedding variants, sequence length, negative
sampling strategy — at 100 epochs each, to save compute. Then I compared each against the
headline number.

The headline number was trained for **200** epochs.

So every ablation was being charged not just for its ablation but for 100 fewer epochs of
training. On sampled HR@10 that budget effect is small (+0.47%) and the table looked fine.
On full ranking it's +5.36%, and correcting it **reversed two conclusions**: dropping
positional embeddings entirely, and halving the sequence length to 100, both looked like
real full-ranking regressions (−7.43% / −5.21%) and are in fact inside noise.

Read literally, after the correction: **maxlen 100 and maxlen 200 are indistinguishable on
full ranking at this budget**, at less than half the per-epoch cost. That's a genuinely
useful finding that the uncorrected table was hiding.

### The thing that made "inside noise" mean something

I could only say "inside noise" because I'd measured the noise. Five seeds of my SASRec on
ML-1M, varying *only* the training seed — weight init and the training negative sampler —
with the evaluation negatives frozen so this is training noise alone:

| Metric | rel. std | comparison floor (2·√2·σ) |
|---|---|---|
| sampled HR@10 | 0.28% | **0.96%** |
| full HR@10 | 1.19% | **3.37%** |

**Full-ranking metrics are ~4x noisier than sampled ones.** That makes sense in hindsight —
separating 3,416 items is far more sensitive to initialization than separating 101 — but I
had been reading full-ranking deltas as if they were as trustworthy as sampled ones.

Measuring the floor cost me two Week 3 conclusions and confirmed several others. It's
maybe four hours of compute. I'd now consider it non-optional before claiming any margin,
and I'd put the number in the README so readers can check my claims against it. (I did:
`scripts/seed_variance.py` prints every margin the repo claims against the floor.)

---

## The part where sampled metrics stop being a footnote

Everyone cites Krichene & Rendle (2020) — ranking one positive against ~100 sampled
negatives isn't a monotone proxy for ranking against the full catalogue — and then everyone
reports sampled metrics anyway, because that's what the baselines used.

I got a clean instance of *how bad* it can be, for free, because I had two independent
SASRec implementations.

**They agree to +0.61% on sampled HR@10.** By any reasonable standard, that's a successful
cross-validation — it was literally my correctness criterion for the reproduction.

**On full ranking they differ by +40% HR@10 / +53% NDCG@10.**

Same data. Same split. Same protocol shape. The divergence grows monotonically with how
much the metric cares about *where* in the ranking the target lands. The obvious suspect is
the training objective — RecBole trains cross-entropy over the full catalogue, mine trains
BCE against one sampled negative per position, following the original paper — and the
pattern is consistent with that. I didn't run the loss-only ablation, so I can't claim it.
It remains the largest unexplained effect in the project, and it's listed as such.

The lesson I'd actually transfer: **agreement under the sampled protocol is not agreement.**
If your cross-implementation validation only checks sampled metrics, it can pass while the
two implementations disagree by 40% on the thing you'd actually deploy.

---

## Semantic IDs, and a negative result I believe

Then the fun part. TIGER-style generative recommendation replaces the atomic item ID with a
short sequence of discrete codes derived from the item's content embedding, and generates
the next item token by token.

The pitch is compelling. An atomic embedding table gives every item a free parameter
vector — which means an item seen twice in training has a vector trained on two gradient
steps, and an item seen zero times has a vector sitting at its initialization, structurally
unrankable. Semantic IDs assemble a representation from codes that thousands of other items
share, so a brand-new item isn't a blank.

My pipeline: item text (title + category path + brand for Amazon Beauty) → `all-MiniLM-L6-v2`
→ residual KMeans, 3 levels × 256 codes, plus a 4th token to disambiguate items landing on
an identical 3-token code.

I'd planned RQ-VAE as a stretch goal specifically to mitigate codebook collapse. **RQ-KMeans
produced 0 dead codes on both datasets**, so there was no collapse to fix and I dropped the
stretch goal rather than performing it. KMeans initializes from data and gives every
centroid a non-empty cluster by construction; collapse is a gradient-descent-on-a-codebook
problem, not an inherent property of residual quantization.

Two things the codes learned that I didn't design:

- **Amazon Beauty collides at 11.78%.** For roughly one item in eight, the only thing
  separating it from a catalogue neighbour is a token carrying no content signal at all.
- **ML-1M's codes encode release year at least as strongly as genre.** Items sharing a
  2-token prefix are 1.68 years apart on average against a 15.85-year baseline — because
  MovieLens titles have the year embedded in the string. The text format picked a
  clustering axis for me.

Then I trained the generative model on **the same backbone** as SASRec — same hidden dim,
depth, heads, dropout, optimizer, budget — so the only variable is how an item is
represented. And it lost:

| Amazon Beauty, test, k=10 | sampled HR@10 | full HR@10 | parameters |
|---|---|---|---|
| SASRec (atomic) | **0.5097** | **0.0594** | 828,352 |
| GenRec (semantic) | 0.3621 | 0.0250 | **113,472** |
| relative | −28.96% | −57.8% | **13.7%** |

That parameter column is what makes it interesting rather than merely negative. 12,101 item
embeddings collapse into 782 token embeddings — a 15.5x smaller table — so the generative
model gives up 29% of sampled HR@10 while running on an eighth of the parameters. That
trade can't be controlled away: matching parameter counts would mean crippling SASRec's item
table or inflating GenRec's hidden dimension, and the compression *is* the method under
test.

### And the cold-start hypothesis failed in the direction it was supposed to win

Semantic IDs should help precisely where atomic IDs are weakest. Clear prediction: GenRec
loses on the head, closes the gap on the tail.

I bucketed every test user by how often the target item appeared in training:

| bucket | users | SASRec | GenRec |
|---|---|---|---|
| unseen (0) | 138 | 0.0000 | 0.0072 |
| tail (1–4) | 4,594 | **0.0185** | 0.0026 |
| torso (5–19) | 9,539 | **0.0427** | 0.0060 |
| head (20+) | 8,092 | **0.1033** | 0.0606 |

**The gap widens as items get rarer.** The opposite of the prediction, and by a lot — 86%
worse on tail, 41% worse on head.

So I looked at what the model was actually recommending, and found the mechanism:

| model | distinct items across all top-10s | median train freq | % head |
|---|---|---|---|
| SASRec (atomic) | **9,221** (76% of catalogue) | 22 | 54.2% |
| GenRec (semantic) | **1,749** (14%) | 63 | 74.1% |

Generation had collapsed onto a small set of high-probability code sequences, and the median
item it recommends is 3x more popular than SASRec's. The tail result is a symptom of that
collapse, not of anything about semantic IDs specifically.

Which suggested a fix worth testing: rank by `log P(item | history) − α · log P_prior(item)`,
subtracting out the training-frequency prior. Same model, same weights, different scoring
rule. At α=1:

**On items never seen in training, the debiased generative model retrieves 7.25% at HR@10
where SASRec retrieves 0.00%.** Ten hits in 138 against none — Fisher exact, one-sided,
**p = 0.0008**. On tail items it becomes statistically indistinguishable from SASRec
(p = 0.059).

So the cold-start claim survives, **in its narrowest form**. Semantic IDs really can reach
items an atomic embedding table structurally cannot — that's a categorical difference, not
a marginal one, because an untrained embedding can't be ranked at all. The price is more
than half the overall accuracy, paid on the head where the traffic is.

But the mechanism I proposed is only *partly* right, and I want to flag that rather than
declare victory. If the popularity prior were the whole story, removing it would restore
diversity — and it doesn't.

Debiasing changes *what* gets recommended enormously. The head share of recommended items
falls from 74.1% to 7.0%. The median recommended item goes from 63 training appearances to 5.
11.6% of slots go to items never seen in training at all. The scoring rule is doing exactly
what I asked it to do.

And yet **coverage moves only 1,749 → 1,976 out of 12,101** — a 13% gain on a number that
needs to grow 5x. Debiasing slides the model along a fixed frontier, trading head accuracy for
tail accuracy, without making it more *discriminative*. **Something about training a model to
emit four codes leaves it with far less resolution over the catalogue than 12,101 free
embeddings have**, and that isn't a scoring-time artifact a better decoder repairs.

---

## Control #3: my beam search was flattering the model I was arguing against

Here's the one that stings, and the reason this post is titled the way it is.

The generative model can't be scored by dot product — you have to decode. I used constrained
beam search over a Trie of legal code sequences (beam 20), and I was appropriately worried
that the beam was costing the generative model score. So I ran a control: **sweep the beam
width from 20 to 200 and see where the metric stops moving.**

It was flat. Completely flat. I wrote, in the README, that the beam approximation "can only
cost the generative side."

That sentence is false, and false in the direction that mattered. Later — after building a
history KV cache that made it affordable — I scored **every catalogue item for every user**,
exhaustively, and compared on the same 1,500 users:

| | HR@10 |
|---|---|
| beam 20 | 0.0407 |
| exhaustive | **0.0240** |

Of the 61 users where beam reported a top-10 hit, exhaustive agreed on 27. And the **mean
true rank of a beam-reported top-10 hit was 167.**

The mechanism is pruning at level 1. The beam keeps 20 of 256 first codes, so every
high-scoring item sitting behind a discarded prefix vanishes from the returned list. Those
vanished items are exactly the *competitors* that should have pushed the target down. In a
20-item list assembled from a pruned space, the target looks far better than it is.

**Why the sweep didn't catch it:** widening the beam does two opposing things simultaneously.
It finds more true targets (helps HR@10) and it finds more competitors that outrank them
(hurts HR@10). They roughly cancel. A flat curve reads as "converged" when it's really two
biases in balance.

My control tested whether the beam **finds** the target. It never tested whether the beam's
**ranking** is faithful. Those are different questions, and only one of them was the one I
needed answered.

Every superseded number moved the same way: the overall loss is −57.8%, not the −44.6% the
beam reported. Correcting it made my negative result *more* negative. The old numbers are
still in the README, next to the correction.

And there's a mechanism here worth generalizing beyond beam search. GenRec ranks by
P(item | history) — a normalized probability, which *contains* the popularity prior. SASRec
ranks by an unnormalized dot product, which carries no such prior. **Swapping atomic IDs for
semantic IDs also silently swaps an unnormalized scorer for a normalized one**, and that
second change — which nobody sets out to make — is doing much of the damage. It arrived with
the architecture the same way the dropout default arrived with the framework.

---

## The checklist I wish I'd started with

If you're reproducing or comparing sequential recommenders, or honestly any pair of models:

1. **Diff the configs, not just the protocol.** Before attributing a gap to architecture,
   confirm the two models are configured comparably. Framework defaults are per-model and
   are not chosen for your comparison. One dropout value moved my headline more than the
   architecture it was credited to.
2. **Measure your noise floor before claiming any margin.** A handful of seeds, evaluation
   held fixed. My full-ranking metrics were 4x noisier than my sampled ones; I'd been
   reading both as equally trustworthy. Publish the floor so readers can check you.
3. **Never difference numbers from different negative draws.** I quantified my own: same
   predictions, two negative draws, +2.28% HR@10 / +5.38% NDCG@10 apart — the same order as
   every margin under discussion.
4. **Report sampled and full ranking side by side, always.** Two implementations agreed to
   0.61% on one and differed by 40% on the other. Whichever you report alone, you're
   reporting the flattering one by accident.
5. **Make sure both sides of every delta share a budget.** Including epochs. Especially when
   you cut the budget on one side to save compute and forget you did.
6. **When you approximate a ranking, validate against the exact ranking — not against a
   wider approximation.** Sweeping a knob on an approximation tells you the approximation is
   stable. It cannot tell you it's unbiased. This is the one I'd tattoo somewhere.
7. **Ask what your control would look like if it were failing.** All three of mine looked
   identical whether they were working or not. That's the tell.

---

## What I'd tell you the project actually shows

Three claims, in decreasing order of confidence:

1. **BERT4Rec's win over SASRec on ML-1M is a baseline-configuration artifact.** One
   framework default, worth +3.71% / +6.33%, decides the comparison.
2. **Sampled metrics can hide a 40% implementation difference.** Mine did.
3. **On Amazon Beauty, semantic IDs cost 58% of full-ranking accuracy to buy a bucket that
   is 0.6% of test users.** That's a bad trade on this catalogue. Flip the bucket sizes —
   marketplaces with continuous listing churn, UGC, news — and the arithmetic flips too. The
   unseen-bucket result is real and categorical; it's just small here.

And the honest gaps, which are in the README's limitations section: the training-budget
curve at the heart of the BERT4Rec controversy was cut on GPU cost, so I have only the 1x
point and can't speak to the scaling claim. The 40%/53% full-ranking divergence is
unexplained. Every RecBole number is a single seed, using a noise floor borrowed from a
different model on a different dataset.

The negative result isn't the interesting part. **The interesting part is that I got the
answer wrong three times with controls that passed**, and what saved it each time was
running the expensive check that the cheap one was standing in for.

---

*Code, every number, and the full debugging trail (including the wrong versions):
[github.com/…/from-sasrec-to-tiger](https://github.com/). The reproduction log is written
as it happened — hypothesis, change, result, next step — which means the mistakes above are
in there with timestamps.*
