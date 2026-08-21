# LinkedIn post

Short version of `medium-draft.md`. Target ~1,300 characters before the link so it doesn't
truncate in-feed. Pick one of the three variants below.

---

## Variant A — the dropout hook (recommended)

I spent six weeks reproducing sequential recommenders — SASRec, BERT4Rec, and a TIGER-style
generative model. The result I did not expect:

**BERT4Rec's win over SASRec on MovieLens-1M is decided by a dropout default nobody chose.**

RecBole's SASRec and BERT4Rec configs are identical on every architectural default — same
depth, heads, hidden size, loss — except one line. SASRec gets dropout 0.5, BERT4Rec gets
0.2. Changing that one value and nothing else:

→ at dropout 0.5: BERT4Rec wins by +3.4% HR@10
→ at dropout 0.2: SASRec wins
→ effect of the default alone: +3.7% HR@10 / +6.3% NDCG@10

The default is worth more than the entire margin it was supposed to explain. Same BERT4Rec
run, three different conclusions.

I walked into this by matching the *evaluation* protocol across models with great care and
never checking that the *model* hyperparameters were comparable — which is exactly the
failure mode the reproducibility literature describes. I reproduced it by accident, then
diagnosed it.

Two more things I'd hand to anyone comparing models:

• Two SASRec implementations agreed to 0.6% on sampled HR@10 and differed by 40% on full
ranking. Agreement under the sampled protocol is not agreement.

• I later found my own beam search was inflating the generative model's scores — the
beam-width sweep I ran to check for exactly that had passed, because it tested whether the
beam *finds* the target, never whether its *ranking* is faithful.

Three controls. All cheap, all plausible, all answering a slightly different question than
the one I was asking. Full write-up, code, and the wrong versions of every number:

[link]

#RecSys #MachineLearning #Reproducibility

---

## Variant B — the negative-result hook

My generative recommender lost to a 2018 baseline. Here's why I published it anyway.

I rebuilt SASRec as a TIGER-style generative model: instead of one embedding per item,
each item becomes 4 discrete codes quantized from its content embedding. Same backbone,
same protocol, same training budget — the only variable is how an item is represented.

On Amazon Beauty it lost 29% of sampled HR@10.

It also ran on **13.7% of the parameters**. 12,101 item embeddings collapse into 782 token
embeddings.

And the cold-start hypothesis — semantic IDs should help most where atomic IDs are weakest —
failed in the direction it was supposed to win. The gap got *wider* as items got rarer.

Then I found the mechanism. The generative model's entire output covered 14% of the
catalogue against SASRec's 76%; generation had collapsed onto a handful of high-probability
code sequences. Correcting for the popularity prior at ranking time recovered the claim in
its narrowest form: **on items never seen in training, 7.25% HR@10 vs SASRec's 0.00%**
(10 hits in 138 vs none, Fisher exact p = 0.0008).

So semantic IDs really can reach items an atomic embedding table structurally cannot. On
this catalogue that's 0.6% of users, and it costs more than half the overall accuracy.
Flip the bucket sizes — marketplaces, UGC, news — and the arithmetic flips with it.

A negative result with a measured mechanism is worth more than a positive one you can't
explain.

[link]

#RecSys #MachineLearning

---

## Variant C — short, for a comment or repost

Three times in one project I ran a control that passed while the thing it was controlling
for was actively breaking my conclusion:

1. Matched the evaluation protocol across two models, never diffed their configs — a
dropout default was worth more than the architecture gap it was credited to.
2. Ran ablations at 100 epochs against a 200-epoch baseline — charging every ablation for
100 missing epochs reversed two conclusions.
3. Swept beam width to check whether my beam search was costing the generative model, and
it was flat — because widening a beam finds more targets *and* more competitors, which
cancel. It was inflating scores, not costing them.

Each control was cheap, plausible, and answered a slightly different question than the one
I was asking. The tell, in hindsight: all three would have looked identical whether they
were working or not.

[link]
