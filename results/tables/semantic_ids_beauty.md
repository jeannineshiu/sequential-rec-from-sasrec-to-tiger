# Semantic ID spot check -- beauty

12101 items, 3 levels x 256 codes, seed 42, L2-normalized embeddings: True.

## Are prefixes semantically coherent?

Mean cosine similarity between two items drawn from the same code prefix, against two items drawn at random. Same distribution would mean the codes carry nothing.

| prefix depth | groups with >=2 items | mean within-group cosine | vs. random pair |
|---|---|---|---|
| 1 token | 256 | 0.605 | +0.318 |
| 2 tokens | 2487 | 0.738 | +0.451 |
| 3 tokens | 1054 | 0.871 | +0.583 |
| random pairs | -- | 0.288 | -- |

## Codebook health

| level | codes used | dead codes | median items/code | max items/code |
|---|---|---|---|---|
| 1 | 256/256 | 0 | 40 | 179 |
| 2 | 256/256 | 0 | 22 | 337 |
| 3 | 256/256 | 0 | 9 | 380 |

Collisions: **1426/12101 items (11.78%)** share a full 3-token code and are separated by the disambiguation token; largest colliding group is 12, so that token needs a vocabulary of at least 12.

Residual norm 1.000 -> 0.513 (48.7% of the embedding norm explained by 3 tokens).

## Sampled depth-2 prefix groups

**prefix [7, 219]**

- 5 Second Brush On Nail Glue .2 fl oz (6 g). Category: Makeup > Nails > Nail Art. Brand: A.I.I. CLUBMAN
- Cutex Regular Jar Twist &amp; Scrub Sponge 7 oz.. Category: Makeup > Nails > Nail Polish Remover. Brand: Cutex
- SuperNail Brush Cleaner - 2oz / 59ml. Category: Tools & Accessories > Makeup Brushes & Tools > Brushes & Applicators. Brand: Super Nail

**prefix [9, 39]**

- Super Solano Professional Hair Dryer - Black. Category: Hair Care > Styling Tools > Hair Dryers. Brand: Solano
- FHI Heat Nano Weight Pro 1900 Turbo Hair Dryer, Black. Category: Hair Care > Styling Tools > Hair Dryers. Brand: FHI
- HOT TOOLS 1023 Ionic Turbo Dryer, Black. Category: Hair Care > Styling Tools > Hair Dryers. Brand: Hot Tools
- Bio Ionic Power Light Dryer - Color: Black. Category: Hair Care > Styling Tools > Hair Dryers. Brand: Bio Ionic

**prefix [11, 228]**

- Joico - *Save 27%* Color Endure Violet Shampoo/Conditioner Duo (10.1 oz). Category: Hair Care > Shampoos. Brand: Joico
- Joico - Moisture Recovery Shampoo and Conditioner Liter Duo Set(33.8oz). Category: Hair Care > Shampoo & Conditioner Sets. Brand: Joico
- Joico Moisture Recovery Shampoo/Conditioner Duo 10.1 Oz. Bottles. Category: Hair Care > Shampoo & Conditioner Sets. Brand: Joico
- Joico K Pak Reconstruct Repair Damage Shampoo &amp; Conditioner Duo 33.8 oz. Category: Hair Care > Shampoo & Conditioner Sets. Brand: Joico
- Joico K-pak Shampoo and Conditioner Liter Duo 33.8 oz Set. Category: Hair Care > Shampoo & Conditioner Sets. Brand: Joico
- Joico K-pak Color Therapy Shampoo &amp; Conditioner (10.1 Oz). Category: Hair Care > Shampoos. Brand: Joico K-pak Color Therapy Shampoo &amp; Conditio

**prefix [33, 136]**

- China Glaze LUBU HEELS 77064. Category: Makeup > Nails > Nail Polish. Brand: China Glaze
- The Wizard of Ohh Ahhz Retuns 6 Pieces / China Glaze / Nail Polish / Laquer / Enamel. Category: Makeup > Nails > Nail Polish
- China Glaze up &amp; Away Collection: Re-fresh Mint #867/80937. Category: Makeup > Nails > Nail Polish. Brand: China Glaze
- China Glaze up &amp; Away Collection: Light As Air #863/80933. Category: Makeup > Nails > Nail Polish. Brand: China Glaze
- China Glaze up &amp; Away Collection: High Hopes #869/80939. Category: Makeup > Nails > Nail Polish. Brand: China Glaze
- China Glaze Four Leaf Clover 80936 [Health and Beauty]. Category: Makeup > Nails > Nail Polish. Brand: China Glaze

**prefix [50, 4]**

- Sigma F80 - Flat Kabuki TM. Category: Tools & Accessories > Makeup Brushes & Tools > Brushes & Applicators. Brand: Sigma Beauty
- Sigma F82 - Round Kabuki TM. Category: Tools & Accessories > Makeup Brushes & Tools > Brushes & Applicators. Brand: Sigma Beauty
- Sigma F84 - Angled Kabuki TM. Category: Tools & Accessories > Makeup Brushes & Tools > Brushes & Applicators. Brand: Sigma Beauty
- Sigma F86 - Tapered Kabuki TM. Category: Tools & Accessories > Makeup Brushes & Tools > Brushes & Applicators. Brand: Sigma Beauty
- Sigma Synthetic Kabuki Kit 4 Brushes. Category: Tools & Accessories > Makeup Brushes & Tools > Brushes & Applicators
- Sigma F88 - Flat Angled Kabuki TM. Category: Tools & Accessories > Makeup Brushes & Tools > Brushes & Applicators. Brand: Sigma Beauty

**prefix [71, 59]**

- CoverGirl Perfect Point Plus Self-Sharpening Eye Pencil, Espresso 210 - 1 ea. Category: Makeup > Eyes > Liner & Shadow Combinations. Brand: COVERGIRL
- CoverGirl Perfect Blend Pencil Mink(W) 115, 1 Count. Category: Makeup > Eyes > Eyeliner. Brand: COVERGIRL
- Pixi Eye Bright Liner, No.1 Nude. Category: Makeup > Eyes > Eyeliner. Brand: Pixi Beauty

**prefix [89, 177]**

- Designer Skin BombShell, 100XXBronzer, 13.5-Ounce Bottle. Category: Skin Care > Sun > Tanning Oils. Brand: Designer Skin
- Designer Skin Bellezza, 13.5-Ounce Bottle. Category: Skin Care > Sun > Self-Tanners & Bronzers. Brand: Designer Skin
- Designer Skin BombShell, 100XXBronzer, 13.5-Ounce Bottle. Category: Skin Care > Sun > Tanning Oils. Brand: Designer Skin
- Designer Skin Black, 13.5-Ounce Bottle. Category: Skin Care > Sun > Self-Tanners & Bronzers. Brand: Designer Skin
- Designer Skin Phoenician, 13.5-Ounce Bottle. Category: Skin Care > Sun > Tanning Oils. Brand: Designer Skin
- Designer Skin Secret Rapture, 13.5-Ounce Bottle. Category: Skin Care > Sun > Tanning Oils

**prefix [89, 252]**

- Body Drench TAN FX Tanning Accelerator - 10 oz.. Category: Skin Care > Sun > Self-Tanners & Bronzers. Brand: Body Drench
- Body Drench Quick Tan Sunless Tanning Spray. Category: Skin Care > Sun > Self-Tanners & Bronzers. Brand: Body Drench
- Body Drench Quick Tan Mist. Category: Skin Care > Sun > Self-Tanners & Bronzers. Brand: Body Drench
- Body Drench Quick Tan * 3 - Pack * Instant Self-tanning Spray * 6 Oz Can NEW PACKAGING. Category: Skin Care > Sun > Self-Tanners & Bronzers. Brand: Bo
- Body Drench Quick Tan Instant Self Tanner Bronzing Spray, Medium/Dark, 6 Ounce. Category: Skin Care > Body > Moisturizers > Lotions. Brand: Body Drenc
- Body Drench Spray Quick Tan Tanning Mist Sunless Self Tanner 3 Pack. Category: Skin Care > Sun > Self-Tanners & Bronzers. Brand: Body Drench

**prefix [153, 245]**

- Liquid Push Down Alcohol Dispenser- Clear Bottle- Labeled - 9 Oz Bottle. Category: Bath & Body > Sets. Brand: Tech-Med
- Soft 'N Style Clear Spray Bottle 16 oz. (Pack of 6). Category: Tools & Accessories > Hair Coloring Tools > Applicator Bottles. Brand: Soft &#39;N Styl
- Debra Lynn Professional 4 oz. Clear Pump Dispenser Bottle. Category: Tools & Accessories > Bags & Cases > Refillable Containers. Brand: Debra Lynn Pro

**prefix [218, 50]**

- Skin Obsession Jessner's Chemical Peel Kit Anti-aging and Anti-acne Skin Care Treatment. Category: Skin Care > Face > Sets & Kits
- The Regimen - Complete acne treatment kit from Acne.org (Travel Kit - 4-5 days). Category: Skin Care > Face > Treatments & Masks
- Exposed Acne Treatment - Basic Kit. Category: Skin Care > Face > Treatments & Masks

