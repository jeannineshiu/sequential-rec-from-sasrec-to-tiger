# Semantic ID spot check -- ml-1m

3416 items, 3 levels x 256 codes, seed 42, L2-normalized embeddings: True.

## Are prefixes semantically coherent?

Mean cosine similarity between two items drawn from the same code prefix, against two items drawn at random. Same distribution would mean the codes carry nothing.

| prefix depth | groups with >=2 items | mean within-group cosine | vs. random pair |
|---|---|---|---|
| 1 token | 256 | 0.641 | +0.202 |
| 2 tokens | 377 | 0.722 | +0.282 |
| 3 tokens | 49 | 0.753 | +0.313 |
| random pairs | -- | 0.439 | -- |

## Codebook health

| level | codes used | dead codes | median items/code | max items/code |
|---|---|---|---|---|
| 1 | 256/256 | 0 | 10 | 55 |
| 2 | 256/256 | 0 | 4 | 148 |
| 3 | 256/256 | 0 | 1 | 432 |

Collisions: **50/3416 items (1.46%)** share a full 3-token code and are separated by the disambiguation token; largest colliding group is 3, so that token needs a vocabulary of at least 3.

Residual norm 1.000 -> 0.443 (55.7% of the embedding norm explained by 3 tokens).

## Sampled depth-2 prefix groups

**prefix [2, 94]**

- G.I. Jane (1997). Genres: Action, Drama, War
- Peacemaker, The (1997). Genres: Action, Thriller, War
- Welcome To Sarajevo (1997). Genres: Drama, War
- Bent (1997). Genres: Drama, War

**prefix [2, 218]**

- Thousand Acres, A (1997). Genres: Drama
- Washington Square (1997). Genres: Drama
- Eve's Bayou (1997). Genres: Drama
- Hanging Garden, The (1997). Genres: Drama

**prefix [23, 218]**

- Journey of August King, The (1995). Genres: Drama
- Browning Version, The (1994). Genres: Drama
- Madness of King George, The (1994). Genres: Drama
- Nell (1994). Genres: Drama
- Quiz Show (1994). Genres: Drama
- Beans of Egypt, Maine, The (1994). Genres: Drama

**prefix [24, 92]**

- Heartburn (1986). Genres: Comedy, Drama
- Soul Man (1986). Genres: Comedy
- Crimes of the Heart (1986). Genres: Comedy, Drama

**prefix [25, 189]**

- Farewell My Concubine (1993). Genres: Drama, Romance
- Shadowlands (1993). Genres: Drama, Romance
- Sommersby (1993). Genres: Drama, Mystery, Romance

**prefix [26, 218]**

- Swingers (1996). Genres: Comedy, Drama
- Kolya (1996). Genres: Comedy
- Different for Girls (1996). Genres: Comedy

**prefix [39, 142]**

- Deadly Friend (1986). Genres: Horror
- Psycho III (1986). Genres: Horror, Thriller
- Rawhead Rex (1986). Genres: Horror, Thriller

**prefix [56, 224]**

- Foxfire (1996). Genres: Drama
- Substance of Fire, The (1996). Genres: Drama
- This World, Then the Fireworks (1996). Genres: Crime, Drama, Film-Noir

**prefix [115, 227]**

- Ready to Wear (Pret-A-Porter) (1994). Genres: Comedy
- Crooklyn (1994). Genres: Comedy
- Maverick (1994). Genres: Action, Comedy, Western
- Clean Slate (1994). Genres: Comedy

**prefix [151, 40]**

- Alaska (1996). Genres: Adventure, Children's
- Adventures of Pinocchio, The (1996). Genres: Adventure, Children's
- Incredible Journey, The (1963). Genres: Adventure, Children's

