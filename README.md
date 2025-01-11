Goal: Show that discourse on Bluesky has a trajectory within embeddings space

Steps:
- Embed one day of tweets using Alibaba-NLP/gte-Qwen2-7B-instruct
- Cluster embeddings
- Look at a 1 hour sliding window (sliding forward by 10 minutes). Pick a centroid and find the dimension of movement (in 8000 dimensional space) over a time period. Use PCA to reduce dimensionality and plot embeddings across that dimension of movement. Use something like plotly to make an animation that shows the
- Why is this useful? **If** discourse can be characterized as a trajectory, you can predict discourse using existing trajectory forecasting methods.

Questions:
- Try weighing likes x replies when clustering?
- Filter out tweets that don't get some amount of likes or replies?

Notes:
- A lot of tweets also don't gain that much engagement. Rather than only weighting like x replies when trying to find a centroid, maybe I can try filtering tweets out if they get too few likes and replies. Maybe this would result in less noise?
- There's probably some ideal region + time period that shows a cluster moving strongly in a specific direction. Should I write a script to do that (later) so that I have a strong example to show?
- I could also try looking at how *all* the tweets move. This would be done by applying PCA across all embeddings, then plotting embeddings for each time slice. 
- Existing embedding models might be fundamentally flawed for this kind of task. For example, if pro and anti-israel (relating to the current israel-palestine conflict) are embedded very far apart, clustering would make it look like two separate discussions.