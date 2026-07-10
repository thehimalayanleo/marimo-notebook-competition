# Submission Kit

## Title

Can We Replace the First-Token Attention Sink?

## Short Description

This interactive notebook reproduces the attention-sink and over-mixing intuition from "Why do LLMs attend to the first token?" It then tests a new hypothesis across SmolLM and Qwen: distribute sink attention over a small causal bank to reduce single-token concentration while retaining containment.

## Main Finding

A four-position causal sink bank retained 98.0%-99.7% of the attention-rollout containment in local GPU tests spanning 135M to 1.5B parameters. This is an attention-surgery feasibility result, not evidence that an untrained intervention preserves language-model quality.

## Five-Minute Video Script

**0:00-0:30 - The puzzle**

*Show the title, then scroll to the first attention map.*

"Why do language models spend so much attention on the first token, even when that token carries almost no useful meaning? This bright first column is called an attention sink. My notebook reproduces the explanation from the paper *Why do LLMs attend to the first token?*, then asks a new question: can we replace one extreme sink with a small bank of routing positions?"

**0:30-1:05 - The paper's explanation**

*Open the 'Why this can happen' accordion.*

"A Transformer repeatedly mixes information between tokens. Across many layers, too much mixing can blur distinct representations or let a small perturbation spread everywhere. Position zero is special because every later token can see it under causal attention. The paper argues that the model can use this shared position as a low-information outlet: attention can go somewhere harmless instead of being forced onto meaningful content tokens."

**1:05-1:35 - Start the experiment**

*Select SmolLM2-135M, keep the packed-documents prompt, and click 'Run selected experiment'.*

"The app lets us change the model, prompt shape, context length, and the paper's sink threshold epsilon. Expensive computation waits behind this run button, so the app opens quickly. This run uses SmolLM2-135M. The runtime panel confirms the active accelerator, and every plot below reacts to the selected settings."

**1:35-2:10 - Detect the sink**

*Show section 1 and the current-run summary.*

"Each cell in this heatmap is one attention head. Brighter cells send more average attention to position zero. The summary also reports the paper-style metric: the percentage of heads whose sink mass exceeds epsilon. The pattern is not uniform. Sink behavior is concentrated in particular layers and heads, which we can inspect later in the optional head inspector."

**2:10-2:55 - Why the sink may be useful**

*Show the perturbation plot, followed by the rollout counterfactual.*

"Detection alone does not explain why the sink exists. Here I replace one token safely inside the sequence and measure how much the final hidden representations change. I compare the normal sequence with a version where the first token is removed while every remaining token keeps its original position ID. Deletion still changes the causal graph, so the next plot performs a cleaner fixed-length counterfactual: it removes only attention to position zero, renormalizes each row, and multiplies attention through depth. When the no-sink curve spreads farther, the first-token route is containing mixing."

**2:55-3:25 - Context length**

*Briefly show section 3 and enable the context sweep if it is already cached.*

"The paper predicts stronger pressure to control mixing as context length and model scale grow. This optional sweep keeps the trained model fixed and increases the visible context. It is not a reproduction of pretraining, but it lets us compare sink strength and the extra rollout spread caused by removing position zero at several lengths."

**3:25-4:15 - Our extension: a distributed sink bank**

*Move the distributed-bank slider from 2 to 4, then show both tradeoff bars.*

"Now for the extension. Do we actually need one token to absorb nearly all of this routing attention? I take the mass sent to position zero and distribute it over the first K causally available positions. The intervention never gives a query access to a future token. We judge it on two axes: how much over-mixing containment survives, and how much concentration is removed from position zero. At K equals four, the bank sharply reduces single-token concentration while retaining nearly all of the rollout containment in this run."

**4:15-4:40 - Does it survive larger models?**

*Show section 5's SmolLM and Qwen comparison.*

"The reference study repeats the same idea across SmolLM and Qwen checkpoints from 135 million to 1.5 billion parameters. The four-position bank retained between 98.0 and 99.7 percent of the rollout containment in these tests. The live scaling checkbox reruns the comparison on the current accelerator instead of relying only on recorded values."

**4:40-5:00 - Honest conclusion**

*Finish on the final model-design takeaway.*

"This does not prove that attention surgery improves language-model quality. It is a feasibility result. The next experiment is to pretrain K learned routing embeddings, exclude them from prediction targets, balance attention across the bank, and evaluate perplexity, long-context quality, and exact perturbation Jacobians. The paper explains why sinks form; this notebook turns that explanation into an architectural hypothesis we can test."

## 90-Second Video Script

**0:00-0:10 - Hook**

"Why does so much Transformer attention go to the very first token, even when that token carries little meaning? This bright column is an attention sink."

Show section 1; the controls open on the strongest sink head, then select a nearby head for comparison.

**0:10-0:28 - Paper result**

"The paper argues that the sink is useful. Repeated attention can over-mix token representations, so the model learns a shared low-information routing target. Here we reproduce the paper-style sink metric and inspect it head by head."

Change epsilon once and select the strongest head.

**0:28-0:45 - Perturbation**

"Now change one token. With the first-token route available, the change stays more contained. Remove attention to position zero and its influence spreads farther through the model."

Show the perturbation and rollout plots.

**0:45-1:08 - Extension**

"But do we need one extreme sink? My extension redistributes that attention over the first K causally available positions. The key comparison measures two things at once: how much containment survives, and how much concentration at position zero disappears."

Move the bank-size slider from 2 to 4 or 8.

**1:08-1:22 - Scale test**

"One click repeats the intervention from SmolLM2-135M through Qwen2.5-1.5B on the GPU. The result is not tied to one tiny checkpoint or one model family."

Enable the 135M-1.5B scaling study and show the paired bars.

**1:22-1:30 - Close**

"Attention surgery is only a feasibility check. The next experiment is to pretrain learned sink banks and test perplexity, long-context quality, and exact perturbation Jacobians."

Show the final training recipe.

## Submission Checklist

- Upload `attention_sinks.py` to molab and verify a clean GPU runtime.
- Run the default SmolLM path and the optional scaling study in molab.
- Confirm every plot renders and no model row contains an error or NaN.
- Record the video using the sequence above.
- Submit the molab link and video through the competition form before July 9, 2026, 11:59 PM PST.
