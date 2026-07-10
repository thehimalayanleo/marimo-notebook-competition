# Submission Kit

## Title

Can We Replace the First-Token Attention Sink?

## Short Description

This interactive notebook reproduces the attention-sink and over-mixing intuition from "Why do LLMs attend to the first token?" It then tests a new hypothesis across SmolLM and Qwen: distribute sink attention over a small causal bank to reduce single-token concentration while retaining containment.

## Main Finding

A four-position causal sink bank retained 98.0%-99.7% of the attention-rollout containment in local GPU tests spanning 135M to 1.5B parameters. This is an attention-surgery feasibility result, not evidence that an untrained intervention preserves language-model quality.

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
