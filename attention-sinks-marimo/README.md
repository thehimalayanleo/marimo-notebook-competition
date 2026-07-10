# Attention Sinks Marimo Notebook

Competition notebook for:

- Paper: "Why do LLMs attend to the first token?"
- arXiv: https://arxiv.org/abs/2504.02732
- alphaXiv: https://www.alphaxiv.org/abs/2504.02732

## Run

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
marimo edit attention_sinks.py
```

Default demo: `HuggingFaceTB/SmolLM2-135M`.

Choose the model and prompt, then click **Run selected experiment**. Expensive model work is gated behind this button so the published app opens immediately.

Fast smoke test: `sshleifer/tiny-gpt2`.

Other small-model demos:

- `HuggingFaceTB/SmolLM2-135M`
- `HuggingFaceTB/SmolLM2-360M-Instruct`
- `Qwen/Qwen2.5-0.5B`
- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-1.5B`
- `distilgpt2`
- `gpt2`

Competition title, short description, video script, and final checklist are in [SUBMISSION.md](SUBMISSION.md).

## Notebook Story

The app turns attention sinks into a five-part evidence trail:

1. Show where attention sinks appear across layers and heads.
2. Perturb one token and compare representation spread with vs without the first token.
3. Sweep context length and measure sink use plus rollout spread.
4. Use causal attention surgery to test whether a distributed sink bank can retain over-mixing protection while reducing position-0 concentration.
5. Run the same intervention from SmolLM2-135M through Qwen2.5-1.5B.

Optional probes then test dummy sink candidates and alternative early sink locations.

## Learning Goals

- Define an attention sink.
- Read `sink_mass_to_pos0`, `prev_token_mass`, `sink_over_prev`, and the paper-style percent-of-heads metric.
- Interpret a bright first heatmap column.
- Explain the paper's why: sinks can help route attention without over-mixing token information.
- Connect attention sinks to a real perturbation experiment.
- Test whether alternative early positions can become sink targets.
- Explore distributed sink-bank designs as a model-building implication.
- Test whether the sink-bank result survives a cross-family model scaling study.
- Compare models, prompts, lengths, and layers.
- Compare library prompts and test their own prompts.
- Use dummy prefix tokens as an intervention.
