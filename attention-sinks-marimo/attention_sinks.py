# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "accelerate>=0.30",
#     "marimo>=0.13",
#     "matplotlib>=3.8",
#     "numpy>=1.26",
#     "pandas>=2.2",
#     "torch>=2.2",
#     "transformers>=4.41",
# ]
# ///

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import functools
    import math
    import textwrap

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    return functools, mo, np, pd, plt, textwrap


@app.cell
def _(mo, textwrap):
    mo.md(
        textwrap.dedent(
            """
    # Can we replace the first-token attention sink?

    Reproducing *Why do LLMs attend to the first token?*, then testing a new design: can one
    extreme attention sink become a small causal bank without losing its protection against over-mixing?

    [Read and discuss the paper on alphaXiv](https://www.alphaxiv.org/abs/2504.02732) · [arXiv](https://arxiv.org/abs/2504.02732)
    """
        )
    )
    return


@app.cell
def _(mo, textwrap):
    mo.md(
        textwrap.dedent(
            """
    ## Experiment map

    - **1. Attention sink**: find which layer/head sends attention to position `0`.
    - **2. Perturbation**: change one token and compare spread with vs without the first token.
    - **3. Context length**: test whether longer visible context changes sink use and rollout spread.
    - **4. Distributed sink bank**: replace one extreme sink with several causally valid routing positions.
    - **5. Scale test**: repeat the extension from SmolLM2-135M through Qwen2.5-1.5B.

    Optional probes later test dummy prefixes and alternative sink locations.
    """
        )
    )
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md("## Result in one sentence"),
            mo.md(
                """
    Across SmolLM and Qwen, position `0` absorbs substantial attention. A causal bank of early
    routing positions can sharply reduce that single-token concentration while retaining most
    of the rollout containment associated with the original sink.
    """
            ),
            mo.callout(
                "The extension is a design hypothesis, not a trained-model claim: learned sink banks should be tested against perplexity, downstream quality, and exact Jacobian measurements.",
                kind="success",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Why this can happen": mo.md(
                """
    The short version:

    - Each transformer layer mixes token states through attention.
    - If every token strongly mixes with every other token, distinct information can blur together.
    - Position 0 is special because every later token can attend to it in a causal LM.
    - A head can use position 0 as a stable place to route attention when it does not want to copy content from nearby tokens.
    - This can help avoid over-mixing: the model spends some attention on a harmless shared location instead of forcing all attention onto meaningful tokens.

    So an attention sink is not necessarily a bug. It can be a learned routing pattern.

    What this notebook tests:

    - Does attention concentrate on position 0?
    - What percent of heads pass the selected sink threshold epsilon?
    - Does that depend on model, prompt, layer, or head?
    - If we add dummy prefix tokens, does the routing move?
    """
            )
        }
    )
    return


@app.cell
def _(mo):
    model_name = mo.ui.dropdown(
        options=[
            "HuggingFaceTB/SmolLM2-135M",
            "HuggingFaceTB/SmolLM2-360M-Instruct",
            "Qwen/Qwen2.5-0.5B",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B",
            "sshleifer/tiny-gpt2",
            "distilgpt2",
            "gpt2",
        ],
        value="HuggingFaceTB/SmolLM2-135M",
        label="Model",
    )
    prompt_style = mo.ui.dropdown(
        options=[
            "single passage: paper intuition",
            "packed docs: mixed topics",
            "story: object tracking",
            "code: state tracking",
            "dialogue: role tracking",
            "list: key-value lookup",
            "long context: distractors",
            "instruction: answer extraction",
        ],
        value="packed docs: mixed topics",
        label="Prompt library",
    )
    max_tokens = mo.ui.slider(32, 256, value=128, step=8, label="Max tokens")
    sink_threshold = mo.ui.slider(0.05, 0.8, value=0.3, step=0.05, label="Sink threshold epsilon")
    use_custom_prompt = mo.ui.checkbox(value=False, label="Use custom prompt")
    custom_prompt = mo.ui.text_area(
        value=(
            "Document A: The robot put the green cube on the shelf.\n"
            "Document B: The scientist changed the model architecture.\n"
            "Question: Which document mentions a model?"
        ),
        label="Custom prompt",
        rows=5,
    )
    perturb_replacement = mo.ui.text(value=" best", label="Perturb replacement token")
    dummy_sink_tokens = mo.ui.slider(0, 12, value=0, step=1, label="Dummy prefix tokens")
    sink_scan_width = mo.ui.slider(4, 24, value=12, step=1, label="Early positions to scan")
    sink_bank_size = mo.ui.slider(2, 16, value=4, step=1, label="Distributed sink bank size")
    run_context_sweep = mo.ui.checkbox(value=False, label="Run context-length sweep")
    run_scaling_study = mo.ui.checkbox(value=False, label="Run 135M-1.5B scaling study")
    run_accelerator_benchmark = mo.ui.checkbox(value=False, label="Run accelerator benchmark")
    run_experiment = mo.ui.run_button(
        label="Run selected experiment",
        kind="success",
        full_width=True,
        tooltip="Load the selected model and update every experiment below.",
    )
    controls = mo.hstack(
        [
            mo.vstack([model_name, prompt_style, max_tokens, sink_threshold, use_custom_prompt, custom_prompt]),
            mo.vstack(
                [
                    perturb_replacement,
                    dummy_sink_tokens,
                    sink_scan_width,
                    sink_bank_size,
                    run_context_sweep,
                    run_scaling_study,
                    run_accelerator_benchmark,
                ]
            ),
        ],
        gap=2,
    )
    mo.vstack(
        [
            controls,
            mo.md("Choose the model and prompt first. Changing a setting pauses expensive computation until you run again."),
            run_experiment,
        ]
    )
    return (
        custom_prompt,
        dummy_sink_tokens,
        max_tokens,
        model_name,
        perturb_replacement,
        prompt_style,
        run_accelerator_benchmark,
        run_context_sweep,
        run_experiment,
        run_scaling_study,
        sink_bank_size,
        sink_scan_width,
        sink_threshold,
        use_custom_prompt,
    )


@app.cell
def _(mo):
    controls_glossary = mo.md(
        """
    - **Model**: the LM whose attention patterns we inspect.
    - **Prompt library**: preset input shapes for comparing `sink_mass_to_pos0`.
    - **Use custom prompt**: ignore the preset and use your own text.
    - **Custom prompt**: text sent to the model when custom mode is on.
    - **Max tokens**: how much of the prompt the model sees.
    - **Sink threshold epsilon**: cutoff for counting a head as having an attention sink.
    - **Perturb replacement token**: token text used to replace the selected source token in the perturbation experiment.
    - **Dummy prefix tokens**: artificial tokens added before the real prompt.
    - **Early positions to scan**: number of early token positions to inspect as possible sink locations.
    - **Distributed sink bank size**: number of early positions used in the attention-surgery sink-bank proxy.
    - **Run context-length sweep**: run extra forward passes at multiple token limits.
    - **Run 135M-1.5B scaling study**: run the same prompt and sink-bank intervention across two SmolLM sizes and two Qwen sizes.
    - **Run accelerator benchmark**: time repeated Transformer forward passes on the active device.
    - **Run selected experiment**: load the chosen model and recompute the notebook after settings change.
    - **Layer**: one transformer block in the model.
    - **Head**: one attention sub-module inside a layer.
    - **Normalize heatmap**: rescales each row so each query token's strongest attention is easy to see.
    """
    )
    return (controls_glossary,)


@app.cell
def _(functools):
    @functools.lru_cache(maxsize=6)
    def load_causal_lm(name):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(name)
        _probe_ids = tokenizer("attention sink", add_special_tokens=False)["input_ids"]
        if not _probe_ids and name.startswith("Qwen/Qwen2.5-"):
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        kwargs = {}
        if torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"
            target_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            kwargs["torch_dtype"] = torch.float32 if "Qwen2.5-1.5B" in name else torch.float16
            target_device = "mps"
        else:
            target_device = "cpu"
        try:
            model = AutoModelForCausalLM.from_pretrained(
                name,
                attn_implementation="eager",
                **kwargs,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        if not hasattr(model, "hf_device_map"):
            model = model.to(target_device)
        model.eval()
        return tokenizer, model

    return (load_causal_lm,)


@app.cell
def _(
    custom_prompt,
    dummy_sink_tokens,
    max_tokens,
    prompt_style,
    textwrap,
    use_custom_prompt,
):
    PAPER_INTUITION = """
    The first token in a Transformer sequence is unusual: every later token can attend to it, and the model can reuse it
    as a low-risk place to route excess attention. The paper argues that this behavior helps prevent over-mixing as
    information propagates through many layers.
    """

    PACKED_DOCS = """
    Document A: A city council approved a new transit plan after months of debate.
    Document B: A protein-folding model improved after researchers changed its training curriculum.
    Document C: A small language model solved a puzzle by tracking symbols across a long context.
    Question: Which document mentions a model?
    """

    STORY_TRACKING = """
    Alice put the copper key in the blue box.
    Bruno moved the red key to the shelf.
    Clara placed the silver coin under the lamp.
    Alice checked the blue box again. The copper key was still there.
    Bruno moved the red key from the shelf into the desk drawer.
    Clara left the silver coin under the lamp.
    A note said the blue box should be opened after sunset.
    Question: Where is the copper key?
    """

    CODE_STATE = """
    state = {}
    state['A'] = 4
    state['B'] = 9
    state['A'] = state['A'] + 3
    state['C'] = state['B'] - state['A']
    query = state['C']
    """

    DIALOGUE_ROLES = """
    User: I need the shipment sent to Berlin.
    Assistant: Noted. The destination is Berlin.
    User: Change the carrier to rail, but keep the destination unchanged.
    Assistant: The carrier is rail. The destination is still Berlin.
    User: What destination should appear on the final label?
    """

    KEY_VALUE_LIST = """
    alpha = copper
    beta = violet
    gamma = silver
    delta = amber
    epsilon = green
    zeta = blue
    Query: What value is assigned to gamma?
    """

    LONG_DISTRACTORS = """
    Important fact: the access code is 4827.
    Distractor: the meeting moved to Tuesday.
    Distractor: the blue folder contains invoices.
    Distractor: the lab ordered new sensors.
    Distractor: the garden gate was painted red.
    Distractor: the archive has four shelves.
    Distractor: the backup drive is encrypted.
    Question: What is the access code?
    """

    INSTRUCTION_EXTRACTION = """
    Task: Extract the final answer only.
    Context: The first candidate answer was Paris. A later correction changed it to Lisbon. Ignore the first candidate.
    Evidence: The verified destination is Lisbon because the booking confirmation uses LIS.
    Final answer:
    """

    prompt_bank = {
        "single passage: paper intuition": PAPER_INTUITION,
        "packed docs: mixed topics": PACKED_DOCS,
        "story: object tracking": STORY_TRACKING,
        "code: state tracking": CODE_STATE,
        "dialogue: role tracking": DIALOGUE_ROLES,
        "list: key-value lookup": KEY_VALUE_LIST,
        "long context: distractors": LONG_DISTRACTORS,
        "instruction: answer extraction": INSTRUCTION_EXTRACTION,
    }
    dummy_prefix = " ".join(["<sink>"] * dummy_sink_tokens.value)
    preset_prompt = textwrap.dedent(prompt_bank[prompt_style.value]).strip()
    custom_text = custom_prompt.value.strip()
    clean_prompt = custom_text if use_custom_prompt.value and custom_text else preset_prompt
    prompt_source = "custom" if use_custom_prompt.value and custom_text else prompt_style.value
    raw_prompt = f"{dummy_prefix}\n{clean_prompt}"
    active_prompt = raw_prompt.strip()
    token_limit = max_tokens.value
    return active_prompt, clean_prompt, prompt_source, token_limit


@app.cell
def _(active_prompt, mo, prompt_source):
    mo.vstack(
        [
            mo.md("## Prompt"),
            mo.md(f"Selected: `{prompt_source}`"),
            mo.accordion({"Prompt text": active_prompt}),
        ]
    )
    return


@app.cell
def _(dummy_sink_tokens, mo, prompt_source):
    if dummy_sink_tokens.value == 0:
        _intervention_hint = "Increase dummy prefix tokens and check whether `sink_mass_to_pos0` changes."
    else:
        _intervention_hint = "Compare against zero dummy prefix tokens."
    mo.vstack(
        [
            mo.md("## How to use this"),
            mo.md(
                f"""
    - Current prompt: `{prompt_source}`.
    - Click **Run selected experiment** after choosing the model and prompt.
    - Find heads with high `sink_mass_to_pos0`.
    - Set the layer/head sliders to inspect one head.
    - Heatmap: rows are query tokens, columns are key tokens.
    - Intervention: {_intervention_hint}
    """
            ),
        ]
    )
    return


@app.cell
def _(
    active_prompt,
    load_causal_lm,
    mo,
    model_name,
    run_experiment,
    token_limit,
):
    if not run_experiment.value:
        tokenizer = None
        model = None
        tokens = []
        attentions = []
        runtime_details = {}
        load_error = None
        load_notice = mo.callout(
            "Ready. Choose settings above, then click **Run selected experiment** to load the model.",
            kind="info",
        )
    else:
        try:
            import torch

            tokenizer, model = load_causal_lm(model_name.value)
            encoded = tokenizer(
                active_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=token_limit,
            )
            device = next(model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded, output_attentions=True)
            tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].detach().cpu().tolist())
            attentions = [attn[0].float().detach().cpu().numpy() for attn in outputs.attentions]
            _first_param = next(model.parameters())
            runtime_details = {
                "device": str(device),
                "dtype": str(_first_param.dtype).replace("torch.", ""),
                "cuda_available": bool(torch.cuda.is_available()),
                "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
                "accelerator_available": bool(
                    torch.cuda.is_available()
                    or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
                ),
                "tokens": int(encoded["input_ids"].shape[-1]),
                "forward_passes": 1,
            }
            load_error = None
            load_notice = mo.callout(
                f"Experiment complete on `{device}` with {runtime_details['tokens']} tokens.",
                kind="success",
            )
        except Exception as exc:
            tokenizer = None
            model = None
            tokens = []
            attentions = []
            runtime_details = {}
            load_error = exc
            load_notice = mo.callout(
                f"Model dependencies or weights are not available yet: `{type(load_error).__name__}: {load_error}`",
                kind="warn",
            )
    experiment_ready = model is not None and bool(attentions)
    load_notice
    return attentions, experiment_ready, model, runtime_details, tokens


@app.cell
def _(attentions, mo, tokens):
    if attentions:
        n_layers = len(attentions)
        n_heads = attentions[0].shape[0]
        _best_score, _best_layer, _best_head = max(
            (
                float(_attention[_head, 1:, 0].mean()),
                _layer,
                _head,
            )
            for _layer, _attention in enumerate(attentions)
            for _head in range(_attention.shape[0])
        )
    else:
        n_layers = 1
        n_heads = 1
        _best_layer = 0
        _best_head = 0
    _token_count = max(len(tokens), 1)
    layer_ix = mo.ui.slider(0, max(n_layers - 1, 0), value=_best_layer, step=1, label=f"Layer (0-{max(n_layers - 1, 0)})")
    head_ix = mo.ui.slider(0, max(n_heads - 1, 0), value=_best_head, step=1, label=f"Head (0-{max(n_heads - 1, 0)})")
    perturb_ix = mo.ui.slider(
        0,
        max(_token_count - 1, 0),
        value=min(1, _token_count - 1),
        step=1,
        label=f"Source token (0-{max(_token_count - 1, 0)})",
    )
    normalize_rows = mo.ui.checkbox(value=True, label="Normalize heatmap per query token")
    mo.vstack(
        [
            mo.md("## View controls"),
            mo.hstack([layer_ix, head_ix, perturb_ix], gap=2),
            normalize_rows,
        ]
    )
    return head_ix, layer_ix, n_heads, n_layers, normalize_rows, perturb_ix


@app.cell
def _(
    attentions,
    head_ix,
    layer_ix,
    mo,
    model,
    n_heads,
    n_layers,
    runtime_details,
):
    if model is not None and attentions:
        active_layer = min(layer_ix.value, n_layers - 1)
        active_head = min(head_ix.value, n_heads - 1)
        _runtime_line = (
            f"- Device: `{runtime_details.get('device', '?')}` | "
            f"Parameters dtype: `{runtime_details.get('dtype', '?')}` | "
            f"CUDA: `{runtime_details.get('cuda_available', False)}` | "
            f"MPS: `{runtime_details.get('mps_available', False)}` | "
            f"Tokens in current run: `{runtime_details.get('tokens', '?')}`"
        )
        model_details = mo.md(
            f"""
    **Loaded model:** `{model.config.name_or_path}`

    - Parameters: `{sum(_parameter.numel() for _parameter in model.parameters()) / 1e6:.1f}M`
    - Layers: `{n_layers}`
    - Heads: `{n_heads}`
    - Hidden size: `{getattr(model.config, "hidden_size", getattr(model.config, "n_embd", "?"))}`
    - Selected view: layer `{active_layer}`, head `{active_head}`
    {_runtime_line}
    """
        )
    else:
        active_layer = 0
        active_head = 0
        model_details = mo.md("No model output yet.")
    return active_head, active_layer, model_details


@app.cell
def _(
    active_prompt,
    controls_glossary,
    mo,
    model_details,
    prompt_source,
    tokens,
):
    preview_tokens = tokens[:80]
    token_rows = [
        {"position": ix, "token": token.replace("\n", "\\n")}
        for ix, token in enumerate(preview_tokens)
    ]
    mo.accordion(
        {
            "Run details": mo.vstack(
                [
                    mo.md("### Model"),
                    model_details,
                    mo.md("### Prompt"),
                    mo.md(
                        f"""
    - Source: `{prompt_source}`
    - Prompt: exact text sent to the model.
    - First tokens: tokenized view with numeric positions.
    - Position `0` is the candidate sink.
    """
                    ),
                    mo.accordion(
                        {
                            "Prompt text": active_prompt,
                            "First tokens": mo.ui.table(token_rows, pagination=False),
                            "Controls glossary": controls_glossary,
                        }
                    ),
                ]
            ),
        }
    )
    return


@app.cell
def _(attentions, np, pd):
    def sink_metrics(attention_stack):
        rows = []
        for layer, attn in enumerate(attention_stack):
            # attn shape: heads, query positions, key positions
            if attn.shape[-1] < 2:
                continue
            sink_mass = attn[:, 1:, 0].mean(axis=1)
            recent_mass = np.stack(
                [
                    attn[head, np.arange(1, attn.shape[1]), np.arange(0, attn.shape[1] - 1)].mean()
                    for head in range(attn.shape[0])
                ]
            )
            for head, sink_value in enumerate(sink_mass):
                rows.append(
                    {
                        "layer": layer,
                        "head": head,
                        "sink_mass_to_pos0": float(sink_value),
                        "prev_token_mass": float(recent_mass[head]),
                        "sink_over_prev": float(sink_value / max(recent_mass[head], 1e-8)),
                    }
                )
        return pd.DataFrame(rows)

    metrics_df = sink_metrics(attentions) if attentions else pd.DataFrame()
    return (metrics_df,)


@app.cell
def _(metrics_df, mo, runtime_details, sink_threshold, tokens):
    if len(metrics_df):
        _best = metrics_df.sort_values("sink_mass_to_pos0", ascending=False).iloc[0]
        _global_sink = metrics_df["sink_mass_to_pos0"].mean()
        _global_prev = metrics_df["prev_token_mass"].mean()
        _threshold = sink_threshold.value
        _paper_sink_metric = float((metrics_df["sink_mass_to_pos0"] >= _threshold).mean() * 100.0)
        _token0 = tokens[0].replace("\n", "\\n") if tokens else "position 0"
        _relative = _global_sink / max(_global_prev, 1e-8)
        if _relative > 1.2:
            _comparison = "higher than"
        elif _relative < 0.8:
            _comparison = "lower than"
        else:
            _comparison = "about the same as"
        run_summary_view = mo.vstack(
            [
                mo.md("## Current run"),
                mo.md(
                    f"""
    - First token: `{_token0}`
    - Mean `sink_mass_to_pos0`: **{_global_sink:.3f}**
    - Mean previous-token mass: **{_global_prev:.3f}**
    - Sink vs local baseline: **{_comparison}**
    - Paper-style metric at epsilon `{_threshold:.2f}`: **{_paper_sink_metric:.1f}%** of heads
    - Strongest head by `sink_mass_to_pos0`: **layer {int(_best['layer'])}, head {int(_best['head'])}** with **{_best['sink_mass_to_pos0']:.3f}**
    - Runtime: **{runtime_details.get('device', '?')}**, dtype **{runtime_details.get('dtype', '?')}**, **{runtime_details.get('tokens', '?')}** tokens, accelerator available: **{runtime_details.get('accelerator_available', False)}**
    """
                ),
                mo.callout(
                    "Theory link: a bright first column means many tokens are routing attention to the same shared early position.",
                    kind="neutral",
                ),
            ]
        )
    else:
        run_summary_view = mo.md("## Current run\n\nLoad a model to generate a summary.")
    run_summary_view
    return


@app.cell
def _(
    active_prompt,
    experiment_ready,
    load_causal_lm,
    mo,
    model_name,
    run_accelerator_benchmark,
    runtime_details,
    token_limit,
):
    mo.stop(
        not experiment_ready,
        mo.md("## Accelerator use\n\nRun the selected experiment first."),
    )
    if run_accelerator_benchmark.value:
        try:
            import time as _time
            import torch as _torch

            _tokenizer, _benchmark_model = load_causal_lm(model_name.value)
            _encoded = _tokenizer(
                active_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=token_limit,
            )
            _device = next(_benchmark_model.parameters()).device
            _encoded = {key: value.to(_device) for key, value in _encoded.items()}
            _batch = {
                key: value.repeat(4, 1) if value.ndim == 2 else value
                for key, value in _encoded.items()
            }

            def _sync():
                if _device.type == "cuda":
                    _torch.cuda.synchronize()
                elif _device.type == "mps" and hasattr(_torch, "mps"):
                    _torch.mps.synchronize()

            with _torch.no_grad():
                for _ in range(2):
                    _benchmark_model(**_batch, output_attentions=True)
            _sync()
            _start = _time.perf_counter()
            _passes = 6
            with _torch.no_grad():
                for _ in range(_passes):
                    _benchmark_model(**_batch, output_attentions=True)
            _sync()
            _elapsed = _time.perf_counter() - _start
            _tokens = int(_batch["input_ids"].numel() * _passes)
            _tokens_per_second = _tokens / max(_elapsed, 1e-8)
            accelerator_view = mo.vstack(
                [
                    mo.md("## Accelerator benchmark"),
                    mo.md(
                        f"""
    - Active device: `{_device}`
    - Batch size: `4`
    - Timed forward passes: `{_passes}`
    - Tokens processed: `{_tokens}`
    - Elapsed time: **{_elapsed:.2f}s**
    - Throughput: **{_tokens_per_second:.1f} tokens/sec**
    """
                    ),
                    mo.callout(
                        "This benchmark uses the same attention-returning Transformer forward pass as the notebook experiments, so it measures the actual workload behind the visualizations.",
                        kind="info",
                    ),
                ]
            )
        except Exception as exc:
            accelerator_view = mo.callout(
                f"Accelerator benchmark could not run: `{type(exc).__name__}: {exc}`",
                kind="warn",
            )
    else:
        _device = runtime_details.get("device", "?")
        _accelerator = runtime_details.get("accelerator_available", False)
        accelerator_view = mo.vstack(
            [
                mo.md("## Accelerator use"),
                mo.md(
                    f"""
    - Current run device: `{_device}`
    - Accelerator available: `{_accelerator}`
    - The model loader uses CUDA when available, MPS when available, and CPU as fallback.
    - Perturbation, context sweep, alternative-sink scan, and the optional benchmark all run repeated Transformer forward passes on that active device.
    - Enable **Run accelerator benchmark** to time batched attention-returning forward passes.
    """
                ),
            ]
        )
    accelerator_view
    return


@app.cell
def _(mo, textwrap):
    mo.md(
        textwrap.dedent(
            """
    ## Metric definitions

    - `sink_mass_to_pos0`: average attention from later tokens to position 0.
    - `prev_token_mass`: average attention to the immediately previous token.
    - `sink_over_prev`: `sink_mass_to_pos0` divided by `prev_token_mass`.
    - `paper-style metric`: percent of heads with `sink_mass_to_pos0` at or above epsilon.
    - `query token`: the token doing the attending.
    - `key token`: the token being attended to.
    - These are inspection metrics, not accuracy metrics.
    """
        )
    )
    return


@app.cell
def _(metrics_df, mo, plt):
    if len(metrics_df):
        _pivot = metrics_df.pivot(index="layer", columns="head", values="sink_mass_to_pos0")
        _fig, _ax = plt.subplots(figsize=(10, 4))
        _im = _ax.imshow(_pivot.values, aspect="auto", cmap="magma", vmin=0)
        _ax.set_title("Section 1: attention sent to position 0")
        _ax.set_xlabel("Head")
        _ax.set_ylabel("Layer")
        _ax.set_xticks(range(_pivot.shape[1]))
        _ax.set_yticks(range(_pivot.shape[0]))
        _fig.colorbar(_im, ax=_ax, label="sink_mass_to_pos0")
        _fig.tight_layout()
        section1_sink_view = mo.vstack(
            [
                mo.md("## 1. Attention sink"),
                mo.md(
                    """
    - Question: does attention concentrate at the first token?
    - Plot: each cell is one layer/head.
    - Brighter cell: that head sends more average attention to position `0`.
    - Use this section to pick a layer/head before inspecting the detailed heatmaps below.
    """
                ),
                _fig,
            ]
        )
    else:
        section1_sink_view = mo.md("## 1. Attention sink\n\nLoad a model to see whether attention concentrates at position `0`.")
    section1_sink_view
    return


@app.cell
def _(
    active_prompt,
    experiment_ready,
    load_causal_lm,
    mo,
    model_name,
    np,
    perturb_ix,
    perturb_replacement,
    plt,
    token_limit,
):
    mo.stop(
        not experiment_ready,
        mo.md("## 2. Over-mixing under perturbation\n\nRun the selected experiment to start this analysis."),
    )

    def _run_final_hidden(input_ids, attention_mask, model):
        import torch

        device = next(model.parameters()).device
        _inputs = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
        }
        with torch.no_grad():
            _outputs = model(**_inputs, output_hidden_states=True)
        return _outputs.hidden_states[-1][0].float().detach().cpu().numpy()

    def _perturbation_spread():
        import torch

        tokenizer, model = load_causal_lm(model_name.value)
        _encoded = tokenizer(
            active_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=token_limit,
        )
        _input_ids = _encoded["input_ids"]
        _attention_mask = _encoded["attention_mask"]
        _seq_len = int(_input_ids.shape[-1])
        if _seq_len < 3:
            raise ValueError("Need at least three tokens for a meaningful perturbation comparison.")

        _source = min(max(1, perturb_ix.value), _seq_len - 1)
        _replacement_ids = tokenizer.encode(perturb_replacement.value, add_special_tokens=False)
        if not _replacement_ids:
            _replacement_ids = tokenizer.encode(" best", add_special_tokens=False)
        _replacement_id = int(_replacement_ids[0])

        _perturbed_ids = _input_ids.clone()
        _original_id = int(_perturbed_ids[0, _source])
        _perturbed_ids[0, _source] = _replacement_id

        _with_first_base = _run_final_hidden(_input_ids, _attention_mask, model)
        _with_first_perturbed = _run_final_hidden(_perturbed_ids, _attention_mask, model)
        _with_first_delta = np.linalg.norm(_with_first_perturbed - _with_first_base, axis=-1)

        _without_first_ids = _input_ids[:, 1:].clone()
        _without_first_mask = _attention_mask[:, 1:].clone()
        _without_first_perturbed_ids = _without_first_ids.clone()
        _without_source = _source - 1
        _without_first_perturbed_ids[0, _without_source] = _replacement_id
        _without_first_base = _run_final_hidden(_without_first_ids, _without_first_mask, model)
        _without_first_perturbed = _run_final_hidden(_without_first_perturbed_ids, _without_first_mask, model)
        _without_first_delta = np.linalg.norm(_without_first_perturbed - _without_first_base, axis=-1)

        _tokens = tokenizer.convert_ids_to_tokens(_input_ids[0].detach().cpu().tolist())
        _replacement_token = tokenizer.convert_ids_to_tokens([_replacement_id])[0]
        _original_token = tokenizer.convert_ids_to_tokens([_original_id])[0]
        return {
            "source": _source,
            "without_source": _without_source,
            "tokens": _tokens,
            "original_token": _original_token,
            "replacement_token": _replacement_token,
            "with_first_delta": _with_first_delta,
            "without_first_delta": _without_first_delta,
        }

    def _short_token(token, max_len=12):
        _cleaned = token.replace("Ġ", "").replace("▁", "").replace("\n", "\\n")
        return _cleaned if len(_cleaned) <= max_len else _cleaned[: max_len - 1] + "…"

    try:
        _result = _perturbation_spread()
        _with_first_delta = _result["with_first_delta"]
        _without_first_delta = _result["without_first_delta"]
        _source = _result["source"]
        _tokens = _result["tokens"]
        _with_exclude = np.ones(len(_with_first_delta), dtype=bool)
        _with_exclude[0] = False
        _with_exclude[_source] = False
        _without_exclude = np.ones(len(_without_first_delta), dtype=bool)
        _without_exclude[_result["without_source"]] = False
        _with_spread = float(_with_first_delta[_with_exclude].mean())
        _without_spread = float(_without_first_delta[_without_exclude].mean())
        _ratio = _without_spread / max(_with_spread, 1e-8)
        if _ratio > 1.1:
            _takeaway = "Removing the first token makes the perturbation spread more through the final representations."
            _kind = "info"
        elif _ratio < 0.9:
            _takeaway = "Removing the first token reduces perturbation spread for this prompt/model."
            _kind = "neutral"
        else:
            _takeaway = "The final-layer perturbation spread is similar with and without the first token."
            _kind = "neutral"

        _fig, _ax = plt.subplots(figsize=(10, 3.8))
        _ax.plot(np.arange(len(_with_first_delta)), _with_first_delta, marker="o", label="with first token")
        _ax.plot(np.arange(1, len(_tokens)), _without_first_delta, marker="o", label="remove first token")
        _ax.axvline(_source, color="#444444", linewidth=1, linestyle="--", alpha=0.7)
        _ax.set_title("Section 2: final-layer change after one token perturbation")
        _ax.set_xlabel("Token position in original prompt")
        _ax.set_ylabel("Hidden-state change norm")
        _tick_stride = max(1, len(_tokens) // 18)
        _tick_positions = list(range(0, len(_tokens), _tick_stride))
        _ax.set_xticks(_tick_positions)
        _ax.set_xticklabels(
            [f"{_ix}:{_short_token(_tokens[_ix])}" for _ix in _tick_positions],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        _ax.grid(alpha=0.25)
        _ax.legend()
        _fig.tight_layout()

        _zoom_positions = np.arange(len(_with_first_delta))
        _zoom_mask = np.ones(len(_with_first_delta), dtype=bool)
        _zoom_mask[0] = False
        _zoom_mask[_source] = False
        _zoom_fig, _zoom_ax = plt.subplots(figsize=(10, 3.2))
        _zoom_ax.plot(
            _zoom_positions[_zoom_mask],
            _with_first_delta[_zoom_mask],
            marker="o",
            label="with first token",
        )
        _without_positions = np.arange(1, len(_tokens))
        _without_zoom_mask = np.ones(len(_without_first_delta), dtype=bool)
        _without_zoom_mask[_result["without_source"]] = False
        _zoom_ax.plot(
            _without_positions[_without_zoom_mask],
            _without_first_delta[_without_zoom_mask],
            marker="o",
            label="remove first token",
        )
        _zoom_ax.set_title("Perturbation spread, excluding the changed token")
        _zoom_ax.set_xlabel("Other token positions")
        _zoom_ax.set_ylabel("Hidden-state change norm")
        _zoom_ax.grid(alpha=0.25)
        _zoom_ax.legend()
        _zoom_fig.tight_layout()

        perturbation_view = mo.vstack(
            [
                mo.md("## 2. Over-mixing under perturbation"),
                mo.md(
                    f"""
    - Perturbation: token `{_source}` changes from `{_short_token(_result['original_token'])}` to `{_short_token(_result['replacement_token'])}`.
    - With first token: run the original sequence and measure final hidden-state changes.
    - Remove first token: run the same comparison after deleting position `0`.
    - Mean spread away from the perturbed token: **{_with_spread:.4f}** with first token vs **{_without_spread:.4f}** without it.
    """
                ),
                mo.callout(_takeaway, kind=_kind),
                _fig,
                _zoom_fig,
                mo.md(
                    """
    - Why this matters: the paper argues that sinks reduce uncontrolled mixing.
    - If the orange curve is higher across many positions, deleting the first-token sink lets the perturbation contaminate more representations.
    - The zoomed plot removes the directly changed token so the spread to other tokens is easier to see.
    - Caveat: deleting the first token also shifts token positions. Section 2b isolates the attention route with a fixed-length counterfactual.
    """
                ),
            ]
        )
    except Exception as exc:
        perturbation_view = mo.callout(
            f"Perturbation experiment could not run: `{type(exc).__name__}: {exc}`",
            kind="warn",
        )
    perturbation_view
    return


@app.cell
def _(attentions, mo, np, perturb_ix, plt, tokens):
    def _rollout_from_attention_stack(attention_stack, remove_position0=False, residual_weight=0.5):
        _seq_len = attention_stack[0].shape[-1]
        _rollout = np.eye(_seq_len)
        for _layer_attention in attention_stack:
            _avg_attention = _layer_attention.mean(axis=0).astype(float)
            _avg_attention = _avg_attention / np.maximum(_avg_attention.sum(axis=1, keepdims=True), 1e-8)
            if remove_position0 and _seq_len > 1:
                _avg_attention = _avg_attention.copy()
                _avg_attention[1:, 0] = 0.0
                _row_sums = _avg_attention.sum(axis=1, keepdims=True)
                _empty_rows = _row_sums[:, 0] <= 1e-8
                _avg_attention = _avg_attention / np.maximum(_row_sums, 1e-8)
                if _empty_rows.any():
                    _avg_attention[_empty_rows] = np.eye(_seq_len)[_empty_rows]
            _transition = residual_weight * np.eye(_seq_len) + (1.0 - residual_weight) * _avg_attention
            _rollout = _transition @ _rollout
        return _rollout

    def _token_label(token, max_len=13):
        _cleaned = token.replace("\n", "\\n")
        return _cleaned if len(_cleaned) <= max_len else _cleaned[: max_len - 1] + "…"

    if attentions and tokens:
        _source = min(perturb_ix.value, len(tokens) - 1)
        _normal_rollout = _rollout_from_attention_stack(attentions, remove_position0=False)
        _without_sink_rollout = _rollout_from_attention_stack(attentions, remove_position0=True)
        _normal_influence = _normal_rollout[:, _source]
        _without_sink_influence = _without_sink_rollout[:, _source]
        _exclude = np.ones(len(tokens), dtype=bool)
        _exclude[_source] = False
        _exclude[0] = False
        _normal_spread = float(_normal_influence[_exclude].sum())
        _without_sink_spread = float(_without_sink_influence[_exclude].sum())
        _delta = _without_sink_spread - _normal_spread
        if _delta > 0.02:
            _takeaway = "Removing the position-0 outlet makes the selected token's influence spread more."
            _kind = "info"
        elif _delta < -0.02:
            _takeaway = "Removing the position-0 outlet reduces spread for this prompt/model."
            _kind = "neutral"
        else:
            _takeaway = "The rollout proxy shows little spread change for this prompt/model."
            _kind = "neutral"

        _positions = np.arange(len(tokens))
        _labels = [f"{_ix}:{_token_label(_token)}" for _ix, _token in enumerate(tokens)]
        _fig, _ax = plt.subplots(figsize=(10, 3.8))
        _ax.plot(_positions, _normal_influence, marker="o", label="actual attention")
        _ax.plot(_positions, _without_sink_influence, marker="o", label="remove position-0 attention")
        _ax.axvline(_source, color="#444444", linewidth=1, linestyle="--", alpha=0.7)
        _ax.set_title(f"Rollout influence from token {_source}: {_token_label(tokens[_source])}")
        _ax.set_xlabel("Final token position")
        _ax.set_ylabel("Attention-rollout influence")
        _tick_stride = max(1, len(tokens) // 18)
        _tick_positions = list(range(0, len(tokens), _tick_stride))
        _ax.set_xticks(_tick_positions)
        _ax.set_xticklabels([_labels[_ix] for _ix in _tick_positions], rotation=45, ha="right", fontsize=8)
        _ax.grid(alpha=0.25)
        _ax.legend()
        _fig.tight_layout()

        overmixing_view = mo.vstack(
            [
                mo.md("## 2b. Attention-rollout proxy"),
                mo.md(
                    f"""
    - Paper question: if token `{_source}` changes slightly, how much can it change other token representations after all layers?
    - Full Jacobian is expensive, so this uses attention rollout: layer-averaged attention maps multiplied through depth, with a residual identity term.
    - Counterfactual: remove attention to position `0`, renormalize each row, then run the same rollout.
    - Spread beyond token `{_source}` and position `0`: **{_normal_spread:.3f}** actual vs **{_without_sink_spread:.3f}** without position-0 attention.
    """
                ),
                mo.callout(_takeaway, kind=_kind),
                _fig,
                mo.md(
                    """
    - Why this matters: position `0` can act like a safe routing target when a head should avoid copying too much content.
    - Model-building angle: if sinks reduce uncontrolled mixing, then better long-context models may need explicit no-op/routing capacity, not just more attention everywhere.
    - Caveat: this is an interpretable rollout proxy, not the exact Jacobian from the theorem.
    """
                ),
            ]
        )
    else:
        overmixing_view = mo.md("## 2b. Attention-rollout proxy\n\nLoad a model to compare actual attention against a no-position-0 counterfactual.")
    overmixing_view
    return


@app.cell
def _(
    clean_prompt,
    experiment_ready,
    load_causal_lm,
    max_tokens,
    mo,
    model_name,
    np,
    pd,
    plt,
    run_context_sweep,
    sink_threshold,
):
    mo.stop(
        not experiment_ready,
        mo.md("## 3. Context length and over-mixing\n\nRun the selected experiment first."),
    )

    def _rollout_spread_for_stack(attention_stack, remove_position0=False):
        _seq_len = attention_stack[0].shape[-1]
        if _seq_len < 3:
            return 0.0
        _source = 1
        _rollout = np.eye(_seq_len)
        for _layer_attention in attention_stack:
            _avg_attention = _layer_attention.mean(axis=0).astype(float)
            _avg_attention = _avg_attention / np.maximum(_avg_attention.sum(axis=1, keepdims=True), 1e-8)
            if remove_position0:
                _avg_attention = _avg_attention.copy()
                _avg_attention[1:, 0] = 0.0
                _row_sums = _avg_attention.sum(axis=1, keepdims=True)
                _empty_rows = _row_sums[:, 0] <= 1e-8
                _avg_attention = _avg_attention / np.maximum(_row_sums, 1e-8)
                if _empty_rows.any():
                    _avg_attention[_empty_rows] = np.eye(_seq_len)[_empty_rows]
            _transition = 0.5 * np.eye(_seq_len) + 0.5 * _avg_attention
            _rollout = _transition @ _rollout
        _influence = _rollout[:, _source]
        _exclude = np.ones(_seq_len, dtype=bool)
        _exclude[0] = False
        _exclude[_source] = False
        return float(_influence[_exclude].sum())

    def _context_length_sweep(prompt):
        import math
        import torch

        tokenizer, model = load_causal_lm(model_name.value)
        _target_limit = max_tokens.value
        _base_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        _repeat_count = max(1, math.ceil((_target_limit + 16) / max(int(_base_ids.numel()), 1)))
        _long_prompt = "\n\n".join([prompt] * _repeat_count)
        _candidate_lengths = [32, 48, 64, 96, 128, 192, 256]
        _lengths = [length for length in _candidate_lengths if length <= _target_limit]
        if _target_limit not in _lengths:
            _lengths.append(_target_limit)
        _lengths = sorted(set(length for length in _lengths if length >= 8))
        _rows = []
        device = next(model.parameters()).device
        for _limit in _lengths:
            _encoded = tokenizer(
                _long_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=_limit,
            )
            _encoded = {key: value.to(device) for key, value in _encoded.items()}
            with torch.no_grad():
                _outputs = model(**_encoded, output_attentions=True)
            _attention_stack = [attn[0].float().detach().cpu().numpy() for attn in _outputs.attentions]
            _per_head = []
            for _attn in _attention_stack:
                if _attn.shape[-1] > 1:
                    _per_head.extend(_attn[:, 1:, 0].mean(axis=1).tolist())
            _actual_spread = _rollout_spread_for_stack(_attention_stack, remove_position0=False)
            _no_sink_spread = _rollout_spread_for_stack(_attention_stack, remove_position0=True)
            _rows.append(
                {
                    "tokens_seen": int(_encoded["input_ids"].shape[-1]),
                    "mean_sink_mass_to_pos0": float(np.mean(_per_head)) if _per_head else 0.0,
                    "paper_style_metric": float(np.mean(np.array(_per_head) >= sink_threshold.value) * 100.0)
                    if _per_head
                    else 0.0,
                    "rollout_spread": _actual_spread,
                    "extra_spread_without_pos0": _no_sink_spread - _actual_spread,
                }
            )
        return pd.DataFrame(_rows).drop_duplicates(subset=["tokens_seen"])

    if run_context_sweep.value:
        try:
            _sweep_df = _context_length_sweep(clean_prompt)
            _fig, (_ax, _ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            _ax.plot(
                _sweep_df["tokens_seen"],
                _sweep_df["mean_sink_mass_to_pos0"],
                marker="o",
                label="mean sink_mass_to_pos0",
            )
            _ax.set_title("Section 3: context length, sinks, and over-mixing")
            _ax.set_ylabel("Mean attention to position 0")
            _ax.grid(alpha=0.25)
            _ax2.plot(
                _sweep_df["tokens_seen"],
                _sweep_df["extra_spread_without_pos0"],
                marker="s",
                color="#d94f30",
                label="extra spread without position 0",
            )
            _ax2.axhline(0.0, color="#555555", linewidth=1, alpha=0.5)
            _ax2.set_xlabel("Tokens visible to the model")
            _ax2.set_ylabel("No-position-0 extra spread")
            _ax2.grid(alpha=0.25)
            _ax.legend(loc="best")
            _ax2.legend(loc="best")
            _fig.tight_layout()
            _start = float(_sweep_df["mean_sink_mass_to_pos0"].iloc[0])
            _end = float(_sweep_df["mean_sink_mass_to_pos0"].iloc[-1])
            _spread_end = float(_sweep_df["extra_spread_without_pos0"].iloc[-1])
            _direction = "increased" if _end > _start else "decreased"
            _context_sweep_view = mo.vstack(
                [
                    mo.md("## 3. Context length and over-mixing"),
                    mo.md(
                        f"""
    - Paper prediction: longer context increases the pressure to control mixing, so trained models should form stronger sinks.
    - Notebook proxy: keep the model fixed, grow the visible context, and track both `sink_mass_to_pos0` and attention-rollout spread.
    - This does **not** reproduce training from scratch; it checks whether the loaded model uses position 0 more as the prompt gets longer.
    - In this run, mean `sink_mass_to_pos0` {_direction}: **{_start:.3f} -> {_end:.3f}**.
    - At the longest length, removing position `0` changes rollout spread by **{_spread_end:.3f}**.
    """
                    ),
                    _fig,
                    mo.ui.table(_sweep_df.round(4).to_dict("records"), pagination=False),
                ]
            )
        except Exception as exc:
            _context_sweep_view = mo.callout(
                f"Context-length sweep could not run: `{type(exc).__name__}: {exc}`",
                kind="warn",
            )
    else:
        _context_sweep_view = mo.vstack(
            [
                mo.md("## 3. Context length and over-mixing"),
                mo.md(
                    """
    - Paper claim: over-squashing pressure grows with context length and model scale.
    - Prediction: longer-context training should make attention sinks more likely or stronger.
    - Notebook proxy: enable **Run context-length sweep** to test the loaded model at several token limits.
    - Readout 1: if `sink_mass_to_pos0` rises with visible tokens, the model is using position 0 more as context grows.
    - Readout 2: if removing position `0` creates more rollout spread, the first token is helping contain mixing.
    - Caveat: this is an inference-time sweep, not a retraining experiment.
    """
                ),
            ]
        )
    _context_sweep_view
    return


@app.cell
def _(attentions, mo, np, pd, plt, sink_bank_size):
    def _mean_position0_mass(_attention_stack):
        _values = [
            _layer[:, 1:, 0].mean()
            for _layer in _attention_stack
            if _layer.shape[-1] > 1
        ]
        return float(np.mean(_values)) if _values else 0.0

    def _rollout_spread(_attention_stack, _source=1, _residual=0.5):
        _seq_len = _attention_stack[0].shape[-1]
        _source = min(_source, _seq_len - 1)
        _rollout = np.eye(_seq_len)
        for _layer_attention in _attention_stack:
            _avg_attention = _layer_attention.mean(axis=0).astype(float)
            _avg_attention = _avg_attention / np.maximum(_avg_attention.sum(axis=1, keepdims=True), 1e-8)
            _transition = _residual * np.eye(_seq_len) + (1.0 - _residual) * _avg_attention
            _rollout = _transition @ _rollout
        _influence = _rollout[:, _source]
        _exclude = np.ones(_seq_len, dtype=bool)
        _exclude[0] = False
        _exclude[_source] = False
        return float(_influence[_exclude].sum())

    def _remove_position0(_attention_stack):
        _rewired = []
        for _layer_attention in _attention_stack:
            _layer = _layer_attention.copy()
            if _layer.shape[-1] > 1:
                _layer[:, 1:, 0] = 0.0
                _layer = _layer / np.maximum(_layer.sum(axis=-1, keepdims=True), 1e-8)
            _rewired.append(_layer)
        return _rewired

    def _distribute_position0(_attention_stack, _bank_size):
        _rewired = []
        for _layer_attention in _attention_stack:
            _layer = _layer_attention.copy()
            _bank = min(_bank_size, _layer.shape[-1])
            if _bank > 1:
                _sink_mass = _layer[:, 1:, 0].copy()
                _layer[:, 1:, 0] = 0.0
                for _query in range(1, _layer.shape[-2]):
                    _available = min(_bank, _query + 1)
                    _layer[:, _query, :_available] += _sink_mass[:, _query - 1, None] / _available
                _layer = _layer / np.maximum(_layer.sum(axis=-1, keepdims=True), 1e-8)
            _rewired.append(_layer)
        return _rewired

    if attentions:
        _bank_values = sorted(set([2, 4, 8, sink_bank_size.value]))
        _original_spread = _rollout_spread(attentions)
        _no_sink_spread = _rollout_spread(_remove_position0(attentions))
        _original_pos0_mass = _mean_position0_mass(attentions)
        _rows = [
            {
                "condition": "original learned sink",
                "bank_size": 1,
                "rollout_spread": _original_spread,
                "containment_vs_no_sink": 1.0,
                "mean_pos0_mass": _original_pos0_mass,
                "pos0_reduction": 0.0,
            },
            {
                "condition": "remove position 0",
                "bank_size": 0,
                "rollout_spread": _no_sink_spread,
                "containment_vs_no_sink": 0.0,
                "mean_pos0_mass": 0.0,
                "pos0_reduction": 1.0,
            },
        ]
        _denom = max(_no_sink_spread - _original_spread, 1e-8)
        for _bank in _bank_values:
            _bank_stack = _distribute_position0(attentions, _bank)
            _spread = _rollout_spread(_bank_stack)
            _bank_pos0_mass = _mean_position0_mass(_bank_stack)
            _rows.append(
                {
                    "condition": f"distributed bank K={_bank}",
                    "bank_size": _bank,
                    "rollout_spread": _spread,
                    "containment_vs_no_sink": (_no_sink_spread - _spread) / _denom,
                    "mean_pos0_mass": _bank_pos0_mass,
                    "pos0_reduction": 1.0 - _bank_pos0_mass / max(_original_pos0_mass, 1e-8),
                }
            )
        _bank_df = pd.DataFrame(_rows)
        _selected_row = _bank_df[_bank_df["bank_size"] == sink_bank_size.value].iloc[0]
        _fig, _ax = plt.subplots(figsize=(9, 3.8))
        _colors = ["#4c78a8" if row["bank_size"] > 0 else "#d94f30" for _, row in _bank_df.iterrows()]
        _condition_positions = np.arange(len(_bank_df))
        _ax.bar(_condition_positions, _bank_df["rollout_spread"], color=_colors)
        _ax.set_title("Can a distributed sink bank replace one extreme sink?")
        _ax.set_ylabel("Rollout spread from source token")
        _ax.set_xticks(_condition_positions)
        _ax.set_xticklabels(_bank_df["condition"], rotation=30, ha="right", fontsize=8)
        _ax.grid(axis="y", alpha=0.25)
        _fig.tight_layout()

        _tradeoff_values = [
            100 * _selected_row["containment_vs_no_sink"],
            100 * _selected_row["pos0_reduction"],
        ]
        _tradeoff_fig, _tradeoff_ax = plt.subplots(figsize=(8, 2.5))
        _tradeoff_ax.barh(
            ["Containment retained", "Position-0 concentration removed"],
            _tradeoff_values,
            color=["#2b6f8a", "#e3a12f"],
        )
        _tradeoff_ax.set_xlim(0, max(105, max(_tradeoff_values) * 1.08))
        _tradeoff_ax.set_xlabel("Percent relative to the original/no-sink gap")
        _tradeoff_ax.set_title(f"The design tradeoff at K={sink_bank_size.value}")
        for _bar, _value in zip(_tradeoff_ax.patches, _tradeoff_values):
            _tradeoff_ax.text(
                _bar.get_width() + 1,
                _bar.get_y() + _bar.get_height() / 2,
                f"{_value:.1f}%",
                va="center",
                fontsize=9,
            )
        _tradeoff_ax.grid(axis="x", alpha=0.25)
        _tradeoff_fig.tight_layout()
        distributed_sink_view = mo.vstack(
            [
                mo.md("## 4. Distributed sink bank"),
                mo.md(
                    f"""
    - Research question: can we preserve the useful no-op/routing effect without concentrating attention on one token?
    - Attention surgery: take attention mass originally sent to position `0` and spread it over the first `K` causally available positions.
    - Selected `K`: **{sink_bank_size.value}**
    - Original rollout spread: **{_original_spread:.4f}**
    - No-position-0 spread: **{_no_sink_spread:.4f}**
    - Distributed bank spread at `K={sink_bank_size.value}`: **{_selected_row['rollout_spread']:.4f}**
    - Containment retained vs no-sink: **{100 * _selected_row['containment_vs_no_sink']:.1f}%**
    - Position-0 concentration reduced: **{100 * _selected_row['pos0_reduction']:.1f}%**
    """
                ),
                mo.callout(
                    f"At K={sink_bank_size.value}, this run retains {100 * _selected_row['containment_vs_no_sink']:.1f}% of containment while removing {100 * _selected_row['pos0_reduction']:.1f}% of the position-0 concentration.",
                    kind="success",
                ),
                _tradeoff_fig,
                _fig,
                mo.callout(
                    "This is attention surgery, not a trained architecture. It establishes feasibility and motivates learned sink banks; it does not establish language-model quality.",
                    kind="info",
                ),
                mo.ui.table(_bank_df.round(4).to_dict("records"), pagination=False),
            ]
        )
    else:
        distributed_sink_view = mo.md("## 4. Distributed sink bank\n\nLoad a model to test distributed sink attention surgery.")
    distributed_sink_view
    return


@app.cell
def _(
    clean_prompt,
    experiment_ready,
    load_causal_lm,
    max_tokens,
    mo,
    np,
    pd,
    plt,
    run_scaling_study,
    sink_bank_size,
    sink_threshold,
):
    def _scaling_rollout_spread(_attention_stack, _source=1, _residual=0.5):
        _seq_len = _attention_stack[0].shape[-1]
        if _seq_len < 3:
            return 0.0
        _source = min(_source, _seq_len - 1)
        _rollout = np.eye(_seq_len)
        for _layer_attention in _attention_stack:
            _avg_attention = _layer_attention.mean(axis=0).astype(float)
            _avg_attention /= np.maximum(_avg_attention.sum(axis=1, keepdims=True), 1e-8)
            _transition = _residual * np.eye(_seq_len) + (1.0 - _residual) * _avg_attention
            _rollout = _transition @ _rollout
        _influence = _rollout[:, _source]
        _keep = np.ones(_seq_len, dtype=bool)
        _keep[0] = False
        _keep[_source] = False
        return float(_influence[_keep].sum())

    def _scaling_rewire(_attention_stack, _bank_size):
        _rewired = []
        for _layer_attention in _attention_stack:
            _layer = _layer_attention.copy()
            if _layer.shape[-1] > 1:
                _sink_mass = _layer[:, 1:, 0].copy()
                _layer[:, 1:, 0] = 0.0
                if _bank_size > 0:
                    _bank = min(_bank_size, _layer.shape[-1])
                    for _query in range(1, _layer.shape[-2]):
                        _available = min(_bank, _query + 1)
                        _layer[:, _query, :_available] += _sink_mass[:, _query - 1, None] / _available
                _layer /= np.maximum(_layer.sum(axis=-1, keepdims=True), 1e-8)
            _rewired.append(_layer)
        return _rewired

    def _run_scaling_model(_name, _label, _prompt, _limit):
        import time
        import torch

        _tokenizer, _model = load_causal_lm(_name)
        _device = next(_model.parameters()).device
        _encoded = _tokenizer(
            _prompt,
            return_tensors="pt",
            truncation=True,
            max_length=_limit,
        )
        _encoded = {_key: _value.to(_device) for _key, _value in _encoded.items()}
        if _device.type == "cuda":
            torch.cuda.synchronize()
        elif _device.type == "mps":
            torch.mps.synchronize()
        _started = time.perf_counter()
        with torch.no_grad():
            _outputs = _model(**_encoded, output_attentions=True)
        if _device.type == "cuda":
            torch.cuda.synchronize()
        elif _device.type == "mps":
            torch.mps.synchronize()
        _seconds = time.perf_counter() - _started
        _stack = [_attn[0].float().detach().cpu().numpy() for _attn in _outputs.attentions]
        _sink_values = np.concatenate(
            [_attn[:, 1:, 0].mean(axis=1) for _attn in _stack if _attn.shape[-1] > 1]
        )
        _original = _scaling_rollout_spread(_stack)
        _without_sink = _scaling_rollout_spread(_scaling_rewire(_stack, 0))
        _bank_stack = _scaling_rewire(_stack, sink_bank_size.value)
        _bank = _scaling_rollout_spread(_bank_stack)
        _bank_sink_values = np.concatenate(
            [_attn[:, 1:, 0].mean(axis=1) for _attn in _bank_stack if _attn.shape[-1] > 1]
        )
        _denom = max(_without_sink - _original, 1e-8)
        _retained = (_without_sink - _bank) / _denom
        _sink_reduction = 1.0 - _bank_sink_values.mean() / max(_sink_values.mean(), 1e-8)
        _config = _model.config
        return {
            "model": _label,
            "family": "Qwen2.5" if _label.startswith("Qwen") else "SmolLM2",
            "parameters_m": sum(_parameter.numel() for _parameter in _model.parameters()) / 1e6,
            "layers": int(getattr(_config, "num_hidden_layers", len(_stack))),
            "heads": int(getattr(_config, "num_attention_heads", _stack[0].shape[0])),
            "tokens": int(_encoded["input_ids"].shape[-1]),
            "mean_sink_mass": float(_sink_values.mean()),
            "heads_above_epsilon_pct": float(100.0 * (_sink_values >= sink_threshold.value).mean()),
            "original_spread": _original,
            "no_sink_spread": _without_sink,
            "bank_spread": _bank,
            "containment_retained_pct": float(100.0 * _retained),
            "pos0_sink_reduction_pct": float(100.0 * _sink_reduction),
            "device": str(_device),
            "forward_seconds": _seconds,
        }

    if run_scaling_study.value and experiment_ready:
        _suite = [
            ("HuggingFaceTB/SmolLM2-135M", "SmolLM2 135M"),
            ("HuggingFaceTB/SmolLM2-360M-Instruct", "SmolLM2 360M"),
            ("Qwen/Qwen2.5-0.5B", "Qwen2.5 500M"),
            ("Qwen/Qwen2.5-1.5B", "Qwen2.5 1.5B"),
        ]
        _comparison_limit = min(max_tokens.value, 96)
        _comparison_prompt = "\n\n".join([clean_prompt] * 4)
        _scaling_rows = []
        _scaling_errors = []
        for _model_id, _model_label in _suite:
            try:
                _scaling_rows.append(
                    _run_scaling_model(
                        _model_id,
                        _model_label,
                        _comparison_prompt,
                        _comparison_limit,
                    )
                )
            except Exception as _exc:
                _scaling_errors.append(f"{_model_label}: {type(_exc).__name__}: {_exc}")

        if _scaling_rows:
            _scaling_df = pd.DataFrame(_scaling_rows)
            _plot_df = _scaling_df.sort_values("parameters_m")
            _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 4.1))
            for _family, _color in [("SmolLM2", "#2b6f8a"), ("Qwen2.5", "#d95f43")]:
                _family_df = _plot_df[_plot_df["family"] == _family]
                _ax1.plot(
                    _family_df["parameters_m"],
                    _family_df["mean_sink_mass"],
                    marker="o",
                    linewidth=2,
                    color=_color,
                    label=_family,
                )
                for _, _row in _family_df.iterrows():
                    _ax1.annotate(_row["model"], (_row["parameters_m"], _row["mean_sink_mass"]), xytext=(4, 5), textcoords="offset points", fontsize=8)
            _ax1.set_xlabel("Model parameters (millions)")
            _ax1.set_ylabel("Mean attention to position 0")
            _ax1.set_title("Does sink strength scale?")
            _ax1.grid(alpha=0.25)
            _ax1.legend()

            _bar_x = np.arange(len(_plot_df))
            _bar_width = 0.38
            _ax2.bar(
                _bar_x - _bar_width / 2,
                _plot_df["containment_retained_pct"],
                width=_bar_width,
                color="#4c78a8",
                label="containment retained",
            )
            _ax2.bar(
                _bar_x + _bar_width / 2,
                _plot_df["pos0_sink_reduction_pct"],
                width=_bar_width,
                color="#e3a12f",
                label="position-0 sink reduced",
            )
            _ax2.axhline(100, color="#444444", linewidth=1, linestyle="--", alpha=0.6)
            _ax2.set_xticks(_bar_x)
            _ax2.set_xticklabels(_plot_df["model"], rotation=20, ha="right")
            _ax2.set_ylabel("Percent (%)")
            _ax2.set_title(f"Distributed bank K={sink_bank_size.value}")
            _ax2.grid(axis="y", alpha=0.25)
            _ax2.legend(fontsize=8)
            _fig.tight_layout()

            _successful = len(_scaling_df)
            _best_row = _scaling_df.loc[_scaling_df["containment_retained_pct"].idxmax()]
            _min_retention = _scaling_df["containment_retained_pct"].min()
            _min_reduction = _scaling_df["pos0_sink_reduction_pct"].min()
            _error_note = ""
            if _scaling_errors:
                _error_note = "\n\nModels that could not run: " + "; ".join(_scaling_errors)
            scaling_study_view = mo.vstack(
                [
                    mo.md("## 5. Does the sink-bank result survive larger models?"),
                    mo.md(
                        f"""
    - Same repeated prompt, up to **{_comparison_limit} tokens**, across **{_successful} models**.
    - Models span **{_scaling_df['parameters_m'].min():.0f}M to {_scaling_df['parameters_m'].max():.0f}M parameters**.
    - The paper's sink metric uses the current epsilon: **{sink_threshold.value:.2f}**.
    - The intervention spreads position-0 attention over the first `K` causally available positions; it does not retrain weights.
    - Strongest retention in this run: **{_best_row['model']} at {_best_row['containment_retained_pct']:.1f}%**.
    - Across successful models, the bank retains at least **{_min_retention:.1f}%** of containment while reducing position-0 concentration by at least **{_min_reduction:.1f}%**.
    {_error_note}
    """
                    ),
                    mo.callout(
                        "A result that survives both SmolLM and Qwen is stronger evidence for the mechanism. It still motivates a training experiment; attention surgery shows feasibility, not final model quality.",
                        kind="success",
                    ),
                    _fig,
                    mo.ui.table(_scaling_df.round(4).to_dict("records"), pagination=False),
                ]
            )
        else:
            scaling_study_view = mo.callout(
                "The scaling study could not run. " + "; ".join(_scaling_errors),
                kind="warn",
            )
    else:
        _reference_df = pd.DataFrame(
            [
                {"model": "SmolLM2 135M", "parameters_m": 134.5, "mean_sink_mass": 0.3709, "heads_above_epsilon_pct": 52.2, "k4_containment_retained_pct": 99.6},
                {"model": "SmolLM2 360M", "parameters_m": 361.8, "mean_sink_mass": 0.5178, "heads_above_epsilon_pct": 79.2, "k4_containment_retained_pct": 99.7},
                {"model": "Qwen2.5 500M", "parameters_m": 494.0, "mean_sink_mass": 0.3867, "heads_above_epsilon_pct": 64.0, "k4_containment_retained_pct": 98.0},
                {"model": "Qwen2.5 1.5B", "parameters_m": 1543.7, "mean_sink_mass": 0.4202, "heads_above_epsilon_pct": 72.3, "k4_containment_retained_pct": 99.3},
            ]
        )
        _reference_fig, _reference_ax = plt.subplots(figsize=(9, 3.2))
        _reference_positions = np.arange(len(_reference_df))
        _reference_ax.bar(
            _reference_positions,
            _reference_df["k4_containment_retained_pct"],
            color=["#2b6f8a", "#2b6f8a", "#d95f43", "#d95f43"],
        )
        _reference_ax.set_ylim(0, 105)
        _reference_ax.set_xticks(_reference_positions)
        _reference_ax.set_xticklabels(_reference_df["model"], rotation=18, ha="right")
        _reference_ax.set_ylabel("Containment retained (%)")
        _reference_ax.set_title("Validated reference run: causal bank K=4")
        for _bar, _value in zip(_reference_ax.patches, _reference_df["k4_containment_retained_pct"]):
            _reference_ax.text(
                _bar.get_x() + _bar.get_width() / 2,
                _value + 0.15,
                f"{_value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        _reference_ax.grid(axis="y", alpha=0.25)
        _reference_fig.tight_layout()
        scaling_study_view = mo.vstack(
            [
                mo.md("## 5. Does the sink-bank result survive larger models?"),
                mo.md(
                    """
    The reference run spans two SmolLM and two Qwen checkpoints. Every model formed a substantial
    position-0 sink, and a causal K=4 bank retained at least **98.0%** of the rollout containment.

    Enable **Run 135M-1.5B scaling study** to reproduce the comparison on the current prompt and accelerator.
    """
                ),
                mo.callout(
                    "Reference configuration: 48 tokens, epsilon 0.30, K=4, Apple MPS. These values are a recorded baseline; the checkbox replaces them with a live GPU-backed run.",
                    kind="info",
                ),
                _reference_fig,
                mo.ui.table(_reference_df.to_dict("records"), pagination=False),
            ]
        )
    scaling_study_view
    return


@app.cell
def _(metrics_df, mo):
    if len(metrics_df):
        _top_heads = metrics_df.sort_values("sink_mass_to_pos0", ascending=False).head(12)
        _table = mo.ui.table(
            _top_heads.round(4).to_dict("records"),
            pagination=False,
            selection=None,
        )
        sink_table_view = mo.vstack(
            [
                mo.md("## Detailed sink heads"),
                mo.md(
                    """
    - Each row is one layer/head pair.
    - Higher `sink_mass_to_pos0` means more attention to the first token.
    - The paper-style metric counts how many rows pass epsilon.
    - Use the top row to choose a layer/head for the plots below.
    """
                ),
                _table,
            ]
        )
    else:
        sink_table_view = mo.md("## Detailed sink heads\n\nRun the model to populate this table.")
    sink_table_view
    return


@app.cell
def _(metrics_df, mo, plt):
    if len(metrics_df):
        _pivot = metrics_df.pivot(index="layer", columns="head", values="sink_mass_to_pos0")
        _fig, _ax = plt.subplots(figsize=(10, 4))
        _im = _ax.imshow(_pivot.values, aspect="auto", cmap="magma", vmin=0)
        _ax.set_title("Mean attention mass sent to token position 0")
        _ax.set_xlabel("Head")
        _ax.set_ylabel("Layer")
        _ax.set_xticks(range(_pivot.shape[1]))
        _ax.set_yticks(range(_pivot.shape[0]))
        _fig.colorbar(_im, ax=_ax, label="sink_mass_to_pos0")
        _fig.tight_layout()
        sink_heatmap_view = mo.vstack(
            [
                mo.md("## Detailed sink map"),
                mo.md(
                    """
    - Rows: layers.
    - Columns: heads.
    - Brighter cells: heads that send more attention to position 0.
    - Interpretation: only some heads may show high `sink_mass_to_pos0`; the pattern can be specialized.
    """
                ),
                _fig,
            ]
        )
    else:
        sink_heatmap_view = mo.md("No sink heatmap yet.")
    sink_heatmap_view
    return


@app.cell
def _(
    clean_prompt,
    dummy_sink_tokens,
    experiment_ready,
    load_causal_lm,
    mo,
    model_name,
    np,
    token_limit,
):
    mo.stop(
        not experiment_ready,
        mo.md("## Optional probe A: prompt-time sink candidates\n\nRun the selected experiment first."),
    )

    def mean_sink_for_prompt(prompt):
        import torch

        tokenizer, model = load_causal_lm(model_name.value)
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=token_limit,
        )
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded, output_attentions=True)
        attention_stack = [attn[0].float().detach().cpu().numpy() for attn in outputs.attentions]
        per_layer = [float(attn[:, 1:, 0].mean()) for attn in attention_stack if attn.shape[-1] > 1]
        return float(np.mean(per_layer)), per_layer

    if dummy_sink_tokens.value:
        try:
            clean_score, _ = mean_sink_for_prompt(clean_prompt)
            prefixed_prompt = f"{' '.join(['<sink>'] * dummy_sink_tokens.value)}\n{clean_prompt}"
            prefixed_score, _ = mean_sink_for_prompt(prefixed_prompt)
            delta = prefixed_score - clean_score
            direction = "increased" if delta > 0 else "decreased"
            if abs(delta) < 0.002:
                _takeaway = "Little change."
            elif delta > 0:
                _takeaway = "Prefix increased `sink_mass_to_pos0`."
            else:
                _takeaway = "Prefix reduced `sink_mass_to_pos0`."
            intervention_view = mo.vstack(
                [
                    mo.md("## Optional probe A: prompt-time sink candidates"),
                    mo.callout(
                        f"Mean `sink_mass_to_pos0` {direction} by {abs(delta):.4f} "
                        f"({clean_score:.4f} -> {prefixed_score:.4f}). {_takeaway}",
                        kind="info",
                    ),
                    mo.md(
                        "- This measures whether the new position `0` attracts attention. A change alone does not prove a new sink was created; probe B shows where the attention went."
                    ),
                ]
            )
        except Exception as exc:
            intervention_view = mo.callout(
                f"Intervention comparison could not run yet: `{type(exc).__name__}: {exc}`",
                kind="warn",
            )
    else:
        intervention_view = mo.md(
            """
    ## Optional probe A: prompt-time sink candidates

    - Set **Dummy prefix tokens** above 0.
    - This adds the ordinary text `<sink>` before the real prompt; it is not a learned special token.
    - The resulting subword tokens are prompt-time sink candidates.
    - If `sink_mass_to_pos0` changes, use probe B to check whether attention moved or merely diluted.
    - Model-building angle: future architectures could expose explicit no-op or routing tokens instead of relying on accidental first-token sinks.
    """
        )
    intervention_view
    return


@app.cell
def _(
    clean_prompt,
    dummy_sink_tokens,
    experiment_ready,
    load_causal_lm,
    mo,
    model_name,
    np,
    plt,
    sink_scan_width,
    token_limit,
):
    mo.stop(
        not experiment_ready,
        mo.md("## Optional probe B: can a prompt create a new sink?\n\nRun the selected experiment first."),
    )

    def _early_position_profile(prompt):
        import torch

        tokenizer, model = load_causal_lm(model_name.value)
        _encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=token_limit,
        )
        _device = next(model.parameters()).device
        _encoded = {key: value.to(_device) for key, value in _encoded.items()}
        with torch.no_grad():
            _outputs = model(**_encoded, output_attentions=True)
        _tokens = tokenizer.convert_ids_to_tokens(_encoded["input_ids"][0].detach().cpu().tolist())
        _attention_stack = [attn[0].float().detach().cpu().numpy() for attn in _outputs.attentions]
        _seq_len = len(_tokens)
        _width = min(sink_scan_width.value, _seq_len)
        _rows = []
        for _pos in range(_width):
            _per_head = []
            for _attn in _attention_stack:
                if _attn.shape[-1] > _pos and _attn.shape[1] > _pos + 1:
                    _per_head.extend(_attn[:, _pos + 1 :, _pos].mean(axis=1).tolist())
            _rows.append(
                {
                    "position": _pos,
                    "token": _tokens[_pos].replace("Ġ", "").replace("▁", "").replace("\n", "\\n"),
                    "attention_received": float(np.mean(_per_head)) if _per_head else 0.0,
                }
            )
        return _rows

    def _plot_profile(_base_rows, _prefixed_rows=None):
        _positions = [row["position"] for row in _base_rows]
        _base_values = [row["attention_received"] for row in _base_rows]
        _fig, _ax = plt.subplots(figsize=(10, 3.8))
        _width = 0.38 if _prefixed_rows else 0.65
        if _prefixed_rows:
            _prefixed_positions = [row["position"] for row in _prefixed_rows]
            _prefixed_values = [row["attention_received"] for row in _prefixed_rows]
            _ax.bar([pos - _width / 2 for pos in _positions], _base_values, width=_width, label="original prompt")
            _ax.bar(
                [pos + _width / 2 for pos in _prefixed_positions],
                _prefixed_values,
                width=_width,
                label="with dummy sink tokens",
            )
        else:
            _ax.bar(_positions, _base_values, width=_width, label="original prompt")
        if _prefixed_rows:
            _labels = [str(row["position"]) for row in _base_rows]
            _xlabel = "Position (token identity changes after prefixing)"
        else:
            _labels = [f"{row['position']}:{row['token'][:10]}" for row in _base_rows]
            _xlabel = "Early key position"
        _ax.set_title("Can prompt-time tokens create a new sink?")
        _ax.set_xlabel(_xlabel)
        _ax.set_ylabel("Average attention received from later tokens")
        _ax.set_xticks(_positions)
        _ax.set_xticklabels(_labels, rotation=45, ha="right", fontsize=8)
        _ax.grid(axis="y", alpha=0.25)
        _ax.legend()
        _fig.tight_layout()
        return _fig

    try:
        _base_rows = _early_position_profile(clean_prompt)
        _prefixed_rows = None
        if dummy_sink_tokens.value > 0:
            _prefixed_prompt = f"{' '.join(['<sink>'] * dummy_sink_tokens.value)}\n{clean_prompt}"
            _prefixed_rows = _early_position_profile(_prefixed_prompt)
        _figure = _plot_profile(_base_rows, _prefixed_rows)
        _best_base = max(_base_rows, key=lambda row: row["attention_received"])
        if _prefixed_rows:
            _best_prefixed = max(_prefixed_rows, key=lambda row: row["attention_received"])
            _takeaway = (
                f"Original strongest early sink: position {_best_base['position']} "
                f"`{_best_base['token']}` ({_best_base['attention_received']:.3f}). With dummy tokens: position "
                f"{_best_prefixed['position']} `{_best_prefixed['token']}` ({_best_prefixed['attention_received']:.3f})."
            )
        else:
            _takeaway = (
                f"Original strongest early sink in this scan: position {_best_base['position']} "
                f"with {_best_base['attention_received']:.3f} average attention received."
            )
        alternate_sink_view = mo.vstack(
            [
                mo.md("## Optional probe B: can a prompt create a new sink?"),
                mo.md(
                    """
    - Research question: is position `0` uniquely special, or can newer early positions become sink targets?
    - This scans early key positions and measures how much attention each receives from later tokens.
    - Turn up **Dummy prefix tokens** to add ordinary prompt text as early sink candidates, then compare the position profiles.
    - If attention moves to dummy positions, sink behavior is partly about available routing slots.
    - If attention stays at position `0`, prompt-time sink creation is failing; the model may need training-time or architectural sink tokens.
    """
                ),
                mo.callout(_takeaway, kind="info"),
                _figure,
            ]
        )
    except Exception as exc:
        alternate_sink_view = mo.callout(
            f"Alternative sink-location scan could not run: `{type(exc).__name__}: {exc}`",
            kind="warn",
        )
    alternate_sink_view
    return


@app.cell
def _(active_head, active_layer, attentions, mo, np, plt, tokens):
    def _short_label(token, max_len=12):
        _cleaned = token.replace("\n", "\\n")
        return _cleaned if len(_cleaned) <= max_len else _cleaned[: max_len - 1] + "…"

    if attentions:
        _attn = attentions[active_layer][active_head]
        _key_mass = _attn[1:, :].mean(axis=0)
        _top_positions = np.argsort(_key_mass)[-12:][::-1]
        _labels = [f"{int(pos)}:{_short_label(tokens[int(pos)])}" for pos in _top_positions]
        _values = [_key_mass[int(pos)] for pos in _top_positions]
        _colors = ["#d94f30" if int(pos) == 0 else "#4c78a8" for pos in _top_positions]

        _fig, _ax = plt.subplots(figsize=(9, 3.6))
        _ax.bar(range(len(_values)), _values, color=_colors)
        _ax.set_title(f"Where layer {active_layer}, head {active_head} sends attention on average")
        _ax.set_ylabel("Average attention received")
        _ax.set_xticks(range(len(_labels)))
        _ax.set_xticklabels(_labels, rotation=40, ha="right", fontsize=8)
        _ax.grid(axis="y", alpha=0.25)
        _fig.tight_layout()
        key_mass_view = mo.vstack(
            [
                mo.md("## Selected head: key-position summary"),
                mo.md(
                    """
    - Collapses the full heatmap into one bar chart.
    - Each bar is a key position.
    - Red marks position 0.
    - Taller bar: this head sends more attention there.
    - Interpretation: a tall red bar means this head prefers the shared first position.
    """
                ),
                _fig,
            ]
        )
    else:
        key_mass_view = mo.md("## Selected head: key-position summary\n\nLoad a model to see where the selected head sends attention.")
    key_mass_view
    return


@app.cell
def _(
    active_head,
    active_layer,
    attentions,
    mo,
    normalize_rows,
    np,
    plt,
    tokens,
):
    def compact_token_label(token, max_len=10):
        cleaned = token.replace("\n", "\\n")
        return cleaned if len(cleaned) <= max_len else cleaned[: max_len - 1] + "…"

    if attentions:
        _matrix = attentions[active_layer][active_head].copy()
        if normalize_rows.value:
            _matrix = _matrix / np.maximum(_matrix.max(axis=-1, keepdims=True), 1e-8)

        _labels = [compact_token_label(token) for token in tokens]
        _fig, _ax = plt.subplots(figsize=(9, 8))
        _im = _ax.imshow(_matrix, aspect="auto", cmap="viridis", vmin=0)
        _ax.set_title(f"Attention heatmap: layer {active_layer}, head {active_head}")
        _ax.set_xlabel("Key token attended to")
        _ax.set_ylabel("Query token")
        _tick_stride = max(1, len(_labels) // 24)
        _tick_positions = list(range(0, len(_labels), _tick_stride))
        _ax.set_xticks(_tick_positions)
        _ax.set_yticks(_tick_positions)
        _ax.set_xticklabels([_labels[i] for i in _tick_positions], rotation=75, ha="right", fontsize=8)
        _ax.set_yticklabels([_labels[i] for i in _tick_positions], fontsize=8)
        _fig.colorbar(_im, ax=_ax, label="row-scaled attention" if normalize_rows.value else "attention")
        _fig.tight_layout()
        head_heatmap_view = mo.vstack(
            [
                mo.md("## Selected head: full attention heatmap"),
                mo.md(
                    """
    - Rows: query tokens, the tokens asking where to attend.
    - Columns: key tokens, the tokens being attended to.
    - Bright first column: many tokens attend to position 0.
    - Diagonal band: tokens mostly attend to nearby previous tokens.
    - Interpretation: attention to position 0 and attention to previous tokens are different routing choices.
    """
                ),
                _fig,
            ]
        )
    else:
        head_heatmap_view = mo.md("No attention matrix yet.")
    head_heatmap_view
    return


@app.cell
def _(attentions, mo, plt):
    if attentions:
        _layer_means = [float(attn[:, 1:, 0].mean()) for attn in attentions if attn.shape[-1] > 1]
        _fig, _ax = plt.subplots(figsize=(8, 3))
        _ax.plot(range(len(_layer_means)), _layer_means, marker="o")
        _ax.set_title("Average sink_mass_to_pos0 by layer")
        _ax.set_xlabel("Layer")
        _ax.set_ylabel("Mean attention to position 0")
        _ax.grid(alpha=0.25)
        _fig.tight_layout()
        layer_trend_view = mo.vstack(
            [
                mo.md("## sink_mass_to_pos0 by layer"),
                mo.md(
                    """
    - X-axis: layer depth.
    - Y-axis: average attention to position 0.
    - Upward trend: later layers have higher `sink_mass_to_pos0`.
    - Downward trend: later layers have lower `sink_mass_to_pos0`.
    - Interpretation: depth matters because repeated mixing accumulates across layers.
    """
                ),
                _fig,
            ]
        )
    else:
        layer_trend_view = mo.md("No layer trend yet.")
    layer_trend_view
    return


@app.cell
def _(mo, textwrap):
    mo.md(
        textwrap.dedent(
            """
    ## Takeaway: from observation to model design

    - **Paper result:** the first token can absorb attention and help limit uncontrolled mixing through depth.
    - **Notebook replication:** removing that route increases perturbation or rollout spread for the tested models.
    - **Notebook extension:** one extreme sink is not the only possible routing pattern. A small causal bank can retain containment while reducing concentration on position `0`.
    - **Negative result:** merely typing dummy prefix tokens usually does not create equivalent trained routing slots.

    ### Training experiment this suggests

    1. Prepend `K` learned routing embeddings during pretraining.
    2. Keep them causally visible, but exclude them from next-token prediction targets.
    3. Add a small balance loss so routing attention is shared across the bank instead of collapsing onto one slot.
    4. Compare against the original architecture on validation loss, long-context tasks, exact perturbation Jacobians, and maximum attention received by any one token.

    ```python
    hidden = concat(learned_sink_bank[K], token_embeddings)
    logits, attention = transformer(hidden)
    loss = language_model_loss(logits[:, K:], targets)
    loss += balance_weight * bank_attention_imbalance(attention[..., :K])
    ```

    The attention surgery here is the feasibility check. Training determines whether the bank preserves language quality and becomes a real architectural improvement.

    [Discuss the paper on alphaXiv](https://www.alphaxiv.org/abs/2504.02732)
    """
        )
    )
    return


if __name__ == "__main__":
    app.run()
