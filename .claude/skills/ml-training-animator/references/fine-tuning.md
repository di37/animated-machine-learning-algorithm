# Fine-tuning pretrained transformers, LLMs and VLMs

Read with `SKILL.md`; its contract, truthfulness rules, branding and pipeline all apply. The model is
pretrained, so the story is **what fine-tuning changed**: the same held-out prompts (and images) answered
before, during and after, beside the real training curves.

Covers Hugging Face **transformers** (`Trainer`), **PEFT** (LoRA, QLoRA), **TRL** (`SFTTrainer`, and DPO/GRPO
trainers for preference and RL tuning) and **Unsloth** (faster LoRA/QLoRA with the same TRL trainers).

## Default story (about 108 s, the lesson pacing)

1. **Title:** the base model, the task, and what success looks like.
2. **Data:** a few training examples rendered through the model's chat template; token counts; which tokens are
   trained (with response-only training, the prompt tokens are masked and shown greyed).
3. **Before:** the base model's answers to the fixed held-out prompts (greedy decoding, settings shown).
4. **Training:** the logged loss, learning rate and gradient norm advancing step by step; tokens seen;
   the trainable share of parameters (for LoRA, often well under 1%).
5. **Checkpoints:** the same held-out prompts answered by each saved checkpoint, so the change is visible.
6. **Final model:** training vs evaluation loss, and the checkpoint kept.
7. **Inference:** one held-out prompt (and image) answered token by token, each step showing the real top-k
   next-token probabilities.
8. **Evaluation:** held-out loss or perplexity and a task metric (exact match, accuracy, ROUGE, a judged
   score with the judge named), before vs after, on the same evaluation set.
9. **What changed inside (optional):** LoRA update size per layer and module, or one attention map.
10. **Takeaway and limitation:** for example a small evaluation set, forgetting of other skills, or
    overfitting on a narrow dataset.

For a **VLM**, chapters 2, 3, 5 and 7 use image plus question pairs. Show the image as the processor
delivered it to the model (resized, cropped or tiled, with the real number of image tokens or patches), and
say whether the vision layers, language layers or both were tuned.

## Recording a run

**Logged curves come from the trainer, not from a re-computation.** Add a callback (transformers v5 hooks
include `on_log`, `on_evaluate`, `on_save`, `on_step_end`, `on_pre_optimizer_step`) that appends every
`logs` dict (loss, learning_rate, grad_norm, epoch) and every evaluation to the states file. Set
`logging_steps=1` for an honest curve, `eval_strategy` and `eval_steps` for evaluation points, and
`save_steps` for the checkpoints the video shows. `trainer.state.log_history` holds the same records after
the run.

```python
class Recorder(TrainerCallback):
    def __init__(self):
        self.logs, self.evals, self.checkpoints = [], [], []
    def on_log(self, args, state, control, logs=None, **kwargs):
        self.logs.append({"step": state.global_step, **(logs or {})})
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.evals.append({"step": state.global_step, **(metrics or {})})
    def on_save(self, args, state, control, **kwargs):
        self.checkpoints.append(state.global_step)
```

**Answers at each checkpoint:** after training, load each saved checkpoint (for LoRA, the base model plus
`PeftModel.from_pretrained(base, checkpoint_dir)`) and answer the fixed held-out prompts with the same
decoding settings; save the text verbatim. Generating inside `on_save` also works, but switch to eval mode and
back, and keep it out of the timed run.

**Next-token probabilities:** generate with `return_dict_in_generate=True, output_scores=True` (or
`output_logits=True`), then apply `softmax` to each step's scores for the top-k bars, or use
`model.compute_transition_scores(...)` for the chosen tokens' log-probabilities. With sampling, the scores
are after temperature and top-p processing; say which.

**Attention maps:** sdpa (the default) and flash attention do not return weights, and transformers v5 then
hands back an **empty tuple without an error**. For the probe pass, load with `attn_implementation="eager"`
(or call `model.set_attn_implementation("eager")`), ask for `output_attentions=True`, check that each map's
rows sum to 1, and state the layer, head and checkpoint.

**LoRA inside the model:** the update a LoRA module adds is `(B @ A) * (lora_alpha / r)`. Its norm per
layer and module at each checkpoint is a real, small summary. Base weights are frozen and do not change; with
QLoRA they are also stored in 4-bit. Say both.

**Version notes (transformers v5):** `eval_strategy` (not `evaluation_strategy`), `dtype=` in
`from_pretrained`, and `attn_implementation=` to choose attention. Check the installed versions of
transformers, PEFT, TRL and Unsloth before writing code, and follow their current examples.

## Unsloth

Unsloth wraps model loading and LoRA set-up, then trains with TRL's trainers, so the callback above records it
unchanged. The usual calls are below; **check them against the installed version**, whose API moves fast
(newer releases also offer `FastModel` for multimodal models):

```python
from unsloth import FastLanguageModel           # FastVisionModel for VLMs
model, tokenizer = FastLanguageModel.from_pretrained(name, max_seq_length=2048, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16, target_modules=[...],
                                         use_gradient_checkpointing="unsloth", random_state=42)
trainer = SFTTrainer(model=model, train_dataset=train, eval_dataset=held_out,
                     args=SFTConfig(logging_steps=1, eval_strategy="steps", save_steps=..., seed=42),
                     callbacks=[recorder])
FastLanguageModel.for_inference(model)          # before generating; for_training(model) to resume
```

For VLMs: `FastVisionModel.get_peft_model(..., finetune_vision_layers=..., finetune_language_layers=...)` and
`UnslothVisionDataCollator`. With Unsloth's response-only training helper, show which tokens are masked.

**Hardware:** Unsloth is built for NVIDIA GPUs (CUDA). On a Mac with Apple Silicon (MPS, no CUDA), first
check whether the installed Unsloth imports and runs. If it does not, either fine-tune a small model with
transformers and PEFT on `mps`, use a community MLX port of the Unsloth API, or train on a CUDA machine and
bring back the states file and checkpoint answers to render locally. Say in the video where it was trained.

## Visual grammar

- **Token strip:** each token as a chip, prompt tokens and masked tokens greyed, trained tokens in colour.
- **Next-token bars:** the top-k candidates with their real probabilities; the chosen token highlighted.
- **Answer panels:** before and after side by side for the same prompt, verbatim, cut marks where trimmed.
- **Curves:** loss per step (raw, faint, with a labelled smoothing), learning-rate strip (warmup and decay),
  gradient norm; checkpoint markers on the x axis, linked to the answer panels.
- **Adapter map:** layers × modules, cell colour = LoRA update norm at the shown checkpoint, one fixed scale.

## Truthfulness specific to fine-tuning

- Answers are verbatim and come from the checkpoint named on screen, asked the same prompts with the same
  decoding settings. Never write, improve or choose between answers by hand.
- Held-out prompts and images are not in the training set: check exact and near duplicates and say so.
- Training loss is next-token cross-entropy on the unmasked tokens only. It is not accuracy. Perplexity is
  `exp(mean evaluation loss)`, computed from the evaluation loss.
- Say which model answers at inference: base plus adapter, or a merged model. Do not mix them up between
  chapters.
- An attention map is one layer and head of one checkpoint on one input. Treat it as an illustration, not an
  explanation of why the model answered.
- Report the real run: base model, dataset and size, steps, sequence length, LoRA rank and alpha,
  quantisation, hardware and wall time.
