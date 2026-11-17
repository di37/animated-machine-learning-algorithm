# Deep learning: networks trained by backpropagation (PyTorch)

Read with `SKILL.md`; its contract, truthfulness rules, branding and pipeline all apply. A network has too
many weights to show, so the story is told through what the network **does** to a fixed probe input at
real checkpoints, alongside logged curves.

## Default story

title → **data** (examples, splits, one fixed held-out **probe** set chosen up front) → **untrained network**
(its real outputs on the probe: random predictions, noisy reconstructions) → **training** (logged loss and
metrics advancing, probe outputs updating at each checkpoint) → **final model** (train vs validation, the
checkpoint kept and why) → **inference** (one held-out input through the network in eval mode, layer by
layer, with real activations and the output) → **evaluation** (test metric, confusion matrix or
reconstructions) → takeaway and limitation.

## Recording a run

Record everything in the training loop and write it to the states file; render from that file.

- **Per optimiser step:** loss (`loss.item()`), learning rate (`scheduler.get_last_lr()`), gradient norm (the
  value `torch.nn.utils.clip_grad_norm_` returns, or the norm computed before `optimizer.step()`), and
  examples seen.
- **Per epoch, or at chosen steps:** train and validation metrics computed in eval mode; a CPU copy of the
  `state_dict` at each checkpoint the video shows (`{k: v.detach().cpu().clone() ...}`).
- **At each checkpoint, on the probe set:** outputs (predictions, probabilities, reconstructions, generated
  samples), and activations from forward hooks (`register_forward_hook`) for the layers the video draws.
- **Decision surfaces (small networks on 2-D data):** model outputs on one fixed grid at each checkpoint.
- **Embeddings:** a penultimate-layer embedding of the probe set at each checkpoint, drawn through **one**
  projection (a PCA fitted once, for example on the final checkpoint) applied to every checkpoint. Refitting
  the projection per frame makes the picture jump for no reason in the model.
- **Determinism:** seed `torch`, `numpy` and `random`; set `torch.use_deterministic_algorithms(True)` where the
  ops allow it. Record the device. MPS has no float64; train in float32.

Lesson snapshots must be real states: a checkpoint is a real model; a loss point is a logged value. Between
checkpoints, cut (or crossfade), never interpolate weights or outputs.

## What to animate, by architecture

| Model | Training | Inference |
| --- | --- | --- |
| MLP (tabular or 2-D toy data) | Decision surface or fitted curve per checkpoint; weights as edge colour/width (sign and size) for small nets | Forward pass: input values, each layer's activations (ReLU zeros greyed), output |
| CNN (images) | Loss/accuracy curves; first-layer filters per checkpoint; feature maps of one probe image; confusion matrix per epoch | One held-out image through the layers: feature maps, then softmax probabilities (top-k) |
| RNN / LSTM / GRU (sequences) | Loss curves; predictions along a probe sequence per checkpoint | Step through the sequence: hidden state per time step, prediction per step |
| Autoencoder / VAE | Reconstructions of the probe set per checkpoint; reconstruction loss (and KL term for a VAE, logged separately); latent map of the probe set | Encode a held-out input, show its latent point, decode it |
| Small transformer from scratch | Loss curves; attention weights for one probe input at chosen layers and heads per checkpoint | Token-by-token prediction with next-token probabilities |
| GAN | Generator and discriminator losses (both, labelled); samples from one **fixed** noise batch per checkpoint | New samples from new noise, labelled as generated |

Explanations such as saliency maps or Grad-CAM are computed for a stated checkpoint and input, and labelled as
an attribution method, not as "what the network sees".

## Pitfalls (each one easy to get wrong)

- **Eval mode for anything shown as inference or evaluation:** `model.eval()` and `torch.no_grad()`. Dropout
  and batch norm behave differently in train mode; a train-mode output shown as a prediction is wrong.
- **Per-batch vs per-epoch loss:** a per-batch curve is noisy, an epoch average is smooth. Label which one is
  drawn. If you smooth it (EMA), say so and keep the raw curve faint behind it.
- **Train vs validation:** compute the validation metric on the validation set in eval mode, at the checkpoint
  shown. Mark the checkpoint kept (best validation, or early stopping) on the curve.
- **Augmentation and normalisation:** augmentation on training batches only; normalisation statistics from
  the training split. Probe images are shown un-augmented.
- **Big models:** do not draw every weight. Summarise with per-layer weight or update norms, a few filters, or
  one layer's activations, and say what is shown.
- **Learning-rate schedules:** show warmup and decay on the LR strip if used; a jump in the loss curve at a
  schedule boundary should be explained, not hidden.
- **Cost:** train once, then render from the states file. Never retrain while rendering.

## Takeaways and limitations worth stating

Overfitting (training loss keeps falling while validation turns), sensitivity to the learning rate and
initialisation, how much data the network needed, and that the probe set is a few examples, not the test set.
