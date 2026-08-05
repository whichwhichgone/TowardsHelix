"""
base_strategy_cogact.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""
import re
import torch
import torchvision.transforms.functional as TF
import torch.distributed as dist
import numpy as np
import wandb

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, Union
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast
from collections import OrderedDict
from PIL import Image, ImageDraw  
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import Metrics, VLAMetrics
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import PaddedCollator, PaddedCollatorForLanguageModeling, IGNORE_INDEX

from vla import CogACT
  

def _draw_latent_action_overlay(img: Image.Image, latent_action) -> Optional[Image.Image]:
    if img is None or latent_action is None:
        return None

    overlay = img.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)
    points = np.asarray(latent_action, dtype=np.float32)
    if points.size == 0:
        return None

    if points.ndim == 2:
        points = points[None, ...]
    points = points.reshape(points.shape[0], -1, 2)  # (T, N, 2)
    T, N = points.shape[0], points.shape[1]

    timestep_colors = ["red", "orange", "yellow", "lime", "cyan",
                       "blue", "purple", "magenta", "pink", "green"]
    track_colors = ["red", "lime", "cyan", "yellow", "magenta", "orange",
                    "blue", "purple", "pink", "brown"]

    def _to_pixel(x, y):
        px = int(round(float(x) / 200.0 * overlay.width))
        py = int(round(float(y) / 200.0 * overlay.height))
        return max(0, min(overlay.width - 1, px)), max(0, min(overlay.height - 1, py))

    # ── 1) Draw trajectory lines first (under dots) ──
    for n in range(N):  # each "track": same-index point across timesteps
        track_color = track_colors[n % len(track_colors)]
        prev = None
        for t in range(T):
            px, py = _to_pixel(points[t, n][0], points[t, n][1])
            if prev is not None:
                draw.line([prev, (px, py)], fill=track_color, width=2)
            prev = (px, py)

    # ── 2) Draw time-step colored dots on top ──
    for timestep, timestep_points in enumerate(points):
        color = timestep_colors[timestep % len(timestep_colors)]
        for x, y in timestep_points:
            px, py = _to_pixel(x, y)
            radius = 3
            draw.ellipse((px - radius, py - radius, px + radius, py + radius),
                         fill=color, outline="white")

    return overlay


def _has_wandb_tracker(metrics: VLAMetrics) -> bool:
    return any(tracker.__class__.__name__ == "WeightsBiasesTracker" for tracker in getattr(metrics, "trackers", []))


def _log_latent_action_overlay(batch, metrics: VLAMetrics, global_step: int) -> None:
    if not overwatch.is_rank_zero() or not _has_wandb_tracker(metrics):
        return

    debug_imgs = batch.get("debug_imgs")
    debug_latent_actions = batch.get("debug_latent_actions")
    if not debug_imgs or not debug_latent_actions:
        return

    overlay = _draw_latent_action_overlay(debug_imgs[0], debug_latent_actions[0])
    if overlay is None:
        return

    wandb.log({"VLA Train/Latent Action Overlay": wandb.Image(overlay)}, step=global_step)


def _parse_latent_action_text(text: Optional[str]) -> Optional[np.ndarray]:
    """Invert process_latent_action: parse a string like

        [(x1,y1) (x2,y2) ...];[(...)];[(...)]

    back into a (T, N, 2) array of points in the 200x200 space expected by
    _draw_latent_action_overlay. Coordinates in the text live in Qwen's 1000x1000
    grounding space, so we divide by 5 (1000 -> 200). Robust to partial / garbled
    decodes: returns None if nothing parses, and skips timesteps whose point count
    does not match the modal count rather than crashing on ragged rows.
    """
    if not text:
        return None

    coord_re = re.compile(r"\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\)")
    timesteps = []
    for seg in text.split(";"):
        pairs = coord_re.findall(seg)
        if not pairs:
            continue
        timesteps.append([(float(x), float(y)) for x, y in pairs])

    if not timesteps:
        return None

    # Keep only timesteps that share the modal point count so np.array is rectangular.
    counts = [len(t) for t in timesteps]
    modal = max(set(counts), key=counts.count)
    timesteps = [t for t in timesteps if len(t) == modal]
    if not timesteps:
        return None

    points = np.array(timesteps, dtype=np.float32)  # (T, N, 2) in 1000-space
    points = points / 5.0                           # -> 200-space
    return points


def _decode_predicted_assistant_text(output, batch, tokenizer, sample_idx: int = 0) -> Optional[str]:
    """Teacher-forced argmax decode of the supervised assistant tokens for one sample.

    Logits at position t predict the token at t+1, so we align argmax(logits[:-1])
    with labels[1:] and keep only the positions that carry LM supervision
    (labels != IGNORE_INDEX). Returns the decoded text, or None if the sample has no
    supervised tokens.
    """
    logits = getattr(output, "logits", None)
    if logits is None:
        return None

    labels = batch.get("labels")
    if labels is None:
        return None

    pred_ids = logits[sample_idx, :-1].argmax(dim=-1)          # [seq_len-1]
    lbl = labels[sample_idx, 1:].to(pred_ids.device)            # [seq_len-1]
    mask = lbl != IGNORE_INDEX
    if not bool(mask.any()):
        return None

    selected = pred_ids[mask].cpu()
    return tokenizer.decode(selected, skip_special_tokens=True)


def _log_predicted_latent_action_overlay(batch, output, model, metrics: VLAMetrics, global_step: int) -> None:
    """Overlay the model's *predicted* latent action points on batch[0]'s scene image.

    Mirrors _log_latent_action_overlay but decodes the points from the model's own LM
    token predictions (teacher-forced argmax) instead of the ground truth. A failure
    here must never interrupt training, so the body is wrapped in try/except.
    """
    if not overwatch.is_rank_zero() or not _has_wandb_tracker(metrics):
        return

    debug_imgs = batch.get("debug_imgs")
    if not debug_imgs:
        return

    try:
        vlm = getattr(model, "vlm", None)
        vlm_backbone = getattr(vlm, "vlm_backbone", None)
        tokenizer = getattr(vlm_backbone, "tokenizer", None)
        if tokenizer is None:
            return

        text = _decode_predicted_assistant_text(output, batch, tokenizer)
        points = _parse_latent_action_text(text)
        overlay = _draw_latent_action_overlay(debug_imgs[0], points)
        if overlay is None:
            return

        wandb.log({"VLA Train/Predicted Latent Action Overlay": wandb.Image(overlay)}, step=global_step)
    except Exception as e:  # noqa: BLE001 - viz must never crash training
        overwatch.warning(f"Failed to log predicted latent action overlay: {e}")


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    def __init__(
        self,
        vlm: Union[PrismaticVLM, CogACT],
        device_id: int,
        stage: str,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        repeated_diffusion_steps: int = 4,
        **_: str,
    ) -> None:
        self.vlm, self.device_id, self.stage = vlm, device_id, stage

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        self.llm_transformer_layer_cls = self.vlm.llm_backbone.transformer_layer_cls

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype
        self.repeated_diffusion_steps = repeated_diffusion_steps

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None: ...

    @abstractmethod
    def load_optimizer_and_scheduler(self, checkpoint_path: str) -> None: ...
    
    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None: ...

    @abstractmethod
    def clip_grad_norm(self) -> None: ...

    def run_training(
        self,
        dataset: Dataset,
        collator: PaddedCollatorForLanguageModeling,
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
    ) -> None:
        """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`"""
        if "finetune" in stage and batch_construction_strategy == "split-modality":
            # Instantiate the split-modality sampler; if you want to extend with other batch construction schemes,
            #   (e.g., grouping by length) =>> can easily add them here!
            modality_lengths = dataset.get_modality_lengths()
            sampler = SplitModalitySampler(
                dataset,
                modality_lengths,
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )

        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        dataloader = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
        )

        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                self.vlm.train()
                sampler.set_epoch(epoch)

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dataloader):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):
                        loss, output = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            multimodal_indices=batch["multimodal_indices"],
                            repeated_diffusion_steps = self.repeated_diffusion_steps
                        )

                    # Commit Loss (Prior to Gradient Accumulation Normalization)
                    metrics.commit(loss=loss)
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)

                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                        # Push Metrics
                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                        status = metrics.push()

                        # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                        if self.max_steps is not None and metrics.global_step >= self.max_steps:
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                            dist.barrier(device_ids=[torch.cuda.current_device()])

                            return

                        # Update Progress Bar
                        progress.update()
                        progress.set_description(status)

            # Save checkpoint at end each epoch (if `self.max_steps` is None)
            if self.max_steps is None:
                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                dist.barrier(device_ids=[torch.cuda.current_device()])

    # === VLA Training ===

    def run_vla_training(
        self,
        vla_dataset: IterableDataset,
        collator: PaddedCollator,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        save_full_model: bool = True,
        action_model: bool = True,
        latent_action_viz_interval: int = 1000,
    ) -> None:
        """Run the VLA training loop for the given `dataset` and `collator`; log losses, action metrics to `metrics`."""
        assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"
        #assert self.grad_accumulation_steps == 1, "VLA training does not support gradient accumulation!"

        # Create a DataLoader =>> Set `num_workers` to 0; RLDS loader handles parallelism!
        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(
                self.epochs * (len(dataloader) // metrics.hparams["vla"]["expected_world_size"] // self.grad_accumulation_steps)
                if self.max_steps is None
                else self.max_steps
            ),
            initial=metrics.global_step,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vlm.train()

            # Zero Gradients (just in case)
            if self.vlm.use_ema is not None and self.vlm.use_ema == True:
                self.vlm.ema_diffusion.eval()
            self.optimizer.zero_grad()

            # [Contract] DataLoader wraps RLDS Loader (`.as_numpy_iterator() =>> implicit `.repeat()`)
            #   => This means looping over the DataLoader is basically "infinite" (so no outer loop over epochs).
            #      Slightly breaks default PyTorch semantics, which is why we adaptively compute `epoch` below.
            for train_idx, batch in enumerate(dataloader):
                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                with torch.autocast(
                    "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
                ):
                    if action_model:
                        loss, output, action_loss = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            actions=batch["actions"],
                            pixel_values=batch["pixel_values"],
                            pixel_utils=batch["pixel_utils"],
                            action_masks=batch["action_masks"],
                            labels=batch["labels"],
                            output_hidden_states = True,
                            repeated_diffusion_steps = 8,
                            state=batch["state"],
                            image_grid_thw=batch["image_grid_thw"],
                        )
                    else:
                        # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                        )
                        loss = output.loss
                        action_loss = None

                # Commit Loss =>> Backward!
                metric_kwargs = {"loss": loss}
                if action_loss is not None:
                    metric_kwargs["action_loss"] = action_loss
                if output.loss is not None:
                    metric_kwargs["lm_loss"] = output.loss
                metrics.commit(**metric_kwargs)
                
                normalized_loss = loss / self.grad_accumulation_steps
                normalized_loss.backward()

                # === Gradient Step ===
                # Step =>> Only if Done w/ Gradient Accumulation
                if (train_idx + 1) % self.grad_accumulation_steps == 0:
                    # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                    self.clip_grad_norm()

                    # Optimizer & LR Scheduler Step
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    if self.vlm.use_ema is not None and self.vlm.use_ema == True:
                        update_ema(self.vlm.ema_diffusion, self.vlm.action_model)
                    self.optimizer.zero_grad()
                    # Compute epoch value using number of completed gradient steps
                    prev_step_epoch = metrics.global_step // (len(vla_dataset) // self.global_batch_size)
                    epoch = (metrics.global_step + 1) // (len(vla_dataset) // self.global_batch_size)

                    # Push Metrics
                    metrics.commit(update_step_time=True, global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                    status = metrics.push()
                    if latent_action_viz_interval > 0 and metrics.global_step % latent_action_viz_interval == 0:
                        _log_latent_action_overlay(batch, metrics, metrics.global_step)
                        _log_predicted_latent_action_overlay(batch, output, self.vlm, metrics, metrics.global_step)

                    # Check for Save Interval or Max Steps & Save Checkpoint
                    if self.max_steps is not None:
                        should_terminate = metrics.global_step >= self.max_steps
                    else:
                        should_terminate = self.epochs is not None and epoch >= self.epochs

                    should_save = (
                        (metrics.global_step % save_interval) == 0 or 
                        (epoch - prev_step_epoch == 1)
                    )

                    if should_terminate or should_save:
                        self.save_checkpoint(
                            metrics.run_dir, metrics.global_step, epoch, loss.item(), only_trainable=not save_full_model
                        )
                        dist.barrier(device_ids=[torch.cuda.current_device()])

                    if should_terminate:
                        return

                # Update Progress Bar
                progress.update()
                progress.set_description(status)