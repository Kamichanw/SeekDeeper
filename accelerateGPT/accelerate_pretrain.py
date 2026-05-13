import argparse
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator,InitProcessGroupKwargs
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.utils.deepspeed import DummyOptim, DummyScheduler
from dotenv import load_dotenv
from datetime import timedelta
from transformers import get_cosine_schedule_with_warmup

import config
from data import load_data
from modules import GPT


def parse_args():
    parser = argparse.ArgumentParser(description="GPT pretraining with accelerate")
    parser.add_argument(
        "--loading_ratio",
        type=float,
        default=0.01,
        help="Fraction of BookCorpus to use (default: 0.01)",
    )
    parser.add_argument(
        "--use_tensorboard",
        action="store_true",
        default=False,
        help="Enable tensorboard logging",
    )
    parser.add_argument(
        "--restore_iteration",
        type=int,
        default=-1,
        help="Checkpoint iteration to restore from (-1 = no restore)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 1. Initialize Accelerator
# ---------------------------------------------------------------------------
set_seed(seed=42)

config_project = ProjectConfiguration(
    project_dir=str(config.checkpoint_dir),
    automatic_checkpoint_naming=True,
    total_limit=10,
)

kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))

accelerator = Accelerator(
    project_config=config_project,
    kwargs_handlers=[kwargs]
)

if accelerator.state.deepspeed_plugin is None:
    accelerator.gradient_accumulation_steps = config.PretrainConfig.accumulate_grad_batches

# ---------------------------------------------------------------------------
# 2. Prepare Model, Dataloader, Optimizer and Scheduler
# ---------------------------------------------------------------------------
load_dotenv()

args = parse_args()

with accelerator.main_process_first():
    tokenizer, dataloader = load_data(
        "bookcorpus", loading_ratio=args.loading_ratio, num_proc=5
    )

model = GPT(
    vocab_size=config.vocab_size,
    max_len=config.max_len,
    hidden_size=config.hidden_size,
    num_attention_heads=config.num_attention_heads,
    num_hidden_layers=config.num_hidden_layers,
    dropout=config.dropout,
)

if accelerator.state.deepspeed_plugin is not None:
    ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
else:
    ds_config = None

num_processes = accelerator.num_processes

# DummyOptim when DeepSpeed config specifies an optimizer, else Adam
if ds_config is not None and "optimizer" in ds_config:
    optimizer = DummyOptim(
        model.parameters(),
        lr=config.PretrainConfig.lr,
        weight_decay=config.weight_decay,
    )
else:
    optimizer = model.configure_optimizers(
        lr=config.PretrainConfig.lr,
        weight_decay=config.weight_decay,
        device_type=accelerator.device.type,
    )

total_steps = (
    len(dataloader)
    // num_processes
    * config.PretrainConfig.n_epoch
    // accelerator.gradient_accumulation_steps
)
warmup_steps = int(
    config.PretrainConfig.warmup_steps
    * args.loading_ratio
    // accelerator.gradient_accumulation_steps
)

# DummyScheduler when DeepSpeed config specifies a scheduler, else cosine
if ds_config is not None and "scheduler" in ds_config:
    ds_config["scheduler"]["params"]["warmup_num_steps"] = warmup_steps
    ds_config["scheduler"]["params"]["total_num_steps"] = total_steps
    scheduler = DummyScheduler(
        optimizer,
        total_num_steps=total_steps,
        warmup_num_steps=warmup_steps,
    )
else:
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_training_steps=total_steps,
        num_warmup_steps=warmup_steps,
    )

model, optimizer, scheduler, dataloader = accelerator.prepare(
    model, optimizer, scheduler, dataloader
)

# ---------------------------------------------------------------------------
# 3. Trainer Class
# ---------------------------------------------------------------------------


class TrainingStatus:
    def __init__(self, checkpoint_interval):
        self.global_step = 0
        self.checkpoint_interval = checkpoint_interval

    def state_dict(self):
        return {
            "global_step": self.global_step,
            "checkpoint_interval": self.checkpoint_interval,
        }

    def load_state_dict(self, state_dict):
        self.global_step = state_dict["global_step"]
        self.checkpoint_interval = state_dict["checkpoint_interval"]


class GPTTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        dataloader,
        optimizer,
        scheduler,
        accelerator,
        use_tensorboard=False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.status = TrainingStatus(config.checkpoint_interval)
        accelerator.register_for_checkpointing(self.status)
        self.criterion = nn.CrossEntropyLoss()

        self.writer = None
        if accelerator.is_main_process and use_tensorboard:
            self.writer = SummaryWriter(log_dir=config.checkpoint_dir / "logs")

    def split_batch(self, batch):
        src, tgt = batch[:, :-1], batch[:, 1:]
        return src, tgt

    def train(self, epoch, dataloader):
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        accelerator = self.accelerator
        status = self.status
        writer = self.writer
        criterion = self.criterion

        model.train()
        total_loss = 0
        optimizer.zero_grad()

        process_bar = tqdm(
            dataloader,
            desc=f"Training Epoch {epoch}",
            disable=not accelerator.is_main_process,
        )

        for batch in process_bar:
            with accelerator.accumulate(model):
                src, tgt = self.split_batch(batch)

                outputs = model(src)
                outputs = outputs.contiguous().view(-1, self.tokenizer.get_vocab_size())
                loss = criterion(outputs, tgt.contiguous().view(-1))

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=config.clip)
                    status.global_step += 1

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()

                if writer is not None:
                    writer.add_scalar("Loss/train", loss.item(), status.global_step)
                    writer.add_scalar(
                        "Learning_rate",
                        scheduler.get_last_lr()[0],
                        status.global_step,
                    )

        return total_loss, len(dataloader)

    def training_loop(self, restore_iteration=-1):
        accelerator = self.accelerator
        dataloader = self.dataloader
        writer = self.writer
        status = self.status

        restore_path = config.checkpoint_dir / f"checkpoints/checkpoint_{restore_iteration}"
        if restore_iteration != -1 and os.path.exists(restore_path):
            accelerator.load_state(restore_path)
            total_batches = status.global_step * accelerator.gradient_accumulation_steps
            restore_epoch = total_batches // len(dataloader)
            skip_batches = total_batches % len(dataloader)
            skipped_dataloader = accelerator.skip_first_batches(dataloader, skip_batches)
        else:
            restore_epoch = 0
            skipped_dataloader = dataloader

        for epoch in range(restore_epoch, config.PretrainConfig.n_epoch):
            if epoch == restore_epoch:
                current_dataloader = skipped_dataloader
            else:
                current_dataloader = dataloader

            local_loss, local_batch_size = self.train(epoch + 1, current_dataloader)

            if (epoch + 1) % config.checkpoint_interval == 0:
                accelerator.save_state()

            gathered_loss = accelerator.gather(torch.tensor(local_loss))
            total_samples = accelerator.gather(torch.tensor(local_batch_size))

            if writer and accelerator.is_main_process:
                avg_train_loss = gathered_loss.sum() / total_samples.sum()
                writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)

    def save_model(self):
        accelerator = self.accelerator
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_path = config.save_model_dir / "gpt_pretrained.pth"
            unwrapped_model = accelerator.unwrap_model(self.model)
            accelerator.save(unwrapped_model.state_dict(), save_path)


# ---------------------------------------------------------------------------
# 4. Main Program
# ---------------------------------------------------------------------------


def main():
    trainer = GPTTrainer(
        model,
        tokenizer,
        dataloader,
        optimizer,
        scheduler,
        accelerator,
        use_tensorboard=args.use_tensorboard,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    start_time=time.time()
    trainer.training_loop(restore_iteration=args.restore_iteration)
    end_time=time.time()

    peak_memory_byte = torch.cuda.max_memory_allocated()
    peak_memory_mb = peak_memory_byte / (1024 ** 2)

    accelerator.print(f"Training uses {end_time-start_time} seconds")
    accelerator.print(f"Training uses {peak_memory_mb} mb")

    accelerator.print(f"Training uses {end_time-start_time} seconds")

if __name__ == "__main__":
    main()
