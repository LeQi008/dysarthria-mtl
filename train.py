import argparse
import itertools
import json
from pathlib import Path
from datetime import datetime
import pickle

from sklearn import metrics
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix
import numpy as np
import tqdm
import pandas as pd
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchaudio
from jiwer import wer
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Model,
)


def _prepare_cfg(raw_args=None):
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the model. \n\n"
        ),
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="N/A"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="N/A"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="N/A"
    )
    parser.add_argument(
        "--num_classes", type=int, default=2, help="N/A"
    )
    parser.add_argument(
        "--ctc_weight", type=float, default=0.5, help="N/A"
    )
    parser.add_argument(
        "--cls_weight", type=float, default=1.0, help="N/A"
    )
    parser.add_argument(
        "--csv_path", type=Path, default=Path("dataset.csv"),
    )
    parser.add_argument(
        "--target_metric", type=str, default="average_loss"
    )
    parser.add_argument(
        "--target_metric_bigger_the_better", type=bool, default=False,
    )
    parser.add_argument(
        "--root_dir", type=str, default=None,
        help="If provided, continue training. If not, start training from scratch."
    )
    parser.add_argument(
        "--prefix", type=str, default='',
        help="Custom string to add to the experiment name."
    )
    parser.add_argument(
        "--save_all_epochs", type=bool, default=False,
        help="Save all the epoch-wise models during training.",
    )
    parser.add_argument(
        "--freeze_feature_extractor", type=bool, default=True,
        help="Freeze convolution models in wav2vec2.",
    )
    parser.add_argument(
        "--pretrained_weights", type=Path, default=None,
        help="If provided, continue training from the model."
    )
    parser.add_argument(
        "--enable_cls_epochs", type=int, default=0,
    )

    args = parser.parse_args(raw_args)  # Default to sys.argv
    args.exp_name = f"{args.prefix}_cls={args.num_classes}_e={args.num_epochs}_bs={args.batch_size}_ctcW={args.ctc_weight}"

    if args.root_dir is None:
        # Train from scratch
        args.root_dir = Path(f'exp_results/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{args.exp_name}')
        args.root_dir.mkdir(parents=True, exist_ok=False)
        args.train_from_ckpt = False

    else:
        # Train from checkpoint
        args.root_dir = Path(args.root_dir)
        assert args.root_dir.exists()
        args.train_from_ckpt = True

    return args


class DysarthriaDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        row = self.df.iloc[index]

        audio, fs = torchaudio.load(row.path)
        audio = torchaudio.functional.resample(
            waveform=audio, orig_freq=fs, new_freq=16_000,
        )[0]

        audio_len = len(audio)

        cls_label = {
            0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
        }[row.category]

        ctc_label = self.tokenizer.encode(row.text)
        
        return {
            "audio": audio,
            "audio_len": audio_len,
            "cls_label": cls_label,
            "ctc_label": ctc_label,
        }

    def __len__(self):
        return len(self.df)


def _collator(batch):
    return {
        "input_values": torch.nn.utils.rnn.pad_sequence(
            [x["audio"] for x in batch],
            batch_first=True,
            padding_value=0.0,
        ),
        "input_lengths": torch.LongTensor(
            [x["audio_len"] for x in batch]
        ),
        "cls_labels": torch.LongTensor(
            [x["cls_label"] for x in batch]
        ),
        "ctc_labels": torch.nn.utils.rnn.pad_sequence(
            [torch.IntTensor(x["ctc_label"]) for x in batch],
            batch_first=True,
            padding_value=-100,
        ),
    }


def get_tokenizer(root_dir, df):
    vocabs = sorted(list(set(itertools.chain.from_iterable(
        df.text.apply(lambda x: x.split())))))
    vocab_dict = {v: i for i, v in enumerate(vocabs)}
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open(root_dir / "vocab.json", "w") as f:
        json.dump(vocab_dict, f)

    return Wav2Vec2CTCTokenizer(
        root_dir / "vocab.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token=" ",
    )


def _get_dataset(tokenizer, target_df, batch_size, **kwargs):
    return torch.utils.data.DataLoader(
        DysarthriaDataset(target_df, tokenizer),
        batch_size=batch_size, collate_fn=_collator, pin_memory=True, **kwargs,
    )


def _prepare_dataset(root_dir, df, train_from_ckpt, batch_size):
    if train_from_ckpt:
        train_df = pd.read_csv(root_dir / "train.csv")
        valid_df = pd.read_csv(root_dir / "valid.csv")
        test_df = pd.read_csv(root_dir / "test.csv")
    else:
        if "split" in df:
            # Predefined split
            train_df = df[df.split == "train"]
            valid_df = df[df.split == "valid"]
            test_df = df[df.split == "test"]
        else:
            # Random split
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=101, stratify=df["category"])
            train_df, valid_df = train_test_split(train_df, test_size=0.25, random_state=101, stratify=train_df["category"])

        train_df.to_csv(root_dir / "train.csv", index=False)
        valid_df.to_csv(root_dir / "valid.csv", index=False)
        test_df.to_csv(root_dir / "test.csv", index=False)

    tokenizer = get_tokenizer(root_dir, df)
    train_ds = _get_dataset(tokenizer, train_df, batch_size, shuffle=True)
    valid_ds = _get_dataset(tokenizer, valid_df, batch_size, shuffle=False)
    test_ds = _get_dataset(tokenizer, test_df, batch_size, shuffle=False)

    return tokenizer, train_ds, valid_ds, test_ds


class Wav2Vec2MTL(Wav2Vec2ForCTC):
    def __init__(self, config):
        super().__init__(config)
        self.cfg = config.task_specific_params

        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.cls_head = nn.Linear(config.hidden_size, self.cfg["num_classes"])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self._vocab_size = config.vocab_size
        self.enable_cls = True

    def forward(
        self,
        input_values,
        input_lengths,
        cls_labels,
        ctc_labels,
    ):
        outputs = self.wav2vec2(
            input_values,
            attention_mask=input_lengths[:, None],
            return_dict=False,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        # Classification
        _, max_state_len, _ = hidden_states.shape
        state_lens = self.wav2vec2._get_feat_extract_output_lengths(input_lengths)
        mask = (torch.arange(max_state_len)[None, :].to(state_lens.device) < state_lens[:, None])[:, :, None]
        avg_states = torch.sum(hidden_states * mask, dim=1) / torch.sum(mask, dim=1)
        cls_logits = self.cls_head(avg_states)
        cls_loss = F.cross_entropy(cls_logits, cls_labels)

        # CTC
        labels_mask = ctc_labels >= 0
        target_lengths = labels_mask.sum(-1)
        flattened_targets = ctc_labels.masked_select(labels_mask)
        ctc_logits = self.lm_head(hidden_states)
        log_probs = nn.functional.log_softmax(ctc_logits, dim=-1, dtype=torch.float32).transpose(0, 1)
        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = nn.functional.ctc_loss(
                log_probs,
                flattened_targets,
                state_lens,
                target_lengths,
                blank=self.config.pad_token_id,
                reduction=self.config.ctc_loss_reduction,
                zero_infinity=self.config.ctc_zero_infinity,
            )

        # Final loss
        if self.enable_cls:
            loss = self.cfg["cls_weight"] * cls_loss + self.cfg["ctc_weight"] * ctc_loss
        else:
            loss = self.cfg["ctc_weight"] * ctc_loss

        return (
            loss,
            cls_loss,
            ctc_loss,
            avg_states,
            hidden_states,
            cls_logits,
            ctc_logits,
        )

def _prepare_model_optimizer(args_cfg, tokenizer):
    if args_cfg.pretrained_weights is not None:
        model = Wav2Vec2MTL.from_pretrained(args_cfg.pretrained_weights)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args_cfg.learning_rate, betas=(0.9, 0.98), eps=1e-08)
    else:
        cfg, *_ = transformers.PretrainedConfig.get_config_dict("facebook/wav2vec2-xls-r-300m")
        cfg["gradient_checkpointing"] = True
        cfg["task_specific_params"] = {
            "num_classes": args_cfg.num_classes,
            "ctc_weight": args_cfg.ctc_weight,
            "cls_weight": args_cfg.cls_weight,
        }
        cfg["vocab_size"] = len(tokenizer)
        cfg["mask_time_prob"] = 0.0  # Disable time masking # Due to short audios (containing 1 word) - Leqi
        cfg["mask_feature_prob"] = 0.0  # Disable feature masking
        model = Wav2Vec2MTL.from_pretrained(
            "facebook/wav2vec2-xls-r-300m",
            config=transformers.Wav2Vec2Config.from_dict(cfg),
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args_cfg.learning_rate, betas=(0.9, 0.98), eps=1e-08)

    if args_cfg.freeze_feature_extractor:
        model.freeze_feature_encoder()

    return model, optimizer


def _eval(model, ds, tokenizer):
    def _ctc_decode(token_ids):
        # Output string with space in-between.
        return tokenizer.decode(
            token_ids=token_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
        )

    loop = tqdm.tqdm(enumerate(ds))

    cls_all, ctc_all = [], []
    cls_labels, ctc_labels = [], []
    losses, cls_losses, ctc_losses = [], [], []

    for step, x in loop:
        assert len(x["cls_labels"]) == 1

        cls_labels.append(x["cls_labels"].item())
        ctc_labels.append(_ctc_decode(x["ctc_labels"].cpu().numpy()[0]))

        x = {k: v.to(device) for k, v in x.items()}
        loss, cls_loss, ctc_loss, *_, cls_logits, ctc_logits = model(**x)

        cls_all.append(cls_logits.detach().cpu().numpy())
        pred_ids = np.argmax(ctc_logits.detach().cpu().numpy(), axis=-1)[0]
        ctc_all.append(_ctc_decode(pred_ids))

        losses.append(loss.item())
        cls_losses.append(cls_loss.item())
        ctc_losses.append(ctc_loss.item())

    cls_all = np.concatenate(cls_all)
    prob_all = softmax(cls_all, axis=1)

    return {
        "average_loss": np.array(losses).mean(),
        "average_cls_loss": np.array(cls_losses).mean(),
        "average_ctc_loss": np.array(ctc_losses).mean(),
        "accuracy": metrics.accuracy_score(
            y_true=cls_labels, y_pred=prob_all.argmax(1)),
        "precision": metrics.precision_score(
            y_true=cls_labels, y_pred=prob_all.argmax(1), average="macro"),
        "recall": metrics.recall_score(
            y_true=cls_labels, y_pred=prob_all.argmax(1), average="macro"),
        "f1": metrics.f1_score(
            y_true=cls_labels, y_pred=prob_all.argmax(1), average="macro"),
        "per": wer(truth=ctc_labels, hypothesis=ctc_all),
        "cls_labels": cls_labels,
        "cls_preds": prob_all.argmax(1).tolist(),
    }


def _train(cfg, model, train_ds, valid_ds, tokenizer, optimizer, best_ckpt_path, last_ckpt_path, all_ckpt_path, logger):
    grad_acc = cfg.batch_size
    eval_target = None
    steps = 0
    start_epoch = 0
    print(f"Batch size: {cfg.batch_size}")
    train_loss_per_epoch = []
    val_loss_per_epoch = []
    train_acc_per_epoch = []
    val_acc_per_epoch = []

    if cfg.train_from_ckpt:
        scheduler = torch.load(last_ckpt_path / "scheduler.pt")
        model.load_state_dict(Wav2Vec2MTL.from_pretrained(last_ckpt_path).to(device).state_dict())
        optimizer.load_state_dict(torch.load(last_ckpt_path / "optimizer.pt"))

        start_epoch = scheduler["last_epoch"]
        steps = start_epoch * len(train_ds)

    for epoch in range(start_epoch, cfg.num_epochs):
        epoch_train_losses = []
        correct, total = 0, 0
        # Train
        model.enable_cls = epoch >= cfg.enable_cls_epochs
        model.train()
        train_loop = tqdm.tqdm(enumerate(train_ds))

        for step, x in train_loop:
            x = {k: v.to(device) for k, v in x.items()}

            loss, cls_loss, ctc_loss, _, _, cls_logits, _ = model(**x)
            (loss / grad_acc).backward()

            _losses = {"loss": loss.item(), "ctc_loss": ctc_loss.item(), "cls_loss": cls_loss.item()}
            train_loop.set_description(
                " | ".join([f"Epoch [{epoch}] "] + [f"{k} {v:.4f}" for k, v in _losses.items()])
            )
            
            # Graph
            epoch_train_losses.append(loss.item()) 
            if model.enable_cls: # Check if classification head is activated
                preds = cls_logits.argmax(dim=1)
                labels = x["cls_labels"]
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            for k, v in _losses.items():
                logger(f"train/{k}", v, steps)
            steps += 1

            # NOTE: gradient accumulation step == batch size in this code.
            if step % grad_acc == (grad_acc - 1):
                optimizer.step()
                optimizer.zero_grad()
            
        # Graph
        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_loss_per_epoch.append(avg_train_loss)
        if model.enable_cls and total > 0:
            train_acc_per_epoch.append(correct / total)
        else:
            train_acc_per_epoch.append(None)

        # Evaluation
        model.eval()
        eval_results = _eval(model, valid_ds, tokenizer)
        for k, v in eval_results.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                logger(f"eval/{k}", v, epoch)
            else:
                # Optionally print or log that this key was skipped
                print(f"Skipping non-scalar metric: {k}")
        # Graph
        val_loss_per_epoch.append(eval_results.get("average_loss", 0.0))
        val_acc_per_epoch.append(eval_results.get("accuracy", 0.0))

        # Bestkeeping
        if (
            (eval_target is None)
            or (cfg.target_metric_bigger_the_better and eval_target <= eval_results[cfg.target_metric])
            or (not cfg.target_metric_bigger_the_better and eval_target >= eval_results[cfg.target_metric])
        ):
            # For pretty printing
            if eval_target is None:
                eval_target = 0.0
            print(
                f"Updating the model with better {cfg.target_metric}.\n"
                f"Prev: {eval_target:.4f}, Curr (epoch={epoch}): {eval_results[cfg.target_metric]:.4f}\n"
                f"Removing the previous checkpoint.\n"
            )
            eval_target = eval_results[cfg.target_metric]
            model.save_pretrained(best_ckpt_path)
            best_epoch = epoch

        # Saving everything
        if cfg.save_all_epochs:
            model.save_pretrained(all_ckpt_path / f"e-{epoch:04d}")

    # Save last model
    model.save_pretrained(last_ckpt_path)
    torch.save(optimizer.state_dict(), last_ckpt_path / "optimizer.pt")
    torch.save({"last_epoch": cfg.num_epochs}, last_ckpt_path / "scheduler.pt")

    # Save accuracy and losses to df
    df = pd.DataFrame({
            "epoch": list(range(0, len(train_loss_per_epoch))),
            "train_loss": train_loss_per_epoch,
            "val_loss": val_loss_per_epoch,
            "train_acc": train_acc_per_epoch,
            "val_acc": val_acc_per_epoch
        })
    df.to_csv("training_stats.csv", index=False)

    return best_epoch


def _get_logger(tb_path):
    writer = SummaryWriter(log_dir=tb_path)
    def _log(name, value, step=0):
        writer.add_scalar(name, value, step)
    return _log


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.version.cuda)           # Should show your CUDA version (e.g., '11.8')
    print(torch.backends.cudnn.enabled) # Should be True if CUDA works
    print(torch.cuda.device_count())    # Should be >= 1 if a GPU is available

    cfg = _prepare_cfg()
    print(cfg)

    pickle.dump(cfg, open(cfg.root_dir / "experiment_args.pkl", "wb"))
    best_ckpt_path = cfg.root_dir / "best-model-ckpt"
    last_ckpt_path = cfg.root_dir / "last-model-ckpt"
    all_ckpt_path = cfg.root_dir / "model-ckpts"
    logger = _get_logger(cfg.root_dir)

    tokenizer, train_ds, valid_ds, test_ds = _prepare_dataset(cfg.root_dir, pd.read_csv(cfg.csv_path), cfg.train_from_ckpt, cfg.batch_size)

    model, optimizer = _prepare_model_optimizer(cfg, tokenizer)
    print("Using device:", device)
    print("Model device:", next(model.parameters()).device)

    # Train & Validation loop
    best_epoch = _train(
        cfg, model, train_ds, valid_ds, tokenizer, optimizer,
        best_ckpt_path=best_ckpt_path, last_ckpt_path=last_ckpt_path, all_ckpt_path=all_ckpt_path, logger=logger)

    # Test on the best model
    best_model = Wav2Vec2MTL.from_pretrained(best_ckpt_path).to(device)
    test_results = _eval(best_model, test_ds, tokenizer)

    # Save a confusion matrix as well
    y_true = test_results["cls_labels"]
    y_pred = test_results["cls_preds"]
    cm = confusion_matrix(y_true, y_pred)
    labels = list(map(str, sorted(set(y_true) | set(y_pred))))  # Ensure all classes covered
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=labels,  # predicted classes
        y=labels,  # true classes
        annotation_text=cm.astype(str),
        colorscale='Blues'
    )
    fig.update_layout(
        title_text="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label"
    )
    fig.write_image(str(cfg.root_dir / "confusion_matrix.png"))

    print(test_results)
    for k, v in test_results.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            logger(f"test/{k}", v, 0)
        else:
            # Optionally print or log that this key was skipped
            print("We actually made it to test_result_items")
            print(f"Skipping non-scalar metric: {k}")

    test_results["best_epoch"] = best_epoch
    with open(cfg.root_dir / "test_metric_results.json", "w") as f:
        json.dump(test_results, f, indent=4)
