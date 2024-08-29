#!/usr/bin/env python3
"""Recipe for training an emotion recognition system from speech data only using IEMOCAP.
The system classifies 4 emotions ( anger, happiness, sadness, neutrality) with wav2vec2.

To run this recipe, do the following:
> python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml --data_folder /path/to/IEMOCAP_full_release

For more wav2vec2/HuBERT results, please see https://arxiv.org/pdf/2111.02735.pdf

Authors
 * Yingzhi WANG 2021
"""

import os
import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
# 增加
import csv
import torch
from enum import Enum, auto
from torch.utils.data import DataLoader
from tqdm.contrib import tqdm


class EmoIdBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + emotion classifier.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig
        #原outputs = self.modules.wav2vec2(wavs, lens)
        outputs = self.modules.mean_var_norm(wavs, lens)
        outputs = self.modules.wav2vec2(outputs)
        # print(outputs.shape)

        # last dim will be used for AdaptativeAVG pool
        # print(outputs.shape)


        # # 交換
        # outputs = torch.transpose(outputs, 1, 2)
        # print(outputs.shape)

        outputs = self.modules.output_encoder(outputs)
        # print(outputs.shape)

        # # 交換
        # outputs = torch.transpose(outputs, 1, 2)
        # print(outputs.shape)


        outputs = self.hparams.avg_pool(outputs, lens)
        outputs = outputs.view(outputs.shape[0], -1)
        outputs = self.modules.output_mlp(outputs)
        # print(outputs.shape)
        outputs = self.hparams.log_softmax(outputs)        
        # print(outputs.shape)
        return outputs
    

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        emoid, _ = batch.emo_encoded

        """to meet the input form of nll loss"""
        emoid = emoid.squeeze(1)
        loss = self.hparams.compute_cost(predictions, emoid)
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, emoid)

        return loss

    def fit_batch(self, batch):
        """Trains the parameters given a single batch in input"""

        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.wav2vec2_optimizer.step()
            self.encoder_optimizer.step()
            self.mlp_optimizer.step()

        self.wav2vec2_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        self.mlp_optimizer.zero_grad()

        return loss.detach()

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "error_rate": self.error_metrics.summarize("average"),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:

            old_lr, new_lr = self.hparams.lr_annealing(stats["error_rate"])
            sb.nnet.schedulers.update_learning_rate(self.encoder_optimizer, new_lr)
            sb.nnet.schedulers.update_learning_rate( self.mlp_optimizer, new_lr)

            (
                old_lr_wav2vec2,
                new_lr_wav2vec2,
            ) = self.hparams.lr_annealing_wav2vec2(stats["error_rate"])
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec2_optimizer, new_lr_wav2vec2
            )

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr, "wave2vec_lr": old_lr_wav2vec2},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta=stats, min_keys=["error_rate"]
            )

        # We also write statistics about test data to stdout and to logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.wav2vec2_optimizer = self.hparams.wav2vec2_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.encoder_optimizer = self.hparams.opt_class(self.modules.output_encoder.parameters())
        self.mlp_optimizer = self.hparams.opt_class(self.modules.output_mlp.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec2_opt", self.wav2vec2_optimizer
            )
            self.checkpointer.add_recoverable("encoder_optimizer", self.encoder_optimizer)
            self.checkpointer.add_recoverable("mlp_optimizer", self.mlp_optimizer)

    def zero_grad(self, set_to_none=False):
        self.wav2vec2_optimizer.zero_grad(set_to_none)
        self.encoder_optimizer.zero_grad(set_to_none)
        self.mlp_optimizer.zero_grad(set_to_none)

# ----------------------------------------------------------------------------------------------------
    def output_predictions_test_set(
            self,
            test_set,
            max_key=None,
            min_key=None,
            progressbar=None,
            test_loader_kwargs={},
        ):
            """Iterate test_set and create output file (id, predictions, true values).

            Arguments
            ---------
            test_set : Dataset, DataLoader
                If a DataLoader is given, it is iterated directly. Otherwise passed
                to ``self.make_dataloader()``.
            max_key : str
                Key to use for finding best checkpoint, passed to
                ``on_evaluate_start()``.
            min_key : str
                Key to use for finding best checkpoint, passed to
                ``on_evaluate_start()``.
            progressbar : bool
                Whether to display the progress in a progressbar.
            test_loader_kwargs : dict
                Kwargs passed to ``make_dataloader()`` if ``test_set`` is not a
                DataLoader. NOTE: ``loader_kwargs["ckpt_prefix"]`` gets
                automatically overwritten to ``None`` (so that the test DataLoader
                is not added to the checkpointer).
            """
            if progressbar is None:
                progressbar = not self.noprogressbar

            if not isinstance(test_set, DataLoader):
                test_loader_kwargs["ckpt_prefix"] = None
                test_set = self.make_dataloader(
                    test_set, Stage.TEST, **test_loader_kwargs
                )

                save_file = os.path.join(
                    self.hparams.output_folder, "predictions.csv"
                )
                with open(save_file, "w", newline="") as csvfile:
                    outwriter = csv.writer(csvfile, delimiter=",")
                    outwriter.writerow(["id", "prediction", "true_value"])

            self.on_evaluate_start(max_key=max_key, min_key=min_key)  # done before
            self.modules.eval()
            with torch.no_grad():
                for batch in tqdm(
                    test_set, dynamic_ncols=True, disable=not progressbar
                ):
                    self.step += 1

                    emo_ids = batch.id
                    true_vals = batch.emo_encoded.data.squeeze(dim=1).tolist()
                    output = self.compute_forward(batch, stage=Stage.TEST)
                    output = output.unsqueeze(1)
                    predictions = (
                        torch.argmax(output, dim=-1).squeeze(dim=1).tolist()
                    )

                    with open(save_file, "a", newline="") as csvfile:
                        outwriter = csv.writer(csvfile, delimiter=",")
                        for emo_id, prediction, true_val in zip(
                            emo_ids, predictions, true_vals
                        ):
                            outwriter.writerow([emo_id, prediction, true_val])

                    # Debug mode only runs a few batches
                    if self.debug and self.step == self.debug_batches:
                        break
            self.step = 0


class Stage(Enum):
    """Simple enum to track stage of experiments."""

    TRAIN = auto()
    VALID = auto()
    TEST = auto()
# ----------------------------------------------------------------------------------------------------

# # 新增的datapreprocessing
# from pydub import AudioSegment
# from pydub.silence import detect_nonsilent
# from pydub.utils import mediainfo
# import json
# import os
# import librosa
# import numpy as np
# import soundfile as sf
# import random
# import itertools

# def split_train_json_by_emotion(original_json_path, hparams):
#     with open(original_json_path, 'r') as f:
#         data = json.load(f)

#     emotion_data = {"neu": {}, "hap": {}, "sad": {}, "ang": {}}
#     for key, value in data.items():
#         emo = value['emo']
#         if emo in emotion_data:
#             emotion_data[emo][key] = value

#     for emo in emotion_data.keys():
#         output_path = hparams[f"{emo}_annotation"]
#         with open(output_path, 'w') as f:
#             json.dump(emotion_data[emo], f, indent=4)

# # def load_and_trim(audio_path):
# #     """Load an audio file and trim leading and trailing silence."""
# #     audio, sr = librosa.load(audio_path, sr=None)
# #     audio, _ = librosa.effects.trim(audio)
# #     return audio, sr

# def load_and_trim(audio_path, silence_threshold=-50, chunk_size=10):
#     """Load an audio file, trim leading and trailing silence, and return a numpy array and sample rate."""
#     audio = AudioSegment.from_file(audio_path)
#     sample_rate = audio.frame_rate

#     nonsilent_chunks = detect_nonsilent(
#         audio,
#         min_silence_len=chunk_size,
#         silence_thresh=silence_threshold
#     )

#     if nonsilent_chunks:
#         start = nonsilent_chunks[0][0]
#         end = nonsilent_chunks[-1][1]
#         trimmed_audio = audio[start:end]
#     else:
#         trimmed_audio = audio

#     # Convert to numpy array
#     channel_sounds = trimmed_audio.split_to_mono()
#     samples = [s.get_array_of_samples() for s in channel_sounds]

#     fp_arr = np.array(samples).T.astype(np.float32)
#     fp_arr /= np.iinfo(samples[0].typecode).max  # Normalize to [-1, 1]

#     return fp_arr, sample_rate
    
# def split_in_half(audio):
#     mid_idx = len(audio) // 2
#     return audio[:mid_idx], audio[mid_idx:]

# def process_json_file(json_path, combined_data, emo_label, base_output_folder):
#     with open(json_path, 'r') as f:
#         data = json.load(f)

#     all_audios = []
#     for key, value in data.items():
#         audio_path = value['wav']
#         audio, sr= load_and_trim(audio_path)
#         half1, half2 = split_in_half(audio)
#         all_audios.append((half1, key, "First half"))
#         all_audios.append((half2, key, "Second half"))

#     # 从 base_output_folder 中提取最后的文件夹名称
#     base_folder_name = os.path.basename(base_output_folder)
#     # 获取 base_output_folder 的上级目录名称
#     parent_folder_name = os.path.basename(os.path.dirname(base_output_folder))

#     # 构建新的 output_folder 名称
#     output_folder = os.path.join(f"output_audios_{parent_folder_name}_{base_folder_name}", f"concatenated_audios_{emo_label}")
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     max_pairs_per_segment = 4  # 每个片段的最大配对数量

#     for i, (segment, key, _) in enumerate(all_audios):
#         # 随机选择最多 max_pairs_per_segment 个其他片段进行配对
#         other_segments = all_audios[:i] + all_audios[i+1:]
#         random.shuffle(other_segments)
#         selected_pairs = other_segments[:max_pairs_per_segment]

#         for other_segment, other_key, _ in selected_pairs:
#             if data[key]["emo"] == data[other_key]["emo"]:
#                 concatenated_segment = np.concatenate([segment, other_segment])
#                 output_path = os.path.join(output_folder, f"concatenated_{emo_label}_{key}_{other_key}.wav")
#                 sf.write(output_path, concatenated_segment, sr)

#                 audio_length = len(concatenated_segment) / sr
#                 unique_key = f"{emo_label}_{key}_{other_key}"
#                 combined_data[unique_key] = {
#                     "wav": output_path,
#                     "length": audio_length,
#                     "emo": data[key]["emo"]
#                 }

#     return combined_data
    
# def merge_and_shuffle_data(original_json_path, combined_data, output_path):
#     with open(original_json_path, 'r') as f:
#         original_data = json.load(f)

#     # 将原始数据和处理后的数据合并
#     merged_data = {**original_data, **combined_data}
#     # 打乱合并后的数据
#     shuffled_keys = list(merged_data.keys())
#     random.shuffle(shuffled_keys)
#     shuffled_data = {key: merged_data[key] for key in shuffled_keys}
#     print(f"Total data after merging: {len(shuffled_data)} items")

#     # 保存打乱后的数据
#     with open(output_path, 'w') as f:
#         json.dump(shuffled_data, f, indent=4)


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined
    functions. We expect `prepare_mini_librispeech` to have been called before
    this, so that the `train.json`, `valid.json`,  and `valid.json` manifest
    files are available.
    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.
    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    # Initialization of the label encoder. The label encoder assignes to each
    # of the observed label a unique index (e.g, 'spk01': 0, 'spk02': 1, ..)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("emo")
    @sb.utils.data_pipeline.provides("emo", "emo_encoded")
    def label_pipeline(emo):
        yield emo
        emo_encoded = label_encoder.encode_label_torch(emo)
        yield emo_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "emo_encoded"],
        )
    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mappinng.

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="emo",
    )

    return datasets


# RECIPE BEGINS!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    from iemocap_prepare import prepare_data  # noqa E402

    # Data preparation, to be run on only one process.
    sb.utils.distributed.run_on_main(
        prepare_data,
        kwargs={
            "data_original": hparams["data_folder"],
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "save_json_test": hparams["test_annotation"],
            "split_ratio": [80, 10, 10],
            "different_speakers": hparams["different_speakers"],
            "test_spk_id": hparams["test_spk_id"],
            "seed": hparams["seed"],
        },
    )
    # # 新增的datapreprocessing
    # # 获取 base_output_folder 的上级目录名称和最后的文件夹名称
    # base_folder_name = os.path.basename(hparams["output_folder"])
    # parent_folder_name = os.path.basename(os.path.dirname(hparams["output_folder"]))
    # base_output_audios_folder = f"output_audios_{parent_folder_name}_{base_folder_name}"

    # # 检查所有情绪类别的文件夹是否已存在
    # emotions = ["neu", "hap", "sad", "ang"]
    # folders_exist = all(os.path.exists(os.path.join(base_output_audios_folder, f"concatenated_audios_{emo}")) for emo in emotions)

    # if not folders_exist:
    #     # 如果任何文件夹不存在，执行数据预处理步骤
    #     # 使用随机种子
    #     seed_value = hparams["seed"]
    #     random.seed(seed_value)
    #     np.random.seed(seed_value)

    #     # 切分原始的 train.json
    #     split_train_json_by_emotion(hparams["train_annotation"], hparams)

    #     # 处理每个情绪类别的数据并合并
    #     combined_data = {}
    #     for emo in emotions:
    #         json_file_path = hparams[f"{emo}_annotation"]
    #         process_json_file(json_file_path, combined_data, emo, hparams["output_folder"])

    #     # 将处理后的数据和原始数据合并，并进行打乱
    #     merge_and_shuffle_data(hparams["train_annotation"], combined_data, hparams["train_annotation"])
    # else:
    #     print("Skipping data preprocessing as all required folders exist.")

    # Create dataset objects "train", "valid", and "test".
    datasets = dataio_prep(hparams)

    hparams["wav2vec2"] = hparams["wav2vec2"].to("cuda:0")
    # freeze the feature extractor part when unfreezing
    if not hparams["freeze_wav2vec2"] and hparams["freeze_wav2vec2_conv"]:
        hparams["wav2vec2"].model.feature_extractor._freeze_parameters()

    # Initialize the Brain object to prepare for mask training.
    emo_id_brain = EmoIdBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    emo_id_brain.fit(
        epoch_counter=emo_id_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Load the best checkpoint for evaluation
    test_stats = emo_id_brain.evaluate(
        test_set=datasets["test"],
        min_key="error_rate",
        test_loader_kwargs=hparams["dataloader_options"],
    )
# -----------------------------------------------------------------
    # Create output file with predictions
    emo_id_brain.output_predictions_test_set(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["dataloader_options"],
    )
# -----------------------------------------------------------------

    #plot loss curve
    from plot_curve_wav import plot
    plot(hparams['output_folder'], hparams['number_of_epochs'])
    #plot confusionmatrix
    from plot_cmatrix import plotc
    plotc(hparams['output_folder'])