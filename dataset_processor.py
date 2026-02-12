from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch
from datasets import load_dataset, concatenate_datasets
from omegaconf import OmegaConf
from transformers import AutoTokenizer
import locale
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import math
import random
import numpy as np
import json



def load_config(config_path: str = './configs/dataset_config.yaml'):
    """Load configuration from a YAML file using OmegaConf.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Any: The loaded OmegaConf DictConfig.
    """
    resolved_path = os.path.abspath(config_path)
    print(f'📁 CONFIG: Loading configuration from {resolved_path}')
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Config file not found: {resolved_path}")
    config = OmegaConf.load(resolved_path)
    print(f'✅ CONFIG: Successfully loaded configuration with {len(config.hf_datasets)} datasets')
    return config


class TrainDataPreProcessor:
    def __init__(self, tokenizer_name: str, max_dur: int, language_tag: str= None) -> None:
        self.text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_dur = max_dur
        self.language_tag = language_tag
        locale.getpreferredencoding = lambda: "UTF-8"

        self.tokeniser_length = 64400
        self.start_of_text = 1
        self.end_of_text = 2
        self.start_of_speech = self.tokeniser_length + 1
        self.end_of_speech = self.tokeniser_length + 2
        self.start_of_human = self.tokeniser_length + 3
        self.end_of_human = self.tokeniser_length + 4
        self.start_of_ai = self.tokeniser_length + 5
        self.end_of_ai = self.tokeniser_length + 6
        self.pad_token = self.tokeniser_length + 7
        self.audio_tokens_start = self.tokeniser_length + 10
        self.codebook_size = 4032

    def add_codes(self, example) -> list:
        snac_layers = ['nano_layer_1', 'nano_layer_2', 'nano_layer_3', 'nano_layer_4']
        codes = [example[i] for i in snac_layers]
        codes = np.array(codes).T
        all_codes = codes + np.array([self.codebook_size * i for i in range(4)])

        # remove duplicates
        all_codes = self.remove_consecutive_duplicates_np(all_codes)

        # flatten to sequence
        all_codes = all_codes + self.audio_tokens_start
        example["codes_list"] = all_codes.flatten().tolist()
        return example

    def remove_consecutive_duplicates_np(self, arr: np.ndarray)->np.ndarray:
        if arr.ndim != 2:
            raise ValueError("2D array expected [num_frames, frame_size]")

        mask = np.any(arr[1:] != arr[:-1], axis=1)
        keep = np.insert(mask, 0, True)
        return arr[keep]


    def create_input_ids(self, example):
        if self.language_tag is not None:
            text_prompt = f"{self.language_tag.lower()}: {example['text']}"
        else:
            text_prompt = example["text"]

        text_ids = self.text_tokenizer.encode(text_prompt, add_special_tokens=True)
        text_ids.append(self.end_of_text)

        example["text_tokens"] = text_ids
        input_ids = (
            [self.start_of_human]
            + example["text_tokens"]
            + [self.end_of_human]
            + [self.start_of_ai]
            + [self.start_of_speech]
            + example["codes_list"]
            + [self.end_of_speech]
            + [self.end_of_ai]
        )
        example["input_ids"] = input_ids
        example["labels"] = input_ids
        example["attention_mask"] = [1] * len(input_ids)
        return example

    def __call__(self, dataset: Dataset) -> Dataset:
        print(f'🔄 SHARD PROCESSING: Processing shard with {len(dataset)} samples...')
        
        if self.max_dur:
            print(f'📊 FILTER: max duration is -- {self.max_dur} sec --')
            dataset_len = len(dataset)
            dataset = dataset.filter(lambda i: i['encoded_len']/12.5 <= self.max_dur)
            filtred_len = len(dataset)
            print(f'✅ COMPLETE {filtred_len} rows from {dataset_len}')

        dataset = dataset.map(  self.add_codes,
                                remove_columns=['nano_layer_1', 'nano_layer_2', 'nano_layer_3', 'nano_layer_4'],
                                desc='Add Audio Codes: ')
        dataset = dataset.filter(lambda x: x["codes_list"] is not None, desc='Check codes list')
        dataset = dataset.filter(lambda x: len(x["codes_list"]) > 0, desc='Check Codes list lenght')
        dataset = dataset.map(self.create_input_ids, remove_columns=["text", "codes_list"],
                                desc='Create input ids: ')
        
        columns_to_keep = ["input_ids", "labels", "attention_mask", "encoded_len", "speaker_emb"]
        columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
        dataset = dataset.remove_columns(columns_to_remove)
        
        print(f'✅ SHARD PROCESSING: Completed shard with {len(dataset)} samples')
        return dataset


def process_shard(shard_idx, shard_data, tokenizer_name, max_dur, language_tag):
    print(f'🚀 WORKER {shard_idx}: Starting processing...')
    processor = TrainDataPreProcessor(tokenizer_name, max_dur, language_tag)
    processed_shard = processor(shard_data)
    print(f'✅ WORKER {shard_idx}: Completed processing')
    return processed_shard


class ItemDataset:
    def __init__(self, item_cfg: OmegaConf, tokenizer_name: str, max_dur: int, n_shards: int = None):
        print(f'📦 DATASET: Loading dataset "{item_cfg.name}" from {item_cfg.reponame}...')
        self.item_cfg = item_cfg
        self.tokenizer_name = tokenizer_name
        self.max_dur = max_dur
        self.language_tag = self.item_cfg.get('language_tag')
        self.max_len = self.item_cfg.get('max_len')
        
        if n_shards is None:
            self.n_shards = min(mp.cpu_count(), 8)
        else:
            self.n_shards = n_shards
            
        self.dataset = load_dataset(
            self.item_cfg.reponame,
            self.item_cfg.name,
            split=self.item_cfg.split,
            num_proc=10
            )

        print(f'📊 DATASET: Loaded {len(self.dataset)} samples from {item_cfg.name}')
        print(f'🔧 DATASET: Will process with {self.n_shards} shards')
        
        print(f'🔄 DATASET: Renaming columns...')
        rename_dict = {
            self.item_cfg.text_col_name: 'text',
            self.item_cfg.nano_layer_1: 'nano_layer_1',
            self.item_cfg.nano_layer_2: 'nano_layer_2',
            self.item_cfg.nano_layer_3: 'nano_layer_3',
            self.item_cfg.nano_layer_4: 'nano_layer_4',
            self.item_cfg.encoded_len: 'encoded_len',
            self.item_cfg.speaker_emb: 'speaker_emb',
        }
        self.dataset = self.dataset.rename_columns(rename_dict)
        print(f'✅ DATASET: Column renaming completed for {item_cfg.name}')



    def __call__(self):
        print(f'🔄 DATASET: Starting parallel processing of {self.item_cfg.name}...')
        
        shards = []
        for i in range(self.n_shards):
            shard = self.dataset.shard(num_shards=self.n_shards, index=i)
            shards.append((shard, i))
            print(f'📦 SHARD {i}: Created with {len(shard)} samples')

        processed_shards = []
        
        with ProcessPoolExecutor(max_workers=self.n_shards) as executor:

            future_to_shard = {
                executor.submit(process_shard, shard_idx, shard, self.tokenizer_name, self.max_dur, self.language_tag): shard_idx 
                for shard, shard_idx in shards
            }
            
            for future in as_completed(future_to_shard):
                shard_idx = future_to_shard[future]
                try:
                    processed_shard = future.result()
                    processed_shards.append((shard_idx, processed_shard))
                    print(f'✅ COMPLETED: Shard {shard_idx} processing finished')
                except Exception as exc:
                    print(f'❌ ERROR: Shard {shard_idx} generated an exception: {exc}')
                    raise exc

        processed_shards.sort(key=lambda x: x[0])
        final_shards = [shard for _, shard in processed_shards]
        
        print(f'🔗 DATASET: Concatenating {len(final_shards)} processed shards...')
        final_dataset = concatenate_datasets(final_shards)
        if self.max_len is not None:
            final_dataset = final_dataset.shuffle(seed=42).select(range(int(self.max_len)))
        print(f'✅ DATASET: {self.item_cfg.name} processing completed! Final size: {len(final_dataset)} samples')
        
        return final_dataset


class DatasetProcessor:
    def __init__(self, tokenizer_name: str, n_shards_per_dataset: int = None):
        print(f'🚀 INIT: Initializing DatasetProcessor...')
        self.cfg = load_config()
        self.tokenizer_name = tokenizer_name
        self.n_shards_per_dataset = n_shards_per_dataset
        self.all_audio_lengths = []  # Store all audio lengths for statistics
        print(f'✅ INIT: DatasetProcessor initialized with {len(self.cfg.hf_datasets)} datasets to process')
        if n_shards_per_dataset:
            print(f'🔧 INIT: Each dataset will be processed with {n_shards_per_dataset} shards')


    def __call__(self):
        print(f'🔄 MASTER: Starting master dataset processing...')
        datasets = []

        for i, item_cfg in enumerate(self.cfg.hf_datasets, 1):
            print(f'📦 MASTER: Processing dataset {i}/{len(self.cfg.hf_datasets)}: {item_cfg.name}')
            item_ds_maker = ItemDataset(
                item_cfg=item_cfg,
                tokenizer_name=self.tokenizer_name,
                max_dur = self.cfg.max_duration_sec,
                n_shards=self.n_shards_per_dataset
            )
            processed_dataset = item_ds_maker()
            datasets.append(processed_dataset)

            # Collect audio lengths if statistics are enabled
            if self.cfg.lenght_statistics:
                print(f'📊 STATISTICS: Collecting audio lengths from {item_cfg.name}...')
                encoded_lengths = processed_dataset['encoded_len']
                self.all_audio_lengths.extend(encoded_lengths)
                print(f'✅ STATISTICS: Collected {len(encoded_lengths)} audio length samples')

        print(f'🔗 MASTER: Concatenating all datasets...')
        final_dataset = concatenate_datasets(datasets)
        print(f'🎉 MASTER: All datasets processed and concatenated! Final dataset size: {len(final_dataset)} samples')
        return final_dataset

    def save_length_statistics(self, output_path: str):
        """Calculate and save audio length statistics to JSON file.

        Args:
            output_path: Path where to save the statistics JSON file
        """
        if not self.all_audio_lengths:
            print(f'⚠️ STATISTICS: No audio lengths collected, skipping statistics')
            return

        print(f'📊 STATISTICS: Calculating statistics for {len(self.all_audio_lengths)} audio samples...')

        # Convert encoded_len to duration in seconds
        audio_durations_sec = np.array(self.all_audio_lengths) / 12.5

        # Calculate statistics
        statistics = {
            'total_samples': len(audio_durations_sec),
            'duration_seconds': {
                'mean': float(np.mean(audio_durations_sec)),
                'std': float(np.std(audio_durations_sec)),
                'min': float(np.min(audio_durations_sec)),
                'max': float(np.max(audio_durations_sec)),
                'median': float(np.median(audio_durations_sec)),
                'quartiles': {
                    'q25': float(np.percentile(audio_durations_sec, 25)),
                    'q50': float(np.percentile(audio_durations_sec, 50)),
                    'q75': float(np.percentile(audio_durations_sec, 75))
                }
            },
            'encoded_length': {
                'mean': float(np.mean(self.all_audio_lengths)),
                'std': float(np.std(self.all_audio_lengths)),
                'min': float(np.min(self.all_audio_lengths)),
                'max': float(np.max(self.all_audio_lengths)),
                'median': float(np.median(self.all_audio_lengths)),
                'quartiles': {
                    'q25': float(np.percentile(self.all_audio_lengths, 25)),
                    'q50': float(np.percentile(self.all_audio_lengths, 50)),
                    'q75': float(np.percentile(self.all_audio_lengths, 75))
                }
            }
        }

        # Save to JSON file
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)

        print(f'✅ STATISTICS: Statistics saved to {output_path}')
        print(f'📊 STATISTICS SUMMARY:')
        print(f'   - Total samples: {statistics["total_samples"]}')
        print(f'   - Duration mean: {statistics["duration_seconds"]["mean"]:.2f}s')
        print(f'   - Duration std: {statistics["duration_seconds"]["std"]:.2f}s')
        print(f'   - Duration range: [{statistics["duration_seconds"]["min"]:.2f}s, {statistics["duration_seconds"]["max"]:.2f}s]')
        print(f'   - Duration median: {statistics["duration_seconds"]["median"]:.2f}s')
        print(f'   - Duration Q25/Q50/Q75: {statistics["duration_seconds"]["quartiles"]["q25"]:.2f}s / {statistics["duration_seconds"]["quartiles"]["q50"]:.2f}s / {statistics["duration_seconds"]["quartiles"]["q75"]:.2f}s')



