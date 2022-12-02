import gc
import gzip
import jax
import numpy as np
from Bio import SeqIO
from jax_unirep import get_reps
from omegaconf import DictConfig, OmegaConf
from os import path
from tqdm import tqdm
from typing import Optional, Sequence, Union

__DIR__ = path.dirname(path.realpath(__file__))


class UniRepFeatureExtractor:
    def __init__(self,
        config: Union[DictConfig, str] = path.join(__DIR__, "config.yaml")
    ) -> None:

        if isinstance(config, str):
            config = OmegaConf.load(config)

        self.data_directory = config.data_directory.format(__dir__=__DIR__)
        self.source_files_directory = path.join(self.data_directory,
                                                config.source_files.subdirectory)
        self.unirep_config = config.unirep_features


    def extract_features(self,
        organism: int = 5664,
        proteins: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        to_npz: bool = True
    ) -> None:

        # Load protein sequences
        source = path.join(self.source_files_directory,
                           self.unirep_config.source.format(org=organism))
        sequences = dict()

        with gzip.open(source, "rt") as f:
            seq_records = (filter(lambda seq_record: seq_record.id in proteins,
                                  SeqIO.parse(f, "fasta"))
                           if proteins is not None else SeqIO.parse(f, "fasta"))

            for seq_record in seq_records:
                sequences[seq_record.id] = str(seq_record.seq)

        proteins = np.array(list(sequences.keys()))
        sequences = np.array(list(sequences.values()))

        if limit is not None:
            proteins = proteins[:limit]
            sequences = sequences[:limit]

        # Extract UniRep vectors in batches
        batches = np.arange(0, len(sequences), self.unirep_config.batch_size)
        if batches[-1] < len(sequences):
            batches = np.append(batches, len(sequences))

        H_avg = []
        progress = tqdm(zip(batches[:-1], batches[1:]), total=len(batches)-1)
        progress.set_description(f"Extracting {len(sequences)} UniRep vectors for {organism}")
        for b_start, b_end in progress:
            UniRepFeatureExtractor.reset_device_memory()
            h_avg, _, _ = get_reps(sequences[b_start:b_end])
            H_avg.append(h_avg)
        H_avg = np.concatenate(H_avg)

        # Save to npz
        if to_npz:
            output_file = path.join(self.data_directory,
                                    self.unirep_config.output_file.format(org=organism))
            np.savez_compressed(output_file, proteins=proteins, H_avg=H_avg)

        return proteins, H_avg


    def reset_device_memory(delete_objs: bool = True) -> None:
        dvals = (x for x in gc.get_objects() if isinstance(x, jax.xla.DeviceArray))
        n_deleted = 0

        for dv in dvals:
            if not isinstance(dv, jax.xla.DeviceConstant):
                try:
                    dv._check_if_deleted()
                    dv.device_buffer.delete()
                    n_deleted += 1
                except:
                    pass

            if delete_objs:
                del dv

        del dvals
        gc.collect()
        return n_deleted