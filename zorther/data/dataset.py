"""Dataset loader."""
import csv
import glob
import json
import os
from typing import Any, Dict, Generator, Iterator, List, Union
from torch.utils.data import IterableDataset

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


class ZortherDataset(IterableDataset):

    def __init__(self, file_paths: Union[str, List[str]], file_type: str = "auto", shuffle: bool = False, split: str = "all") -> None:
        super().__init__()
        self.file_paths = self._resolve_paths(file_paths)
        self.file_type = file_type
        self.shuffle = shuffle
        self.split = split
    def _resolve_paths(self, paths: Union[str, List[str]]) -> List[str]:
        resolved: List[str] = []
        if isinstance(paths, str):
            paths = [paths]
        for p in paths:
            if os.path.isdir(p):
                resolved.extend(glob.glob(os.path.join(p, "*.*")))
            elif os.path.isfile(p):
                resolved.append(p)
            else:
                resolved.extend(glob.glob(p))
        return sorted([f for f in resolved if os.path.exists(f)])

    def _detect_file_type(self, path: str) -> str:
        _, ext = os.path.splitext(path.lower())
        if ext == ".jsonl":
            return "jsonl"
        elif ext in {".parquet", ".pq"}:
            return "parquet"
        elif ext == ".csv":
            return "csv"
        elif ext in {".txt", ".raw"}:
            return "text"
        else:
            raise ValueError(f"Unsupported file format: {path}")

    def _stream_jsonl(self, path: str) -> Generator[Dict[str, Any], None, None]:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    yield json.loads(stripped)

    def _stream_parquet(self, path: str) -> Generator[Dict[str, Any], None, None]:
        if pq is None:
            raise ImportError("pyarrow is required to read Parquet files. Install it using 'pip install pyarrow'")
        
        parquet_file = pq.ParquetFile(path)
        for i in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(i)
            dicts = table.to_pylist()
            for record in dicts:
                yield record

    def _stream_csv(self, path: str) -> Generator[Dict[str, Any], None, None]:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield dict(row)

    def _stream_text(self, path: str) -> Generator[Dict[str, Any], None, None]:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    yield {"text": stripped}

    def _get_generator(self, path: str) -> Generator[Dict[str, Any], None, None]:
        ftype = self.file_type if self.file_type != "auto" else self._detect_file_type(path)
        if ftype == "jsonl":
            return self._stream_jsonl(path)
        elif ftype == "parquet":
            return self._stream_parquet(path)
        elif ftype == "csv":
            return self._stream_csv(path)
        elif ftype == "text":
            return self._stream_text(path)
        else:
            raise ValueError(f"Invalid file type configured: {ftype}")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        target_paths = list(self.file_paths)
        if self.shuffle:
            import random
            random.shuffle(target_paths)
            
        counter = 0
        for path in target_paths:
            try:
                for sample in self._get_generator(path):
                    is_val = (counter % 20 == 0)
                    counter += 1
                    if self.split == "val" and is_val:
                        yield sample
                    elif self.split == "train" and not is_val:
                        yield sample
                    elif self.split == "all":
                        yield sample
            except Exception as e:
                raise RuntimeError(f"Error reading dataset file {path}: {str(e)}")