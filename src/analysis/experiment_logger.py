"""Локальный журнал экспериментов.

для каждого запуска создается
отдельная папка с параметрами, метриками, событиями, manifest и копиями ключевых
артефактов. Это помогает восстановить, чем именно был получен конкретный отчет.
"""

from __future__ import annotations

import json
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


class ExperimentLogger:
    """Файловый трекер параметров, метрик, событий и артефактов запуска."""

    def __init__(self, root: Path, run_name: str, tags: dict[str, Any] | None = None) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.started_at = time.time()
        self.run_dir = Path(root) / f"{timestamp}_{run_name}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = self.run_dir / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.metrics_path = self.run_dir / "metrics.csv"
        self.manifest_path = self.run_dir / "manifest.json"
        self.manifest: dict[str, Any] = {
            "run_name": run_name,
            "status": "running",
            "started_at": timestamp,
            "duration_sec": None,
            "tags": _to_plain(tags or {}),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "params": [],
            "metrics": [],
            "artifacts": [],
        }
        self._write_manifest()
        self.log_event("run_started", {"run_name": run_name, "tags": tags or {}})

    def log_params(self, params: dict[str, Any], filename: str = "params.yaml") -> Path:
        """Сохраняет параметры запуска в YAML и регистрирует их в manifest."""
        path = self.run_dir / filename
        with path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(_to_plain(params), file, allow_unicode=True, sort_keys=False)
        self.manifest["params"].append(_path_payload(path, kind="params"))
        self._write_manifest()
        self.log_event("params", {"path": path.as_posix()})
        return path

    def log_metrics(self, metrics: dict[str, Any], step: str | None = None) -> None:
        """Добавляет метрики в JSONL и табличный CSV-журнал."""
        timestamp = time.time()
        rows = [
            {"time": timestamp, "step": step, "metric": str(key), "value": _metric_value(value)}
            for key, value in metrics.items()
        ]
        pd.DataFrame(rows).to_csv(
            self.metrics_path,
            mode="a",
            index=False,
            header=not self.metrics_path.exists(),
            encoding="utf-8-sig",
        )
        self.manifest["metrics"].append({"step": step, "keys": sorted(str(key) for key in metrics.keys())})
        self._write_manifest()
        self.log_event("metrics", {"step": step, "metrics": metrics})

    def log_artifact(self, path: Path, name: str | None = None, copy: bool = True) -> Path:
        """Регистрирует артефакт и при необходимости копирует его внутрь папки запуска."""
        source = Path(path)
        if not source.exists():
            self.log_event("missing_artifact", {"path": source.as_posix(), "name": name or source.name})
            return source

        target = source
        if copy:
            artifact_name = name or source.name
            target = self.artifacts_dir / artifact_name
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)

        payload = _path_payload(target, kind="artifact")
        payload["source_path"] = source.resolve().as_posix()
        payload["name"] = name or source.name
        self.manifest["artifacts"].append(payload)
        self._write_manifest()
        self.log_event("artifact", payload)
        return target

    def log_artifacts(self, paths: list[Path], copy: bool = True) -> None:
        """Регистрирует набор файлов или папок как артефакты."""
        for path in paths:
            self.log_artifact(path, copy=copy)

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Добавляет событие в JSONL-журнал."""
        event = {"time": time.time(), "type": event_type, **payload}
        with self.events_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(_to_plain(event), ensure_ascii=False) + "\n")

    def finish(self, status: str = "completed", error: str | None = None) -> None:
        """Фиксирует финальный статус запуска."""
        self.manifest["status"] = status
        self.manifest["duration_sec"] = round(time.time() - self.started_at, 3)
        if error:
            self.manifest["error"] = error
        self._write_manifest()
        self.log_event("run_finished", {"status": status, "error": error})

    def _write_manifest(self) -> None:
        self.manifest_path.write_text(
            json.dumps(_to_plain(self.manifest), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _path_payload(path: Path, kind: str) -> dict[str, Any]:
    """Возвращает описание файла или папки для manifest."""
    resolved = path.resolve()
    payload: dict[str, Any] = {
        "kind": kind,
        "path": path.as_posix(),
        "absolute_path": resolved.as_posix(),
        "exists": path.exists(),
    }
    if path.is_file():
        payload["size_bytes"] = path.stat().st_size
    return payload


def _metric_value(value: Any) -> Any:
    """Готовит значение метрики для плоского CSV."""
    plain = _to_plain(value)
    if isinstance(plain, (dict, list)):
        return json.dumps(plain, ensure_ascii=False)
    return plain


def _to_plain(value: Any) -> Any:
    """Преобразует Path/numpy-типы в обычные JSON/YAML-значения."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value
