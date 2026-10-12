"""The index of a run's dataset versions, runs/<task>/run.json. Updated each time a version is created; read by run_status and data_report.

    {"task": "...", "spec": "runs/<task>/robot_spec.json",
     "datasets": {"source": "data/<src>", "clean": "data/<src>-clean", "timed": "data/<src>-timed",
                  "labels": {"time-s20": {"train": "data/<src>-time-s20-train", "eval": "data/<src>-time-s20-eval"}}}}

Relative paths are relative to the project root, the parent of runs/. Versions not created yet are absent.
Every dataset path must name an existing LeRobot dataset when it is recorded.

Usage (from the project root):
    python -m vla_tools.run_index runs/<task> --task <name> --spec runs/<task>/robot_spec.json --source data/<src>
    python -m vla_tools.run_index runs/<task> --clean data/<src>-clean
    python -m vla_tools.run_index runs/<task> --timed data/<src>-timed
    python -m vla_tools.run_index runs/<task> --labels time-s20 data/<src>-time-s20-train data/<src>-time-s20-eval
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class LabelVersions:
    train: str
    eval: str


@dataclasses.dataclass(frozen=True)
class RunIndex:
    task: str
    spec: str
    source: str
    clean: str | None = None
    timed: str | None = None
    labels: dict[str, LabelVersions] = dataclasses.field(default_factory=dict)

    @staticmethod
    def file(run_dir: Path) -> Path:
        return Path(run_dir) / "run.json"

    @staticmethod
    def project_root(run_dir: Path) -> Path:
        return Path(run_dir).resolve().parent.parent

    @staticmethod
    def resolve(run_dir: Path, recorded: str) -> Path:
        """A path as recorded in run.json or another artifact, made absolute against the project root."""
        path = Path(recorded)
        return (path if path.is_absolute() else RunIndex.project_root(run_dir) / path).resolve()

    @staticmethod
    def load(run_dir: Path) -> RunIndex:
        raw = json.loads(RunIndex.file(run_dir).read_text(encoding="utf-8"))
        datasets = raw["datasets"]
        return RunIndex(
            task=str(raw["task"]),
            spec=str(raw["spec"]),
            source=str(datasets["source"]),
            clean=datasets.get("clean"),
            timed=datasets.get("timed"),
            labels={k: LabelVersions(train=v["train"], eval=v["eval"]) for k, v in datasets.get("labels", {}).items()},
        )

    def save(self, run_dir: Path) -> None:
        datasets: dict[str, object] = {"source": self.source}
        if self.clean is not None:
            datasets["clean"] = self.clean
        if self.timed is not None:
            datasets["timed"] = self.timed
        if self.labels:
            datasets["labels"] = {k: dataclasses.asdict(v) for k, v in self.labels.items()}
        payload = {"task": self.task, "spec": self.spec, "datasets": datasets}
        RunIndex.file(run_dir).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _recorded(run_dir: Path, given: Path, dataset: bool) -> str:
    """The path to record: relative to the project root when inside it. Refuses paths that do not exist."""
    path = given.resolve()
    if dataset and not (path / "meta" / "info.json").is_file():
        raise SystemExit(f"{given} is not a LeRobot dataset (no meta/info.json)")
    if not dataset and not path.is_file():
        raise SystemExit(f"{given} does not exist")
    root = RunIndex.project_root(run_dir)
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--task", default=None)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--clean", type=Path, default=None)
    parser.add_argument("--timed", type=Path, default=None)
    parser.add_argument("--labels", nargs=3, metavar=("NAME", "TRAIN", "EVAL"), default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    if RunIndex.file(run_dir).exists():
        index = RunIndex.load(run_dir)
    else:
        if args.task is None or args.spec is None or args.source is None:
            raise SystemExit("a new run.json needs --task, --spec and --source")
        index = RunIndex(
            task=args.task,
            spec=_recorded(run_dir, args.spec, dataset=False),
            source=_recorded(run_dir, args.source, dataset=True),
        )
    if args.task is not None:
        index = dataclasses.replace(index, task=args.task)
    if args.spec is not None:
        index = dataclasses.replace(index, spec=_recorded(run_dir, args.spec, dataset=False))
    if args.source is not None:
        index = dataclasses.replace(index, source=_recorded(run_dir, args.source, dataset=True))
    if args.clean is not None:
        index = dataclasses.replace(index, clean=_recorded(run_dir, args.clean, dataset=True))
    if args.timed is not None:
        index = dataclasses.replace(index, timed=_recorded(run_dir, args.timed, dataset=True))
    if args.labels is not None:
        name, train, evaluation = args.labels
        versions = LabelVersions(
            train=_recorded(run_dir, Path(train), dataset=True), eval=_recorded(run_dir, Path(evaluation), dataset=True)
        )
        index = dataclasses.replace(index, labels=index.labels | {name: versions})
    index.save(run_dir)
    print(RunIndex.file(run_dir).read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
