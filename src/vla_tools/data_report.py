"""Write runs/<task>/data_report.md: what the run's data stages produced and why. Reads artifacts only.

The report describes the training-ready dataset handed downstream: dataset version lineage, audit
findings and cleaning decisions, keyframe definitions and corrections, still-frame and end-padding
decisions, coverage and the held-out split, label candidates and the built label sets, and the
decisions a person made. Every number comes from an artifact; nothing is recomputed from data.

Written only when run_status reports every stage S0-S7 as passed; otherwise the statuses are printed
and nothing is written.

Usage (from the project root):
    python -m vla_tools.data_report runs/<task>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from vla_tools.decision_log import load_decisions
from vla_tools.run_index import RunIndex
from vla_tools.run_status import stage_statuses


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return [*lines, ""]


def _detector(detector: dict) -> str:
    parts = [detector["type"]]
    signal = detector.get("signal")
    if signal:
        source = signal["source"]
        parts.append(source if source == "joint_speed" else f"{signal.get('name') or source}[{signal.get('dim', -1)}]")
    for key in ("direction", "threshold", "hold_frames", "reference", "search_frames"):
        if detector.get(key) is not None:
            parts.append(f"{key}={detector[key]}")
    return " ".join(str(p) for p in parts)


class DataReport:
    """Collects the run's artifacts once; each method renders one section."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.index = RunIndex.load(self.run_dir)
        self.audit = _json(self.run_dir / "audit.json")
        self.cleaning = _json(self._path(self.index.clean) / "meta" / "cleaning_report.json")
        self.timing = _json(self._path(self.index.timed) / "meta" / "timing_report.json")
        self.keyframes = _json(self.run_dir / "keyframes.json")
        self.diversity = _json(self.run_dir / "diversity.json")
        self.sweep = _json(self.run_dir / "sweep.json")
        self.decisions = load_decisions(self.run_dir / "decisions.md")

    def _path(self, recorded: str | None) -> Path:
        return RunIndex.resolve(self.run_dir, str(recorded))

    def _size(self, recorded: str) -> tuple[int, int]:
        info = _json(self._path(recorded) / "meta" / "info.json")
        return int(info["total_episodes"]), int(info["total_frames"])

    def lineage(self) -> list[str]:
        repairs = sum(len(e["repaired_jumps"]) + len(e["removed_pulses"]) for e in self.cleaning["episodes"])
        dropped = sum(e["dropped"] for e in self.timing["episodes"].values())
        padding = self.timing["plan"].get("end_padding", 0)
        rows = [
            ["源数据", self.index.source, *self._size(self.index.source), "只读"],
            ["清洗版", self.index.clean, *self._size(self.index.clean),
             f"源数据：修复 {repairs} 处，排除 {len(self.cleaning['excluded'])} 集"],
            ["定时版", self.index.timed, *self._size(self.index.timed),
             f"清洗版：裁掉 {dropped} 帧静止帧，每集末尾补 {padding} 帧"],
        ]  # fmt: skip
        for name, versions in self.index.labels.items():
            excluded = _json(self._path(versions.train) / "meta" / "arc_resample.json")["excluded_source_episodes"]
            rows.append(
                [f"标签 {name} 训练版", versions.train, *self._size(versions.train), f"定时版：去掉留出集 {excluded}"]
            )
            rows.append(
                [
                    f"标签 {name} 评估版",
                    versions.eval,
                    *self._size(versions.eval),
                    "定时版：全部 episode，编号与定时版一致",
                ]
            )
        return ["## 数据集版本", "", *_table(["版本", "路径", "episode", "帧", "由来"], rows)]

    def audit_and_cleaning(self) -> list[str]:
        a = self.audit
        lengths = a["lengths"]
        kinds: dict[str, int] = {}
        for f in a["findings"]:
            kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
        lines = [
            "## 体检与清洗",
            "",
            f"- 规模：{a['episodes']} 集，{a['frames']} 帧，{a['fps']:g} fps；长度 {lengths['min']:.0f} / "
            f"{lengths['median']:.0f} / {lengths['max']:.0f}（最短 / 中位 / 最长）",
            f"- 索引约定：max|action[t] − state[t+1]| = {a['next_state_error_max']:.1e}，max|action[t] − state[t]| = "
            f"{a['same_state_error_max']:.1e}；`action[t]` 对应 `state[t+1]`",
            f"- 一帧运动量：{a['median_step']:.4g}；静止维：{a['static_dims'] or '无'}",
            "",
            *_table(["发现", "数量"], [[k, v] for k, v in sorted(kinds.items())]),
        ]
        plan = self.cleaning["plan"]
        decisions = plan.get("jump_decisions", {})
        lines += [
            "跳变决定：",
            "",
            *(
                _table(["位置（episode:帧）", "决定"], [[k, v] for k, v in decisions.items()])
                if decisions
                else ["无", ""]
            ),
        ]
        excluded = self.cleaning["excluded"]
        lines += [
            "排除的 episode：",
            "",
            *(_table(["episode", "理由"], [[k, v] for k, v in excluded.items()]) if excluded else ["无", ""]),
        ]
        repaired = [
            [
                e["source"],
                ", ".join(str(t) for t in e["repaired_jumps"]) or "—",
                "；".join(f"维 {d}，帧 {s}–{t - 1}" for d, s, t in e["removed_pulses"]) or "—",
            ]
            for e in self.cleaning["episodes"]
            if e["repaired_jumps"] or e["removed_pulses"]
        ]
        lines += [
            "修复：",
            "",
            *(_table(["episode", "修复的孤立跳变（帧）", "去掉的指令毛刺"], repaired) if repaired else ["无", ""]),
        ]
        return lines

    def keyframe_section(self) -> list[str]:
        kspec = self.keyframes["keyframes_spec"]
        rows = [
            [t["name"], t.get("description", ""), _detector(t["detector"]),
             f"−{t['window']['before']} / +{t['window']['after']}" + (f"，到 {t['window']['until']}" if t["window"].get("until") else ""),
             t.get("expected", -1)]
            for t in kspec["types"]
        ]  # fmt: skip
        lines = [
            "## 关键帧",
            "",
            f"任务：{kspec.get('task', '')}；末端执行器：{kspec.get('effector', '')}",
            "",
            *_table(["类型", "含义", "检测器", "窗口（帧）", "每集次数"], rows),
        ]
        overrides = self.keyframes["overrides"]
        edits = [
            [kind, o["episode"], o["type"], o.get("frame", "—"), o["reason"]]
            for kind in ("set", "add", "remove")
            for o in overrides.get(kind, [])
        ]
        lines += [
            "人工修正：",
            "",
            *(_table(["操作", "episode", "类型", "帧", "理由"], edits) if edits else ["无", ""]),
        ]
        accepted = [
            [i["episode"], i["check"], i["detail"], i["accepted"]] for i in self.keyframes["issues"] if i["accepted"]
        ]
        lines += [
            "接受的检查问题：",
            "",
            *(_table(["episode", "检查", "内容", "理由"], accepted) if accepted else ["无", ""]),
        ]
        return lines

    def timing_section(self) -> list[str]:
        plan = self.timing["plan"]
        reasons = plan.get("reasons", {})
        keys = ("threshold", "min_frames", "leading_keep", "pause_keep", "trailing_keep", "end_padding")
        rows = [[k, "全部保留" if plan.get(k) is None else plan[k], reasons.get(k, "—")] for k in keys]
        runs: dict[str, list[int]] = {}
        for episode in self.timing["episodes"].values():
            for run in episode["still_runs"]:
                runs.setdefault(run["kind"], []).append(run["end"] - run["start"])
        kinds = [[k, len(v), sum(v)] for k, v in sorted(runs.items())]
        dropped = sum(e["dropped"] for e in self.timing["episodes"].values())
        return [
            "## 静止帧与末尾补帧",
            "",
            *_table(["计划项", "值", "理由"], rows),
            *_table(["静止段", "段数", "帧数"], kinds),
            f"共裁掉 {dropped} 帧；每集末尾补 {plan.get('end_padding', 0)} 帧，补帧行的 `is_padding` 为 1。",
            "",
        ]

    def coverage_section(self) -> list[str]:
        d = self.diversity
        groups = [
            [
                name,
                g["dims"],
                len(set(g["clusters"])),
                sorted([g["clusters"].count(c) for c in set(g["clusters"])], reverse=True),
            ]
            for name, g in d["groups"].items()
        ]
        combined = d["combined_clusters"]
        sizes = sorted([combined.count(c) for c in set(combined)], reverse=True)
        split = d["split"]
        cluster_of = dict(zip(d["episodes"], combined, strict=True))
        held = [
            [e, cluster_of[e], f"{split['holdout_distance_to_train'][str(e)] / split['median_nearest_neighbour']:.2f}"]
            for e in split["holdout"]
        ]
        return [
            "## 覆盖与留出集",
            "",
            *_table(["描述组", "维数", "簇数", "各簇大小"], groups),
            f"合并后 {len(sizes)} 个簇，大小 {sizes}；近重复：{d['near_duplicates'] or '无'}；离群：{d['isolated'] or '无'}。",
            "",
            f"留出集（{split['strategy']}，seed {split['seed']}）：{split['holdout']}；训练集缺失的簇："
            f"{split['clusters_missing_from_train'] or '无'}。",
            "",
            *_table(["留出 episode", "合并簇", "到最近训练 episode 的距离（× 中位最近邻距离）"], held),
        ]

    def label_section(self) -> list[str]:
        s = self.sweep
        step = s["path_step_rad"]
        cells = {
            (c["config"]["metric"], c["config"]["span_frames"], c["config"]["event_density"]): c for c in s["cells"]
        }
        rows = []
        for span, config in s["shortlist"].items():
            c = cells[(config["metric"], config["span_frames"], config["event_density"])]
            kf = "—" if math.isnan(c["kf_position"][0]) else f"{c['kf_position'][0] / step:.2f}"
            rows.append([span, config["metric"], config["event_density"], f"{c['path_head'][0] / step:.2f}",
                         f"{c['path_full'][0] / step:.2f} / {c['path_full'][1] / step:.2f}", kf,
                         f"{c['speed_ratio'][0]:.2f} / {c['speed_ratio'][1]:.2f}", f"{c['jump_steps']:.1f}",
                         f"{c['pad_fraction']:.1%}"])  # fmt: skip
        horizon = next(iter(s["cells"]))["config"]["action_horizon"]
        lines = [
            "## 动作标签",
            "",
            f"候选（`sweep`，H = {horizon}，执行 {s['head']} 帧后重新请求）：按服务端执行的目标与录制轨迹的几何差异打分，"
            f"不按逐帧时间对齐；距离单位为路径步长 {step:.4g} rad（录制中相邻帧关节位置欧氏距离的中位数），"
            f"速度比为执行目标的最大单帧步长除以录制步长的 p95（{s['reference_step_rad']:.4g} rad）：",
            "",
            *_table(
                [
                    "span",
                    "metric",
                    "density",
                    "前段路径偏差 中位",
                    "整段路径偏差 中位 / p90",
                    "关键帧位置误差 中位",
                    "速度比 中位 / p95",
                    "首点跳变（步）",
                    "补帧点",
                ],
                rows,
            ),
            f"网格搜索选定：{s['selected']}",
            "",
        ]
        for name, versions in self.index.labels.items():
            report = _json(self._path(versions.train) / "meta" / "arc_resample.json")
            stats = list(report["episodes"].values())
            replay = _json(self.run_dir / f"replay_{name}.json")
            r_step = replay["frame_of_motion_rad"]
            replay_rows = [
                [r["execute"], f"{r['err'][0] / r_step:.2f} / {r['err'][1] / r_step:.2f}", f"{r['seam'][0]:.2f} / {r['seam'][1]:.2f}",
                 f"{r['edge_shift'][0]:+.1f} [{r['edge_shift'][1]:+.0f}, {r['edge_shift'][2]:+.0f}]", r["edges_missed"]]
                for r in replay["rows"]
            ]  # fmt: skip
            lines += [
                f"### {name}",
                "",
                *_table(["配置项", "值"], [[k, v] for k, v in report["config"].items()]),
                f"训练版按 episode 平均：覆盖帧数中位 {sum(x['reach_median_frames'] for x in stats) / len(stats):.1f}，"
                f"补帧点 {sum(x['pad_fraction'] for x in stats) / len(stats):.1%}，窗口内点 "
                f"{sum(x['window_fraction'] for x in stats) / len(stats):.1%}，平均 loss 权重 "
                f"{sum(x['weight_mean'] for x in stats) / len(stats):.2f}。",
                "",
                "标签回放（按服务端的执行方式拼接；误差单位为一帧运动量）：",
                "",
                *_table(
                    [
                        "执行帧数",
                        "误差 中位 / p90",
                        "接缝 中位 / 最大（步）",
                        "指令偏移 中位 [最小, 最大]",
                        "漏掉的指令",
                    ],
                    replay_rows,
                ),
            ]
        return lines

    def decision_section(self) -> list[str]:
        rows = [[d.id, d.stage, d.item, d.decision, d.written_to] for d in self.decisions]
        return ["## 人的决定", "", *(_table(["编号", "阶段", "事项", "决定", "写入"], rows) if rows else ["无", ""])]

    def render(self) -> str:
        lines = [
            f"# 数据报告：{self.index.task}",
            "",
            f"由 `python -m vla_tools.data_report` 根据 `runs/{self.run_dir.name}/` 的产物生成；生成时 S0–S7 均已通过。",
            "",
            *self.lineage(),
            *self.audit_and_cleaning(),
            *self.keyframe_section(),
            *self.timing_section(),
            *self.coverage_section(),
            *self.label_section(),
            *self.decision_section(),
        ]
        return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    blocking = [s for s in stage_statuses(args.run_dir) if s.status != "passed"]
    if blocking:
        for s in blocking:
            print(f"{s.stage} {s.name}: {s.status} -- {s.reason}")
        raise SystemExit(2)
    path = args.run_dir / "data_report.md"
    path.write_text(DataReport(args.run_dir).render(), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
