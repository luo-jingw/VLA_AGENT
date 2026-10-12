"""Write the README figures as static SVG, one light and one dark file per figure, from one geometry.

Static files carry concrete colours instead of CSS variables (GitHub shows README images as plain SVG);
the palette is the diagram-svg skill's visual spec. Rerun after changing the geometry:

    python docs/figures/make_figures.py
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

OUT = Path(__file__).resolve().parent
FONT = (
    "-apple-system, 'Segoe UI', 'Noto Sans CJK SC', 'PingFang SC', 'Microsoft YaHei', 'Noto Serif CJK SC', sans-serif"
)


@dataclasses.dataclass(frozen=True)
class Palette:
    fg: str
    muted: str
    card: str
    line: str
    box: tuple[str, str, str, str]  # fill, stroke, title, subtitle
    teal: tuple[str, str, str, str]
    purple: tuple[str, str, str, str]
    coral: tuple[str, str, str, str]
    gray: tuple[str, str, str, str]


LIGHT = Palette(
    fg="#1f1e1b",
    muted="#6b6a64",
    card="#ffffff",
    line="#dedcd3",
    box=("#FFFFFF", "#B4B2A9", "#1f1e1b", "#6b6a64"),
    teal=("#E1F5EE", "#0F6E56", "#085041", "#0F6E56"),
    purple=("#EEEDFE", "#534AB7", "#3C3489", "#534AB7"),
    coral=("#FAECE7", "#993C1D", "#712B13", "#993C1D"),
    gray=("#F1EFE8", "#888780", "#444441", "#5F5E5A"),
)
DARK = Palette(
    fg="#ecebe6",
    muted="#a3a29b",
    card="#232321",
    line="#3a3936",
    box=("#232321", "#5F5E5A", "#ecebe6", "#a3a29b"),
    teal=("#085041", "#5DCAA5", "#E1F5EE", "#9FE1CB"),
    purple=("#3C3489", "#AFA9EC", "#EEEDFE", "#CECBF6"),
    coral=("#712B13", "#F0997B", "#FAECE7", "#F5C4B3"),
    gray=("#444441", "#B4B2A9", "#F1EFE8", "#D3D1C7"),
)


class Canvas:
    def __init__(self, height: int, palette: Palette, label: str) -> None:
        self.p = palette
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 {height}" width="680" height="{height}" '
            f'role="img" aria-label="{label}" font-family="{FONT}">',
            "<defs><marker id='arrow' viewBox='0 0 10 10' refX='8' refY='5' markerWidth='6' markerHeight='6' "
            f"orient='auto-start-reverse'><path d='M2 1L8 5L2 9' fill='none' stroke='{palette.muted}' "
            "stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/></marker></defs>",
            f'<rect x="0.5" y="0.5" width="679" height="{height - 1}" rx="12" fill="{palette.card}" '
            f'stroke="{palette.line}"/>',
        ]

    def text(self, x: float, y: float, s: str, size: int, color: str, weight: int = 400, rotate: bool = False) -> None:
        transform = f' transform="rotate(-90 {x} {y})"' if rotate else ""
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="middle" '
            f'dominant-baseline="central"{transform}>{s}</text>'
        )

    def box(self, x: float, y: float, w: float, h: float, kind: str, lines: list[str]) -> None:
        fill, stroke, title, sub = getattr(self.p, kind)
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="0.7"/>'
        )
        if len(lines) == 1:
            self.text(x + w / 2, y + h / 2, lines[0], 14, title, 600)
            return
        offsets = [19, 38] if len(lines) == 2 else [20, 41, 59]
        for i, (line, dy) in enumerate(zip(lines, offsets, strict=True)):
            self.text(x + w / 2, y + dy, line, 14 if i == 0 else 12, title if i == 0 else sub, 600 if i == 0 else 400)

    def arrow(self, d: str, dashed: bool = False) -> None:
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        self.parts.append(
            f'<path d="{d}" fill="none" stroke="{self.p.muted}" stroke-width="1.2"{dash} marker-end="url(#arrow)"/>'
        )

    def write(self, path: Path) -> None:
        path.write_text("\n".join([*self.parts, "</svg>"]) + "\n", encoding="utf-8")


# Figure 1: how the agent moves through data preprocessing, and where it goes back.
STAGES = [
    ("box", "S0 范围", "vla-pipeline · robot_spec · run.json"),
    ("teal", "S1–S2 体检与清洗", "vla-data-cleaning"),
    ("teal", "S3–S5 关键帧 · 静止帧与补帧", "vla-keyframes · vla-episode-timing"),
    ("teal", "S6 数据评估", "vla-data-diversity · 留出集"),
    ("teal", "S7 动作标签", "vla-action-labels · 网格搜索"),
    ("coral", "训练就绪数据集", "训练版 · 评估版 · data_report.md"),
    ("gray", "S8 下游", "vla-downstream · 训练 评估 交付"),
    ("box", "S9 收尾", "close · docs 与 issues"),
]
X, W, H, TOP, STEP = 170, 340, 56, 30, 82


def center(i: int) -> float:
    return TOP + i * STEP + H / 2


def pipeline(p: Palette) -> Canvas:
    bottom = TOP + len(STAGES) * STEP - (STEP - H)
    c = Canvas(bottom + 64, p, "VLA 数据预处理的推进与回环")
    for i, (kind, title, sub) in enumerate(STAGES):
        y = TOP + i * STEP
        c.box(X, y, W, H, kind, [title, sub])
        if i + 1 < len(STAGES):
            c.arrow(f"M{X + W / 2} {y + H} L{X + W / 2} {y + STEP - 3}")
    left = X
    # S6 -> S1-S2 arrives below S8 -> S1-S2, so the two lanes never cross; S9 -> S0 runs on the outer lane.
    c.arrow(f"M{left} {center(3)} L130 {center(3)} L130 {center(1) + 10} L{left - 3} {center(1) + 10}", dashed=True)
    c.arrow(f"M{left} {center(6)} L100 {center(6)} L100 {center(1) - 10} L{left - 3} {center(1) - 10}", dashed=True)
    c.arrow(f"M{left} {center(7)} L70 {center(7)} L70 {center(0)} L{left - 3} {center(0)}", dashed=True)
    c.text(116, (center(1) + center(3)) / 2, "批次或离群", 12, p.muted, rotate=True)
    c.text(86, (center(3) + center(6)) / 2, "下游发现的数据问题", 12, p.muted, rotate=True)
    c.text(56, (center(0) + center(7)) / 2, "下一个任务", 12, p.muted, rotate=True)
    # Bracket over the data stages and their product: the part of the run that run_status checks.
    top, end = TOP + STEP, TOP + 5 * STEP + H
    c.parts.append(
        f'<path d="M524 {top} L534 {top} L534 {end} L524 {end}" fill="none" stroke="{p.muted}" stroke-width="1.2"/>'
    )
    c.text(552, (top + end) / 2, "数据预处理 · run_status 逐阶段检查", 12, p.muted, rotate=True)
    c.text(
        340,
        bottom + 34,
        "中性：范围与收尾　青：数据阶段　珊瑚：交付的数据　灰：下游　虚线：回到前面的阶段",
        12,
        p.muted,
    )
    return c


# Figure 2: what happens inside one data stage.
def stage_loop(p: Palette) -> Canvas:
    c = Canvas(292, p, "一个数据阶段内部的循环")
    steps = [
        ("box", ["① 运行工具", "写新版本或报告", "源数据只读"]),
        ("box", ["② 观察", "统计与核对图", "一致性检查"]),
        ("teal", ["③ 决定", "写入计划文件", "每条附理由"]),
        ("box", ["④ 放行检查", "run_status", "产物与决定齐全"]),
    ]
    xs = [41, 197, 353, 509]
    for x, (kind, lines) in zip(xs, steps, strict=True):
        c.box(x, 30, 130, 76, kind, lines)
    for x in xs[:-1]:
        c.arrow(f"M{x + 130} 68 L{x + 153} 68")
    c.box(353, 150, 130, 76, "coral", ["人的决定", "decisions.md", "未决时停在本阶段"])
    c.arrow("M398 106 L398 147")
    c.arrow("M438 150 L438 109", dashed=True)
    c.box(553, 150, 86, 44, "box", ["下一阶段"])
    c.arrow("M596 106 L596 147")
    c.arrow("M530 106 L530 244 L106 244 L106 109", dashed=True)
    c.text(232, 228, "未通过：调参数、改定义，或停下报告阻塞", 12, p.muted)
    c.text(340, 270, "青：agent 写入的决定　珊瑚：需要人决定的事项　虚线：回到本阶段或等待", 12, p.muted)
    return c


def main() -> None:
    for name, build in (("pipeline-loop", pipeline), ("stage-loop", stage_loop)):
        for mode, palette in (("light", LIGHT), ("dark", DARK)):
            build(palette).write(OUT / f"{name}-{mode}.svg")
            print(f"wrote {name}-{mode}.svg")


if __name__ == "__main__":
    main()
