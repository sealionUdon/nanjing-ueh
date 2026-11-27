"""从年度评价与逐项评价结果生成报告插图。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
from matplotlib.text import Text
import numpy as np
import pandas as pd
from PIL import Image

OUT = Path(__file__).resolve().parents[1] / "outputs"
GROUPS = ["活力", "组织结构", "恢复力", "服务功能", "人群健康"]
GRADES = ["病态", "一般", "中等", "良好", "优良"]
YEARS = list(range(2020, 2025))
COLORS = ["#146785", "#c26722", "#468456", "#875e99", "#ac465a"]
ANNUAL = "result_annual.csv"
SCORES = "scores_by_indicator.csv"
PREFERRED = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
             "PingFang SC", "DengXian", "SimSun"]


def save(fig, filename, source):
    caption = "数据来源：outputs/" + source
    footer = fig.text(.02, .015, caption, fontsize=8, color="#454545")
    chars = {ord(c) for t in fig.findobj(Text) for c in t.get_text() if not c.isspace()}
    fonts = sorted(font_manager.fontManager.ttflist,
                   key=lambda f: (PREFERRED.index(f.name) if f.name in PREFERRED else len(PREFERRED),
                                  f.style != "normal", f.weight != 400, f.fname))
    for font in fonts:
        if not Path(font.fname).is_file():
            continue
        face = FT2Font(font.fname)
        fallback = font.name != PREFERRED[0]
        note = "  |  回退字体：" + font.name if fallback else ""
        if all(face.get_char_index(c) for c in chars | {ord(c) for c in note}):
            break
    else:
        raise RuntimeError("未找到覆盖图中文字的字体，请安装 Noto Sans CJK SC 后重试")
    footer.set_text(caption + note)
    plt.rcParams["font.family"] = font.name
    for label in fig.findobj(Text):
        prop = label.get_fontproperties().copy()
        prop.set_file(font.fname)
        label.set_fontproperties(prop)
    fig.tight_layout(rect=(0, .065, 1, 1))
    path = OUT / "figures" / filename
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    with Image.open(path) as im:
        im.verify()
    size = path.stat().st_size
    if size <= 30 * 1024:
        raise RuntimeError(f"{path} 仅 {size} 字节")
    print(f"{path.name}  {size:,} 字节  {font.fname}")


def year_axis(ax, ylabel):
    ax.set(xticks=YEARS, xlabel="年份（年）", ylabel=ylabel)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)


def main():
    plt.rcParams.update({"axes.unicode_minus": False, "figure.figsize": (16/2.54, 3.9),
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    annual = pd.read_csv(OUT / ANNUAL)
    annual = annual[(annual["阈值口径"] == "25项探索") & (annual["情景"] == "含假设值")]
    annual = annual.set_index("年份").loc[YEARS]
    if not annual.index.is_unique:
        raise ValueError("所选情景每年应只有一条年度结果")
    scores = pd.read_csv(OUT / SCORES)
    grades = scores.pivot(index="年份", columns="编号", values="判定等级").loc[YEARS]
    if grades.shape[1] != 25 or not grades.isin(GRADES).all().all():
        raise ValueError("每年应有 25 项有效指标等级")

    fig, ax = plt.subplots()
    values = annual["优良"]
    ax.plot(YEARS, values, "o-", color=COLORS[0])
    for year, val in values.items():
        ax.annotate(f"{val:.3f}\n{annual.at[year, '判定等级']}", (year, val),
                    xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9)
    ax.margins(x=.09, y=.3)
    ax.set_title("南京城市生态健康综合优良关联度")
    year_axis(ax, "优良关联度（无量纲）")
    save(fig, "fig1_综合关联度趋势.png", ANNUAL)

    factors = annual[[g + "_得分" for g in GROUPS]]
    fig, ax = plt.subplots()
    for group, color, marker in zip(GROUPS, COLORS, "os^DP"):
        ax.plot(YEARS, factors[group + "_得分"], marker=marker, color=color, label=group)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(.5, 1), frameon=False)
    fig.suptitle("五要素年度得分趋势")
    year_axis(ax, "要素得分（无量纲）")
    save(fig, "fig2_五要素得分趋势.png", ANNUAL)

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(16/2.54, 4.7))
    theta = np.linspace(0, 2*np.pi, len(GROUPS), endpoint=False)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    for year, color in zip((YEARS[0], YEARS[-1]), COLORS):
        row = factors.loc[year].to_numpy()
        angles, radial = np.r_[theta, theta[0]], np.r_[row, row[0]]
        ax.plot(angles, radial, "o-", color=color, label=f"{year}年")
        ax.fill(angles, radial, color=color, alpha=.10)
    ax.set_xticks(theta, GROUPS)
    ax.set_title("五要素得分对比（无量纲）", pad=20)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), frameon=False)
    save(fig, "fig3_五要素雷达图.png", ANNUAL)

    fig, axes = plt.subplots(2, 2, figsize=(16/2.54, 4.9))
    fig.suptitle("关键指标年度变化")
    panels = [("X3", "单位工业增加值能耗", "吨标准煤/万元"), ("X13", "工业废水排放强度", "吨/万元GDP"),
              ("X22", "每万人公交车辆", "辆/万人"), ("X26", "每万人床位", "张/万人")]
    for ax, (code, name, unit), color in zip(axes.flat, panels, COLORS):
        vals = scores[scores["编号"] == code].set_index("年份").loc[YEARS, "值"]
        ax.plot(YEARS, vals, "o-", color=color)
        for year in (YEARS[0], YEARS[-1]):
            ax.annotate(f"{vals[year]:g}", (year, vals[year]), xytext=(0, 5),
                        textcoords="offset points", ha="center", fontsize=8)
        ax.set_title(f"{code}  {name}", fontsize=10)
        ax.margins(x=.16, y=.25)
        ax.tick_params(labelsize=8)
        year_axis(ax, unit)
    save(fig, "fig4_关键指标变化.png", SCORES)

    fig, ax = plt.subplots()
    bottom = np.zeros(len(YEARS))
    for grade, color in zip(GRADES, ["#993c43", "#ca7141", "#e4be68", "#79af9e", "#266b78"]):
        counts = grades.eq(grade).sum(axis=1)
        ax.bar(YEARS, counts, bottom=bottom, label=grade, color=color, width=.62)
        for year, base, n in zip(YEARS, bottom, counts):
            if n:
                ax.text(year, base+n/2, str(n), ha="center", va="center", fontsize=8,
                        color="white" if grade in ("病态", "优良") else "black")
        bottom += counts.to_numpy()
    year_axis(ax, "指标数量（项）")
    fig.suptitle("年度指标等级分布")
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(.5, 1), frameon=False)
    save(fig, "fig5_指标等级分布.png", SCORES)


if __name__ == "__main__":
    main()
