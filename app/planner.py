"""复习规划：按考试倒计时与薄弱考点，生成可执行的周计划。

产品取舍：不做"AI 生成任意计划"，而是用固定节奏模板 + 错题驱动的考点分配，
保证输出结构稳定、可解释，也便于嵌入日历与提醒。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

PHASES = [
    ("通读梳理", "通读教材与课件，产出知识清单，标出看不懂的点", 0.30),
    ("章节刷题", "按章节做真题，错题标注原因（不会 / 记错 / 粗心）", 0.30),
    ("错题攻坚", "只做错题与高频考点，补齐知识清单中的空白", 0.25),
    ("整套模拟", "整套限时模拟，训练答题节奏，不再新增知识点", 0.15),
]


@dataclass
class PlanItem:
    week: int
    phase: str
    focus: list[str]
    task: str
    hours: float


@dataclass
class Plan:
    days_left: int
    start: str
    exam_date: str
    items: list[PlanItem] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "days_left": self.days_left,
            "start": self.start,
            "exam_date": self.exam_date,
            "items": [i.__dict__ for i in self.items],
        }


def build_plan(exam_date: str | date, weak_topics: list[str] | None = None,
               hours_per_day: float = 2.0, today: date | None = None) -> Plan:
    today = today or date.today()
    exam = exam_date if isinstance(exam_date, date) else datetime.strptime(exam_date, "%Y-%m-%d").date()
    days = max((exam - today).days, 0)
    weak = [w for w in (weak_topics or []) if w.strip()] or ["高频考点"]

    if days == 0:
        return Plan(0, today.isoformat(), exam.isoformat(), [])

    weeks = max(1, min(6, round(days / 7)))
    plan = Plan(days, today.isoformat(), exam.isoformat())

    # 距考试越近，权重越偏向后期阶段；按剩余周数截取阶段
    if days <= 7:
        stages = PHASES[-1:]
    elif days <= 14:
        stages = PHASES[-2:]
    elif days <= 21:
        stages = PHASES[-3:]
    else:
        stages = PHASES

    per_week = max(1, len(weak) // max(len(stages), 1) + 1)
    for i in range(weeks):
        stage = stages[min(i * len(stages) // weeks, len(stages) - 1)]
        focus = weak[i * per_week:(i + 1) * per_week] or weak[-1:]
        plan.items.append(PlanItem(
            week=i + 1,
            phase=stage[0],
            focus=focus,
            task=stage[1],
            hours=round(days / weeks * hours_per_day / 7, 1),
        ))
    return plan


if __name__ == "__main__":
    p = build_plan("2027-01-10", ["数据库索引与隔离级别", "TCP 可靠传输", "进程调度算法"])
    for it in p.items:
        print(f"第{it.week}周 {it.phase} 重点={it.focus} 每日{it.hours}h :: {it.task}")
