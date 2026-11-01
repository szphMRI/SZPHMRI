from __future__ import annotations

from typing import Protocol

from app.schemas import Report


class Notifier(Protocol):
    def send(self, report: Report) -> bool: ...


class ConsoleNotifier:
    def send(self, report: Report) -> bool:
        print(f"[通知] 报告 {report.report_id} 已生成，待医生终端 {report.patient_id} 审核。")
        return True