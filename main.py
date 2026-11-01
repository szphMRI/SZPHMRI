from __future__ import annotations

from pathlib import Path

from app.config import PROJECT_ROOT, Config
from app.container import AppContainer
from app.schemas import LesionInfo
from app.services.mock_data import ensure_mock_data


def build_demo_lesion(mask_path: str) -> LesionInfo:
    """构造一个带掩码的示例病灶，方便直接进入可视化环节调试。"""
    return LesionInfo(
        lesion_id="L1",
        location="右侧额叶皮层下、左侧顶叶灰白质交界区",
        count=2,
        max_diameter_mm=3.2,
        morphology="类圆形，边界欠清，无明显占位效应",
        signal_features={"T1WI": "低信号", "T2WI/FLAIR": "高信号", "DWI": "稍高信号"},
        enhancement="环形强化",
        radiomics={"GLCM熵": 7.23, "对比度": 4.89, "球形度": 0.78},
        confidence=0.93,
        mask_path=mask_path,
        reference_sequence="T1WI",
    )


def main() -> None:
    # 1. 加载配置（config.yaml 相对项目根目录）
    cfg = Config.load()

    # 2. 依赖装配
    container = AppContainer(cfg)

    # 3. 准备 mock DICOM 与掩码（首次运行会生成，之后直接复用）
    dicom_dir, mask_path = ensure_mock_data(PROJECT_ROOT / "data")
    print(f"[main] DICOM 目录: {dicom_dir}")
    print(f"[main] 病灶掩码: {mask_path}")

    # 4. 构造病灶信息
    lesion = build_demo_lesion(mask_path)

    # 5. 跑完整流程（此处可打断点逐步调试）
    report = container.pipeline.run(
        study_id="STU-20261028-001",
        patient_id="PAT-0001",
        dicom_dir=dicom_dir,
        lesions=[lesion],
    )

    # 6. 保存 Markdown 报告到 outputs/
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{report.report_id}.md"
    report_path.write_text(report.content_markdown, encoding="utf-8")

    print("=" * 80)
    print(report.content_markdown)
    print("=" * 80)
    print(f"报告已保存至: {report_path}")
    print(f"三视图 PNG 保存在: {cfg.visualization.output_dir}")


if __name__ == "__main__":
    main()