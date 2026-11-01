from __future__ import annotations

from app.schemas import LesionInfo, MRIStudy, RetrievalQuery


"""	
MRIStudy，其中 study.lesions 即病灶信息（位置、数量、最大径、形态、信号、强化、影像组学）
输出	RetrievalQuery：keyword_query / semantic_query / differential_query 三类检索数据
对应 RAG 环节	RAG 的 Query Construction / Query Generation（检索增强生成中的“生成检索意图”）
"""


class LesionQueryBuilder:
    def build(self, study: MRIStudy) -> RetrievalQuery:
        """_describe() 把数值型病灶特征转成自然语言（位置、数量、最大径、形态、信号、强化），_keywords()
        提取关键词（含“微小病灶”这类阈值触发词）。这就是“根据病灶信息生成病灶检索数据”"""
        if not study.lesions:
            return RetrievalQuery(
                keyword_query="正常 脑部 MRI 影像 表现",
                semantic_query="脑部磁共振未见明显异常病灶的影像学表现",
            )

        descriptions = "；".join(self._describe(les) for les in study.lesions)
        keyword = " ".join(self._keywords(study.lesions))
        return RetrievalQuery(
            keyword_query=keyword,
            semantic_query=f"患者脑部 MRI 显示：{descriptions}。请给出符合影像学与循证医学证据的诊断依据。",
            differential_query=f"{keyword} 鉴别诊断 影像学特征 诊断标准",
        )

    def _describe(self, les: LesionInfo) -> str:
        sig = "、".join(f"{k}{v}" for k, v in les.signal_features.items())
        return (
            f"{les.location}见{les.count}个病灶，最大径{les.max_diameter_mm}mm，"
            f"形态{les.morphology or '未描述'}，信号{sig or '未描述'}，"
            f"强化方式{les.enhancement or '未描述'}"
        )

    def _keywords(self, lesions: list[LesionInfo]) -> list[str]:
        kws = ["脑部 MRI", "病灶"]
        for les in lesions:
            if les.location:
                kws.append(les.location.split("、")[0])
            if les.enhancement:
                kws.append(les.enhancement)
            if les.morphology:
                kws.append(les.morphology)
            if les.max_diameter_mm and les.max_diameter_mm < 5:
                kws.append("微小病灶")
            kws.extend(les.signal_features.keys())
        return list(dict.fromkeys([k for k in kws if k]))