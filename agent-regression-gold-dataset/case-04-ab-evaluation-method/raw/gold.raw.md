# 我保存的一份回答

用户当时大概问的是：Skill A/B 是否全程 AI 自迭代？base 可互换是否消除了偏见？指标和盲化方案应该如何设计？

该流程是 human-in-the-loop：AI 运行、评分和分析，人类审阅、反馈并决定停止。base/candidate 可互换只减少身份偏见，不能消除位置偏见和随机性。指标应分为少量 primary 决策指标与 diagnostic 指标。deterministic diff 应作为 primary，盲化或双向 LLM 比较只用于边界 case；Regression 保留确定性 gate。
