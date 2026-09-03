"""GUI 包：本地网页版后端（runner 任务编排 / app 路由 / static 前端）。

边界：GUI 只能通过 subprocess 调 scripts 流水线，不得 import scripts 内部逻辑。
"""
