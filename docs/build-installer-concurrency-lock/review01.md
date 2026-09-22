# 自审报告 review01

## 需求对齐

- 严格按用户语义实现：构建锁在 `_configure_logging()` 后立即获取；备份目录搬出 dist；restore 对丢失备份显式抛 `BuildError`。
- 未引入第三方依赖；锁文件路径在 `dist/.build.lock`，不在 PyInstaller 清理范围内。

## 风险点

- `os.kill(pid, 0)` 在 Windows 上可检测进程是否存在，但可能因权限失败。已用 `ProcessLookupError`/`PermissionError`/`OSError` 兜底，异常时保守视为“存活”以避免数据丢失风险。
- 锁文件内容含 PID 与启动时间，便于人工排查；解析失败视为陈旧锁覆盖。
- 备份根目录 `packaging/build/` 为 PyInstaller workpath，PyInstaller 不会删除它；备份子目录带时间戳，不会与 workpath 文件冲突。

## 测试覆盖

- 活锁拒绝、陈旧锁覆盖、锁异常释放、备份路径迁移、备份丢失抛错均已覆盖。
- 现有 7 个测试同步更新备份路径断言。

## 待后续观察

- 未来若把打包入口改为多阶段并行，需改用文件锁+flock；当前单进程互斥足够。
