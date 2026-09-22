# build_installer.py 并发锁与用户数据保护加固

## 背景

2026-09-22 两个会话并发执行 `packaging/build_installer.py --pack-only`：
- PyInstaller 会整删 `dist/` 重建；
- 备份目录 `dist/_userdata_backup_<ts>/` 与 `dist/` 同根，被连带删除；
- `_restore_user_data` 在 `backup_dir` 丢失时静默早退，未报错；
- 用户 `work/` 数据第二次丢失。

## 目标

消除并发打包导致用户运行数据丢失的根因，让数据丢失场景显式失败、可定位，而非静默跳过。

## 边界

- 只改 `packaging/build_installer.py` 与 `tests/packaging/test_build_installer.py`。
- 不改动 PyInstaller spec、资产补齐、Inno 等其它链路。
- 不引入第三方依赖（仅用标准库实现锁）。
- 兼容 `--pack-only` 与完整链路；兼容首次打包（无 dist/）与覆盖打包。

## 成功标准

1. 同时只能有一个打包进程执行：`dist/.build.lock` 在 `_configure_logging()` 后立即获取，PID 存活则退出码 1，陈旧锁则覆盖。
2. 备份目录移出 `dist/`，放到 `packaging/build/_userdata_backup_<ts>/`，避免 PyInstaller 清 dist 时连带删除备份。
3. `backup_dir is not None 且 not backup_dir.is_dir()` 时 `_restore_user_data` 必须抛出 `BuildError`；`backup_dir is None` 保持静默。
4. finally 兜底发现备份目录存在过但消失时，记录醒目 ERROR。
5. 单元测试覆盖：活锁拒绝、陈旧锁继续、锁正常释放、备份路径迁移、备份丢失抛错、现有备份/恢复流程。
6. 全部测试通过（1172 绿），真打包 `--pack-only` 成功且锁文件不残留。
