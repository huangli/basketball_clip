# 执行计划

## 1. 常量与路径

- 新增 `LOCK_PATH = DIST_DIR / ".build.lock"`。
- 新增 `USERDATA_BACKUP_ROOT = PACKAGING_DIR / "build"`。
- 更新模块 docstring 中对备份路径的描述。

## 2. 锁机制

- 新增 `_acquire_build_lock()`：创建 `dist/`（如不存在），检查锁文件；解析 PID 与启动时间；若 PID 存活则 `raise BuildError`；否则覆盖写新锁（PID + 启动 ISO 时间）。
- 新增 `_release_build_lock()`：忽略错误删除 `LOCK_PATH`。
- `main` 在 `_configure_logging()` 后立即调用 `_acquire_build_lock()`，并置于 `try/finally`，确保任何退出路径都释放锁。

## 3. 备份与恢复

- `_backup_user_data(app_dir, backup_root)` 调用处改为 `backup_root=USERDATA_BACKUP_ROOT`。
- `_restore_user_data`：当 `backup_dir is not None and not backup_dir.is_dir()` 时抛 `BuildError`。
- `_restore_user_data_failsafe`：finally 中记录 ERROR（若 `backup_dir is not None and not backup_dir.is_dir()`）。
- main 中保留 `backup_dir = None` 标记，finally 只在 `backup_dir is not None` 时兜底。

## 4. 测试

- 更新现有测试：备份路径断言从 `dist_dir` 改为 `USERDATA_BACKUP_ROOT`（即 `tmp_path/packaging/build`）。
- 新增：
  - `test_lock_alive_pid_refuses_build`：monkeypatch `_is_pid_alive` 返回 True，断言 `main(["--pack-only"]) == 1` 且锁保留。
  - `test_lock_stale_pid_overrides_and_succeeds`：monkeypatch `_is_pid_alive` 返回 False，mock `ensure_assets`/`run_pyinstaller`/`stage_runtime_files` 等，断言返回 0 且锁删除。
  - `test_lock_released_on_build_error`：mock `ensure_assets` 抛错，断言返回 1 且锁删除。
  - `test_backup_dir_under_packaging_build`：断言备份目录在 `packaging/build/` 下。
  - `test_restore_raises_when_backup_dir_missing`：直接调用 `_restore_user_data` 并删除备份目录，断言抛 `BuildError` 且信息含"备份目录丢失"。

## 5. 验证

- 跑 `ruff format scripts tests gui packaging`。
- 跑 `ruff check --fix scripts tests gui packaging`。
- 跑 `pytest -q`（1172 绿）。
- 实跑 `.venv-spike/Scripts/python packaging/build_installer.py --pack-only`，确认成功且 `dist/.build.lock` 不残留。
