"""技能目录管理：扫描、安装（zip 上传）、删除。

技能遵循 Agent Skills 规范：

    skills/<skill-name>/
    ├── SKILL.md          # 必需，YAML frontmatter（name/description）+ Markdown
    └── 其他辅助文件        # 可选，如 helper.py、参考文档

员工通过上传 zip 提交技能，zip 支持两种布局：

1. 规范布局：归档内顶层就是 ``<skill-name>/SKILL.md``；
2. 扁平布局：``SKILL.md`` 直接在归档根目录，目录名取 frontmatter 的 name。

安装时做严格校验：frontmatter 合法、name 与目录名一致且符合命名规范、
防 zip-slip 路径穿越、总解压大小与文件数上限。
"""

from __future__ import annotations

import io
import logging
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import yaml

logger = logging.getLogger(__name__)

# 上传安全上限
MAX_SKILL_ZIP_BYTES = 10 * 1024 * 1024
MAX_SKILL_FILES = 200
MAX_SKILL_TOTAL_BYTES = 10 * 1024 * 1024
MAX_SKILL_NAME_LENGTH = 64
MAX_SKILL_DESCRIPTION_LENGTH = 1024

# Agent Skills 规范：小写字母数字与连字符，不得以连字符开头/结尾，不得连续连字符
_SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class SkillInstallError(ValueError):
    """技能包不合法或冲突时抛出，消息可直接展示给上传者。"""


class SkillConflictError(SkillInstallError):
    """同名技能已存在且未允许覆盖（对应 HTTP 409）。"""


@dataclass(frozen=True, slots=True)
class SkillInfo:
    """技能元数据，供列表接口与启动日志使用。"""

    name: str
    description: str
    path: str
    files: list[str] = field(default_factory=list)
    updated_at: float = 0.0
    valid: bool = True
    error: str = ""


@dataclass(frozen=True, slots=True)
class SkillZipInspection:
    """zip 技能包的预检结果：校验通过后的元数据，全程不触碰磁盘。"""

    name: str
    description: str
    files: list[str]  # 相对技能根目录的 POSIX 路径
    nested_dir: str | None  # 规范布局的顶层目录名；扁平布局为 None


def validate_skill_name(name: str) -> None:
    """校验技能名：1-64 字符、小写字母数字与单个连字符。"""

    if not name or len(name) > MAX_SKILL_NAME_LENGTH:
        raise SkillInstallError(f"技能名不合法：长度需在 1-{MAX_SKILL_NAME_LENGTH} 之间")
    if not _SKILL_NAME_RE.match(name):
        raise SkillInstallError(
            f"技能名 '{name}' 不合法：只能包含小写字母、数字和连字符（-），"
            "不得以连字符开头或结尾，不得出现连续连字符"
        )


def parse_skill_markdown(content: str) -> tuple[str, str]:
    """解析 SKILL.md，返回 (name, description)。

    Raises:
        SkillInstallError: frontmatter 缺失、YAML 非法或必填项为空。
    """

    match = _FRONTMATTER_RE.match(content)
    if not match:
        raise SkillInstallError("SKILL.md 缺少合法的 YAML frontmatter（应以 --- 包裹）")
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise SkillInstallError(f"SKILL.md 的 YAML 解析失败: {exc}") from exc
    if not isinstance(data, dict):
        raise SkillInstallError("SKILL.md 的 frontmatter 必须是键值映射")
    name = str(data.get("name", "")).strip()
    description = str(data.get("description", "")).strip()
    if not name or not description:
        raise SkillInstallError("SKILL.md 必须包含非空的 name 和 description")
    if len(description) > MAX_SKILL_DESCRIPTION_LENGTH:
        raise SkillInstallError(
            f"description 长度 {len(description)} 超出上限 {MAX_SKILL_DESCRIPTION_LENGTH} 字符"
        )
    return name, description


def _is_within(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
    except ValueError:
        return False
    return True


def _safe_members(zf: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    """过滤并校验 zip 成员，返回 (ZipInfo, 规范化 POSIX 相对路径)。

    拒绝绝对路径、盘符、``..`` 穿越段以及符号链接；跳过目录与系统杂项文件。
    """

    accepted: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
    for info in zf.infolist():
        raw = info.filename.replace("\\", "/")
        posix = PurePosixPath(raw)
        if posix.is_absolute() or posix.drive:
            raise SkillInstallError(f"压缩包包含绝对路径条目: {info.filename!r}")
        parts = posix.parts
        if not parts or any(part in ("", "..") for part in parts):
            raise SkillInstallError(f"压缩包包含非法路径条目: {info.filename!r}")
        if info.is_dir() or "__MACOSX" in parts or parts[-1] in {".DS_Store", "Thumbs.db"}:
            continue
        unix_mode = (info.external_attr >> 16) & 0o170000
        if unix_mode == 0o120000:  # 符号链接
            raise SkillInstallError(f"压缩包不允许包含符号链接: {info.filename!r}")
        accepted.append((info, posix))
    return accepted


def _parse_skill_zip(
    zf: zipfile.ZipFile,
) -> tuple[list[tuple[zipfile.ZipInfo, PurePosixPath]], str | None, str, str]:
    """对打开的 zip 做全部结构校验（不写磁盘）。

    Returns:
        (安全成员列表, 规范布局顶层目录名或 None, 技能名, 描述)。
    """

    bad_file = zf.testzip()
    if bad_file is not None:
        raise SkillInstallError(f"压缩包已损坏（CRC 校验失败: {bad_file}）")
    members = _safe_members(zf)
    if len(members) > MAX_SKILL_FILES:
        raise SkillInstallError(f"压缩包文件数 {len(members)} 超出上限 {MAX_SKILL_FILES}")

    # 定位 SKILL.md：允许 <name>/SKILL.md 或根目录 SKILL.md
    skill_md_entries = [p for _, p in members if p.name == "SKILL.md" and len(p.parts) <= 2]
    if not skill_md_entries:
        raise SkillInstallError(
            "压缩包中未找到 SKILL.md（应位于根目录或唯一的顶层技能目录下）"
        )
    if len(skill_md_entries) > 1:
        raise SkillInstallError("压缩包中存在多个 SKILL.md，请每个压缩包只打包一个技能")
    skill_md_posix = skill_md_entries[0]

    nested_dir = skill_md_posix.parts[0] if len(skill_md_posix.parts) == 2 else None
    if nested_dir is not None:
        # 规范布局：所有文件必须位于该顶层目录内
        for _, posix in members:
            if posix.parts[0] != nested_dir:
                raise SkillInstallError(
                    f"压缩包文件 {posix.as_posix()!r} 位于技能目录 {nested_dir}/ 之外，"
                    "请将技能全部文件放在同一个顶层目录中"
                )

    try:
        skill_md_content = zf.read(skill_md_posix.as_posix()).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SkillInstallError("SKILL.md 必须是 UTF-8 编码的文本文件") from exc

    name, description = parse_skill_markdown(skill_md_content)
    validate_skill_name(name)
    if nested_dir is not None and nested_dir != name:
        raise SkillInstallError(
            f"目录名 '{nested_dir}' 与 SKILL.md 中的 name '{name}' 不一致，两者必须相同"
        )

    # 按声明的未压缩大小做总量预检（实际解压时 install 仍逐字节复核）
    declared_total = sum(info.file_size for info, _ in members)
    if declared_total > MAX_SKILL_TOTAL_BYTES:
        raise SkillInstallError(
            f"解压后总大小超出上限 {MAX_SKILL_TOTAL_BYTES // (1024 * 1024)}MB"
        )
    return members, nested_dir, name, description


def inspect_skill_zip(zip_bytes: bytes) -> SkillZipInspection:
    """上传前预检：对 zip 技能包执行与安装完全一致的全部校验，但不写入任何文件。

    Raises:
        SkillInstallError: 任一校验未通过，消息可直接展示给上传者。
    """

    if len(zip_bytes) == 0:
        raise SkillInstallError("上传内容为空")
    if len(zip_bytes) > MAX_SKILL_ZIP_BYTES:
        raise SkillInstallError(
            f"压缩包过大：{len(zip_bytes)} bytes，上限 {MAX_SKILL_ZIP_BYTES // (1024 * 1024)}MB"
        )

    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise SkillInstallError("文件不是合法的 zip 压缩包") from exc

    with zf:
        members, nested_dir, name, description = _parse_skill_zip(zf)

    files = sorted(
        (PurePosixPath(*posix.parts[1:]) if nested_dir is not None else posix).as_posix()
        for _, posix in members
    )
    return SkillZipInspection(
        name=name, description=description, files=files, nested_dir=nested_dir
    )


def install_skill_from_zip(
    skills_root: Path,
    zip_bytes: bytes,
    *,
    overwrite: bool = False,
) -> SkillInfo:
    """从 zip 字节流安装一个技能到 ``skills_root/<name>/``。

    先通过 inspect_skill_zip 完成全部校验（不触碰磁盘），再解压到
    暂存目录并原子替换目标目录，失败时不会在 skills 目录留下半成品。

    Args:
        skills_root: 技能根目录（通常是项目下的 ``skills/``）。
        zip_bytes: zip 文件的完整字节内容。
        overwrite: 同名技能已存在时是否覆盖。

    Returns:
        安装后的技能元数据。
    """

    inspection = inspect_skill_zip(zip_bytes)
    name = inspection.name

    skills_root = Path(skills_root)
    target = skills_root / name
    if target.exists() and not overwrite:
        raise SkillConflictError(f"技能 '{name}' 已存在，如需替换请勾选覆盖上传")

    # 解压到暂存目录，校验大小/路径后再原子替换，失败不留半成品
    skills_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".staging-{name}-", dir=skills_root))
    total_bytes = 0
    file_count = 0
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            members = _safe_members(zf)  # 与预检一致的安全成员列表
            for info, posix in members:
                rel = (
                    PurePosixPath(*posix.parts[1:])
                    if inspection.nested_dir is not None
                    else posix
                )
                if str(rel) == ".":
                    continue
                dest = (staging / Path(*rel.parts)).resolve()
                if not _is_within(staging, dest):
                    raise SkillInstallError(f"检测到路径穿越条目: {info.filename!r}")
                data = zf.read(info.filename)
                total_bytes += len(data)
                if total_bytes > MAX_SKILL_TOTAL_BYTES:
                    raise SkillInstallError(
                        f"解压后总大小超出上限 {MAX_SKILL_TOTAL_BYTES // (1024 * 1024)}MB"
                    )
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                file_count += 1

        if not (staging / "SKILL.md").is_file():
            raise SkillInstallError("解压后未在技能根目录找到 SKILL.md")

        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    logger.info(
        "技能已安装: %s（%d 个文件，%s）",
        name,
        file_count,
        "覆盖" if overwrite else "新增",
    )
    return _build_skill_info(target, name, inspection.description)


def _build_skill_info(
    skill_dir: Path,
    name: str,
    description: str,
    *,
    valid: bool = True,
    error: str = "",
) -> SkillInfo:
    files = sorted(
        p.relative_to(skill_dir).as_posix()
        for p in skill_dir.rglob("*")
        if p.is_file()
    )
    updated_at = max(
        (p.stat().st_mtime for p in skill_dir.rglob("*") if p.is_file()), default=0.0
    )
    return SkillInfo(
        name=name,
        description=description,
        path=f"/skills/{name}/SKILL.md",
        files=files,
        updated_at=updated_at,
        valid=valid,
        error=error,
    )


def list_skills(skills_root: Path) -> list[SkillInfo]:
    """扫描技能根目录，返回全部技能（含 frontmatter 解析失败的问题技能）。"""

    skills_root = Path(skills_root)
    if not skills_root.is_dir():
        return []
    result: list[SkillInfo] = []
    skill_dirs = sorted(
        p for p in skills_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    for skill_dir in skill_dirs:
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.is_file():
            continue
        name = skill_dir.name
        try:
            content = skill_md.read_text(encoding="utf-8")
            parsed_name, description = parse_skill_markdown(content)
            if parsed_name != name:
                raise SkillInstallError(
                    f"目录名 '{name}' 与 frontmatter name '{parsed_name}' 不一致"
                )
            validate_skill_name(name)
            result.append(_build_skill_info(skill_dir, name, description))
        except (SkillInstallError, UnicodeDecodeError) as exc:
            logger.warning("技能 %s 的 SKILL.md 解析失败: %s", name, exc)
            result.append(_build_skill_info(skill_dir, name, "", valid=False, error=str(exc)))
    return result


def tenant_skills_root(skills_root: Path, tenant_id: str) -> Path:
    """返回指定租户的私有技能目录（skills_root/tenants/<tenant_id>/）。"""

    return Path(skills_root) / "tenants" / tenant_id


def list_skills_for_tenant(
    skills_root: Path, tenant_id: str
) -> list[SkillInfo]:
    """列出租户可见的全部技能：全局共享 + 租户私有。

    布局：
    - 全局技能：``skills_root/<name>/SKILL.md``
    - 租户私有：``skills_root/tenants/<tenant_id>/<name>/SKILL.md``

    同名时租户私有技能覆盖全局技能。
    """

    skills_root = Path(skills_root)
    global_skills = {s.name: s for s in list_skills(skills_root)}
    tenant_root = tenant_skills_root(skills_root, tenant_id)
    tenant_skills = {s.name: s for s in list_skills(tenant_root)}
    # 租户私有覆盖全局同名
    merged = {**global_skills, **tenant_skills}
    return sorted(merged.values(), key=lambda s: s.name)


def delete_skill(skills_root: Path, name: str) -> None:
    """按名称删除技能目录；不存在或名称非法时抛 SkillInstallError。"""

    validate_skill_name(name)
    root = Path(skills_root).resolve()
    target = (root / name).resolve()
    if not _is_within(root, target) or target == root:
        raise SkillInstallError("非法的技能路径")
    if not target.is_dir():
        raise SkillInstallError(f"技能 '{name}' 不存在")
    shutil.rmtree(target)
    logger.info("技能已删除: %s", name)
