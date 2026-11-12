"""技能管理器单元测试：zip 安装、校验、安全防护、扫描与删除。"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from src.skills.manager import (
    SkillConflictError,
    SkillInstallError,
    delete_skill,
    install_skill_from_zip,
    list_skills,
)

SKILL_MD = """---
name: demo-skill
description: 一个用于测试的演示技能
---

# 演示技能

正文内容。
"""


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_install_nested_zip(tmp_path: Path) -> None:
    """规范布局 <name>/SKILL.md + 辅助文件可正常安装。"""

    payload = _zip_bytes(
        {
            "demo-skill/SKILL.md": SKILL_MD.encode("utf-8"),
            "demo-skill/helper.py": b"print('hi')",
        }
    )
    info = install_skill_from_zip(tmp_path, payload)

    assert info.name == "demo-skill"
    assert info.description == "一个用于测试的演示技能"
    assert (tmp_path / "demo-skill" / "SKILL.md").is_file()
    assert (tmp_path / "demo-skill" / "helper.py").read_text() == "print('hi')"
    assert set(info.files) == {"SKILL.md", "helper.py"}


def test_install_flat_zip_uses_frontmatter_name(tmp_path: Path) -> None:
    """扁平布局（SKILL.md 在归档根）按 frontmatter 的 name 建目录。"""

    payload = _zip_bytes({"SKILL.md": SKILL_MD.encode("utf-8")})
    info = install_skill_from_zip(tmp_path, payload)

    assert info.name == "demo-skill"
    assert (tmp_path / "demo-skill" / "SKILL.md").is_file()


def test_list_and_delete(tmp_path: Path) -> None:
    install_skill_from_zip(tmp_path, _zip_bytes({"demo-skill/SKILL.md": SKILL_MD.encode()}))

    skills = list_skills(tmp_path)
    assert len(skills) == 1
    assert skills[0].valid is True
    assert skills[0].path == "/skills/demo-skill/SKILL.md"

    delete_skill(tmp_path, "demo-skill")
    assert list_skills(tmp_path) == []
    with pytest.raises(SkillInstallError):
        delete_skill(tmp_path, "demo-skill")


def test_duplicate_requires_overwrite(tmp_path: Path) -> None:
    payload = _zip_bytes({"demo-skill/SKILL.md": SKILL_MD.encode()})
    install_skill_from_zip(tmp_path, payload)

    with pytest.raises(SkillConflictError):
        install_skill_from_zip(tmp_path, payload)

    info = install_skill_from_zip(tmp_path, payload, overwrite=True)
    assert info.name == "demo-skill"


def test_reject_missing_frontmatter(tmp_path: Path) -> None:
    payload = _zip_bytes({"bad-skill/SKILL.md": "# 没有 frontmatter\n".encode()})
    with pytest.raises(SkillInstallError, match="frontmatter"):
        install_skill_from_zip(tmp_path, payload)
    # 失败不留目录
    assert not (tmp_path / "bad-skill").exists()


def test_reject_name_directory_mismatch(tmp_path: Path) -> None:
    payload = _zip_bytes({"wrong-dir/SKILL.md": SKILL_MD.encode()})
    with pytest.raises(SkillInstallError, match="不一致"):
        install_skill_from_zip(tmp_path, payload)


def test_reject_invalid_skill_name(tmp_path: Path) -> None:
    content = SKILL_MD.replace("name: demo-skill", "name: Demo_Skill")
    payload = _zip_bytes({"SKILL.md": content.encode()})
    with pytest.raises(SkillInstallError, match="不合法"):
        install_skill_from_zip(tmp_path, payload)


def test_reject_zip_slip(tmp_path: Path) -> None:
    """含 ../ 穿越条目的压缩包必须被拒绝。"""

    payload = _zip_bytes(
        {
            "demo-skill/SKILL.md": SKILL_MD.encode(),
            "demo-skill/../../evil.txt": b"pwned",
        }
    )
    with pytest.raises(SkillInstallError):
        install_skill_from_zip(tmp_path, payload)
    assert not (tmp_path.parent / "evil.txt").exists()


def test_reject_multiple_skills_in_one_zip(tmp_path: Path) -> None:
    other = SKILL_MD.replace("demo-skill", "other-skill")
    payload = _zip_bytes(
        {
            "demo-skill/SKILL.md": SKILL_MD.encode(),
            "other-skill/SKILL.md": other.encode(),
        }
    )
    with pytest.raises(SkillInstallError, match="多个 SKILL.md"):
        install_skill_from_zip(tmp_path, payload)


def test_reject_non_zip(tmp_path: Path) -> None:
    with pytest.raises(SkillInstallError, match="zip"):
        install_skill_from_zip(tmp_path, b"not a zip at all")


def test_invalid_skill_listed_with_error(tmp_path: Path) -> None:
    """已落盘但 frontmatter 损坏的技能仍应列出，并标记 valid=False。"""

    skill_dir = tmp_path / "broken-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# broken", encoding="utf-8")

    skills = list_skills(tmp_path)
    assert len(skills) == 1
    assert skills[0].valid is False
    assert skills[0].error
