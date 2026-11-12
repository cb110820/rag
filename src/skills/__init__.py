"""员工技能（Skill）管理。

每个技能是 ``skills/<skill-name>/`` 目录，内含必需的 ``SKILL.md``
（YAML frontmatter + Markdown 指令）和可选的辅助文件。
DeepAgents 的 SkillsMiddleware 会自动扫描并把技能清单注入系统提示词，
模型按需用 read_file 读取完整指令（渐进披露）。
"""

from src.skills.manager import (
    SkillConflictError,
    SkillInfo,
    SkillInstallError,
    delete_skill,
    install_skill_from_zip,
    list_skills,
    list_skills_for_tenant,
    tenant_skills_root,
)

__all__ = [
    "SkillConflictError",
    "SkillInfo",
    "SkillInstallError",
    "delete_skill",
    "install_skill_from_zip",
    "list_skills",
    "list_skills_for_tenant",
    "tenant_skills_root",
]
