"""Agent Skills loader.

Discovers and loads skills from a ``skills/`` directory following the
`Agent Skills <https://agentskills.io>`_ open standard.

Progressive disclosure:
1. At startup, only skill **metadata** (name + description) is loaded
   and shown in the system prompt (~100 tokens per skill).
2. When the agent calls ``load_skill(name)``, the full **instructions**
   (SKILL.md body) are returned into the conversation context.
3. Referenced files in ``scripts/``, ``references/``, ``assets/`` can be
   loaded on demand by the agent.

Usage::

    from chimera_agent_baseline.skills import load_skills, format_skills_summary

    skills = load_skills("skills")
    summary = format_skills_summary(skills)     # for the system prompt
    body = skills["guideline-search"]["body"]   # for load_skill tool
"""

import logging
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


def load_skills(skills_dir: str | Path) -> dict[str, dict]:
    """Load all skills from a directory.

    Returns a dict mapping skill name to {name, description, body, path}.
    """
    skills_dir = Path(skills_dir)
    if not skills_dir.exists():
        log.info("Skills directory not found: %s", skills_dir)
        return {}

    skills = {}
    for skill_path in sorted(skills_dir.iterdir()):
        skill_md = skill_path / "SKILL.md"
        if not skill_md.exists():
            continue

        text = skill_md.read_text()
        frontmatter, body = _parse_frontmatter(text)

        name = frontmatter.get("name")
        if not name:
            log.warning("Skipping skill at %s: missing 'name' in frontmatter", skill_path)
            continue

        skills[name] = {
            "name": name,
            "description": frontmatter.get("description", ""),
            "body": body.strip(),
            "path": str(skill_path),
        }

    log.info("Loaded %d skills from %s", len(skills), skills_dir)
    return skills


def format_skills_summary(skills: dict[str, dict]) -> str:
    """Format skill metadata into a system prompt section.

    Only includes name and description (progressive disclosure level 1).
    The agent must call ``load_skill(name)`` to get the full instructions.
    """
    if not skills:
        return ""

    lines = [
        "## Skills",
        "",
        "The following skills are available. Call `load_skill(name)` to "
        "load the full instructions before using the skill.",
        "",
    ]
    for skill in skills.values():
        lines.append(f"- **{skill['name']}**: {skill['description']}")

    return "\n".join(lines)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a SKILL.md into YAML frontmatter and markdown body."""
    if not text.startswith("---"):
        return {}, text

    end = text.find("---", 3)
    if end == -1:
        return {}, text

    frontmatter_str = text[3:end].strip()
    body = text[end + 3 :]

    try:
        frontmatter = yaml.safe_load(frontmatter_str) or {}
    except yaml.YAMLError:
        log.warning("Failed to parse SKILL.md frontmatter")
        frontmatter = {}

    return frontmatter, body
