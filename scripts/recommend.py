"""Simple resume-based job recommender

This module exposes a function `recommend_jobs(text, n=5)` that returns
a small list of recommended job titles with a heuristic score and reason.

It's intentionally lightweight so it can run without heavy ML deps.
"""
from typing import List, Dict
import re

# Small keyword -> job mapping for heuristic matching
JOB_KEYWORDS = {
    "software": ["software", "engineer", "developer", "javascript", "python", "java", "react", "node"],
    "data_scientist": ["data", "machine learning", "ml", "model", "statistics", "python", "pandas", "numpy"],
    "product_manager": ["product", "roadmap", "stakeholder", "product management", "pm", "strategy"],
    "designer": ["design", "ux", "ui", "figma", "sketch", "prototype", "visual"],
    "devops_engineer": ["devops", "ci/cd", "docker", "kubernetes", "aws", "azure", "gcp", "infrastructure"],
    "qa_engineer": ["test", "qa", "automation", "selenium", "jest", "pytest"],
    "technical_writer": ["documentation", "write", "technical writing", "docs"],
    "teacher": ["teaching", "teacher", "curriculum", "lesson", "classroom", "student", "tutor", "education"],
    "animator": ["animation", "after effects", "aftereffects", "maya", "blender", "3d", "render", "storyboard", "animator"],
    "accountant": ["accounting", "bookkeeping", "finance", "tax", "quickbooks", "audit", "accounts payable", "accounts receivable"],
    "nurse": ["nurse", "rn", "patient care", "clinical", "ward", "patient", "nursing"],
    "doctor": ["doctor", "physician", "md", "medical", "clinical", "patient care"],
    "marketing_manager": ["marketing", "seo", "campaign", "content", "social media", "google analytics", "email marketing", "growth"],
    "sales_rep": ["sales", "quota", "crm", "pipeline", "prospecting", "negotiation", "account executive", "bdm"],
    "hr_manager": ["hr", "human resources", "talent acquisition", "recruiting", "onboarding", "employee relations"],
    "graphic_designer": ["photoshop", "illustrator", "figma", "adobe", "design", "branding", "visual design"],
    "project_manager": ["project manager", "project management", "scrum", "pmp", "gantt", "stakeholder management", "delivery"],
    "social_worker": ["social work", "social worker", "case management", "counseling", "community"],
}

JOB_TITLES = {
    "software": "Software Engineer",
    "data_scientist": "Data Scientist",
    "product_manager": "Product Manager",
    "designer": "Designer / UX",
    "devops_engineer": "DevOps Engineer",
    "qa_engineer": "QA / Test Engineer",
    "technical_writer": "Technical Writer",
    "teacher": "Teacher / Educator",
    "animator": "Animator / Motion Designer",
    "accountant": "Accountant / Bookkeeper",
    "nurse": "Nurse",
    "doctor": "Physician / Doctor",
    "marketing_manager": "Marketing Manager",
    "sales_rep": "Sales Representative",
    "hr_manager": "HR / People Operations",
    "graphic_designer": "Graphic Designer",
    "project_manager": "Project Manager",
    "social_worker": "Social Worker",
}



def _normalize(text: str) -> str:
    # Lowercase and replace common separators with spaces so phrase matching works
    text = (text or "").lower()
    text = re.sub(r"[\/_\-\+]", " ", text)
    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    text = _normalize(text)
    tokens = re.split(r"\W+", text)
    return [t for t in tokens if t]


def recommend_jobs(text: str, n: int = 5) -> List[Dict]:
    """Return up to `n` recommended job dicts: {title, score, reason}

    This is a heuristic matching implementation using keyword overlap.
    """
    if not text:
        return []

    normalized = _normalize(text)
    tokens = _tokenize(text)
    joined = " ".join(tokens)

    scores = {}
    # optional per-keyword weight map to boost specific signals
    SPECIAL_WEIGHTS = {
        "ci cd": 3,
        "ci/cd": 3,
        "cicd": 3,
        "docker": 3,
        "kubernetes": 3,
        "aws": 3,
        "devops": 3,
        "infrastructure": 2,
        "machine learning": 2,
        "python": 2,
        "data": 2,
    }

    for key, kws in JOB_KEYWORDS.items():
        score_sum = 0
        total_possible = 0
        reasons = []
        for kw in kws:
            k = kw.lower()
            # normalize keyword separators too
            k_norm = re.sub(r"[\/_\-\+]", " ", k).strip()
            weight = SPECIAL_WEIGHTS.get(k_norm, 1)
            total_possible += weight

            matched = False
            # If keyword is a short token (single word), match tokens list
            if " " not in k_norm:
                if k_norm in tokens:
                    matched = True
            else:
                # phrase - check in normalized full text
                if k_norm in normalized:
                    matched = True
                else:
                    # if all sub-words of the phrase appear separately in tokens, count it
                    parts = [p for p in k_norm.split() if p]
                    if parts and all(p in tokens for p in parts):
                        matched = True

            if matched:
                score_sum += weight
                reasons.append(kw)

        if score_sum > 0 and total_possible > 0:
            # normalized score to 0-100 range using weights
            norm_score = min(100, int((score_sum / total_possible) * 100))
            # If there was any high-signal keyword matched, boost to a reasonable minimum
            if any(re.sub(r"[\/_\-\+]", " ", r).strip() in SPECIAL_WEIGHTS and SPECIAL_WEIGHTS.get(re.sub(r"[\/_\-\+]", " ", r).strip(), 1) >= 3 for r in reasons):
                norm_score = max(norm_score, 60)

            scores[key] = {
                "score": norm_score,
                "reasons": reasons,
            }

    # also boost if there are explicit job-title words
    for key, info in scores.items():
        if JOB_TITLES[key].split()[0].lower() in joined:
            info["score"] = min(100, info["score"] + 10)

    # produce sorted list
    items = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)

    results = []
    for key, info in items[:n]:
        results.append({
            "title": JOB_TITLES.get(key, key),
            "score": info["score"],
            "reason": f"Matched keywords: {', '.join(info['reasons'])}",
        })

    # fallback: if nothing found, suggest general roles based on words
    if not results:
        results = [
            {"title": "Generalist / Entry-level", "score": 50, "reason": "No strong keyword matches; broad role"}
        ]

    return results


if __name__ == "__main__":
    # simple CLI test
    example = """
    Experienced Python developer with background in machine learning, numpy, pandas and deployment with docker and aws.
    """
    recs = recommend_jobs(example)
    import json
    print(json.dumps(recs, indent=2))
