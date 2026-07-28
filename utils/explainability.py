"""
Explainability utilities for deepfake detection.

Provides structured risk explanations with evidence summaries
for image, video, audio, and multimodal analysis.
"""


def explain_risk(score, model_scores=None):
    """
    Generate structured risk explanation for image/video analysis.

    Args:
        score: Final risk score (0.0 - 1.0).
        model_scores: Optional dict of per-model scores for evidence.

    Returns:
        str with detailed risk explanation.
    """
    if score >= 0.60:
        level = "AI-GENERATED"
        desc = "Strong indicators of AI generation or manipulation"
    else:
        level = "AUTHENTIC"
        desc = "No significant manipulation indicators"

    explanation = f"{level} — {desc}"

    if model_scores:
        evidence = []
        if model_scores.get("vit_prob", 0) > 0.6:
            evidence.append("ViT detected deepfake patterns")
        if model_scores.get("face_prob", 0) > 0.6:
            evidence.append("facial manipulation artifacts found")
        if model_scores.get("forensic_prob", 0) > 0.5:
            evidence.append("noise/ELA inconsistency detected")
        if model_scores.get("frequency_prob", 0) > 0.5:
            evidence.append("frequency-domain anomalies")
        if model_scores.get("eff_prob", 0) > 0.6:
            evidence.append("EfficientNet flagged AI generation")
        if model_scores.get("dino_prob", 0) > 0.6:
            evidence.append("DINOv2 detected synthetic features")

        if evidence:
            explanation += f". Evidence: {'; '.join(evidence)}"

    return explanation


def explain_audio_risk(fake_prob):
    """Explain audio deepfake risk level."""
    if fake_prob >= 0.60:
        return "AI-GENERATED — AI-generated speech detected (voice cloning / TTS)"
    else:
        return "AUTHENTIC — Audio appears authentic"


def explain_multimodal(modality_scores, final_score):
    """
    Generate explanation for multimodal fusion result.

    Args:
        modality_scores: dict with image/video/audio scores.
        final_score: fused risk score.

    Returns:
        str explanation.
    """
    active = {k: v for k, v in modality_scores.items() if v is not None}

    if not active:
        return "No modalities analyzed"

    parts = []
    for mod, score in active.items():
        if score >= 60:
            parts.append(f"{mod}: AI-generated ({score}%)")
        else:
            parts.append(f"{mod}: authentic ({score}%)")

    if final_score >= 0.60:
        verdict = "Strong evidence of manipulation across modalities"
    else:
        verdict = "Content appears authentic across analyzed modalities"

    return f"{verdict}. Per-modality: {'; '.join(parts)}"
