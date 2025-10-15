from typing import Dict, List, Union, Tuple
import math

EmotionProbs = Dict[str, float]
Segment = Dict[str, Union[int, EmotionProbs]]

def fuzzy_mood(
    data: Union[Segment, List[Segment]],
    return_details: bool = False,
) -> Union[str, Dict[str, object]]:
    """
    Compute an interpretable average mood over audio using fuzzy rules and a valence–arousal view.
    
    Input:
      - data: a dict {"start_ms": int, "end_ms": int, "probabilities": {"angry": float, "sad": float, "neutral": float, "positive": float, "other": float}}
               or a list of such dicts (e.g., multiple frames/chunks).
      - return_details: if True, returns a dict with diagnostics (valence, arousal, chosen rule, confidence).
    
    Output:
      - By default, a single mood phrase (str), e.g., "calm positive mood." or "negative background with irritation."
      - If return_details=True, returns a dict:
          {
            "mood": str,
            "valence": float in [-1,1],
            "arousal": float in [0,1],
            "confidence": float in [0,1],
            "avg_probs": dict,
            "fired_rule": str,
            "alternatives": List[Tuple[str, float]]
          }
    """

    # Normalize input to a list
    segments = data if isinstance(data, list) else [data]

    # Ensure required keys and sane values
    def safe_probs(p: EmotionProbs) -> EmotionProbs:
        keys = ["angry", "sad", "neutral", "positive", "other"]
        q = {k: float(max(0.0, min(1.0, p.get(k, 0.0)))) for k in keys}
        s = sum(q.values())
        if s > 0:
            q = {k: v / s for k, v in q.items()}
        return q

    # Duration-weighted pooling across segments
    total_w = 0.0
    acc = {k: 0.0 for k in ["angry", "sad", "neutral", "positive", "other"]}
    for seg in segments:
        start = int(seg.get("start_ms", 0))
        end = int(seg.get("end_ms", start))
        w = max(1, end - start)  # at least 1ms weight
        probs = safe_probs(seg.get("probabilities", {}))
        for k, v in probs.items():
            acc[k] += w * v
        total_w += w

    if total_w <= 0:
        return {"mood": "unknown mood.", "valence": 0.0, "arousal": 0.5, "confidence": 0.0, "avg_probs": {}, "fired_rule": "none", "alternatives": []} if return_details else "unknown mood."

    avg = {k: acc[k] / total_w for k in acc.keys()}

    # Map emotions to valence [-1,1] and arousal [0,1] (circumplex-inspired)
    V = {"angry": -0.8, "sad": -0.7, "neutral": 0.0, "positive": 0.8, "other": 0.0}
    A = {"angry": 0.8,  "sad": 0.2,  "neutral": 0.3, "positive": 0.6, "other": 0.5}

    valence = sum(avg[k] * V[k] for k in avg.keys())
    arousal = sum(avg[k] * A[k] for k in avg.keys())

    # Confidence from entropy + peak concentration
    def entropy(p: Dict[str, float]) -> float:
        eps = 1e-12
        return -sum(pi * math.log(pi + eps) for pi in p.values() if pi > 0)

    H = entropy(avg)
    Hmax = math.log(len(avg))  # log 5
    peak = max(avg.values()) if avg else 0.0
    confidence = 0.5 * peak + 0.5 * (1.0 - (H / Hmax if Hmax > 0 else 1.0))

    # Fuzzy helpers
    def clip01(x: float) -> float:
        return max(0.0, min(1.0, x))

    def soft_more(x: float, thresh: float, width: float = 0.2) -> float:
        # 0 at x<=thresh, 1 at x>=thresh+width
        return clip01((x - thresh) / max(1e-6, width))

    def soft_less(x: float, thresh: float, width: float = 0.2) -> float:
        # 1 at x<=thresh, 0 at x>=thresh+width
        return clip01((thresh + width - x) / max(1e-6, width))

    def approx_equal(a: float, b: float, rel: float = 0.15, abs_tol: float = 0.05) -> float:
        denom = max(abs_tol, rel * (a + b) / 2.0 + 1e-6)
        return clip01(1.0 - abs(a - b) / denom)

    p = avg
    max_prob = max(p.values()) if p else 0.0
    pos_deg = clip01((valence + 1.0) / 2.0)  # 0..1
    neg_deg = 1.0 - pos_deg
    low_ar_deg = soft_less(arousal, 0.45, 0.25)
    hi_ar_deg = soft_more(arousal, 0.60, 0.20)

    # Define fuzzy rules with degrees
    rules: List[Tuple[str, float]] = []

    rules.append((
        "негативный фон с раздражением.",
        clip01(((p["sad"] + p["angry"]) - 0.50) / 0.20)
    ))

    rules.append((
        "спокойное позитивное настроение.",
        min(approx_equal(p["neutral"], p["positive"]), low_ar_deg, pos_deg)
    ))

    # Additional nuanced rules
    rules.append((
        "в основном нейтральное настроение.",
        clip01((p["neutral"] - 0.60) / 0.20)
    ))

    rules.append((
        "энергичное позитивное настроение.",
        min(clip01((p["positive"] - 0.60) / 0.20), hi_ar_deg)
    ))

    rules.append((
        "подавленное настроение.",
        min(clip01((p["sad"] - 0.40) / 0.20), soft_less(arousal, 0.40, 0.20))
    ))

    rules.append((
        "напряженное, раздражительное настроение.",
        min(clip01((p["angry"] - 0.40) / 0.20), hi_ar_deg)
    ))

    rules.append((
        "противоречивое настроение.",
        min(clip01((p["positive"] - 0.30) / 0.20), clip01((p["sad"] - 0.30) / 0.20))
    ))

    rules.append((
        "оптимистичное с ноткой напряжения.",
        min(clip01((p["positive"] - 0.50) / 0.20), clip01((p["angry"] - 0.25) / 0.20))
    ))

    rules.append((
        "теплое, уравновешенное спокойное настроение.",
        min(clip01((p["positive"] - 0.40) / 0.20), clip01((p["neutral"] - 0.30) / 0.20), clip01((0.30 - (p["sad"] + p["angry"])) / 0.20))
    ))

    rules.append((
        "неопределенное настроение со смешанными сигналами.",
        max(clip01((p["other"] - 0.50) / 0.20), clip01((0.35 - max_prob) / 0.15))
    ))

    # Pick best-scoring rule
    rules_sorted = sorted(rules, key=lambda x: x[1], reverse=True)
    best_label, best_score = rules_sorted[0]

    # Fallback if no strong rule fires
    if best_score < 0.30:
        if valence > 0.20:
            best_label = "в общем положительное настроение."
        elif valence < -0.20:
            best_label = "в общем грустное настроение."
        else:
            best_label = "в общем нейтральное настроение."

    if return_details:
            return {
                "mood": best_label,
                "valence": round(valence, 4),
                "arousal": round(arousal, 4),
                "confidence": round(confidence, 4),
                "avg_probs": {k: round(v, 4) for k, v in avg.items()},
                "fired_rule": best_label,
                "alternatives": [{"label": lab, "score": round(sc, 4)} for lab, sc in rules_sorted[1:4]],
            }
    return best_label
