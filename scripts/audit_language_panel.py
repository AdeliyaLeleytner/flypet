"""Build an author-auditable, AI-assisted descriptive audit of the fixed panel.

The quoted-span classifications below are explicit post-hoc judgments, not a
learned evaluator, blinded human labels, or a population accuracy estimate.
"""

from pathlib import Path
import json, hashlib
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "paper/results/language-v1"
OUT = ROOT / "paper/analysis"

# Each flagged span is checked verbatim against the original response.
FLAGS = {
    "sugar": [
        (
            "unsupported_physical",
            "шея и средние ноги чуть подобрались",
            "VNC motor activity is recorded, but no physical neck or leg motion is simulated.",
        )
    ],
    "bitter": [
        (
            "unsupported_explanation",
            "одной вспышки в 250 мс мало",
            "No duration comparison was made; silent named outputs do not identify duration as their cause.",
        )
    ],
    "sugar_bitter": [
        (
            "unsupported_physical",
            "лёгкое напряжение в шее",
            "Neck motor-neuron activity does not measure muscle tension.",
        )
    ],
    "antenna_touch": [
        (
            "unsupported_explanation",
            "просто фон в теле",
            "The record does not establish a background origin or behavioral irrelevance for the wing motor activity.",
        )
    ],
    "sugar_paraphrase": [
        (
            "explicit_trace_contradiction",
            "левый мотонейрон CB0701 частит около 100 Гц, правый — 68 Гц",
            "Recorded output is right=100 Hz, left=68 Hz.",
        )
    ],
    "looming_paraphrase": [
        (
            "nomenclature",
            "гигантские волокна DNp01 и DNp04",
            "The model catalogue identifies DNp01 as giant fibre; DNp04 is a separate looming-associated type.",
        ),
        ("unsupported_explanation", "гудят на пределе", "No saturation bound was established."),
    ],
    "apple": [
        (
            "inference_caution",
            "видимо, это первая встреча",
            "The case was reset, but zero valence delta alone is not sufficient evidence of first exposure.",
        )
    ],
    "vinegar": [
        (
            "delta_absolute_ambiguity",
            "он не тянет и не отталкивает",
            "No learned shift (delta=0) coexists with positive absolute model valence (0.15).",
        )
    ],
    "glass": [
        (
            "explicit_trace_contradiction",
            "обонятельный сигнал такой слабый",
            "No odor input was delivered: resolver unmapped, odor_input_present=false; the only delivered input was cold.",
        ),
        (
            "unsupported_explanation",
            "Стекло почти ничем не пахнет",
            "Lack of a chemical mapping is not a measurement of the object or animal detection.",
        ),
    ],
    "memory_naive_ethyl_acetate": [
        (
            "delta_absolute_ambiguity",
            "запах не тянет и не отталкивает",
            "The learned change is zero, but total model valence is +0.18.",
        )
    ],
    "memory_A_pyrrolidine": [
        (
            "unsupported_physical",
            "просто разворачиваюсь прочь",
            "No body motion or source position is modeled; the left-turn readout is unchanged from the naive probe.",
        ),
        (
            "nomenclature",
            "рецептор VL1",
            "VL1 names a glomerular population in this interface, not an individual receptor.",
        ),
    ],
    "memory_B_ethyl_acetate": [
        (
            "approximation_caution",
            "почти втрое",
            "Avoidance/approach is 772/308=2.506. This approximate wording is not scored as a numerical contradiction.",
        )
    ],
    "memory_B_pyrrolidine": [
        (
            "delta_absolute_ambiguity",
            "запах пока ни притягивает, ни отталкивает",
            "The learned change is zero, but total model valence remains +0.44.",
        )
    ],
}
SUPPORT = {
    "sugar": (
        "оба CB0701 разом загорелись",
        "Bilateral output CB0701 rates right88/left64; other named behavioral channels0.",
    ),
    "bitter": (
        "все поведенческие каналы молчат",
        "All seven named behavioral readouts are0; central activity/VNC are separately present.",
    ),
    "sugar_bitter": (
        "24 и 20 Гц",
        "Recorded right/left CB0701 rates24/20; compared with88/64 in same-seed sugar.",
    ),
    "antenna_touch": (
        "SAD093 76 Гц",
        "Grooming-associated SAD093 rate76; escape/walking/turn readouts0.",
    ),
    "looming": (
        "DNp04 разряжается на 248 Гц",
        "Recorded maximum DNp04 rate248; backward and turn channels active.",
    ),
    "sugar_paraphrase": (
        "Побега, чистки, ходьбы и поворота нет",
        "Those named readouts are0; the separate left/right quotation is wrong.",
    ),
    "looming_paraphrase": (
        "Хоботок и чистка молчат полностью",
        "Both named channels are0; backward/escape channels active.",
    ),
    "apple": ("168 Гц против 80", "Approach/avoidance model sums168/80; delta0 and DNa02 peak20."),
    "vinegar": (
        "память тут ничего не добавила",
        "Valence delta0; proboscis/escape/grooming0 and turn20.",
    ),
    "sound_400hz": (
        "DNp01 28 Гц, DNp02 20 Гц",
        "Recorded right DNp01/DNp02 rates28/20; sound400Hz amplitude0.5 preserved.",
    ),
    "zero_hz_sugar": (
        "ни один канал не шевельнулся",
        "Effective inputs empty, total spikes0, all named channels and VNC rates0.",
    ),
    "glass": (
        "всё молчит",
        "The named behavioral readouts are0; the odor assertion is separately contradicted.",
    ),
    "memory_naive_ethyl_acetate": (
        "DNa02/l 40Гц",
        "Left DNa02 rate40; no current reinforcement, delta0.",
    ),
    "memory_naive_pyrrolidine": (
        "+0.44 что с памятью, что без",
        "Both reported model valences0.44; DNa02 peak8 and VL1 input-population mean144.1Hz.",
    ),
    "memory_A_ethyl_acetate": (
        "(+0.18), а с памятью подскочила до +0.65",
        "Reported valence changes0.18→0.65; delta+0.47; left DNa02 28Hz.",
    ),
    "memory_A_pyrrolidine": (
        "24 Гц против приближения 4 Гц",
        "Avoidance/approach sums24/4; valence+0.44→−0.69, left DNa02 8Hz.",
    ),
    "memory_B_ethyl_acetate": (
        "DNa02/l 24Гц",
        "Left DNa02 rate24; score0.18→−0.43, avoidance772>approach308.",
    ),
    "memory_B_pyrrolidine": (
        "валентность не сдвинулась",
        "Both model scores0.44, delta0; whole neural state need not be unchanged.",
    ),
}


def main():
    rs = [json.loads(x) for x in (RUN / "records.jsonl").read_text().splitlines() if x]
    assert len(rs) == 18
    cases = []
    counts = Counter()
    for r in rs:
        cid = r["case_id"]
        reply = r["reply"]
        s = r["summary"]
        span, explanation = SUPPORT[cid]
        assert span in reply, (cid, span)
        flags = []
        for category, span, note in FLAGS.get(cid, []):
            assert span in reply, (cid, span)
            flags.append({"category": category, "span_ru": span, "reason": note})
            counts[category] += 1
        evidence = {
            "accepted_plan": r["accepted_plan"],
            "odor_source": {
                k: (s.get("odor_source") or {}).get(k)
                for k in ("source", "status", "compound", "mapping")
            },
            "odor_input_present": s.get("odor_input_present"),
            "valence": {
                k: (s.get("valence") or {}).get(k)
                for k in ("naive_score", "score", "delta", "approach_hz", "avoid_hz")
            },
            "behavior_peak_hz": {
                k: v["max_rate_hz"]
                for k, v in s["behaviours"].items()
                if k not in ("other_motor", "descending_all")
            },
        }
        cases.append(
            {
                "case_id": cid,
                "kind": r["kind"],
                "source_result": f"paper/results/language-v1/trials/{cid}/result.json",
                "reply_ru": reply,
                "supported_anchor": {"span_ru": SUPPORT[cid][0], "evidence": explanation},
                "flags": flags,
                "evidence": evidence,
            }
        )
    # Independent artifact checker supplies recomputed arrays, not only run-time flags.
    independent = json.loads((OUT / "language_audit_independent_evidence.json").read_text())
    manifest = json.loads((RUN / "manifest.json").read_text())
    summary = {
        "n_fixed_interactions": 18,
        "completed": sum(r["status"] == "complete_unannotated" for r in rs),
        "natural_language_cases": 12,
        "controlled_memory_cases": 6,
        "actual_llm_calls": manifest["actual_llm_calls"],
        "model_ids_reported_by_cli": manifest["actual_model_ids"],
        "requested_model": "sonnet",
        "reported_api_equivalent_cost_usd": manifest["reported_cost_usd"],
        "explicit_trace_contradiction_responses": [
            r["case_id"]
            for r in cases
            if any(f["category"] == "explicit_trace_contradiction" for f in r["flags"])
        ],
        "flag_counts": dict(counts),
        "responses_with_supported_anchor": len(cases),
        "interpretation": "Counts describe this post-hoc audit of one fixed panel, not a calibrated overall faithfulness percentage or a blinded human study. An anchor does not certify the rest of its response.",
    }
    out = {
        "schema_version": 1,
        "audit_provenance": "AI-assisted quoted-span audit, independently checked against records, for author review. No new LLM calls used for scoring.",
        "records_sha256": hashlib.sha256((RUN / "records.jsonl").read_bytes()).hexdigest(),
        "rubric": {
            "supported_anchor": "At least one specific factual statement checked against trace quantities.",
            "explicit_trace_contradiction": "Unambiguous conflict with delivered-input existence or named numerical/laterality fields.",
            "unsupported_explanation": "Cause, background interpretation, saturation or biological-absence assertion not established by this panel.",
            "unsupported_physical": "Unqualified body motion, tension or source-relative direction without simulated body/source position.",
            "delta_absolute_ambiguity": "Zero learned change expressed as neutral absolute valence; retained separately from unambiguous contradictions.",
            "cautions": "Hedged metaphors, approximations, naming and evidential ambiguity are listed without treating every first-person phrase as a measured claim.",
        },
        "summary": summary,
        "cases": cases,
        "independent_evidence": "language_audit_independent_evidence.json",
    }
    (OUT / "language_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    (OUT / "language_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
