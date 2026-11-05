"""Frozen three-object bilingual instruction-coverage protocol."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


CONDITIONS = ("en_standard", "en_rewrite", "zh_standard", "zh_rewrite")
METHODS = ("astra", "rule", "vlm")
SEEDS = (209101, 209102, 209103)
OBJECTS = {
    "cube": {
        "display": "cube",
        "labels": ("T", "E", "R", "L", "S", "H"),
        "standard": {
            "en": "Reorient the cube so that the face printed with the capital letter {label} points upward.",
            "zh": "请将立方体上印有大写字母 {label} 的那一面转到朝上。",
        },
        "rewrite": {
            "en": "Turn the cube until the face marked {label} is on top.",
            "zh": "把标有字母 {label} 的那一面翻到立方体顶部。",
        },
    },
    "hedgehog": {
        "display": "hedgehog",
        "labels": ("A", "B", "C", "D", "E", "F"),
        "standard": {
            "en": "Reorient the hedgehog so that the surface marked with letter {label} faces upward.",
            "zh": "请将刺猬物体上标有字母 {label} 的那一面转到朝上。",
        },
        "rewrite": {
            "en": "Turn the hedgehog until its {label}-labeled surface is on top.",
            "zh": "把带有字母 {label} 标记的刺猬表面转到顶部。",
        },
    },
    "duck": {
        "display": "duck",
        "labels": ("A", "B", "C", "D", "E", "F"),
        "standard": {
            "en": "Reorient the duck so that the surface marked with letter {label} faces upward.",
            "zh": "请将鸭子物体上标有字母 {label} 的那一面转到朝上。",
        },
        "rewrite": {
            "en": "Turn the duck until its {label}-labeled surface is on top.",
            "zh": "把带有字母 {label} 标记的鸭子表面转到顶部。",
        },
    },
}

RULE_PROMPT_ZH = {
    "cube": "请根据当前 RGB 图像和放大裁剪图，判断请求的大写字母出现在哪种颜色的面上，以及该字母是否清晰可读。物体表面包含六种颜色和字母 T、E、R、L、S、H。不要推断被遮挡的字母。",
    "duck": "请根据当前 RGB 图像和放大裁剪图，判断请求字母位于哪种颜色的表面，以及该字母是否清晰可读。物体表面包含六种颜色和字母 A 到 F。不要假设字母对应固定颜色，也不要推断被遮挡的字母。",
    "hedgehog": "请根据当前 RGB 图像和放大裁剪图，判断请求字母位于哪种颜色的表面，以及该字母是否清晰可读。物体表面包含六种颜色和字母 A 到 F。不要假设字母对应固定颜色，也不要推断被遮挡的字母。",
}

RULE_SCHEMA_ZH = {
    "cube": '仅返回一个 JSON 对象：{"color":"red|green|blue|yellow|magenta|cyan|unknown","visible":true,"evidence":"观察到的字母和颜色"}。color 必须是图像中实际包含请求字母的颜色。证据不足时使用 unknown 和 false。',
    "duck": '仅返回一个 JSON 对象：{"number":1,"visible":true,"evidence":"观察到的字母及颜色"}。number 是承载请求字母的物理动作技能编号；不要把字母序号当作动作编号。无法确认字母与颜色关系时使用 null 和 false。',
    "hedgehog": '仅返回一个 JSON 对象：{"number":1,"visible":true,"evidence":"观察到的字母及颜色"}。number 是承载请求字母的物理动作技能编号；不要把字母序号当作动作编号。无法确认字母与颜色关系时使用 null 和 false。',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def condition_parts(condition):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown prompt condition: {condition}")
    language, style = condition.split("_", 1)
    return language, style


def build_cases(object_name, condition):
    if object_name not in OBJECTS:
        raise ValueError(f"Unknown object: {object_name}")
    language, style = condition_parts(condition)
    obj = OBJECTS[object_name]
    cases = []
    for block, seed in enumerate(SEEDS):
        for episode in range(20):
            goal = (episode + 2 * block) % 6
            label = obj["labels"][goal]
            cases.append(dict(
                index=len(cases), block=block, base_seed=seed, episode=episode,
                trial_seed=seed * 1000 + episode, goal_index=goal, label=label,
                language=language, wording=style, prompt_condition=condition,
                instruction=obj[style][language].format(label=label),
            ))
    counts = Counter(case["goal_index"] for case in cases)
    if len(cases) != 60 or counts != Counter({i: 10 for i in range(6)}):
        raise AssertionError(f"Unbalanced cases for {object_name}/{condition}: {counts}")
    return cases


def build_bundle():
    bundle = {}
    for object_name in OBJECTS:
        bundle[object_name] = {
            condition: build_cases(object_name, condition)
            for condition in CONDITIONS
        }
        reference = bundle[object_name][CONDITIONS[0]]
        pairing = [(c["index"], c["trial_seed"], c["goal_index"]) for c in reference]
        for condition in CONDITIONS[1:]:
            other = [(c["index"], c["trial_seed"], c["goal_index"])
                     for c in bundle[object_name][condition]]
            if other != pairing:
                raise AssertionError(f"Pairing changed in {object_name}/{condition}")
    return bundle


def write_bundle(out_dir):
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)
    bundle = build_bundle()
    paths = {}
    for object_name, conditions in bundle.items():
        for condition, cases in conditions.items():
            path = out_dir / f"{object_name}_{condition}.json"
            payload = dict(
                format="multilingual_case_manifest_v1", object=object_name,
                method_conditions=list(METHODS), prompt_condition=condition,
                cases=cases, protocol_claim="deployed_language_coverage_only",
            )
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
            paths[path.name] = sha(path)
    study = dict(
        format="multilingual_study_v1", status="frozen",
        objects=list(OBJECTS), methods=list(METHODS), prompt_conditions=list(CONDITIONS),
        goals_per_object=6, repeats_per_goal=10, cases_per_condition=60,
        total_episodes=3 * 3 * 4 * 60,
        pairing="same trial seed and target reused across methods and all four language/wording conditions within each object",
        seed_blocks=list(SEEDS), case_manifest_sha256=paths,
        interpretation={
            "astra": "hosted model receives the native-language instruction",
            "rule": "localized task prompt and localized output instructions/schema in Chinese cells",
            "vlm": "English-trained deployed checkpoint; no Chinese retraining",
            "claim": "language coverage of the deployed systems; not a causal comparison of fine-tuning methods",
        },
    )
    (out_dir / "study.json").write_text(json.dumps(study, indent=2, ensure_ascii=False) + "\n")
    return study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_bundle(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
