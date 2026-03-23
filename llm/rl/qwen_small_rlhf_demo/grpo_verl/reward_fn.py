import re


def _normalize_text(value):
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    del data_source, extra_info, kwargs
    solution_text = _normalize_text(solution_str)
    target_text = _normalize_text(ground_truth)
    match = re.search(r"-?\d+", solution_text)
    pred = match.group(0) if match else ""
    return 1.0 if pred == target_text else 0.0
