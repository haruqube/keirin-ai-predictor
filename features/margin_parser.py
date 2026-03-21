"""着差文字列のパーサー"""


def _parse_fraction(s: str) -> float:
    """分数文字列を数値に変換 ('1/2' -> 0.5)"""
    if not s:
        return 0.0
    if '/' in s:
        parts = s.split('/')
        try:
            return float(parts[0]) / float(parts[1])
        except (ValueError, ZeroDivisionError):
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def parse_margin_to_numeric(margin_str) -> float | None:
    """着差文字列を車身数(float)に変換"""
    if not margin_str or not isinstance(margin_str, str):
        return None
    margin = margin_str.split('(')[0].strip()
    if not margin:
        return None
    if 'タイヤ' in margin:
        return 0.05
    if '大差' in margin:
        return 10.0
    if '車輪' in margin:
        num_part = margin.replace('車輪', '').strip()
        return _parse_fraction(num_part) * 0.5
    if '車身' in margin:
        parts = margin.split('車身')
        main = _parse_fraction(parts[0].strip()) if parts[0].strip() else 0
        frac = _parse_fraction(parts[1].strip()) if len(parts) > 1 and parts[1].strip() else 0
        return main + frac
    return None
