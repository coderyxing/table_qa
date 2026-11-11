# 模块职责：严格解析完整单元格中的数值和单位，处理可追溯的数量转换及工具提示。
# 接口约定：样本、操作和结果采用本项目的结构化字典；细节见各入口及字段检查。
"""Explicit quantity conversions; never infer a unit from a question or label.

Only full-cell, whitelisted quantities are accepted. This is an experimental
executor capability, not a semantic parser or evidence of benchmark improvement.
"""
from dataclasses import dataclass
from decimal import Decimal, DecimalException, Inexact, localcontext
import re


# QuantityError：数量解析或单位转换违反约束时抛出的执行错误。
class QuantityError(ValueError):
    pass


# unit -> dimension, base unit, scale. Currency names must be explicit: '$' is
# deliberately unsupported because it does not uniquely identify a currency.
_UNITS = {
    '1': ('ratio', '1', '1'), 'ratio': ('ratio', '1', '1'),
    '%': ('ratio', '1', '.01'), 'percent': ('ratio', '1', '.01'),
    'm': ('length', 'm', '1'), 'km': ('length', 'm', '1000'),
    'cm': ('length', 'm', '.01'), 'mm': ('length', 'm', '.001'),
    'kg': ('mass', 'kg', '1'), 'g': ('mass', 'kg', '.001'),
    'mg': ('mass', 'kg', '.000001'),
    's': ('duration', 's', '1'), 'ms': ('duration', 's', '.001'),
    'min': ('duration', 's', '60'), 'h': ('duration', 's', '3600'),
    'Hz': ('frequency', 'Hz', '1'), 'kHz': ('frequency', 'Hz', '1000'),
    'MHz': ('frequency', 'Hz', '1000000'), 'GHz': ('frequency', 'Hz', '1000000000'),
    'usd': ('currency:USD', 'USD', '1'),
    'eur': ('currency:EUR', 'EUR', '1'),
    'cny': ('currency:CNY', 'CNY', '1'),
}
_FREQUENCY_ALIASES = {'hz': 'Hz', 'khz': 'kHz', 'mhz': 'MHz', 'ghz': 'GHz'}
_SCALES = {'': '1', 'thousand': '1000', 'million': '1000000',
           'billion': '1000000000'}
_NUMBER = r'[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?'
_CELL = re.compile(rf'(?P<number>{_NUMBER})\s*(?P<scale>thousand|million|billion)?\s*(?P<unit>[a-zA-Z]+|%)?', re.IGNORECASE)


# 查找或规范化受支持单位，保留严格的数量类型约束。
# 输入：name。
def _unit(name):
    if not isinstance(name, str):
        raise QuantityError(f'Unsupported explicit unit: {name!r}')
    # SI symbol case matters: M can mean mega, and G is not the gram symbol g.
    key = name.casefold() if name.casefold() in {'usd', 'eur', 'cny', 'ratio', 'percent'} else name
    key = _FREQUENCY_ALIASES.get(key, key)
    if key not in _UNITS:
        raise QuantityError(f'Unsupported explicit unit: {name!r}')
    return _UNITS[key]


# Quantity：保存十进制数值、单位及转换所需的数量类型。
@dataclass(frozen=True)
class Quantity:
    # raw 保留来源值；value 统一到规范基准单位，换算前必须核对 dimension。
    raw: object
    value: Decimal  # Value in the canonical base unit.
    dimension: str
    base_unit: str
    source_unit: str
    # 显式记录是否借用了外部声明单位，以便审核纯数值单元格的解析依据。
    assumed_unit_used: bool

    # 检查数量类型与目标单位兼容后，返回换算后的数量对象。
    # 输入：target_unit。
    # 异常约束：Conversion requires unsupported inexact decimal rounding
    def in_unit(self, target_unit):
        dimension, _, factor = _unit(target_unit)
        if dimension != self.dimension:
            raise QuantityError(f'Incompatible units: {self.dimension} and {dimension}')
        try:
            with localcontext() as ctx:
                ctx.prec = 512
                ctx.traps[Inexact] = True
                return self.value / Decimal(factor)
        except DecimalException as exc:
            raise QuantityError('Conversion requires unsupported inexact decimal rounding') from exc


# 解析完整单元格及显式来源单位，拒绝范围、脚注和混杂文本。
# 输入：value；source_unit。
# 异常约束：Quantity requires a finite number or full-cell string
def parse_quantity(value, source_unit=None):
    """Parse one complete cell; source_unit applies ONLY to a unitless cell.

    Callers may provide a header's explicit unit, recording it in the operation.
    Dates, ranges, footnotes, unknown markers, currency symbols, and mixed text
    fail rather than being partially parsed or replaced by zero.
    """
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise QuantityError('Quantity requires a finite number or full-cell string')
    if source_unit is not None:
        _unit(source_unit)
    text = str(value).strip()
    if len(text) > 256:
        raise QuantityError('Quantity text exceeds the supported bound')
    match = _CELL.fullmatch(text)
    if match is None:
        raise QuantityError(f'Ambiguous or unsupported quantity: {value!r}')
    token = match['number'].replace(',', '')
    try:
        n = Decimal(token)
        # Bound exponent and result to avoid pathological decimal arithmetic.
        if not n.is_finite() or abs(n.adjusted()) > 100:
            raise QuantityError('Quantity magnitude exceeds the supported bound')
        unit = match['unit'] or source_unit or '1'
        dimension, base, factor = _unit(unit)
        with localcontext() as ctx:
            ctx.prec = 512
            ctx.traps[Inexact] = True
            result = n * Decimal(_SCALES[(match['scale'] or '').casefold()]) * Decimal(factor)
    except DecimalException as exc:
        raise QuantityError('Invalid decimal quantity') from exc
    if not result.is_finite() or abs(result.adjusted()) > 100:
        raise QuantityError('Converted quantity exceeds the supported bound')
    return Quantity(value, result, dimension, base, unit,
                    match['unit'] is None and source_unit is not None)


# 把数量转换成原执行器可读的无损十进制文本。
# 输入：value；target_unit；source_unit。
def quantity_text(value, target_unit, source_unit=None):
    """Return a lossless decimal string usable by the existing numeric executor."""
    return format(parse_quantity(value, source_unit).in_unit(target_unit), 'f')


# 检查操作是否使用数量工具，以验证提示和能力开关匹配。
# 输入：value。
def uses_quantities(value):
    if isinstance(value, dict):
        return value.get('op') == 'quantity' or any(uses_quantities(v) for v in value.values())
    return isinstance(value, list) and any(uses_quantities(v) for v in value)


QUANTITY_PROMPT = '''
Optional explicit quantity conversion (quantity-v2):
derive.expression may use {"op":"quantity","args":[{"col":"c1"}],"unit":"km"}.
It returns a numeric string in the requested unit, retaining the original column.
Units: m/km/cm/mm, kg/g/mg, s/ms/min/h, Hz/kHz/MHz/GHz, USD/EUR/CNY, ratio/percent/%.
Lowercase hz/khz/mhz/ghz spellings are accepted.
thousand/million/billion scaling is supported. Dates, ranges, footnotes and '$' are not.
For bare numbers whose header explicitly states a unit, include source_unit, e.g.
{"op":"quantity","args":[{"col":"c1"}],"source_unit":"m","unit":"km"}.
source_unit applies only to bare numbers. Do not guess units or silently remove rows.
Quantity conversion verifies numeric compatibility, not whether the question was understood.
Conversions needing nonterminating decimal rounding are rejected.
'''
