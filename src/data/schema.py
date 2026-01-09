"""
数据配置 + 单条样本结构（精简版），聚焦 Chebyshev Type I 的 LC 滤波器数据生成。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

FilterType = Literal["lowpass", "bandpass", "highpass", "bandstop"]  # 预留多类型
PrototypeType = Literal["cheby1", "butter"]  # 支持 Chebyshev I 与 Butterworth
VariantType = Literal[
    "ideal",               # 原型连续参数缩放得到的理想电路
    "quantized",           # 只做 E12/E24 离散化
    "perturbed",           # 在理想参数附近加扰动
    "quantized_perturbed", # 离散化后再扰动
]


# --------- 全局数据生成配置 ---------


@dataclass
class FilterDatasetConfig:
    """
    生成数据集时使用的全局配置（给 gen_dataset.py 用）。
    数值可由 YAML 读入后构造本 dataclass。
    """

    num_specs: int = 10_000                     # 要生成多少条设计意图
    max_variants_per_spec: int = 3              # 每个设计意图派生的电路变体上限
    filter_types: Tuple[FilterType, ...] = ("lowpass",)
    prototype_types: Tuple[PrototypeType, ...] = ("cheby1",)
    order_min: int = 2
    order_max: int = 7
    ripple_db_min: float = 0.1
    ripple_db_max: float = 1.0
    fc_min_hz: float = 1e6
    fc_max_hz: float = 1e9
    z0: float = 50.0
    num_freqs: int = 256
    random_seed: int = 42
    split_train: float = 0.8
    split_val: float = 0.1
    split_test: float = 0.1
    output_dir: str = "data/processed/filter_v1"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------- 元件定义 ---------


@dataclass
class ComponentSpec:
    """
    表示电路中的单个元件，既可存连续值也可存离散值。
    """

    ctype: Literal["L", "C"]
    role: Literal["series", "shunt"]
    value_si: float
    std_label: Optional[str]
    node1: str
    node2: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------- 单条样本结构（Python 内部使用） ---------


@dataclass
class FilterSample:
    """
    单条样本的“统一视图”，方便在 pipeline 中传递与调试。
    """

    # 必填标量
    sample_id: str
    filter_type: FilterType
    prototype_type: PrototypeType
    order: int
    ripple_db: float
    fc_hz: float
    variant: VariantType
    z0: float
    num_L: int
    num_C: int
    scenario: Optional[str] = None
    scenario_id: Optional[int] = None
    bw_frac: Optional[float] = None
    return_loss_min_db: Optional[float] = None
    notch_freq_hz: Optional[float] = None
    notch_depth_db: Optional[float] = None
    notch_bw_frac: Optional[float] = None
    asymmetry_factor: Optional[float] = None

    # 可选信息
    ideal_components: Optional[List[ComponentSpec]] = None
    discrete_components: Optional[List[ComponentSpec]] = None
    json_components: Optional[str] = None
    freqs_hz: Any = None
     # 波形
    w_ideal_S21_db: Any = None
    w_real_S21_db: Any = None
    w_ideal_S11_db: Any = None
    w_real_S11_db: Any = None
    passband_min_db: Optional[float] = None
    stopband_max_db: Optional[float] = None
    mask_min_db: Any = None
    mask_max_db: Any = None
    # 输出
    vact_tokens: Optional[List[str]] = None  # component-centric VACT-Seq tokens
    vact_struct_tokens: Optional[List[str]] = None  # structured VACT-Struct tokens
    dsl_tokens: Optional[List[str]] = None  # structured Macro/Repeat DSL tokens
    dsl_slot_values: Optional[List[float]] = None  # numeric slots aligned with dsl_tokens
    macro_ir_macros: Optional[List[str]] = None  # expanded Macro-IR (one macro per cell)
    sfci_tokens: Optional[List[str]] = None  # net-centric SFCI tokens
    action_tokens: Optional[List[str]] = None  # action-oriented construction tokens
    spec_id: Optional[int] = None
    circuit_id: Optional[int] = None

    def to_metadata_dict(self) -> Dict[str, Any]:
        """
        只保留标量 & 字符串，方便写入 csv / json 之类的元信息文件。
        波形数组通常单独存储。
        """
        return {
            "spec_id": self.spec_id,
            "circuit_id": self.circuit_id,
            "sample_id": self.sample_id,
            "filter_type": self.filter_type,
            "prototype_type": self.prototype_type,
            "order": self.order,
            "ripple_db": self.ripple_db,
            "fc_hz": self.fc_hz,
            "variant": self.variant,
            "z0": self.z0,
            "num_L": self.num_L,
            "num_C": self.num_C,
            "scenario": self.scenario,
            "scenario_id": self.scenario_id,
            "bw_frac": self.bw_frac,
            "return_loss_min_db": self.return_loss_min_db,
            "notch_freq_hz": self.notch_freq_hz,
            "notch_depth_db": self.notch_depth_db,
            "notch_bw_frac": self.notch_bw_frac,
            "asymmetry_factor": self.asymmetry_factor,
            "passband_min_db": self.passband_min_db,
            "stopband_max_db": self.stopband_max_db,
            "json_components": self.json_components,
            "vact_tokens": self.vact_tokens or [],
            "vact_struct_tokens": self.vact_struct_tokens or [],
            "dsl_tokens": self.dsl_tokens or [],
            "dsl_slot_values": self.dsl_slot_values or [],
            "macro_ir_macros": self.macro_ir_macros or [],
            "sfci_tokens": self.sfci_tokens or [],
            "action_tokens": self.action_tokens or [],
        }
