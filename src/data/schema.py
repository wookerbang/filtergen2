"""
数据配置 + 单条样本结构（精简版），聚焦 Chebyshev Type I 的 LC 滤波器数据生成。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

FilterType = Literal["lowpass", "bandpass"]  # 当前主要用 "lowpass"
PrototypeType = Literal["cheby1"]            # 仅保留 Chebyshev Type I
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
    # 输出
    sfci_sequence: str = ""
    sfci_tokens: Optional[List[str]] = None
    sfci_tokens_net: Optional[List[str]] = None
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
            "json_components": self.json_components,
            "sfci_sequence": self.sfci_sequence,
        }
