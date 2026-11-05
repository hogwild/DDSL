from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class LabelSchemeSpec:
    name: str
    axis1_col: str
    axis2_col: str
    joint_col: str
    axis1_to_id: Dict[str, int]
    axis2_to_id: Dict[str, int]
    joint_to_id: Dict[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "id_to_axis1", {v: k for k, v in self.axis1_to_id.items()})
        object.__setattr__(self, "id_to_axis2", {v: k for k, v in self.axis2_to_id.items()})
        object.__setattr__(self, "id_to_joint", {v: k for k, v in self.joint_to_id.items()})
        object.__setattr__(self, "num_axis1_classes", len(self.axis1_to_id))
        object.__setattr__(self, "num_axis2_classes", len(self.axis2_to_id))
        object.__setattr__(self, "num_joint_classes", len(self.joint_to_id))

        pair_to_joint = {}
        joint_to_pair = {}
        for axis1_name, axis1_id in self.axis1_to_id.items():
            for axis2_name, axis2_id in self.axis2_to_id.items():
                joint_name = f"{axis1_name}_TO_{axis2_name}"
                if joint_name not in self.joint_to_id:
                    raise ValueError(
                        f"Joint label {joint_name!r} not found in joint_to_id "
                        f"for scheme {self.name!r}."
                    )
                joint_id = self.joint_to_id[joint_name]
                pair_to_joint[(axis1_id, axis2_id)] = joint_id
                joint_to_pair[joint_id] = (axis1_id, axis2_id)

        object.__setattr__(self, "_pair_to_joint", pair_to_joint)
        object.__setattr__(self, "_joint_to_pair", joint_to_pair)

    def combine_axes_to_joint_id(self, axis1_id: int, axis2_id: int) -> int:
        return self._pair_to_joint[(int(axis1_id), int(axis2_id))]

    def split_joint_id(self, joint_id: int) -> Tuple[int, int]:
        return self._joint_to_pair[int(joint_id)]
