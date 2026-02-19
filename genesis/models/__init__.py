"""Model handling for Genesis."""

__all__ = [
    "TeacherModel",
    "StudentModel",
    "LoRAManager",
    "LoRAConfig",
]


def __getattr__(name):
    if name == "TeacherModel":
        from genesis.models.teacher import TeacherModel
        return TeacherModel
    if name == "StudentModel":
        from genesis.models.student import StudentModel
        return StudentModel
    if name == "LoRAManager":
        from genesis.models.lora_manager import LoRAManager
        return LoRAManager
    if name == "LoRAConfig":
        from genesis.models.lora_manager import LoRAConfig
        return LoRAConfig
    raise AttributeError(f"module 'genesis.models' has no attribute {name!r}")
