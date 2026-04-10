"""
Локальный рецепт HarfBuzz — belt-and-suspenders поверх clang-17 wrapper.

РОЛЬ В ТЕКУЩЕЙ АРХИТЕКТУРЕ
---------------------------
Основной fix — clang-17 wrapper в main.yml.
Этот рецепт переопределяет get_recipe_env() для standalone HarfBuzz
(используется если harfbuzz подтянется как отдельная зависимость, а не
как часть SDL2_ttf). Второй независимый слой защиты.

ПОЧЕМУ АТРИБУТ cflags = [...] НЕ РАБОТАЛ
-----------------------------------------
HarfbuzzRecipe в p4a наследует Recipe напрямую (не CMakeRecipe /
AutotoolsRecipe). Атрибут класса `cflags` в базовом Recipe нигде не
читается — он является соглашением только для CMake/Autotools рецептов.
HarfbuzzRecipe строит окружение через get_recipe_env() вручную.
Правильный способ добавить флаги — переопределить get_recipe_env().
"""

from pythonforandroid.recipes.harfbuzz import HarfbuzzRecipe as _Upstream
from pythonforandroid.logger import info

_FLAGS = [
    "-Wno-cast-function-type-strict",
    "-Wno-cast-function-type",
    "-Wno-error=cast-function-type-strict",
]


class HarfbuzzRecipe(_Upstream):
    """HarfBuzz с патчем для Clang 17 / NDK r26b."""

    def get_recipe_env(self, arch=None):
        env = super().get_recipe_env(arch)
        extra = " " + " ".join(_FLAGS)
        env["CFLAGS"] = env.get("CFLAGS", "") + extra
        env["CXXFLAGS"] = env.get("CXXFLAGS", "") + extra
        info("HarfBuzz: добавлены Clang17-флаги в CFLAGS/CXXFLAGS")
        return env


recipe = HarfbuzzRecipe()
