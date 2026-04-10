"""
Локальный рецепт SDL2_ttf — belt-and-suspenders поверх clang-17 wrapper.

РОЛЬ В ТЕКУЩЕЙ АРХИТЕКТУРЕ
---------------------------
Основной fix — clang-17 wrapper в main.yml (шаг "Patch NDK clang-17").
Он перехватывает ВСЕ компиляции NDK toolchain и добавляет нужные флаги.

Этот локальный рецепт добавляет флаги ЯВНО в Android.mk SDL2_ttf через
LOCAL_CFLAGS / LOCAL_CPPFLAGS. Это второй независимый слой защиты:
даже если wrapper по каким-то причинам не сработал (старый кэш NDK без
wrapper, нестандартное окружение), Android.mk с нашими флагами обеспечит
корректную компиляцию бандлированного HarfBuzz и FreeType.

ПОЧЕМУ sdl2_ttf, А НЕ ОТДЕЛЬНЫЙ harfbuzz
------------------------------------------
SDL2_ttf 2.22.0 вендорит HarfBuzz внутрь своего архива и строит его
через ndk-build (Android.mk). Внешний p4a-рецепт harfbuzz к этой сборке
отношения не имеет — SDL2_ttf компилирует свой собственный HarfBuzz.

ПРОБЛЕМА CLANG 17 (NDK r26b)
------------------------------
HarfBuzz и FreeType приводят типы функций:
    void (*)(FT_FaceRec_ *)  →  void (*)(void *)
Clang 17 (-Wcast-function-type-strict) считает это фатальной ошибкой.
"""

import os
import re

from pythonforandroid.recipe import BootstrapNDKRecipe
from pythonforandroid.logger import info, warning


_CLANG17_FLAGS = (
    "-Wno-cast-function-type-strict "
    "-Wno-cast-function-type "
    "-Wno-error=cast-function-type-strict"
)
_MARKER = "# [p4a-clang17-fix]"


class LibSDL2TTF(BootstrapNDKRecipe):
    version = "2.22.0"
    url = (
        "https://github.com/libsdl-org/SDL_ttf/releases/download/"
        "release-{version}/SDL2_ttf-{version}.tar.gz"
    )
    dir_name = "SDL2_ttf"

    def prepare_build_dir(self, arch):
        super().prepare_build_dir(arch)
        self._patch_makefiles(arch)

    def _patch_makefiles(self, arch):
        # prepare_build_dir получает arch уже как строку (arch.arch),
        # поэтому get_build_dir вызываем напрямую с arch, без .arch
        build_dir = self.get_build_dir(arch)
        patched_count = 0

        for dirpath, _dirs, files in os.walk(build_dir):
            if "Android.mk" not in files:
                continue
            mk = os.path.join(dirpath, "Android.mk")
            patched_count += self._patch_one_mk(mk, build_dir)

        if patched_count == 0:
            warning(
                "SDL2_ttf Clang17-fix: Android.mk не найден в '{}'. "
                "clang-17 wrapper должен покрывать это самостоятельно.".format(build_dir)
            )
        else:
            info("SDL2_ttf Clang17-fix: пропатчено {} Android.mk файл(ов).".format(patched_count))

    def _patch_one_mk(self, mk_path, build_dir):
        with open(mk_path, "r", encoding="utf-8") as fh:
            src = fh.read()

        if _MARKER in src:
            return 0  # already patched

        # Патчим только корневой Android.mk или те, что содержат harfbuzz/freetype
        is_root = (os.path.dirname(mk_path) == build_dir)
        is_relevant = any(k in src.lower() for k in ("harfbuzz", "freetype", "sdl2_ttf", "sdl_ttf"))
        if not is_root and not is_relevant:
            return 0

        patch = (
            "\n{marker}\n"
            "LOCAL_CFLAGS   += {flags}\n"
            "LOCAL_CPPFLAGS += {flags}\n"
        ).format(marker=_MARKER, flags=_CLANG17_FLAGS)

        patched, n = re.subn(
            r"(LOCAL_MODULE\s*:=\s*[^\n]+\n)",
            r"\1" + patch,
            src,
            count=1,
        )

        if n == 0:
            # Fallback: нет LOCAL_MODULE — вставляем в начало
            patched = patch.lstrip("\n") + "\n" + src
            warning("SDL2_ttf Clang17-fix: LOCAL_MODULE не найден в {}, fallback.".format(
                os.path.relpath(mk_path, build_dir)))

        with open(mk_path, "w", encoding="utf-8") as fh:
            fh.write(patched)

        info("SDL2_ttf Clang17-fix: пропатчен {}".format(
            os.path.relpath(mk_path, build_dir)))
        return 1


recipe = LibSDL2TTF()
