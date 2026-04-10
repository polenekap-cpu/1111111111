[app]
title = AI Playlist Generator
package.name = aiplaylist
package.domain = org.playlistai

source.dir = .
source.include_exts = py,json,png,kv,ttf,tsv,txt

version = 1.3.0

requirements = python3,kivy,mutagen,requests,certifi,charset-normalizer,idna,urllib3,android,openssl

# Android settings
android.permissions = INTERNET,READ_MEDIA_AUDIO
android.api = 35
android.minapi = 21
android.ndk_api = 21

# NDK r26b: последняя стабильная версия, поддерживающая 16 KB page size.
# CI загружает NDK r26b явно (шаг "Download NDK r26b" в main.yml) и
# устанавливает clang-17 wrapper ДО запуска buildozer. Когда buildozer
# видит, что NDK уже лежит в ~/.buildozer/android/platform/android-ndk-r26b,
# он не скачивает его заново.
android.ndk = 26b

# Только arm64-v8a: >95% актуальных устройств.
android.archs = arm64-v8a

android.allow_backup = True
android.accept_sdk_license = True

# SDL2 bootstrap (стандарт для Kivy).
p4a.bootstrap = sdl2

# Локальные рецепты — дополнительный слой защиты (belt-and-suspenders).
#
# Основной fix — clang-17 wrapper в main.yml, который перехватывает
# ВСЕ компиляции через NDK toolchain (Kivy, HarfBuzz, SDL2_ttf, ...).
#
# sdl2_ttf/__init__.py:
#   Патчит Android.mk SDL2_ttf — явно добавляет LOCAL_CFLAGS для
#   нативной ndk-build компиляции бандлированного HarfBuzz.
#
# harfbuzz/__init__.py:
#   Переопределяет get_recipe_env() для standalone HarfBuzz-рецепта
#   (страховка на случай, если harfbuzz подтянется отдельно).
#
# sdl2/__init__.py:
#   Обеспечивает правильную C++ линковку для hidapi компонентов,
#   которые вызывают неопределённые символы операторов new/delete при
#   использовании clang-17 с NDK r26b.
#
# Без этих рецептов clang-17 wrapper уже решает проблему. С рецептами —
# двойная гарантия: флаги попадают и через wrapper, и через env/mk.
p4a.local_recipes = ./p4a-recipes

# Ориентация и отображение
orientation = portrait
fullscreen = 0

log_level = 2

source.exclude_dirs = __pycache__,logs,.git,.github,bin,.buildozer,p4a-recipes
source.exclude_patterns = *.pyc,catalog.tsv,catalog_for_ai.txt,build_log.txt,test_*.py

[buildozer]
warn_on_root = 1
