#!/usr/bin/env python3
"""
AI Playlist Generator — Kivy Android app.
"""

import json
import os
import threading
import traceback

from kivy.app import App
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import (
    BooleanProperty, NumericProperty, StringProperty, ObjectProperty,
)
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.metrics import dp
from kivy.utils import platform

try:
    import catalog_builder
except ImportError:
    catalog_builder = None

try:
    import query_engine
except ImportError:
    query_engine = None

# ---------------------------------------------------------------------------
# Constants & Themes
# ---------------------------------------------------------------------------

APP_VERSION = "1.3.0"

THEMES = {
    "midnight": {
        "bg": (0.06, 0.07, 0.11, 1),
        "surface": (0.12, 0.14, 0.22, 1),
        "card": (0.08, 0.09, 0.15, 1),
        "text": (0.90, 0.90, 0.95, 1),
        "text_secondary": (0.55, 0.55, 0.62, 1),
        "accent": (0.18, 0.55, 0.82, 1),
        "active_tab": (0.35, 0.75, 1.0, 1),
        "inactive_tab": (0.55, 0.55, 0.60, 1),
        "error": (0.90, 0.30, 0.30, 1),
        "warning": (0.90, 0.70, 0.30, 1),
        "success": (0.55, 0.80, 0.55, 1),
        "hint_text": (0.45, 0.45, 0.50, 1),
        "cursor": (0.35, 0.75, 1.0, 1),
        "input_bg": (0.10, 0.12, 0.18, 1),
        "button_bg": (0.18, 0.55, 0.82, 1),
        "button_disabled": (0.25, 0.25, 0.30, 1),
    },
}


class _ThemeData:
    def __init__(self, data):
        for key, value in data.items():
            setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _app_data_dir():
    """Return writable app-private directory, works on Android & desktop."""
    if platform == "android":
        try:
            from android.storage import app_storage_path
            return app_storage_path()
        except ImportError:
            return "/data/data/org.playlistai.aiplaylist/files"
    return os.path.dirname(os.path.abspath(__file__))


def _config_path():
    return os.path.join(_app_data_dir(), "config.json")


# ---------------------------------------------------------------------------
# API providers
# ---------------------------------------------------------------------------

API_PROVIDERS = {
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "default_model": "google/gemini-2.0-flash-lite-preview-02-05:free",
    },
    "google": {
        "url_template": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
        "default_model": "gemini-2.0-flash",
    },
}


def _call_ai(api_provider, api_key, model, system_prompt, user_message):
    """Unified AI call supporting OpenRouter and Google Gemini."""
    import requests

    if api_provider == "google":
        url = API_PROVIDERS["google"]["url_template"].format(
            model=model, key=api_key,
        )
        payload = {
            "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_message}"}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2048},
        }
        resp = requests.post(url, json=payload, timeout=120)
        data = resp.json()
        if resp.status_code != 200:
            err = data.get("error", {}).get("message", resp.text[:300])
            raise ConnectionError(f"Google API error ({resp.status_code}): {err}")
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise ValueError(f"Unexpected Google response: {str(data)[:300]}")

    else:  # openrouter
        url = API_PROVIDERS["openrouter"]["url"]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "X-Title": "AI Playlist Generator",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        data = resp.json()
        if resp.status_code != 200:
            err = data.get("error", {}).get("message", resp.text[:300])
            raise ConnectionError(f"OpenRouter error ({resp.status_code}): {err}")
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise ValueError(f"Unexpected OpenRouter response: {str(data)[:300]}")


# ---------------------------------------------------------------------------
# KV Language (UI)
# ---------------------------------------------------------------------------

KV = '''
#:import dp kivy.metrics.dp

<RoundedButton@Button>:
    background_normal: ''
    background_color: app.theme.get('button_bg', (0.18, 0.55, 0.82, 1))
    color: app.theme.get('text', (1,1,1,1))
    font_size: dp(15)
    size_hint_y: None
    height: dp(48)
    disabled_color: (0.5, 0.5, 0.5, 1)

<StyledInput@TextInput>:
    background_color: app.theme.get('input_bg', (0.1, 0.12, 0.18, 1))
    foreground_color: app.theme.get('text', (1,1,1,1))
    cursor_color: app.theme.get('cursor', (0.35, 0.75, 1, 1))
    hint_text_color: app.theme.get('hint_text', (0.45, 0.45, 0.5, 1))
    font_size: dp(14)
    padding: [dp(12), dp(10)]
    multiline: False

<TabButton@ToggleButton>:
    group: 'tabs'
    background_normal: ''
    background_down: ''
    background_color: (0,0,0,0)
    font_size: dp(13)
    size_hint: 1, None
    height: dp(44)
    color: app.theme.get('active_tab', (0.35,0.75,1,1)) if self.state == 'down' else app.theme.get('inactive_tab', (0.55,0.55,0.6,1))

<MainScreen>:
    name: 'main'
    BoxLayout:
        orientation: 'vertical'
        padding: dp(16)
        spacing: dp(12)
        canvas.before:
            Color:
                rgba: app.theme.get('bg', (0.06, 0.07, 0.11, 1))
            Rectangle:
                pos: self.pos
                size: self.size

        Label:
            text: 'AI Playlist Generator'
            font_size: dp(20)
            color: app.theme.get('text', (1,1,1,1))
            size_hint_y: None
            height: dp(36)
            bold: True

        Label:
            text: 'Запрос:'
            font_size: dp(13)
            color: app.theme.get('text_secondary', (0.55,0.55,0.62,1))
            size_hint_y: None
            height: dp(20)
            halign: 'left'
            text_size: self.size

        StyledInput:
            id: query_input
            hint_text: 'Например: энергичный рок 80-х'
            text: app.query_text
            on_text: app.query_text = self.text
            size_hint_y: None
            height: dp(44)

        BoxLayout:
            size_hint_y: None
            height: dp(40)
            spacing: dp(8)
            Label:
                text: 'Треков: ' + str(int(slider_size.value))
                font_size: dp(13)
                color: app.theme.get('text_secondary', (0.55,0.55,0.62,1))
                size_hint_x: 0.35
            Slider:
                id: slider_size
                min: 5
                max: 100
                step: 5
                value: app.playlist_size
                on_value: app.playlist_size = int(self.value)
                size_hint_x: 0.65

        RoundedButton:
            text: 'Создать плейлист'
            on_release: app.generate_playlist()
            disabled: app.is_busy

        Label:
            text: app.status_text
            font_size: dp(13)
            color: app.theme.get('accent', (0.18,0.55,0.82,1))
            size_hint_y: None
            height: dp(24)

        ScrollView:
            size_hint_y: 1
            Label:
                text: app.results_text
                font_size: dp(12)
                color: app.theme.get('text', (0.9,0.9,0.95,1))
                text_size: self.width, None
                size_hint_y: None
                height: self.texture_size[1]
                halign: 'left'
                valign: 'top'
                markup: True

<CatalogScreen>:
    name: 'catalog'
    BoxLayout:
        orientation: 'vertical'
        padding: dp(16)
        spacing: dp(12)
        canvas.before:
            Color:
                rgba: app.theme.get('bg', (0.06, 0.07, 0.11, 1))
            Rectangle:
                pos: self.pos
                size: self.size

        Label:
            text: 'Каталог'
            font_size: dp(20)
            color: app.theme.get('text', (1,1,1,1))
            size_hint_y: None
            height: dp(36)
            bold: True

        Label:
            text: app.catalog_stats
            font_size: dp(14)
            color: app.theme.get('text_secondary', (0.55,0.55,0.62,1))
            size_hint_y: None
            height: dp(24)
            markup: True

        ProgressBar:
            value: app.progress_value
            max: 100
            size_hint_y: None
            height: dp(6)

        Label:
            text: app.scan_status
            font_size: dp(12)
            color: app.theme.get('text_secondary', (0.55,0.55,0.62,1))
            size_hint_y: None
            height: dp(20)

        RoundedButton:
            text: 'Сканировать библиотеку'
            on_release: app.scan_catalog()
            disabled: app.is_busy

        RoundedButton:
            text: 'Очистить каталог'
            on_release: app.clear_catalog()
            background_color: app.theme.get('error', (0.9,0.3,0.3,1))

        Widget:
            size_hint_y: 1

<SettingsScreen>:
    name: 'settings'
    ScrollView:
        BoxLayout:
            orientation: 'vertical'
            padding: dp(16)
            spacing: dp(12)
            size_hint_y: None
            height: self.minimum_height
            canvas.before:
                Color:
                    rgba: app.theme.get('bg', (0.06, 0.07, 0.11, 1))
                Rectangle:
                    pos: self.pos
                    size: self.size

            Label:
                text: 'Настройки'
                font_size: dp(20)
                color: app.theme.get('text', (1,1,1,1))
                size_hint_y: None
                height: dp(36)
                bold: True

            # --- API Provider ---
            Label:
                text: 'API провайдер'
                font_size: dp(13)
                color: app.theme.get('text_secondary', (0.55,0.55,0.62,1))
                size_hint_y: None
                height: dp(20)
                halign: 'left'
                text_size: self.size

            BoxLayout:
                size_hint_y: None
                height: dp(40)
                spacing: dp(8)
                ToggleButton:
                    text: 'OpenRouter'
                    group: 'api_provider'
                    state: 'down' if app.api_provider == 'openrouter' else 'normal'
                    on_release: app.set_api_provider('openrouter')
                    background_normal: ''
                    background_down: ''
                    background_color: app.theme.get('accent', (0.18,0.55,0.82,1)) if self.state == 'down' else app.theme.get('surface', (0.12,0.14,0.22,1))
                    color: app.theme.get('text', (1,1,1,1))
                    font_size: dp(13)
                ToggleButton:
                    text: 'Google Gemini'
                    group: 'api_provider'
                    state: 'down' if app.api_provider == 'google' else 'normal'
                    on_release: app.set_api_provider('google')
                    background_normal: ''
                    background_down: ''
                    background_color: app.theme.get('accent', (0.18,0.55,0.82,1)) if self.state == 'down' else app.theme.get('surface', (0.12,0.14,0.22,1))
                    color: app.theme.get('text', (1,1,1,1))
                    font_size: dp(13)

            # --- API Key ---
            Label:
                text: 'API ключ'
                font_size: dp(13)
                color: app.theme.get('text_secondary', (0.55,0.55,0.62,1))
                size_hint_y: None
                height: dp(20)
                halign: 'left'
                text_size: self.size

            StyledInput:
                id: api_key_input
                hint_text: 'Вставьте ключ API'
                text: app.api_key
                on_text: app.api_key = self.text
                password: not app.show_api_key
                size_hint_y: None
                height: dp(44)

            BoxLayout:
                size_hint_y: None
                height: dp(30)
                CheckBox:
                    active: app.show_api_key
                    on_active: app.show_api_key = self.active
                    size_hint_x: None
                    width: dp(30)
                    color: app.theme.get('accent', (0.18,0.55,0.82,1))
                Label:
                    text: 'Показать ключ'
                    font_size: dp(12)
                    color: app.theme.get('text_secondary', (0.55,0.55,0.62,1))
                    halign: 'left'
                    text_size: self.size

            # --- Model ---
            Label:
                text: 'Модель ИИ'
                font_size: dp(13)
                color: app.theme.get('text_secondary', (0.55,0.55,0.62,1))
                size_hint_y: None
                height: dp(20)
                halign: 'left'
                text_size: self.size

            StyledInput:
                hint_text: 'Название модели'
                text: app.ai_model
                on_text: app.ai_model = self.text
                size_hint_y: None
                height: dp(44)

            # --- Music dirs ---
            Label:
                text: 'Папки с музыкой (по одной на строку)'
                font_size: dp(13)
                color: app.theme.get('text_secondary', (0.55,0.55,0.62,1))
                size_hint_y: None
                height: dp(20)
                halign: 'left'
                text_size: self.size

            StyledInput:
                hint_text: '/storage/emulated/0/Music'
                text: app.music_dirs_edit
                on_text: app.music_dirs_edit = self.text
                multiline: True
                size_hint_y: None
                height: dp(80)

            # --- Output dir ---
            Label:
                text: 'Папка для плейлистов'
                font_size: dp(13)
                color: app.theme.get('text_secondary', (0.55,0.55,0.62,1))
                size_hint_y: None
                height: dp(20)
                halign: 'left'
                text_size: self.size

            StyledInput:
                hint_text: '/storage/emulated/0/Music/Playlists'
                text: app.output_dir
                on_text: app.output_dir = self.text
                size_hint_y: None
                height: dp(44)

            RoundedButton:
                text: 'Сохранить настройки'
                on_release: app.save_settings()

            Label:
                text: app.settings_status
                font_size: dp(12)
                color: app.theme.get('success', (0.55,0.8,0.55,1))
                size_hint_y: None
                height: dp(20)

            Widget:
                size_hint_y: None
                height: dp(40)

<RootWidget>:
    orientation: 'vertical'
    canvas.before:
        Color:
            rgba: app.theme.get('bg', (0.06, 0.07, 0.11, 1))
        Rectangle:
            pos: self.pos
            size: self.size

    ScreenManager:
        id: sm
        MainScreen:
        CatalogScreen:
        SettingsScreen:

    # Bottom tabs
    BoxLayout:
        size_hint_y: None
        height: dp(50)
        canvas.before:
            Color:
                rgba: app.theme.get('surface', (0.12, 0.14, 0.22, 1))
            Rectangle:
                pos: self.pos
                size: self.size

        TabButton:
            text: 'Плейлист'
            state: 'down'
            on_release: sm.current = 'main'
        TabButton:
            text: 'Каталог'
            on_release: sm.current = 'catalog'
        TabButton:
            text: 'Настройки'
            on_release: sm.current = 'settings'
'''

# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

class MainScreen(Screen):
    pass

class CatalogScreen(Screen):
    pass

class SettingsScreen(Screen):
    pass

class RootWidget(BoxLayout):
    pass


# ---------------------------------------------------------------------------
# Main App Class
# ---------------------------------------------------------------------------

class PlaylistApp(App):
    status_text    = StringProperty("Готов")
    results_text   = StringProperty("Готов к работе.")
    scan_status    = StringProperty("Ожидание...")
    catalog_stats  = StringProperty("Каталог не загружен")
    progress_value = NumericProperty(0)
    is_busy        = BooleanProperty(False)
    playlist_size  = NumericProperty(30)
    settings_status = StringProperty("")

    api_key       = StringProperty("")
    api_provider  = StringProperty("openrouter")
    ai_model      = StringProperty("google/gemini-2.0-flash-lite-preview-02-05:free")
    music_dirs_edit = StringProperty("")
    output_dir    = StringProperty("")
    theme_name    = StringProperty("midnight")
    show_api_key  = BooleanProperty(False)
    query_text    = StringProperty("")

    theme = ObjectProperty(_ThemeData(THEMES["midnight"]), rebind=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def build(self):
        self.title = "AI Playlist Generator"
        self._setup_paths()
        self._load_settings()
        self.set_theme(self.theme_name)
        self._update_catalog_stats()
        Builder.load_string(KV)
        return RootWidget()

    def on_start(self):
        if platform == "android":
            self._request_android_permissions()

    def _request_android_permissions(self):
        """Android permission handling compatible with API 21–36.

        Android 13+ (API 33+): only READ_MEDIA_AUDIO — READ_EXTERNAL_STORAGE
        is deprecated and causes crashes starting from API 36.
        Android 12 and below: legacy READ/WRITE_EXTERNAL_STORAGE.
        """
        try:
            from android.permissions import request_permissions, Permission

            api_level = 33  # safe default if jnius is unavailable
            try:
                from jnius import autoclass
                api_level = autoclass("android.os.Build$VERSION").SDK_INT
            except Exception:
                pass

            if api_level >= 33:
                perms = [Permission.READ_MEDIA_AUDIO]
            else:
                perms = [Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE]

            request_permissions(perms)
        except Exception as e:
            print(f"Permission request failed: {e}")

    # ------------------------------------------------------------------
    # Paths & Config
    # ------------------------------------------------------------------

    def _setup_paths(self):
        if platform == "android":
            try:
                from android.storage import primary_external_storage_path
                base = primary_external_storage_path()
            except ImportError:
                base = "/storage/emulated/0"
            self.output_dir = os.path.join(base, "Music", "Playlists")
            if not self.music_dirs_edit:
                self.music_dirs_edit = os.path.join(base, "Music")
        else:
            self.output_dir = os.path.expanduser("~/Music/Playlists")
            if not self.music_dirs_edit:
                self.music_dirs_edit = os.path.expanduser("~/Music")

    def _load_settings(self):
        cfg_path = _config_path()
        if not os.path.exists(cfg_path):
            return
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.api_key      = cfg.get("api_key", "")
            self.api_provider = cfg.get("api_provider", "openrouter")
            self.ai_model     = cfg.get("ai_model", cfg.get("model", self.ai_model))
            self.playlist_size = cfg.get("playlist_size", 30)

            dirs = cfg.get("music_dirs", [])
            if platform == "android":
                dirs = [d for d in dirs if not (d.startswith("C:") or d.startswith("D:"))]
            if dirs:
                self.music_dirs_edit = "\n".join(dirs)
            self.output_dir = cfg.get("output_dir", self.output_dir)
            self.theme_name  = cfg.get("theme", "midnight")
        except Exception as e:
            print(f"Config load error: {e}")

    def _build_config_dict(self):
        """Build config dict from current app state. Thread-safe (read-only)."""
        data_dir = _app_data_dir()
        return {
            "api_key":          self.api_key,
            "api_provider":     self.api_provider,
            "ai_model":         self.ai_model,
            "music_dirs":       [d.strip() for d in self.music_dirs_edit.split("\n") if d.strip()],
            "output_dir":       self.output_dir,
            "catalog_path":     os.path.join(data_dir, "catalog.tsv"),
            "ai_catalog_path":  os.path.join(data_dir, "catalog_for_ai.txt"),
            "playlist_size":    self.playlist_size,
            "theme":            self.theme_name,
        }

    def _write_config(self):
        """Write config to disk. No UI side effects — safe to call from any thread."""
        data_dir = _app_data_dir()
        os.makedirs(data_dir, exist_ok=True)
        config = self._build_config_dict()
        with open(_config_path(), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

    def save_settings(self):
        """Save settings and update UI status. Must be called from the main thread."""
        try:
            self._write_config()
            self.settings_status = "Настройки сохранены!"
            Clock.schedule_once(lambda dt: setattr(self, "settings_status", ""), 3)
        except Exception as e:
            self.settings_status = f"Ошибка: {e}"

    # ------------------------------------------------------------------
    # API provider switching
    # ------------------------------------------------------------------

    def set_api_provider(self, provider):
        self.api_provider = provider
        defaults = {
            "openrouter": "google/gemini-2.0-flash-lite-preview-02-05:free",
            "google":     "gemini-2.0-flash",
        }
        self.ai_model = defaults.get(provider, self.ai_model)

    # ------------------------------------------------------------------
    # Catalog
    # ------------------------------------------------------------------

    def scan_catalog(self):
        if self.is_busy:
            return
        self.is_busy = True
        self.progress_value = 0
        self.scan_status = "Запуск сканирования..."
        threading.Thread(target=self._scan_thread, daemon=True).start()

    def _scan_thread(self):
        def update_progress(curr, total, msg):
            Clock.schedule_once(lambda dt: self._sync_progress(curr, total, msg))

        try:
            if catalog_builder is None:
                raise ImportError("Модуль catalog_builder не найден")
            # _write_config() is thread-safe (no Kivy property writes)
            self._write_config()
            res = catalog_builder.build_catalog(
                config_path=_config_path(),
                progress_cb=update_progress,
            )
            msg = f"Готово! Найдено треков: {res.get('total', 0)}"
        except Exception as e:
            msg = f"Ошибка: {e}"
            traceback.print_exc()

        Clock.schedule_once(lambda dt: self._end_busy(msg))
        Clock.schedule_once(lambda dt: self._update_catalog_stats())

    def _sync_progress(self, curr, total, msg):
        self.scan_status = msg
        if total > 0:
            self.progress_value = (curr / total) * 100

    def clear_catalog(self):
        data_dir = _app_data_dir()
        for fname in ("catalog.tsv", "catalog_for_ai.txt"):
            p = os.path.join(data_dir, fname)
            if os.path.exists(p):
                os.remove(p)
        self._update_catalog_stats()
        self.scan_status = "Каталог очищен"

    def _update_catalog_stats(self):
        catalog_path = os.path.join(_app_data_dir(), "catalog.tsv")
        if os.path.exists(catalog_path):
            try:
                with open(catalog_path, "r", encoding="utf-8") as f:
                    count = sum(1 for _ in f)
                self.catalog_stats = f"В базе {count} треков"
            except Exception:
                self.catalog_stats = "Ошибка чтения базы"
        else:
            self.catalog_stats = "Каталог пуст"

    # ------------------------------------------------------------------
    # Playlist generation
    # ------------------------------------------------------------------

    def generate_playlist(self):
        if not self.api_key:
            self.status_text = "Ошибка: введите API ключ в настройках"
            return
        if not self.query_text.strip():
            self.status_text = "Ошибка: введите запрос"
            return
        self.is_busy = True
        self.status_text = "Связь с ИИ..."
        self.results_text = ""
        threading.Thread(target=self._generate_thread, daemon=True).start()

    def _generate_thread(self):
        status = "Готово!"
        try:
            catalog_path = os.path.join(_app_data_dir(), "catalog.tsv")
            if query_engine and os.path.exists(catalog_path):
                self._generate_with_engine()
            else:
                self._generate_simple()
        except Exception as e:
            status = f"Ошибка: {e}"
            traceback.print_exc()
            err_text = str(e)
            Clock.schedule_once(lambda dt: setattr(self, "results_text", err_text))

        Clock.schedule_once(lambda dt: self._end_busy(status))

    def _generate_simple(self):
        """Simple generation without local catalog."""
        system = "Ты музыкальный эксперт. Выдай список: Артист - Название."
        user_msg = f"Плейлист на {self.playlist_size} треков: {self.query_text}"

        result = _call_ai(
            self.api_provider, self.api_key, self.ai_model,
            system, user_msg,
        )
        Clock.schedule_once(lambda dt: setattr(self, "results_text", result))

    def _generate_with_engine(self):
        """Full two-step pipeline via query_engine."""
        # _write_config() — only I/O, no Kivy property writes, thread-safe
        self._write_config()

        def progress(msg):
            Clock.schedule_once(lambda dt: setattr(self, "status_text", msg))

        result = query_engine.create_playlist(
            config_path=_config_path(),
            user_query=self.query_text,
            progress_cb=progress,
        )
        summary = (
            f"Плейлист создан: {result['valid']} треков\n"
            f"Файл: {result['output_path']}\n\n"
        )
        track_lines = [
            f"{i}. {t.get('artist', '')} — {t.get('title', '')}"
            for i, t in enumerate(result.get("tracks", []), 1)
        ]
        text = summary + "\n".join(track_lines)
        Clock.schedule_once(lambda dt: setattr(self, "results_text", text))

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _end_busy(self, status):
        self.is_busy = False
        self.status_text = status

    def set_theme(self, name):
        if name in THEMES:
            self.theme_name = name
            self.theme = _ThemeData(THEMES[name])


if __name__ == "__main__":
    PlaylistApp().run()
