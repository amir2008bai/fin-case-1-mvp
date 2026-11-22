# Демо-ролик

`demo.mp4` — 28 с, 1920×1080, 60 fps, озвучка Fish Audio («Энергичный Рассказчик»).

Сборка (из `docs/demo/src`):
1. Кадры: `for k in 0 1 2 3; do node rec.js $((k*420)) $((k*420+420)) 1 & done; wait` (Playwright, 1680 кадров в `f/`).
2. Голос: 5 фраз в `oa.py` (`L`), вставляются с задержками 0.5 / 4.3 / 10.6 / 16.4 / 21.4 с. Сейчас `fish_tts.py`; задача — переозвучить через Gemini TTS (голос Puck/Fenrir, «носитель русского, без акцента, правильные ударения, позитивно»), ключ Google должен быть в заголовке `x-goog-api-key`.
3. Сведение: музыка `music.wav` приглушается под голос (sidechain), см. команду ffmpeg в истории сессии:
   `ffmpeg -framerate 60 -i f/%05d.jpg -i music.wav -i v0 … -i v4` → `adelay` каждой фразы, `amix`, `sidechaincompress`, `alimiter`; H.264 yuv420p crf 18, AAC 192k, `-t 28`.
