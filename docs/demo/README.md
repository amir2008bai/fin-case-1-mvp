# Демо-ролик

`demo.mp4` — 28 с, 1920×1080, 60 fps, русская озвучка Gemini TTS, голос Puck.

Сборка (из `docs/demo/src`):
1. Кадры: `for k in 0 1 2 3; do node rec.js $((k*420)) $((k*420+420)) 1 & done; wait` (Playwright, 1680 кадров в `f/`).
2. Голос: 5 фраз в `oa.py` (`L`), вставляются с задержками 0.5 / 4.3 / 10.6 / 16.4 / 21.4 с. Готовые исходные реплики Gemini Puck лежат в `puck/1.wav` … `puck/5.wav`. Коэффициенты темпа при сведении равны 1.107, 1.314, 1.317, 1.509 и 1.000 соответственно.
3. Сведение: музыка `music.wav` приглушается под голос (sidechain), см. команду ffmpeg в истории сессии:
   `ffmpeg -framerate 60 -i f/%05d.jpg -i music.wav -i v0 … -i v4` → `adelay` каждой фразы, `amix`, `sidechaincompress`, `alimiter`; H.264 yuv420p crf 18, AAC 192k, `-t 28`.
