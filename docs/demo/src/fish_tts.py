"""Voice lines via Fish Audio (voice "Энергичный Рассказчик"); key is injected by the proxy."""
import json, urllib.request
from oa import L
for i, t in enumerate(L):
    r = urllib.request.Request("https://api.fish.audio/v1/tts", data=json.dumps({"text": t, "reference_id": "48ec28a45588422b977a86460c11977f", "format": "mp3"}).encode(),
        headers={"Authorization": "Bearer proxy-injected", "Content-Type": "application/json", "model": "s1"})
    open(f"v{i}.mp3", "wb").write(urllib.request.urlopen(r, timeout=120).read())
