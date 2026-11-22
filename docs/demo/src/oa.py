import json,urllib.request,base64,sys,subprocess
L=["Как понять, что ИИ-аудитор правда умеет находить ошибки?",
"Мы построили для него тренажёр: компания за месяц. Счета, платежи, проводки, курсы валют. И скрытые ошибки.",
"Агент работает пятью инструментами, как живой аудитор. Четырнадцать шагов — и все десять ошибок найдены.",
"Обмануть награду нельзя. Двенадцать атак и перебор двухсот тысяч сидов проигрывают честному аудиту.",
"Одна команда, сорок пять секунд — и всё проверено. Эр-эль среда для ИИ-аудитора."]
voice=sys.argv[1]; model=sys.argv[2]; idx=[int(x) for x in sys.argv[3].split(',')]
for i in idx:
    body={"model":model,"modalities":["text","audio"],"audio":{"voice":voice,"format":"pcm16"},"stream":True,
      "messages":[{"role":"system","content":"Ты профессиональный диктор рекламы технологического продукта. Носитель русского языка, без акцента, правильные ударения, бодро, позитивно, энергично и естественно, с улыбкой в голосе. Прочитай текст пользователя дословно, ничего не добавляя."},{"role":"user","content":L[i]}]}
    r=urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",data=json.dumps(body).encode(),headers={"Authorization":"Bearer proxy-injected","Content-Type":"application/json"})
    pcm=b"";tr=""
    for line in urllib.request.urlopen(r,timeout=180):
        line=line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"): continue
        d=json.loads(line[5:]); ch=d.get("choices") or [{}]; a=ch[0].get("delta",{}).get("audio") or {}
        if a.get("data"): pcm+=base64.b64decode(a["data"])
        if a.get("transcript"): tr+=a["transcript"]
    open(f"g{i}.pcm","wb").write(pcm)
    subprocess.run(["ffmpeg","-v","error","-y","-f","s16le","-ar","24000","-ac","1","-i",f"g{i}.pcm",f"g{i}.wav"])
    print(i,len(pcm)/48000,"s |",tr)
