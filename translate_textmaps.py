from deep_translator import GoogleTranslator
import json
import time

# Load original textmaps
with open("Chisco/json/textmaps.json", "r", encoding="utf-8") as f:
    textmaps = json.load(f)

print(f"Total phrases: {len(textmaps)}")

translator = GoogleTranslator(source='zh-CN', target='en')

phrases = list(textmaps.keys())
categories = list(textmaps.values())

translated_textmaps = {}
batch_size = 50  # translate 50 at a time

for i in range(0, len(phrases), batch_size):
    batch = phrases[i:i+batch_size]
    try:
        # Join with separator, translate all at once
        combined = " ||| ".join(batch)
        translated = translator.translate(combined)
        parts = translated.split(" ||| ")
        
        for j, eng in enumerate(parts):
            if j < len(batch):
                translated_textmaps[eng.strip()] = categories[i+j]
        
        print(f"Progress: {i+batch_size}/{len(phrases)} done ✅")
        time.sleep(0.3)

    except Exception as e:
        print(f"Batch {i} failed: {e} — keeping originals")
        for j, phrase in enumerate(batch):
            translated_textmaps[phrase] = categories[i+j]
        time.sleep(1)

# Save
with open("Chisco/json/textmaps_english.json", "w", encoding="utf-8") as f:
    json.dump(translated_textmaps, f, ensure_ascii=False, indent=2)

print(f"Done! Saved {len(translated_textmaps)} phrases to textmaps_english.json")