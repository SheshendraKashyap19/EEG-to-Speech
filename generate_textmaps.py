import pickle, json, os

pkl_dir = 'ds005170/derivatives/preprocessed_pkl/sub-01/eeg/'
all_texts = set()

for f in os.listdir(pkl_dir):
    if f.endswith('.pkl'):
        try:
            data = pickle.load(open(pkl_dir+f, 'rb'))
            for trial in data:
                all_texts.add(trial['text'].strip())
        except:
            pass

all_texts = sorted(list(all_texts))
print(f'Total unique phrases: {len(all_texts)}')

# Map each phrase to one of 39 categories based on index
textmaps = {}
for idx, text in enumerate(all_texts):
    textmaps[text] = idx % 39

# Save to Chisco/json/textmaps.json
os.makedirs('Chisco/json', exist_ok=True)
with open('Chisco/json/textmaps.json', 'w', encoding='utf-8') as f:
    json.dump(textmaps, f, ensure_ascii=False, indent=2)

print('textmaps.json created successfully!')
print(f'Sample: {list(textmaps.items())[:3]}')