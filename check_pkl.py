import pickle, json, os

pkl_dir = 'ds005170/derivatives/preprocessed_pkl/sub-01/eeg/'
all_texts = set()

for f in os.listdir(pkl_dir):
    if f.endswith('.pkl'):
        try:
            data = pickle.load(open(pkl_dir+f, 'rb'))
            for trial in data:
                all_texts.add(trial['text'].strip())
            print(f'OK: {f} -> {len(data)} trials')
        except Exception as e:
            print(f'SKIP: {f} -> {e}')

print(f'Total unique phrases: {len(all_texts)}')