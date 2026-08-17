import pandas as pd
import re

df = pd.read_csv('data/4chan/raw/manual_check.csv')

def auto_label(text):
    t = text.lower()
    # Если есть мат И упоминание "you" → токсично
    if re.search(r'(fuck|shit|bitch|cunt|dick|asshole|retard|idiot|moron|dumbass)', t) and \
       re.search(r'\b(you|your|u|ya|yall)\b', t):
        return 1
    else:
        return 0

df['label'] = df['text'].apply(auto_label)

# Сохраняем размеченный датасет
df.to_csv('data/4chan/labeled_auto.csv', index=False)
