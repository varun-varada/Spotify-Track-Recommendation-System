# ------------- setup -------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
from tensorflow.keras.models import Model

# ------------- load -------------
df = pd.read_csv("spotify_history.csv")

# Basic clean
df = df.drop_duplicates().copy()
# Parse timestamp
df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
df = df[df['ts'].notna()].copy()

# Safety: fill missing text cols
text_cols = ['track_name','artist_name','album_name','platform','reason_start','reason_end']
for c in text_cols:
    if c in df.columns:
        df[c] = df[c].fillna("Unknown")

# Ensure ms_played/skipped/shuffle exist in correct dtype
if 'ms_played' in df.columns:
    df['ms_played'] = pd.to_numeric(df['ms_played'], errors='coerce').fillna(0).clip(lower=0)
if 'skipped' in df.columns:
    df['skipped'] = df['skipped'].astype(int)  # 0/1
if 'shuffle' in df.columns:
# # sometimes boolean as string; normalize to 0/1
    df['shuffle'] = df['shuffle'].astype(str).str.lower().isin(['true','1','yes']).astype(int)

print("Shape:", df.shape)
print(df.head(3))

# Top artists by total play time
top_artists = (
    df.groupby('artist_name')['ms_played']
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

plt.figure()
top_artists[::-1].plot(kind='barh')
plt.title("Top 10 Artists by Total Play Time (ms)")
plt.xlabel("Total ms_played")
plt.tight_layout()
plt.show()

# Top songs by average engagement (min 10 plays)
plays_per_song = df.groupby('track_name').agg(
    plays=('track_name','count'),
    total_ms=('ms_played','sum')
)
plays_per_song['avg_ms'] = plays_per_song['total_ms'] / plays_per_song['plays']
engaged = plays_per_song[plays_per_song['plays']>=10].sort_values('avg_ms', ascending=False).head(10)
print("Top 10 Songs by Avg Engagement (>=10 plays):\n", engaged)

# Skip rate by artist (min 20 plays)
if 'skipped' in df.columns:
    sk = df.groupby('artist_name').agg(
        plays=('artist_name','count'),
        skip_rate=('skipped','mean')
    )
    sk = sk[sk['plays']>=20].sort_values('skip_rate')
    print("Low skip-rate artists (min 20 plays):\n", sk.head(10))

# Platform popularity
if 'platform' in df.columns:
    plat = df['platform'].value_counts().head(10)
    plt.figure()
    plat[::-1].plot(kind='barh')
    plt.title("Plays by Platform")
    plt.tight_layout()
    plt.show()

# Reasons distribution
for col in ['reason_start','reason_end']:
    if col in df.columns:
        vc = df[col].value_counts().head(10)
        plt.figure()
        vc[::-1].plot(kind='barh')
        plt.title(f"Top {col} values")
        plt.tight_layout()
        plt.show()

work = df.copy()

# Keep only needed columns
need = ['spotify_track_uri','artist_name','album_name','platform','reason_start','reason_end','shuffle','ms_played','skipped']
work = work[[c for c in need if c in work.columns]].dropna(subset=['spotify_track_uri'])

# Optional: filter very rare tracks to stabilize training
track_counts = work['spotify_track_uri'].value_counts()
rare_cut = 3
keep_tracks = track_counts[track_counts >= rare_cut].index
work = work[work['spotify_track_uri'].isin(keep_tracks)].copy()

# Bucketize playtime (optional feature)
work['played_bucket'] = pd.qcut(work['ms_played'].clip(upper=work['ms_played'].quantile(0.99)), q=5, duplicates='drop').astype(str)

# Label encode categoricals
encoders = {}
def fit_encode(col):
    le = LabelEncoder()
    work[col] = le.fit_transform(work[col].astype(str))
    encoders[col] = le
    return le

for col in ['spotify_track_uri','artist_name','album_name','platform','reason_start','reason_end','played_bucket']:
    if col in work.columns:
        fit_encode(col)

# Features & target
feature_cols = [c for c in ['spotify_track_uri','artist_name','album_name','platform','reason_start','reason_end','shuffle','played_bucket'] if c in work.columns]
X = work[feature_cols].copy()
y = work['skipped'].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Cardinalities for embeddings
card = {c: int(work[c].max())+1 for c in ['spotify_track_uri','artist_name','album_name','platform','reason_start','reason_end','played_bucket'] if c in work.columns}

# Build model
inputs = []
embeds = []

def add_embed(col, dim=32):
    inp = Input(shape=(1,), name=f"{col}_in")
    emb = Embedding(card[col], dim, name=f"{col}_emb")(inp)
    emb = Flatten()(emb)
    inputs.append(inp)
    embeds.append(emb)

# Add embeddings
if 'spotify_track_uri' in card: add_embed('spotify_track_uri', dim=48)  # main signal
if 'artist_name' in card:       add_embed('artist_name', dim=24)
if 'album_name' in card:        add_embed('album_name', dim=16)
if 'platform' in card:          add_embed('platform', dim=8)
if 'reason_start' in card:      add_embed('reason_start', dim=8)
if 'reason_end' in card:        add_embed('reason_end', dim=8)
if 'played_bucket' in card:     add_embed('played_bucket', dim=8)

# Numeric input(s)
num_inputs = []
if 'shuffle' in X.columns:
    shuffle_in = Input(shape=(1,), name="shuffle_in")
    inputs.append(shuffle_in)
    num_inputs.append(shuffle_in)

# Concatenate
merged = Concatenate()(embeds + num_inputs) if num_inputs else Concatenate()(embeds)
x = Dense(128, activation='relu')(merged)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
out = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC','accuracy'])
model.summary()

# Prepare input dicts
def to_keras_inputs(df_part):
    d = {}
    for c in card.keys():
        if c in df_part.columns:
            d[f"{c}_in"] = df_part[c].values
    if 'shuffle' in df_part.columns:
        d['shuffle_in'] = df_part['shuffle'].values
    return d

train_inputs = to_keras_inputs(X_train)
test_inputs  = to_keras_inputs(X_test)

hist = model.fit(train_inputs, y_train, validation_data=(test_inputs, y_test), epochs=5, batch_size=2048, verbose=1)

# Get learned track embedding matrix
track_emb_layer = model.get_layer('spotify_track_uri_emb')
track_vectors = track_emb_layer.get_weights()[0]  # shape: [num_tracks, emb_dim]

# Helper: map between encoded id and original track uri & names
track_le = encoders['spotify_track_uri']
id_to_track = dict(enumerate(track_le.classes_))

# Build lookup for display: track_uri -> (track_name, artist_name)
# Take the "mode" (most frequent) name/artist per track uri
meta = (df[['spotify_track_uri','track_name','artist_name']]
        .dropna()
        .groupby('spotify_track_uri')
        .agg(lambda s: s.value_counts().index[0])
        .reset_index())

meta_lut = {r['spotify_track_uri']: (r['track_name'], r['artist_name']) for _, r in meta.iterrows()}

def pretty_track(track_uri):
    tn, ar = meta_lut.get(track_uri, ("Unknown","Unknown"))
    return f"{tn} — {ar}"

def recommend_similar(track_uri, top_k=10):
# # encode given track_uri to id
    if track_uri not in track_le.classes_:
        return []
    tid = np.where(track_le.classes_ == track_uri)[0][0]
    vec = track_vectors[tid:tid+1]
    sims = cosine_similarity(vec, track_vectors)[0]
# # top similar excluding self
    idxs = np.argsort(-sims)
    recs = []
    for i in idxs:
        if i == tid:
            continue
        cand_uri = id_to_track[i]
        recs.append((cand_uri, sims[i], pretty_track(cand_uri)))
        if len(recs) >= top_k:
            break
    return recs

# Example usage:
example_uri = track_le.classes_[0]  # pick any known track
recs = recommend_similar(example_uri, top_k=10)
for uri, score, label in recs:
    print(round(score,4), label)

if 'country' in df.columns:
    top_by_country = (
        df.groupby(['country','track_name'])['ms_played']
          .sum()
          .reset_index()
          .sort_values(['country','ms_played'], ascending=[True, False])
          .groupby('country')
          .head(1)
    )
    print(top_by_country.head(20))

