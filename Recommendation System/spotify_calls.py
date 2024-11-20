import pandas as pd


def get_mood_recommendations(user_name, mood, top_n=10):

    df = pd.read_csv('updated_user_data.csv')

    user_played_songs = df[(df['User Name'] == user_name) & (df['Mood Preferences'] == mood)]
    recommendations = user_played_songs.sort_values(by='Play Count', ascending=False).head(top_n)

    return recommendations[['track_name', 'Mood Preferences', 'duration']]


def get_trending_songs(top_n: int = 10) -> list:
    """
    :param top_n: by default n value is 10.
    :return: list of trending songs. list contains list of dict contains basic info about the song.
    """

    df = pd.read_csv('spotify_tracks_dataset-new.csv')

    df_sorted = df.sort_values(by='popularity', ascending=False)

    songs_list = []
    seen_tracks = set()

    for _, row in df_sorted.iterrows():
        track_name = row['track_name']

        if track_name not in seen_tracks:
            duration_ms = row['duration_ms']
            minutes, seconds = divmod(duration_ms // 1000, 60)
            duration = f"{minutes}:{str(seconds).zfill(2)}"

            song_info = {
                "track_name": track_name,
                "artists": row['artists'],
                "duration": duration
            }
            songs_list.append(song_info)
            seen_tracks.add(track_name)

        if len(songs_list) == top_n:
            break

    return songs_list



