import os
import django
import json
import csv
from spotify_calls import get_trending_songs, get_mood_recommendations
from django.conf import settings
from django.urls import path
from django.shortcuts import render
from django.http import JsonResponse
from django.core.wsgi import get_wsgi_application

settings.configure(
    DEBUG=True,
    ROOT_URLCONF=__name__,
    SECRET_KEY='+0on6r)k0btf3&*+g&3a1++dm*1ga32w$1kn*bn^6bq^%6-zyz',
    TEMPLATES=[{
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': ['.'],
        'APP_DIRS': True,
    }],
)

django.setup()


def load_user_data():
    data = []
    csv_file = 'user_data.csv'

    if os.path.exists(csv_file):
        with open(csv_file, newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    return data


def trending_songs_view(request):
    trending_songs = get_trending_songs()
    return JsonResponse({'status': 'success', 'songs': trending_songs})


def user_data(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            selected_user = data.get('user')
            selected_mood = data.get('mood')

            user_data = load_user_data()

            filtered_songs = [
                row for row in user_data
                if row['User Name'].strip() == selected_user.strip() and row[
                    'Mood Preferences'].strip() == selected_mood.strip()
            ]

            song_data = [
                {
                    'track_name': row['track_name'],
                    'Play Time': row['Play Time'],
                    'Play Count': row['Play Count'],
                    'Like/Dislike': row['Like/Dislike'],
                    'Rating': row['Rating']
                }
                for row in filtered_songs[:10]
            ]
            return JsonResponse({'songs': song_data})

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            return JsonResponse({'error': 'An unexpected error occurred'}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=400)


def index(request):
    return render(request, 'new_edited.html')


def recommend(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user = data.get('user')
        mood = data.get('mood')
        data = get_mood_recommendations(user, mood)
        print(data.head())
        data_json = data.to_dict(orient='records')
        return JsonResponse({'recommendation': data_json})


urlpatterns = [
    path('', index, name='sample_html'),
    path('recommend/', recommend, name='recommend'),
    path('user_data/', user_data, name='user_data'),
    path('trending-songs/', trending_songs_view, name='trending-songs'),
]

application = get_wsgi_application()

if __name__ == "__main__":
    from django.core.management import execute_from_command_line

    execute_from_command_line(['manage.py', 'runserver', '0.0.0.0:8000'])
