from flask import Flask, session, redirect, url_for, request, Response
from auth import get_spotify_client, sp_oauth, cache_handler
from data_collection import collect_data
import config

app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY

@app.route('/')
def home():
    sp = get_spotify_client()
    if isinstance(sp, Response):  # redirect returned
        return sp
    return redirect(url_for('get_playlists'))

@app.route('/callback')
def callback():
    code = request.args.get('code')
    if code:
        sp_oauth.get_access_token(code)  # updates session internally
    return redirect(url_for('get_playlists'))

@app.route('/get_playlists')
def get_playlists():
    sp = get_spotify_client()
    if isinstance(sp, Response):
        return sp
    playlists = sp.current_user_playlists()
    playlists_html = '<br>'.join([f"{pl['name']}: {pl['external_urls']['spotify']}" for pl in playlists['items']])
    playlists_html += "<br><br><a href='/collect'>Collect Track Data</a>"
    return playlists_html

@app.route('/collect')
def collect():
    sp = get_spotify_client()
    if isinstance(sp, Response):
        return sp
    collect_data(sp)
    return f"Data collected and saved to {config.CSV_OUTPUT}"

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
