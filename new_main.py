import streamlit as st
import pandas as pd
import auth
import data_collection
import spotify_utils
import reccobeats_utils
import config
from recommendation_system import recommendation_engine
import dataset_expansion
import os

# Page configuration
st.set_page_config(
    page_title="VibeGenie - AI Music Discovery",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'sp' not in st.session_state:
    st.session_state.sp = None
if 'user_cache' not in st.session_state:
    st.session_state.user_cache = None
if 'recommendation_engine_loaded' not in st.session_state:
    st.session_state.recommendation_engine_loaded = False
if 'current_recommendations' not in st.session_state:
    st.session_state.current_recommendations = None
if 'playlist_creation_result' not in st.session_state:
    st.session_state.playlist_creation_result = None

# Loading the recommendation engine
if not st.session_state.recommendation_engine_loaded:
    with st.spinner("Loading recommendation engine..."):
        recommendation_engine.load_model()
        st.session_state.recommendation_engine_loaded = True
        

# ... existing code ...

def show_landing_page():
    """Show the modern landing page."""
    st.markdown("""
    <div class="hero-content">
        <h1 class="hero-title">Get closer to the Edge.</h1>
        <p class="hero-subtitle">
            Discover your next favorite song with AI-powered recommendations. 
            Connect your Spotify and dive into a world of personalized music discovery.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Functional button for Spotify connection - positioned below the hero content
    col1, col2, col3 = st.columns([1.4, 1, 1])
    with col2:
        st.markdown('<div class="hero-button-container">', unsafe_allow_html=True)
        if st.button("🎵 Connect with Spotify", key="hero_connect", help="Connect your Spotify account to get started", type="primary"):
            try:
                sp, user_cache = auth.get_spotify_client()
                if sp:
                    st.session_state.sp = sp
                    st.session_state.user_cache = user_cache
                    st.session_state.authenticated = True
                    st.success("🎉 Successfully connected to Spotify!")
                    st.rerun()
                else:
                    st.error("❌ Connection failed. Please try again.")
            except Exception as e:
                st.error(f"❌ Connection error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # AI Recommendations section with image
    st.markdown("<br>", unsafe_allow_html=True)
    
    ai_col1, ai_col2 = st.columns([2, 1])
    
    with ai_col1:
        st.markdown("""
        <div class="feature-card" style="height: 100%;">
            <div class="feature-icon">🤖</div>
            <h3 class="feature-title">AI-Powered Recommendations</h3>
            <p class="feature-description">
                Our advanced machine learning model analyzes your music taste and discovers 
                songs you'll love based on audio features and listening patterns.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with ai_col2:
        st.image("Model_Neural_Net.avif")
        
    # Other feature cards
    st.markdown("""
    <div class="feature-grid" style="margin-top: 2rem;">
        <div class="feature-card">
            <div class="feature-icon">🎧</div>
            <h3 class="feature-title">Personalized Discovery</h3>
            <p class="feature-description">
                Connect your Spotify account to get recommendations tailored specifically 
                to your unique music preferences and listening history.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # How it Works section
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="how-it-works-section">
        <h2 style="text-align: center; color: var(--text-primary); margin-bottom: 2rem; font-size: 2rem; font-weight: 600;">How it Works</h2>
        <div class="how-it-works-grid">
            <div class="how-it-works-card">
                <div class="how-it-works-icon">🔌</div>
                <h3 class="how-it-works-title">Connect</h3>
                <p class="how-it-works-description">
                    Connect your Spotify account securely to access your music library and preferences.
                </p>
            </div>
            <div class="how-it-works-card">
                <div class="how-it-works-icon">🔍</div>
                <h3 class="how-it-works-title">Analyze</h3>
                <p class="how-it-works-description">
                    Our AI analyzes your music taste using advanced machine learning algorithms.
                </p>
            </div>
            <div class="how-it-works-card">
                <div class="how-it-works-icon">✨</div>
                <h3 class="how-it-works-title">Discover</h3>
                <p class="how-it-works-description">
                    Get personalized recommendations for songs you'll love, tailored just for you.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    

# ... existing functions (show_dashboard, show_collect_data, etc.) remain the same ...
def show_dashboard(sp):
    """Show the main dashboard."""
    st.markdown("# 📊 Your Music Dashboard")
    
    # Get basic stats
    try:
        playlists = sp.current_user_playlists(limit=50)
        user = sp.current_user()
        
        # Metrics in cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: var(--primary-color); margin: 0;">Total Playlists</h3>
                <h1 style="margin: 0.5rem 0;">{len(playlists['items'])}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: var(--secondary-color); margin: 0;">Welcome Back</h3>
                <h1 style="margin: 0.5rem 0; font-size: 1.5rem;">{user.get('display_name', 'Unknown')}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            followers = user.get('followers', {}).get('total', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: var(--accent-color); margin: 0;">Followers</h3>
                <h1 style="margin: 0.5rem 0;">{followers}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        # Show recent playlists
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🎵 Your Recent Playlists")
        for playlist in playlists['items'][:5]:
            st.markdown(f"""
            <div class="recommendation-card">
                <h4 style="margin: 0; color: var(--text-primary);">{playlist['name']}</h4>
                <p style="margin: 0.5rem 0; color: var(--text-secondary);">{playlist['tracks']['total']} tracks</p>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error loading dashboard: {e}")

def show_collect_data(sp):
    """Show data collection interface."""
    st.markdown("# 📥 Collect Your Music Data")
        
    st.markdown("""
    <div style="background: var(--bg-secondary); padding: 2rem; border-radius: 15px; border: 1px solid var(--border-color); margin-bottom: 2rem;">
        <h3 style="color: var(--primary-color); margin-top: 0;">What we'll collect:</h3>
        <ul style="color: var(--text-secondary); line-height: 1.8;">
            <li>🎵 Your playlists and liked songs</li>
            <li>📊 Top tracks (short, medium, long term)</li>
            <li>🎧 Audio features from Reccobeats API</li>
            <li>📈 Listening patterns and preferences</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Collect All User Data", type="primary", use_container_width=True):
            with st.spinner("Analyzing your Spotify library..."):
                try:
                    df = data_collection.collect_user_tracks(sp)
                    st.success(f"Successfully collected {len(df)} tracks!")
                    st.session_state.last_collection = df
                except Exception as e:
                    st.error(f"Error collecting data: {e}")
    
    with col2:
        if st.button("📝 Collect Playlist Data Only", use_container_width=True):
            with st.spinner("Collecting playlist data..."):
                try:
                    df = data_collection.collect_playlist_data(sp)
                    st.success(f"Successfully collected {len(df)} playlist tracks!")
                except Exception as e:
                    st.error(f"Error collecting playlist data: {e}")

def show_view_data():
    """Show collected data."""
    st.markdown("# 📊 Your Music Data")
    
    # Check for saved files
    try:
        user_tracks_df = pd.read_csv(config.USER_TRACKS_CSV, index_col=0)
        
        # Show basic stats in cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: var(--primary-color); margin: 0;">Total Tracks</h3>
                <h1 style="margin: 0.5rem 0;">{len(user_tracks_df)}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            tracks_with_features = user_tracks_df.dropna(subset=['danceability']).shape[0]
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: var(--secondary-color); margin: 0;">With Features</h3>
                <h1 style="margin: 0.5rem 0;">{tracks_with_features}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            unique_playlists = user_tracks_df['Playlist Name'].nunique()
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: var(--accent-color); margin: 0;">Playlists</h3>
                <h1 style="margin: 0.5rem 0;">{unique_playlists}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        # Show data preview
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📋 Data Preview")
        st.dataframe(user_tracks_df.head(10), use_container_width=True)
        
        # Download button
        csv = user_tracks_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="spotify_user_tracks.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    except FileNotFoundError:
        st.warning("No data found. Please collect data first.")
    except Exception as e:
        st.error(f"Error loading data: {e}")

def show_Music_recommendations():
    """Show Music recommendations with enhanced styling."""
    st.markdown("# 🤖 AI Music Recommendations")
    
    # Check model status
    status = recommendation_engine.get_model_status()
    if not status['is_loaded']:
        st.error("❌ Recommendation system failed to load")
        st.error(f"Error: {status['load_error']}")
        st.info("Please check that your trained model files are in the correct location.")
        return
    
    # Show model info in cards
    model_info = status['model_info']
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: var(--primary-color); margin: 0;">Total Songs</h3>
            <h1 style="margin: 0.5rem 0;">{model_info['total_songs']:,}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: var(--secondary-color); margin: 0;">Embedding Dim</h3>
            <h1 style="margin: 0.5rem 0;">{model_info['embedding_dim']}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: var(--accent-color); margin: 0;">Model Type</h3>
            <h1 style="margin: 0.5rem 0; font-size: 1.2rem;">{model_info['model_type']}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    # Check if user data is collected
    try: 
        user_tracks_df = pd.read_csv(config.USER_TRACKS_CSV, index_col=0)
        if len(user_tracks_df) == 0:
            st.warning("No user data found. Please collect your Spotify data first.")
            return
        
        # Show data summary
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🎧 Your Music Profile")
        
        # Get tracks with features (actual DataFrame, not just count)
        tracks_with_features_df = user_tracks_df.dropna(subset=['danceability'])
        tracks_with_features_count = len(tracks_with_features_df)
        
        st.markdown(f"""
        <div style="background: var(--bg-secondary); padding: 1.5rem; border-radius: 15px; border: 1px solid var(--border-color); margin-bottom: 2rem;">
            <p style="margin: 0; color: var(--text-secondary); font-size: 1.1rem;">
                You have <strong style="color: var(--primary-color);">{len(user_tracks_df)}</strong> total tracks, 
                <strong style="color: var(--secondary-color);">{tracks_with_features_count}</strong> with audio features
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if tracks_with_features_count < 3:
            st.warning("You need at least 3 tracks with audio features for recommendations.")
            st.info("Please collect more data or wait for audio features to be processed.")
            return
        
        # User Music Taste Analysis and Mood Controls
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🎭 Your Music Taste Analysis")
        
        # Analyze user's music taste
        try:
            taste_analysis = recommendation_engine.recommender.analyze_user_music_taste(tracks_with_features_df)
            
            if taste_analysis:
                # Display listening patterns
                st.markdown("""
                <div style="background: var(--bg-secondary); padding: 1.5rem; border-radius: 15px; border: 1px solid var(--border-color); margin-bottom: 2rem;">
                    <h4 style="color: var(--primary-color); margin-top: 0;">🎵 Your Listening Patterns</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Show listening patterns
                for pattern in taste_analysis['listening_patterns']:
                    if pattern['type'] == 'Primary':
                        st.success(f"🥇 **Primary Taste**: {pattern['description']}")
                    elif pattern['type'] == 'Secondary':
                        st.info(f"🥈 **Secondary Taste**: {pattern['description']}")
                    else:
                        st.info(f"↔️ **Diversity**: {pattern['description']}")
                
                # Display mood breakdown in cards
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**📊 Your Music Mood Breakdown:**")
                
                mood_cols = st.columns(len(taste_analysis['mood_statistics']))
                for i, (mood, stats) in enumerate(taste_analysis['mood_statistics'].items()):
                    with mood_cols[i]:
                        st.markdown(f"""
                        <div style="background: var(--bg-secondary); padding: 1rem; border-radius: 10px; border: 1px solid var(--border-color); text-align: center;">
                            <h4 style="color: var(--primary-color); margin: 0; font-size: 1rem;">{mood}</h4>
                            <h2 style="margin: 0.5rem 0; color: var(--text-primary);">{stats['percentage']:.1f}%</h2>
                            <p style="margin: 0; color: var(--text-secondary); font-size: 0.8rem;">{stats['count']} tracks</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # NEW: Mood-based Recommendation Controls
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("🎛️ Recommendation Preferences")
                
                # Mood preference controls
                st.markdown("**🎭 What type of recommendations do you want?**")
                
                # Get available moods from analysis
                available_moods = list(taste_analysis['mood_statistics'].keys())
                
                # Create mood selection with checkboxes
                mood_cols = st.columns(min(len(available_moods), 3))
                selected_moods = []
                
                for i, mood in enumerate(available_moods):
                    with mood_cols[i % 3]:
                        mood_percentage = taste_analysis['mood_statistics'][mood]['percentage']
                        is_selected = st.checkbox(
                            f"{mood} ({mood_percentage:.1f}%)", 
                            value=True,  # Default to selected
                            key=f"mood_{mood}",
                            help=f"You have {mood_percentage:.1f}% {mood.lower()} music in your library"
                        )
                        if is_selected:
                            selected_moods.append(mood)
                
                # If no moods selected, select all
                if not selected_moods:
                    selected_moods = available_moods
                
                # Diversity and energy controls
                col1, col2 = st.columns(2)
                
                with col1:
                    diversity_level = st.slider(
                        "🎨 Diversity Level", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=0.3, 
                        step=0.1,
                        help="Higher values = more diverse recommendations, Lower values = more similar to your taste"
                    )
                
                with col2:
                    energy_preference = st.selectbox(
                        "⚡ Energy Preference",
                        options=["Match My Taste", "High Energy", "Medium Energy", "Low Energy", "Mixed"],
                        help="Override energy level for recommendations"
                    )
                
                # Store preferences in session state
                st.session_state.recommendation_preferences = {
                    'selected_moods': selected_moods,
                    'diversity_level': diversity_level,
                    'energy_preference': energy_preference,
                    'taste_analysis': taste_analysis
                }
                
            else:
                st.warning("Could not analyze your music taste. Using default recommendations.")
                st.session_state.recommendation_preferences = {
                    'selected_moods': ['High Energy', 'Chill', 'Medium Energy'],
                    'diversity_level': 0.3,
                    'energy_preference': 'Match My Taste',
                    'taste_analysis': None
                }
                
        except Exception as e:
            st.warning(f"Error analyzing music taste: {e}. Using default preferences.")
            st.session_state.recommendation_preferences = {
                'selected_moods': ['High Energy', 'Chill', 'Medium Energy'],
                'diversity_level': 0.3,
                'energy_preference': 'Match My Taste',
                'taste_analysis': None
            }
        
        # Recommendation controls
        st.subheader("🎯 Generate Recommendations")
        col1, col2 = st.columns(2)
        
        with col1:
            n_recommendations = st.slider("Number of recommendations", 5, 50, 20)
            
            # Add quick mode option
            quick_mode = st.checkbox("🚀 Quick Mode", value=True, 
                                   help="Faster recommendations with slightly less diversity")
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🎵 Generate Recommendations", type="primary", use_container_width=True):
                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Analyzing user tracks
                    status_text.text("🔍 Analyzing your music tracks...")
                    progress_bar.progress(20)
                    
                    # Step 2: Generating recommendations
                    status_text.text("🎵 Finding similar songs in the dataset...")
                    progress_bar.progress(60)

                    # Get user preferences (with defaults if not set)
                    preferences = st.session_state.get('recommendation_preferences', {
                        'selected_moods': ['High Energy', 'Chill', 'Medium Energy'],
                        'diversity_level': 0.3,
                        'energy_preference': 'Match My Taste',
                        'taste_analysis': None
                    })
                    
                    # Pass preferences to recommendation engine
                    recommendations, error = recommendation_engine.get_recommendations(
                        user_tracks_df, 
                        n_recommendations, 
                        quick_mode=quick_mode,
                        user_preferences=preferences  # NEW: Pass preferences
                    )
                    
                    
                    # Step 3: Finalizing results
                    status_text.text("✨ Finalizing recommendations...")
                    progress_bar.progress(100)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Handle results
                    if error:
                        st.error(f"Error: {error}")
                        st.session_state.current_recommendations = None
                    elif recommendations:
                        st.success(f"Generated {len(recommendations)} personalized recommendations!")
                        # Store recommendations in session state
                        st.session_state.current_recommendations = recommendations
                        # Clear any previous playlist creation result
                        st.session_state.playlist_creation_result = None
                    else:
                        st.warning("No recommendations generated. Please try again.")
                        st.session_state.current_recommendations = None
                    
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Unexpected error: {str(e)}")
                    st.session_state.current_recommendations = None
                        
        # Display recommendations from session state
        if st.session_state.current_recommendations:
            recommendations = st.session_state.current_recommendations
            
            # Display recommendations with enhanced styling
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("🎵 Your Personalized Recommendations")
            
            for i, rec in enumerate(recommendations, 1):
                # Show source track if available
                source_info = ""
                if 'source_track' in rec and 'source_artist' in rec:
                    source_info = f"<p style='margin: 0.25rem 0; color: var(--text-muted); font-size: 0.85rem; font-style: italic;'>Similar to: {rec['source_track']} by {rec['source_artist']}</p>"
                
                st.markdown(f"""
                <div class="recommendation-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="flex: 1;">
                            <h4 style="margin: 0; color: var(--text-primary); font-size: 1.2rem;">
                                {i}. {rec['track_name']}
                            </h4>
                            <p style="margin: 0.5rem 0; color: var(--text-secondary);">
                                by {rec['artists']}
                            </p>
                            {source_info}
                        
                </div>
                """, unsafe_allow_html=True)
                        
            # Save as Spotify Playlist section
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("💾 Save Recommendations")
            
            # Note about re-authentication if needed
            st.info("💡 **Note:** If you encounter permission errors, you may need to log out and log back in to grant playlist creation permissions.")
            
            # Playlist options
            col1, col2 = st.columns([2, 1])
            
            with col1:
                playlist_name = st.text_input(
                    "Playlist Name", 
                    value="VibeGenie Playlist",
                    help="Name for your new Spotify playlist"
                )
            
            with col2:
                is_public = st.toggle(
                    "Make Public", 
                    value=False,
                    help="Make the playlist public and shareable"
                )
            
            # Save as playlist button
            if st.button("🎵 Save as Spotify Playlist", type="primary", use_container_width=True):
                if playlist_name.strip():
                    try:
                        # Debug: Check if we have recommendations
                        st.write(f"Debug: Creating playlist with {len(recommendations)} recommendations")
                        
                        # Import the playlist creation function
                        from spotify_utils import create_playlist_from_recommendations
                        
                        # Create the playlist
                        result = create_playlist_from_recommendations(
                            sp=st.session_state.sp,
                            recommendations=recommendations,
                            playlist_name=playlist_name.strip(),
                            is_public=is_public
                        )
                        
                        # Store result in session state
                        st.session_state.playlist_creation_result = result
                        
                        if result['success']:
                            # Show success message with playlist info
                            playlist_info = result['playlist_info']
                            stats = result['stats']
                            
                            st.success(f"🎉 Playlist '{playlist_info['name']}' created successfully!")
                            
                            # Display playlist info
                            st.markdown(f"""
                            <div style="background: var(--bg-secondary); padding: 1.5rem; border-radius: 15px; border: 1px solid var(--border-color); margin: 1rem 0;">
                                <h4 style="color: var(--primary-color); margin-top: 0;">📋 Playlist Details</h4>
                                <p style="margin: 0.5rem 0;"><strong>Name:</strong> {playlist_info['name']}</p>
                                <p style="margin: 0.5rem 0;"><strong>Tracks Added:</strong> {playlist_info['total_tracks']}</p>
                                <p style="margin: 0.5rem 0;"><strong>Privacy:</strong> {'Public' if playlist_info['public'] else 'Private'}</p>
                                <p style="margin: 0.5rem 0;"><strong>Spotify URL:</strong> <a href="{playlist_info['url']}" target="_blank">Open in Spotify</a></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show statistics
                            if stats['tracks_skipped'] > 0 or stats['tracks_failed'] > 0:
                                st.info(f"ℹ️ {stats['tracks_skipped']} tracks were skipped and {stats['tracks_failed']} tracks failed to add.")
                            
                        else:
                            st.error(f"❌ {result['error']}")
                            
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {str(e)}")
                        st.error("Please check your Spotify connection and try again.")
                else:
                    st.warning("Please enter a playlist name.")
            
            # Show previous playlist creation result if it exists
            if st.session_state.playlist_creation_result:
                result = st.session_state.playlist_creation_result
                if result['success']:
                    playlist_info = result['playlist_info']
                    st.info(f"✅ Last created playlist: [{playlist_info['name']}]({playlist_info['url']})")
            
            # Optional: Keep CSV download as secondary option
            with st.expander("📥 Download as CSV (Alternative)"):
                rec_df = pd.DataFrame(recommendations)
                csv = rec_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Recommendations CSV",
                    data=csv,
                    file_name="ai_music_recommendations.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # Show user's top tracks for context
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🎧 Your Top Tracks (Used for Recommendations)")
        
        # Filter to show only tracks with audio features AND from user's top tracks

        required_features= ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence','Loudness', 'tempo', 'key', 'mode', 'popularity']

        # Checking if features are present in the dataframe
        available_features = [col for col in required_features if col in user_tracks_df.columns]
        tracks_with_features = user_tracks_df.dropna(subset=available_features)

        # Filter to show only tracks from user's top tracks
        top_tracks_pattern = r'Top Tracks \(.*\)'
        top_tracks_only = tracks_with_features[tracks_with_features['Playlist Name'].str.contains(top_tracks_pattern, na=False)]        

        # Mapping column names to match the CSV structure
        display_columns = []
        if 'Track Name' in top_tracks_only.columns:
            display_columns.append('Track Name')
        elif 'track_name' in top_tracks_only.columns:
            display_columns.append('track_name')
            
        if 'Artist(s)' in top_tracks_only.columns:
            display_columns.append('Artist(s)')
        elif 'artists' in top_tracks_only.columns:
            display_columns.append('artists')
            
        if 'Popularity' in top_tracks_only.columns:
            display_columns.append('Popularity')
        elif 'popularity' in top_tracks_only.columns:
            display_columns.append('popularity')
            
        # Add Playlist Name to show which time period the track is from

        if 'Playlist Name' in top_tracks_only.columns:
            display_columns.append('Playlist Name')
        

        if display_columns and len(top_tracks_only) > 0:
            # showing up to 15 top tracks
            top_user_tracks = top_tracks_only.head(15)[display_columns]
            st.dataframe(top_user_tracks, use_container_width=True)

            # Show summary info
            st.info(f"📊 Showing {len(top_user_tracks)} tracks from your top tracks.")
        else:
            st.warning("Could not find track display columns in the data.")
        
        # Analyze user's music characteristics
        try:
            tracks_with_features = user_tracks_df.dropna(subset=['danceability'])
            if len(tracks_with_features) > 0:
                # Calculate average characteristics
                avg_energy = tracks_with_features['energy'].mean() if 'energy' in tracks_with_features.columns else 0
                avg_danceability = tracks_with_features['danceability'].mean() if 'danceability' in tracks_with_features.columns else 0
                avg_valence = tracks_with_features['valence'].mean() if 'valence' in tracks_with_features.columns else 0
                avg_popularity = tracks_with_features['Popularity'].mean() if 'Popularity' in tracks_with_features.columns else 0
                
                                
                # Music style description
                style_description = []
                if avg_energy > 0.7:
                    style_description.append("High Energy")
                elif avg_energy < 0.3:
                    style_description.append("Low Energy")
                
                if avg_danceability > 0.7:
                    style_description.append("Danceable")
                elif avg_danceability < 0.3:
                    style_description.append("Not Danceable")
                
                if avg_valence > 0.7:
                    style_description.append("Positive Mood")
                elif avg_valence < 0.3:
                    style_description.append("Negative Mood")
                
                if style_description:
                    st.info(f"🎵 **Your Music Style**: {', '.join(style_description)}")
                
                    
        except Exception as e:
            st.warning(f"Could not analyze music profile: {e}")
        
    except FileNotFoundError:
        st.warning("No user data found. Please collect your Spotify data first.")
        st.info("Go to 'Collect Data' to get started!")
    except Exception as e:
        st.error(f"Error loading user data: {e}")


# Define page wrapper functions for st.navigation
def dashboard_page():
    """Dashboard page."""
    if st.session_state.authenticated:
        show_dashboard(st.session_state.sp)
    else:
        st.warning("Please authenticate with Spotify first.")

def collect_data_page():
    """Data collection page."""
    if st.session_state.authenticated:
        show_collect_data(st.session_state.sp)
    else:
        st.warning("Please authenticate with Spotify first.")

def view_data_page():
    """View data page."""
    show_view_data()

def recommendations_page():
    """AI recommendations page."""
    if st.session_state.authenticated:
        show_Music_recommendations()
    else:
        st.warning("Please authenticate with Spotify first.")

# Define navigation structure
pages = [
    st.Page(dashboard_page, icon='🏠', title='Dashboard'),
    st.Page(collect_data_page, title="📥 Collect Data"),
    st.Page(view_data_page, title="📊 View Data"),
    st.Page(recommendations_page, title="🤖 AI Recommendations"),
    # st.Page(debug_tools_page, title="🔧 Debug Tools"),
    # st.Page(settings_page, title="⚙️ Settings"),
]

# Check for OAuth callback first
if 'code' in st.query_params or 'error' in st.query_params:
    # Handle OAuth callback
    try:
        sp, user_cache = auth.get_spotify_client()
        if sp:
            st.session_state.sp = sp
            st.session_state.user_cache = user_cache
            st.session_state.authenticated = True
            st.success("🎉 Successfully connected to Spotify!")
            # Clear URL parameters
            st.query_params.clear()
            st.rerun()
        else:
            st.error("❌ Authentication failed. Please try again.")
    except Exception as e:
        st.error(f"❌ Authentication error: {e}")

# Main application logic
if not st.session_state.authenticated:
    # Show the new landing page
    show_landing_page()
else:
    # Top right logout button and user info
    user_info = auth.get_current_user_info(st.session_state.sp)
    
    # Create a top bar with user info and logout button
    col1, col2 = st.columns([1, 0.1])
    
    with col1:
        if user_info:
            st.markdown(f"""
            <div style="text-align: left; margin-bottom: 1rem;">
                <h4 style="margin: 0; color: var(--secondary-color); display: inline;">👋 Welcome back, {user_info.get('display_name', 'Unknown')}!</h4>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🚪 Logout", key="top_logout", help="Logout from Spotify"):
            # Clear session state properly
            if st.session_state.user_cache and os.path.exists(st.session_state.user_cache):
                os.remove(st.session_state.user_cache)
            st.session_state.authenticated = False
            st.session_state.sp = None
            st.session_state.user_cache = None
            st.success("You have been logged out successfully!")
            st.rerun()
    
    # Navigation
    pg = st.navigation(pages, position="top")
    pg.run()