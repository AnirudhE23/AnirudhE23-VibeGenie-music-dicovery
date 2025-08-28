import streamlit as st
import pandas as pd
import auth
import data_collection
import spotify_utils
import reccobeats_utils
import config

def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="Spotify Data Extractor", page_icon="🎵")
    st.title("Spotify Data Extractor")
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'sp' not in st.session_state:
        st.session_state.sp = None
    if 'user_cache' not in st.session_state:
        st.session_state.user_cache = None

    # Authentication section
    if not st.session_state.authenticated:
        st.header("🔐 Authentication")
        st.write("Please authenticate with Spotify to continue.")
        
        if st.button("Login to Spotify"):
            try:
                sp, user_cache = auth.get_spotify_client()
                if sp:
                    st.session_state.sp = sp
                    st.session_state.user_cache = user_cache
                    st.session_state.authenticated = True
                    st.success("Successfully authenticated with Spotify!")
                    st.rerun()
                else:
                    st.error("Authentication failed. Please try again.")
            except Exception as e:
                st.error(f"Authentication error: {e}")
    
    else:
        # Main application after authentication
        sp = st.session_state.sp
        user_cache = st.session_state.user_cache
        
        # Get user info
        user_info = auth.get_current_user_info(sp)
        if user_info:
            st.sidebar.success(f"Logged in as: {user_info.get('display_name', 'Unknown')}")
        
        # Sidebar for navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Choose a page:",
            ["Dashboard", "Collect Data", "View Data", "Debug Tools", "Settings"]
        )
        
        if page == "Dashboard":
            show_dashboard(sp)
        elif page == "Collect Data":
            show_collect_data(sp)
        elif page == "View Data":
            show_view_data()
        elif page == "Debug Tools":
            show_debug_tools(sp)
        elif page == "Settings":
            show_settings(user_cache)
        
        # Logout button in sidebar
        if st.sidebar.button("Logout"):
            auth.logout(user_cache)

def show_dashboard(sp):
    """Show the main dashboard."""
    st.header("📊 Dashboard")
    
    # Get basic stats
    try:
        playlists = sp.current_user_playlists(limit=50)
        st.metric("Total Playlists", len(playlists['items']))
        
        # Get user profile
        user = sp.current_user()
        st.metric("User", user.get('display_name', 'Unknown'))
        
        # Show recent playlists
        st.subheader("Recent Playlists")
        for playlist in playlists['items'][:5]:
            st.write(f"• {playlist['name']} ({playlist['tracks']['total']} tracks)")
            
    except Exception as e:
        st.error(f"Error loading dashboard: {e}")

def show_collect_data(sp):
    """Show data collection interface."""
    st.header("📥 Collect Data")
    
    st.write("This will collect all your Spotify data including:")
    st.write("• Your playlists")
    st.write("• Liked songs")
    st.write("• Top tracks (short, medium, long term)")
    st.write("• Audio features from Reccobeats API")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Collect All User Data", type="primary"):
            with st.spinner("Collecting your Spotify data..."):
                try:
                    df = data_collection.collect_user_tracks(sp)
                    st.success(f"Successfully collected {len(df)} tracks!")
                    st.session_state.last_collection = df
                except Exception as e:
                    st.error(f"Error collecting data: {e}")
    
    with col2:
        if st.button("Collect Playlist Data Only"):
            with st.spinner("Collecting playlist data..."):
                try:
                    df = data_collection.collect_playlist_data(sp)
                    st.success(f"Successfully collected {len(df)} playlist tracks!")
                except Exception as e:
                    st.error(f"Error collecting playlist data: {e}")

def show_view_data():
    """Show collected data."""
    st.header("📋 View Data")
    
    # Check for saved files
    try:
        user_tracks_df = pd.read_csv(config.USER_TRACKS_CSV, index_col=0)
        st.success(f"Found user tracks data: {len(user_tracks_df)} tracks")
        
        # Show basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tracks", len(user_tracks_df))
        with col2:
            tracks_with_features = user_tracks_df.dropna(subset=['danceability']).shape[0]
            st.metric("Tracks with Features", tracks_with_features)
        with col3:
            unique_playlists = user_tracks_df['Playlist Name'].nunique()
            st.metric("Unique Playlists", unique_playlists)
        
        # Show data preview
        st.subheader("Data Preview")
        st.dataframe(user_tracks_df.head(10))
        
        # Download button
        csv = user_tracks_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="spotify_user_tracks.csv",
            mime="text/csv"
        )
        
    except FileNotFoundError:
        st.warning("No data found. Please collect data first.")
    except Exception as e:
        st.error(f"Error loading data: {e}")

def show_debug_tools(sp):
    """Show debugging tools."""
    st.header("🔧 Debug Tools")
    
    st.subheader("Test Individual Song")
    if st.button("Test Danger Zone"):
        # Test the find_recco_id_from_artist function directly
        recco_id = reccobeats_utils.find_recco_id_from_artist(
            spotify_id="test",
            artist_name="Kenny Loggins", 
            track_name="Danger Zone"
        )
        if recco_id:
            st.success(f"Found Reccobeats ID: {recco_id}")
            # Get features
            features = reccobeats_utils.get_features_from_reccoid(recco_id, "test")
            if features is not None:
                st.write("Audio Features:", features)
            else:
                st.error("Could not fetch features")
        else:
            st.error("Could not find Reccobeats ID for Danger Zone")
    
    st.subheader("Cache Information")
    try:
        cache_df = data_collection.load_features_cache()
        if not cache_df.empty:
            st.write(f"Cache contains {len(cache_df)} tracks")
            st.dataframe(cache_df.head(5))
        else:
            st.write("Cache is empty")
    except Exception as e:
        st.error(f"Error loading cache: {e}")

def show_settings(user_cache):
    """Show settings and configuration."""
    st.header("⚙️ Settings")
    
    st.subheader("Configuration")
    st.write(f"Client ID: {config.CLIENT_ID[:10]}..." if config.CLIENT_ID else "Not set")
    st.write(f"Redirect URI: {config.REDIRECT_URI}")
    st.write(f"Scope: {config.SCOPE}")
    
    st.subheader("Cache Management")
    if st.button("Clear Features Cache"):
        try:
            data_collection.save_features_cache(pd.DataFrame())
            st.success("Cache cleared successfully!")
        except Exception as e:
            st.error(f"Error clearing cache: {e}")

if __name__ == "__main__":
    main()
