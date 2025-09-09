#!/usr/bin/env python3
"""
Scheduled Model Retraining Script

This script can be run during scheduled downtimes to retrain the model
with accumulated user tracks. This improves long-term recommendation quality
without impacting user experience.

Usage:
    python scheduled_retraining.py [--force] [--check-only]
    
Options:
    --force: Force retraining even if no new tracks are detected
    --check-only: Only check if retraining is needed, don't actually retrain
"""

import os
import sys
import pandas as pd
import argparse
from datetime import datetime, timedelta
import json

def check_retraining_needed():
    """
    Check if model retraining is needed based on:
    1. New user tracks added since last retraining
    2. Time since last retraining
    3. Dataset size changes
    """
    retraining_info = {
        'needed': False,
        'reason': '',
        'new_tracks': 0,
        'days_since_retraining': 0,
        'dataset_size': 0
    }
    
    try:
        # Check if training dataset exists
        if not os.path.exists("Final_training_dataset.csv"):
            retraining_info['needed'] = True
            retraining_info['reason'] = "Training dataset not found"
            return retraining_info
        
        # Load current training dataset
        current_df = pd.read_csv("Final_training_dataset.csv")
        retraining_info['dataset_size'] = len(current_df)
        
        # Check last retraining timestamp
        last_retraining_file = "last_retraining.json"
        if os.path.exists(last_retraining_file):
            with open(last_retraining_file, 'r') as f:
                last_retraining = json.load(f)
            
            last_retraining_date = datetime.fromisoformat(last_retraining['timestamp'])
            days_since = (datetime.now() - last_retraining_date).days
            retraining_info['days_since_retraining'] = days_since
            
            # Check if dataset has grown significantly
            last_size = last_retraining.get('dataset_size', 0)
            new_tracks = len(current_df) - last_size
            retraining_info['new_tracks'] = new_tracks
            
            # Determine if retraining is needed
            if days_since >= 7:  # Retrain weekly
                retraining_info['needed'] = True
                retraining_info['reason'] = f"Weekly retraining due (last: {days_since} days ago)"
            elif new_tracks >= 1000:  # Retrain if 1000+ new tracks
                retraining_info['needed'] = True
                retraining_info['reason'] = f"Significant dataset growth ({new_tracks} new tracks)"
            elif new_tracks >= 100 and days_since >= 3:  # Retrain if 100+ tracks and 3+ days
                retraining_info['needed'] = True
                retraining_info['reason'] = f"Moderate growth ({new_tracks} new tracks, {days_since} days)"
        else:
            # First time retraining
            retraining_info['needed'] = True
            retraining_info['reason'] = "First time retraining"
        
        return retraining_info
        
    except Exception as e:
        retraining_info['needed'] = True
        retraining_info['reason'] = f"Error checking retraining status: {e}"
        return retraining_info

def update_retraining_timestamp(dataset_size):
    """Update the last retraining timestamp"""
    retraining_info = {
        'timestamp': datetime.now().isoformat(),
        'dataset_size': dataset_size,
        'status': 'completed'
    }
    
    with open("last_retraining.json", 'w') as f:
        json.dump(retraining_info, f, indent=2)

def retrain_model():
    """Retrain the model using the existing retrain_model.py script"""
    try:
        import subprocess
        result = subprocess.run([sys.executable, "retrain_model.py"], 
                              capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        if result.returncode == 0:
            print("✅ Model retraining completed successfully!")
            return True
        else:
            print(f"❌ Model retraining failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Model retraining timed out (30 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error during model retraining: {e}")
        return False

def main():
    """Main scheduled retraining function"""
    parser = argparse.ArgumentParser(description='Scheduled model retraining')
    parser.add_argument('--force', action='store_true', 
                       help='Force retraining even if not needed')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check if retraining is needed, don\'t retrain')
    
    args = parser.parse_args()
    
    print("🔄 Scheduled Model Retraining Check")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if retraining is needed
    retraining_info = check_retraining_needed()
    
    print(f"\n📊 Retraining Status:")
    print(f"   Dataset Size: {retraining_info['dataset_size']:,} tracks")
    print(f"   New Tracks: {retraining_info['new_tracks']:,}")
    print(f"   Days Since Last Retraining: {retraining_info['days_since_retraining']}")
    print(f"   Retraining Needed: {'Yes' if retraining_info['needed'] else 'No'}")
    print(f"   Reason: {retraining_info['reason']}")
    
    if args.check_only:
        print("\n✅ Check completed. Use --force to retrain if needed.")
        return retraining_info['needed']
    
    # Determine if we should retrain
    should_retrain = retraining_info['needed'] or args.force
    
    if not should_retrain:
        print("\n✅ No retraining needed at this time.")
        return True
    
    if args.force:
        print("\n🔄 Force retraining requested...")
    else:
        print(f"\n🔄 Retraining needed: {retraining_info['reason']}")
    
    # Perform retraining
    print("\n🔄 Starting model retraining...")
    success = retrain_model()
    
    if success:
        # Update retraining timestamp
        update_retraining_timestamp(retraining_info['dataset_size'])
        print("\n🎉 Scheduled retraining completed successfully!")
        print("💡 Model has been updated with latest user tracks and dataset expansions.")
    else:
        print("\n❌ Scheduled retraining failed!")
        print("💡 Check the error messages above and try again later.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
