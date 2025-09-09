#!/usr/bin/env python3
"""
Setup Script for Scheduled Model Retraining

This script helps set up automated retraining during scheduled downtimes.
It creates cron jobs or scheduled tasks to run model retraining periodically.

Usage:
    python setup_scheduled_retraining.py [--frequency daily|weekly|monthly]
"""

import os
import sys
import argparse
import platform
from datetime import datetime

def create_cron_job(frequency="weekly"):
    """
    Create a cron job for scheduled retraining
    """
    if platform.system() == "Windows":
        print("❌ Cron jobs are not available on Windows.")
        print("💡 Use Windows Task Scheduler instead:")
        print("   1. Open Task Scheduler")
        print("   2. Create Basic Task")
        print("   3. Set trigger to your desired frequency")
        print("   4. Set action to run: python scheduled_retraining.py")
        return False
    
    # Determine cron schedule based on frequency
    cron_schedules = {
        "daily": "0 2 * * *",      # 2 AM daily
        "weekly": "0 2 * * 0",     # 2 AM every Sunday
        "monthly": "0 2 1 * *"     # 2 AM on 1st of every month
    }
    
    if frequency not in cron_schedules:
        print(f"❌ Invalid frequency: {frequency}. Use: daily, weekly, or monthly")
        return False
    
    cron_schedule = cron_schedules[frequency]
    
    # Get current directory
    current_dir = os.getcwd()
    python_path = sys.executable
    
    # Create cron job entry
    cron_entry = f"{cron_schedule} cd {current_dir} && {python_path} scheduled_retraining.py >> retraining.log 2>&1"
    
    print("🔄 Setting up scheduled retraining...")
    print(f"📅 Frequency: {frequency}")
    print(f"⏰ Schedule: {cron_schedule}")
    print(f"📁 Directory: {current_dir}")
    print(f"🐍 Python: {python_path}")
    
    # Add to crontab
    try:
        import subprocess
        
        # Get current crontab
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
        current_crontab = result.stdout if result.returncode == 0 else ""
        
        # Check if our job already exists
        if "scheduled_retraining.py" in current_crontab:
            print("⚠️ Scheduled retraining job already exists in crontab")
            response = input("Do you want to replace it? (y/N): ")
            if response.lower() != 'y':
                print("❌ Setup cancelled")
                return False
            
            # Remove existing entry
            lines = current_crontab.split('\n')
            lines = [line for line in lines if "scheduled_retraining.py" not in line]
            current_crontab = '\n'.join(lines)
        
        # Add new entry
        new_crontab = current_crontab.rstrip() + '\n' + cron_entry + '\n'
        
        # Write to temporary file
        with open('temp_crontab', 'w') as f:
            f.write(new_crontab)
        
        # Install new crontab
        result = subprocess.run(['crontab', 'temp_crontab'], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Scheduled retraining job added successfully!")
            print(f"📝 Cron entry: {cron_entry}")
            print("📋 To view your crontab: crontab -l")
            print("🗑️ To remove: crontab -e (then delete the line)")
            
            # Clean up
            os.remove('temp_crontab')
            return True
        else:
            print(f"❌ Failed to add cron job: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error setting up cron job: {e}")
        return False

def create_windows_task(frequency="weekly"):
    """
    Create a Windows scheduled task for retraining
    """
    if platform.system() != "Windows":
        print("❌ Windows Task Scheduler is only available on Windows")
        return False
    
    print("🔄 Setting up Windows scheduled task...")
    print("📋 Manual setup required for Windows Task Scheduler:")
    print()
    print("1. Open Task Scheduler (taskschd.msc)")
    print("2. Click 'Create Basic Task'")
    print("3. Name: 'Music Model Retraining'")
    print("4. Description: 'Retrain music recommendation model'")
    print()
    
    # Determine trigger based on frequency
    triggers = {
        "daily": "Daily at 2:00 AM",
        "weekly": "Weekly on Sunday at 2:00 AM", 
        "monthly": "Monthly on 1st at 2:00 AM"
    }
    
    print(f"5. Trigger: {triggers.get(frequency, 'Weekly on Sunday at 2:00 AM')}")
    print()
    print("6. Action: Start a program")
    print(f"   Program: {sys.executable}")
    print(f"   Arguments: scheduled_retraining.py")
    print(f"   Start in: {os.getcwd()}")
    print()
    print("7. Check 'Open the Properties dialog' and click Finish")
    print("8. In Properties:")
    print("   - General tab: Check 'Run whether user is logged on or not'")
    print("   - Settings tab: Check 'Allow task to be run on demand'")
    print("   - Actions tab: Verify the program path is correct")
    print()
    print("✅ Task will run automatically during scheduled downtimes")
    
    return True

def test_scheduled_retraining():
    """
    Test the scheduled retraining script
    """
    print("🧪 Testing scheduled retraining script...")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, "scheduled_retraining.py", "--check-only"], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Scheduled retraining script test passed!")
            print("📋 Output:")
            print(result.stdout)
            return True
        else:
            print("❌ Scheduled retraining script test failed!")
            print("📋 Error:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error testing scheduled retraining: {e}")
        return False

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description='Setup scheduled model retraining')
    parser.add_argument('--frequency', choices=['daily', 'weekly', 'monthly'], 
                       default='weekly', help='Retraining frequency')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test the scheduled retraining script')
    
    args = parser.parse_args()
    
    print("🔄 Scheduled Model Retraining Setup")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Platform: {platform.system()}")
    print(f"Frequency: {args.frequency}")
    
    if args.test_only:
        return test_scheduled_retraining()
    
    # Test the script first
    print("\n1. Testing scheduled retraining script...")
    if not test_scheduled_retraining():
        print("❌ Setup aborted due to test failure")
        return False
    
    # Set up scheduling
    print(f"\n2. Setting up {args.frequency} retraining schedule...")
    
    if platform.system() == "Windows":
        success = create_windows_task(args.frequency)
    else:
        success = create_cron_job(args.frequency)
    
    if success:
        print("\n🎉 Scheduled retraining setup completed!")
        print("💡 Your model will be automatically retrained during scheduled downtimes")
        print("📊 Check retraining.log for execution logs")
        print("🔧 To run manually: python scheduled_retraining.py")
    else:
        print("\n❌ Scheduled retraining setup failed!")
        print("💡 You can still run manual retraining: python scheduled_retraining.py")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
