"""Test script for checkpoint saving API.

This script demonstrates how to interact with the checkpoint saving endpoints.
Run this while training is active in the web app.
"""

import requests
import json
import time


def check_training_status():
    """Check if training is running."""
    try:
        response = requests.get('http://localhost:5002/status')
        status = response.json()
        print("\nTraining Status:")
        print(f"  Running: {status['running']}")
        print(f"  Mode: {status['mode']}")
        print(f"  Epoch: {status['epoch']}")
        print(f"  Batch: {status['batch']}")
        print(f"  Loss: {status['metrics']['loss']:.4f}")
        print(f"  Accuracy: {status['metrics']['accuracy']:.2f}%")
        return status['running'] and status['mode'] == 'train'
    except Exception as e:
        print(f"Error checking status: {e}")
        return False


def save_checkpoint(name=None):
    """Request a checkpoint save."""
    try:
        data = {}
        if name:
            data['name'] = name
        
        response = requests.post('http://localhost:5002/save_checkpoint',
                               json=data)
        result = response.json()
        
        if response.status_code == 200:
            print(f"\n✓ Checkpoint save requested: {result['checkpoint_name']}")
            print(f"  Status: {result['status']}")
            return True
        else:
            print(f"\n✗ Error: {result.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"\nError saving checkpoint: {e}")
        return False


def list_checkpoints():
    """List all saved checkpoints."""
    try:
        response = requests.get('http://localhost:5002/list_checkpoints')
        result = response.json()
        
        checkpoints = result['checkpoints']
        
        if not checkpoints:
            print("\nNo checkpoints found.")
            return
        
        print(f"\n{'='*100}")
        print(f"SAVED CHECKPOINTS ({len(checkpoints)} total)")
        print(f"{'='*100}")
        
        for i, cp in enumerate(checkpoints, 1):
            print(f"\n{i}. {cp['filename']}")
            print(f"   Size: {cp['size_mb']} MB")
            print(f"   Modified: {cp['modified']}")
            print(f"   Epoch: {cp.get('epoch', 'N/A')}, Batch: {cp.get('batch', 'N/A')}")
            print(f"   Task: {cp.get('task_type', 'N/A')}, Bottleneck: {cp.get('bottleneck_type', 'N/A')}")
            if cp.get('val_acc'):
                print(f"   Validation Accuracy: {cp['val_acc']:.2f}%")
            if cp.get('error'):
                print(f"   ⚠ Error: {cp['error']}")
        
        print(f"\n{'='*100}\n")
        return checkpoints
    except Exception as e:
        print(f"Error listing checkpoints: {e}")
        return []


def monitor_and_save_periodically(interval_seconds=60, checkpoint_prefix="auto_save"):
    """Monitor training and save checkpoints periodically."""
    print(f"\nMonitoring training and saving checkpoints every {interval_seconds} seconds...")
    print("Press Ctrl+C to stop monitoring.\n")
    
    save_count = 0
    
    try:
        while True:
            # Check if training is running
            if not check_training_status():
                print("\nTraining is not running. Waiting...")
                time.sleep(10)
                continue
            
            # Wait for interval
            print(f"\nWaiting {interval_seconds} seconds before next save...")
            time.sleep(interval_seconds)
            
            # Save checkpoint
            save_count += 1
            checkpoint_name = f"{checkpoint_prefix}_{save_count}"
            
            if save_checkpoint(checkpoint_name):
                print(f"✓ Saved checkpoint #{save_count}")
            else:
                print(f"✗ Failed to save checkpoint #{save_count}")
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        print(f"Total checkpoints saved: {save_count}")


def interactive_menu():
    """Interactive menu for checkpoint operations."""
    while True:
        print("\n" + "="*60)
        print("CHECKPOINT MANAGEMENT")
        print("="*60)
        print("\n1. Check training status")
        print("2. Save checkpoint with custom name")
        print("3. Save checkpoint with default name")
        print("4. List all checkpoints")
        print("5. Monitor and save periodically")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            check_training_status()
        
        elif choice == '2':
            name = input("Enter checkpoint name: ").strip()
            if name:
                save_checkpoint(name)
            else:
                print("No name provided. Cancelled.")
        
        elif choice == '3':
            save_checkpoint()
        
        elif choice == '4':
            list_checkpoints()
        
        elif choice == '5':
            try:
                interval = int(input("Enter save interval in seconds (default 60): ").strip() or "60")
                prefix = input("Enter checkpoint name prefix (default 'auto_save'): ").strip() or "auto_save"
                monitor_and_save_periodically(interval, prefix)
            except ValueError:
                print("Invalid interval. Using default (60 seconds).")
                monitor_and_save_periodically()
        
        elif choice == '6':
            print("\nExiting...")
            break
        
        else:
            print("\nInvalid choice. Please try again.")


def main():
    """Main function."""
    import sys
    
    if len(sys.argv) > 1:
        # Command-line mode
        command = sys.argv[1]
        
        if command == 'status':
            check_training_status()
        
        elif command == 'save':
            name = sys.argv[2] if len(sys.argv) > 2 else None
            save_checkpoint(name)
        
        elif command == 'list':
            list_checkpoints()
        
        elif command == 'monitor':
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
            prefix = sys.argv[3] if len(sys.argv) > 3 else "auto_save"
            monitor_and_save_periodically(interval, prefix)
        
        else:
            print("Usage:")
            print("  python test_checkpoint_api.py status")
            print("  python test_checkpoint_api.py save [name]")
            print("  python test_checkpoint_api.py list")
            print("  python test_checkpoint_api.py monitor [interval] [prefix]")
            print("\nOr run without arguments for interactive mode.")
    
    else:
        # Interactive mode
        interactive_menu()


if __name__ == '__main__':
    main()

