#!/usr/bin/env python3
"""
🚀 OMNI AI CINEMATIC GUI LAUNCHER
Quick launcher for the Hollywood-style interface
"""

import sys
import os
from pathlib import Path

def main():
    """Launch the cinematic GUI"""
    print("🎬 Launching Omni AI Cinematic Interface...")
    print("🏰 Loading fortress control systems...")
    
    try:
        # Add gui directory to path
        gui_dir = Path(__file__).parent / "gui"
        sys.path.append(str(gui_dir))
        
        # Import and run GUI
        from cinematic_gui import CinematicGUI
        
        print("✅ GUI modules loaded successfully")
        print("🎭 Initializing Hollywood-style interface...")
        print("\n🤖 OMNI AI FORTRESS CONTROL CENTER STARTING...\n")
        
        # Create and run the GUI
        app = CinematicGUI()
        app.run()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("📦 Please ensure all dependencies are installed:")
        print("   pip install customtkinter")
    except Exception as e:
        print(f"❌ GUI Launch Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()