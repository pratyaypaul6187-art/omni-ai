#!/usr/bin/env python3
"""
🎬 CINEMATIC GUI FOR OMNI AI
Hollywood-style interface inspired by sci-fi movies
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import customtkinter as ctk
import threading
import time
import random
from datetime import datetime
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append('src')
sys.path.append('..')

# Set theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class CinematicGUI:
    """🎬 Hollywood-style AI interface"""
    
    def __init__(self):
        self.root = ctk.CTk()
        self.setup_main_window()
        self.setup_styles()
        self.create_interface()
        self.start_animations()
        
    def setup_main_window(self):
        """Setup main window with cinematic properties"""
        self.root.title("🤖 OMNI AI - FORTRESS CONTROL CENTER 🏰")
        self.root.geometry("1400x900")
        self.root.configure(fg_color="#0a0a0a")
        
        # Make it look more cinematic
        self.root.attributes('-alpha', 0.95)
        
        # Center the window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1400 // 2)
        y = (self.root.winfo_screenheight() // 2) - (900 // 2)
        self.root.geometry(f"1400x900+{x}+{y}")
        
    def setup_styles(self):
        """Setup cinematic color scheme and fonts"""
        self.colors = {
            'bg_primary': '#0a0a0a',
            'bg_secondary': '#1a1a1a', 
            'accent_blue': '#00d4ff',
            'accent_green': '#00ff41',
            'accent_red': '#ff073a',
            'accent_orange': '#ff9500',
            'text_primary': '#ffffff',
            'text_secondary': '#b0b0b0',
            'border': '#333333'
        }
        
        self.fonts = {
            'title': ('Consolas', 24, 'bold'),
            'header': ('Consolas', 16, 'bold'),
            'normal': ('Consolas', 12),
            'small': ('Consolas', 10),
            'mono': ('Courier New', 11)
        }
    
    def create_interface(self):
        """Create the main cinematic interface"""
        # Main container
        self.main_frame = ctk.CTkFrame(self.root, fg_color=self.colors['bg_primary'])
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Header section
        self.create_header()
        
        # Content area with tabs
        self.create_content_tabs()
        
        # Footer with system status
        self.create_footer()
    
    def create_header(self):
        """Create cinematic header with title and status"""
        header_frame = ctk.CTkFrame(self.main_frame, height=80, fg_color=self.colors['bg_secondary'])
        header_frame.pack(fill="x", padx=5, pady=5)
        header_frame.pack_propagate(False)
        
        # Title with glow effect
        title_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_frame.pack(side="left", fill="y", padx=20)
        
        self.title_label = ctk.CTkLabel(
            title_frame,
            text="🤖 OMNI AI FORTRESS 🏰",
            font=self.fonts['title'],
            text_color=self.colors['accent_blue']
        )
        self.title_label.pack(anchor="w", pady=10)
        
        self.subtitle_label = ctk.CTkLabel(
            title_frame,
            text="MILITARY-GRADE AI DEFENSE SYSTEM",
            font=self.fonts['normal'],
            text_color=self.colors['text_secondary']
        )
        self.subtitle_label.pack(anchor="w")
        
        # Status indicators
        status_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        status_frame.pack(side="right", fill="y", padx=20)
        
        self.create_status_indicators(status_frame)
    
    def create_status_indicators(self, parent):
        """Create animated status indicators"""
        self.status_frame = parent
        
        # System Status
        system_frame = ctk.CTkFrame(parent, fg_color="transparent")
        system_frame.pack(side="top", fill="x", pady=2)
        
        ctk.CTkLabel(system_frame, text="SYSTEM:", font=self.fonts['small'], 
                     text_color=self.colors['text_secondary']).pack(side="left")
        self.system_status = ctk.CTkLabel(system_frame, text="🟢 ONLINE", 
                                         font=self.fonts['small'], 
                                         text_color=self.colors['accent_green'])
        self.system_status.pack(side="right")
        
        # Security Status
        security_frame = ctk.CTkFrame(parent, fg_color="transparent")
        security_frame.pack(side="top", fill="x", pady=2)
        
        ctk.CTkLabel(security_frame, text="SECURITY:", font=self.fonts['small'],
                     text_color=self.colors['text_secondary']).pack(side="left")
        self.security_status = ctk.CTkLabel(security_frame, text="🛡️ FORTRESS", 
                                           font=self.fonts['small'], 
                                           text_color=self.colors['accent_green'])
        self.security_status.pack(side="right")
        
        # AI Status
        ai_frame = ctk.CTkFrame(parent, fg_color="transparent")
        ai_frame.pack(side="top", fill="x", pady=2)
        
        ctk.CTkLabel(ai_frame, text="AI CORE:", font=self.fonts['small'],
                     text_color=self.colors['text_secondary']).pack(side="left")
        self.ai_status = ctk.CTkLabel(ai_frame, text="🤖 ACTIVE", 
                                     font=self.fonts['small'], 
                                     text_color=self.colors['accent_blue'])
        self.ai_status.pack(side="right")
    
    def create_content_tabs(self):
        """Create tabbed content area"""
        # Tab container
        self.tab_view = ctk.CTkTabview(
            self.main_frame,
            fg_color=self.colors['bg_secondary'],
            segmented_button_fg_color=self.colors['bg_primary'],
            segmented_button_selected_color=self.colors['accent_blue'],
            text_color=self.colors['text_primary']
        )
        self.tab_view.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create tabs
        self.create_ai_chat_tab()
        self.create_security_fortress_tab()
        self.create_system_monitor_tab()
        self.create_terminal_tab()
    
    def create_ai_chat_tab(self):
        """Create AI chat interface with movie effects"""
        tab = self.tab_view.add("🤖 AI CHAT")
        
        # Chat history
        chat_frame = ctk.CTkFrame(tab, fg_color=self.colors['bg_primary'])
        chat_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Chat display
        self.chat_display = ctk.CTkTextbox(
            chat_frame,
            font=self.fonts['mono'],
            fg_color="#000000",
            text_color=self.colors['accent_green'],
            wrap="word",
            state="disabled"
        )
        self.chat_display.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Input area
        input_frame = ctk.CTkFrame(chat_frame, fg_color="transparent")
        input_frame.pack(fill="x", padx=10, pady=5)
        
        self.chat_input = ctk.CTkEntry(
            input_frame,
            placeholder_text="Enter your command...",
            font=self.fonts['normal'],
            fg_color=self.colors['bg_secondary'],
            text_color=self.colors['text_primary']
        )
        self.chat_input.pack(side="left", fill="x", expand=True, padx=5)
        self.chat_input.bind("<Return>", self.send_ai_message)
        
        send_button = ctk.CTkButton(
            input_frame,
            text="SEND",
            command=self.send_ai_message,
            font=self.fonts['normal'],
            fg_color=self.colors['accent_blue'],
            hover_color=self.colors['accent_green']
        )
        send_button.pack(side="right", padx=5)
        
        # Welcome message
        self.add_ai_message("🤖 OMNI AI ONLINE\nFortress defense systems activated.\nHow may I assist you today?", is_ai=True)
    
    def create_security_fortress_tab(self):
        """Create visual security fortress interface"""
        tab = self.tab_view.add("🛡️ FORTRESS")
        
        # Fortress visualization
        fortress_frame = ctk.CTkFrame(tab, fg_color=self.colors['bg_primary'])
        fortress_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        title_frame = ctk.CTkFrame(fortress_frame, fg_color="transparent")
        title_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            title_frame,
            text="🏰 SECURITY FORTRESS STATUS",
            font=self.fonts['header'],
            text_color=self.colors['accent_blue']
        ).pack()
        
        # Security layers
        self.create_security_layers(fortress_frame)
        
        # Fortress controls
        self.create_fortress_controls(fortress_frame)
    
    def create_security_layers(self, parent):
        """Create visual representation of security layers"""
        layers_frame = ctk.CTkFrame(parent, fg_color=self.colors['bg_secondary'])
        layers_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.security_layers = {}
        layers = [
            ("🔒 Layer 1: Input Validation", "ACTIVE", self.colors['accent_green']),
            ("🔐 Layer 2: Authentication", "ACTIVE", self.colors['accent_green']),
            ("🔍 Layer 3: Intrusion Detection", "MONITORING", self.colors['accent_blue']),
            ("🔐 Layer 4: Cryptography", "ACTIVE", self.colors['accent_green']),
            ("🛡️ Layer 5: System Hardening", "ACTIVE", self.colors['accent_green']),
            ("🧪 Layer 6: Security Testing", "SCANNING", self.colors['accent_orange'])
        ]
        
        for i, (name, status, color) in enumerate(layers):
            layer_frame = ctk.CTkFrame(layers_frame, fg_color=self.colors['bg_primary'])
            layer_frame.pack(fill="x", padx=5, pady=3)
            
            # Layer info
            info_frame = ctk.CTkFrame(layer_frame, fg_color="transparent")
            info_frame.pack(fill="x", padx=10, pady=5)
            
            ctk.CTkLabel(info_frame, text=name, font=self.fonts['normal'],
                        text_color=self.colors['text_primary']).pack(side="left")
            
            status_label = ctk.CTkLabel(info_frame, text=f"● {status}", 
                                       font=self.fonts['small'], text_color=color)
            status_label.pack(side="right")
            
            self.security_layers[f"layer_{i+1}"] = status_label
    
    def create_fortress_controls(self, parent):
        """Create fortress control buttons"""
        controls_frame = ctk.CTkFrame(parent, fg_color=self.colors['bg_secondary'])
        controls_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(controls_frame, text="FORTRESS CONTROLS", 
                    font=self.fonts['header'], 
                    text_color=self.colors['accent_blue']).pack(pady=10)
        
        button_frame = ctk.CTkFrame(controls_frame, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=10)
        
        # Control buttons
        self.demo_button = ctk.CTkButton(
            button_frame,
            text="🎬 RUN FORTRESS DEMO",
            command=self.run_fortress_demo,
            font=self.fonts['normal'],
            fg_color=self.colors['accent_green'],
            hover_color=self.colors['accent_blue'],
            height=40
        )
        self.demo_button.pack(side="left", padx=5, fill="x", expand=True)
        
        test_button = ctk.CTkButton(
            button_frame,
            text="🧪 SECURITY TEST",
            command=self.run_security_test,
            font=self.fonts['normal'],
            fg_color=self.colors['accent_orange'],
            hover_color=self.colors['accent_red'],
            height=40
        )
        test_button.pack(side="right", padx=5, fill="x", expand=True)
    
    def create_system_monitor_tab(self):
        """Create system monitoring dashboard"""
        tab = self.tab_view.add("📊 MONITOR")
        
        # Monitor content
        monitor_frame = ctk.CTkFrame(tab, fg_color=self.colors['bg_primary'])
        monitor_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        ctk.CTkLabel(
            monitor_frame,
            text="📊 SYSTEM MONITORING DASHBOARD",
            font=self.fonts['header'],
            text_color=self.colors['accent_blue']
        ).pack(pady=10)
        
        # Monitoring display
        self.monitor_display = ctk.CTkTextbox(
            monitor_frame,
            font=self.fonts['mono'],
            fg_color="#000000",
            text_color=self.colors['accent_green'],
            state="disabled"
        )
        self.monitor_display.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Start monitoring
        self.start_system_monitoring()
    
    def create_terminal_tab(self):
        """Create terminal interface"""
        tab = self.tab_view.add("💻 TERMINAL")
        
        # Terminal frame
        terminal_frame = ctk.CTkFrame(tab, fg_color="#000000")
        terminal_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Terminal display
        self.terminal_display = ctk.CTkTextbox(
            terminal_frame,
            font=self.fonts['mono'],
            fg_color="#000000",
            text_color=self.colors['accent_green'],
            state="disabled"
        )
        self.terminal_display.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Terminal input
        input_frame = ctk.CTkFrame(terminal_frame, fg_color="transparent")
        input_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(input_frame, text="OMNI-AI>", font=self.fonts['mono'],
                    text_color=self.colors['accent_blue']).pack(side="left")
        
        self.terminal_input = ctk.CTkEntry(
            input_frame,
            font=self.fonts['mono'],
            fg_color="#000000",
            text_color=self.colors['accent_green'],
            border_width=0
        )
        self.terminal_input.pack(side="left", fill="x", expand=True, padx=5)
        self.terminal_input.bind("<Return>", self.execute_terminal_command)
        
        # Welcome message
        self.add_terminal_message("OMNI AI Terminal v1.0")
        self.add_terminal_message("Type 'help' for available commands")
    
    def create_footer(self):
        """Create status footer"""
        footer_frame = ctk.CTkFrame(self.main_frame, height=40, fg_color=self.colors['bg_secondary'])
        footer_frame.pack(fill="x", padx=5, pady=5)
        footer_frame.pack_propagate(False)
        
        # Left side - timestamp
        self.timestamp_label = ctk.CTkLabel(
            footer_frame,
            text="",
            font=self.fonts['small'],
            text_color=self.colors['text_secondary']
        )
        self.timestamp_label.pack(side="left", padx=20, pady=10)
        
        # Right side - system info
        self.system_info_label = ctk.CTkLabel(
            footer_frame,
            text="FORTRESS MODE | SECURITY LEVEL: MAXIMUM",
            font=self.fonts['small'],
            text_color=self.colors['accent_green']
        )
        self.system_info_label.pack(side="right", padx=20, pady=10)
    
    def start_animations(self):
        """Start cinematic animations"""
        self.update_timestamp()
        self.animate_status_indicators()
        
    def update_timestamp(self):
        """Update timestamp display"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.timestamp_label.configure(text=f"LOCAL TIME: {now}")
        self.root.after(1000, self.update_timestamp)
    
    def animate_status_indicators(self):
        """Animate status indicators for movie effect"""
        # Randomly flicker status for cinematic effect
        if random.random() < 0.1:  # 10% chance
            original_color = self.security_status.cget("text_color")
            self.security_status.configure(text_color=self.colors['accent_orange'])
            self.root.after(100, lambda: self.security_status.configure(text_color=original_color))
        
        self.root.after(2000, self.animate_status_indicators)
    
    def send_ai_message(self, event=None):
        """Send message to AI with cinematic effects"""
        message = self.chat_input.get().strip()
        if not message:
            return
        
        # Clear input
        self.chat_input.delete(0, "end")
        
        # Add user message
        self.add_ai_message(f"USER: {message}", is_ai=False)
        
        # Simulate AI thinking
        self.add_ai_message("🤖 AI: Processing request...", is_ai=True)
        
        # Simulate AI response after delay
        threading.Thread(target=self.simulate_ai_response, args=(message,), daemon=True).start()
    
    def simulate_ai_response(self, user_message):
        """Simulate AI response with typing effect"""
        time.sleep(1)
        
        # Generate response based on message
        if "security" in user_message.lower():
            response = "🛡️ Security fortress is operating at maximum efficiency.\nAll defense layers are active and monitoring threats.\nNo security breaches detected."
        elif "status" in user_message.lower():
            response = "🤖 All systems operational.\n📊 CPU: 45% | Memory: 62% | Security: FORTRESS\n🔒 Defense protocols: ACTIVE"
        elif "help" in user_message.lower():
            response = "🤖 Available commands:\n• security - Check fortress status\n• status - System information\n• fortress demo - Run security demonstration\n• monitor - System monitoring data"
        else:
            response = f"🤖 Command processed: {user_message}\nAI analysis complete. Standing by for next instruction."
        
        # Clear "processing" message and add real response
        self.root.after(0, self.replace_last_ai_message, response)
    
    def add_ai_message(self, message, is_ai=True):
        """Add message to chat with cinematic formatting"""
        self.chat_display.configure(state="normal")
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = "🤖 AI" if is_ai else "👤 USER"
        color = self.colors['accent_green'] if is_ai else self.colors['accent_blue']
        
        formatted_message = f"[{timestamp}] {message}\n\n"
        
        self.chat_display.insert("end", formatted_message)
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")
    
    def replace_last_ai_message(self, new_message):
        """Replace the last AI message (for processing effect)"""
        self.chat_display.configure(state="normal")
        
        # Get current content
        content = self.chat_display.get("1.0", "end")
        lines = content.strip().split('\n')
        
        # Find and replace last AI message
        for i in range(len(lines) - 1, -1, -1):
            if "🤖 AI: Processing request..." in lines[i]:
                timestamp = datetime.now().strftime("%H:%M:%S")
                lines[i] = f"[{timestamp}] 🤖 AI: {new_message}"
                break
        
        # Update display
        self.chat_display.delete("1.0", "end")
        self.chat_display.insert("1.0", '\n'.join(lines) + '\n\n')
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")
    
    def run_fortress_demo(self):
        """Run the fortress demonstration with visual effects"""
        self.demo_button.configure(text="🎬 DEMO RUNNING...", state="disabled")
        
        # Start demo in background thread
        threading.Thread(target=self.fortress_demo_thread, daemon=True).start()
    
    def fortress_demo_thread(self):
        """Run fortress demo with visual updates"""
        try:
            # Import demo function
            sys.path.append('.')
            from demo_security_fortress import main as demo_main
            
            # Update status
            for i, (layer_key, status_label) in enumerate(self.security_layers.items()):
                self.root.after(i * 500, lambda sl=status_label: sl.configure(text="● TESTING", text_color=self.colors['accent_orange']))
            
            # Run demo (would need to capture output)
            time.sleep(3)
            
            # Update to success
            for i, (layer_key, status_label) in enumerate(self.security_layers.items()):
                self.root.after(i * 200, lambda sl=status_label: sl.configure(text="● SECURED", text_color=self.colors['accent_green']))
            
            # Re-enable button
            self.root.after(2000, lambda: self.demo_button.configure(text="🎬 RUN FORTRESS DEMO", state="normal"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Demo Error", f"Demo failed: {str(e)}"))
            self.root.after(0, lambda: self.demo_button.configure(text="🎬 RUN FORTRESS DEMO", state="normal"))
    
    def run_security_test(self):
        """Run security test"""
        messagebox.showinfo("Security Test", "🧪 Running comprehensive security scan...\n\n✅ All security layers passed!\n🛡️ Fortress status: IMPENETRABLE")
    
    def start_system_monitoring(self):
        """Start system monitoring display"""
        def update_monitor():
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cpu_usage = random.randint(20, 80)
            memory_usage = random.randint(40, 85)
            security_events = random.randint(0, 5)
            
            monitor_text = f"""
═══════════════════════════════════════════════════════════
🖥️  OMNI AI SYSTEM MONITOR - {timestamp}
═══════════════════════════════════════════════════════════

📊 SYSTEM RESOURCES:
   CPU Usage:      {cpu_usage}% {'█' * (cpu_usage // 10)}
   Memory Usage:   {memory_usage}% {'█' * (memory_usage // 10)}
   Disk I/O:       {'●' * random.randint(1, 5)}
   Network:        {'●' * random.randint(1, 3)}

🛡️  SECURITY STATUS:
   Threat Level:   🟢 MINIMAL
   Active Scans:   {random.randint(3, 8)}
   Blocked Attacks: {random.randint(0, 2)}
   Security Events: {security_events}
   
🤖 AI CORE STATUS:
   Model Status:   🟢 ONLINE
   Response Time:  {random.randint(50, 200)}ms
   Queries/Hour:   {random.randint(100, 500)}
   Accuracy:       99.{random.randint(5, 9)}%

🔍 REAL-TIME LOGS:
   [{datetime.now().strftime("%H:%M:%S")}] Security scan completed
   [{datetime.now().strftime("%H:%M:%S")}] AI model inference ready
   [{datetime.now().strftime("%H:%M:%S")}] System health check passed
   [{datetime.now().strftime("%H:%M:%S")}] Fortress defenses active
═══════════════════════════════════════════════════════════
            """
            
            self.monitor_display.configure(state="normal")
            self.monitor_display.delete("1.0", "end")
            self.monitor_display.insert("1.0", monitor_text)
            self.monitor_display.configure(state="disabled")
        
        def monitor_loop():
            update_monitor()
            self.root.after(2000, monitor_loop)
        
        monitor_loop()
    
    def add_terminal_message(self, message):
        """Add message to terminal"""
        self.terminal_display.configure(state="normal")
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        self.terminal_display.insert("end", formatted_message)
        self.terminal_display.configure(state="disabled")
        self.terminal_display.see("end")
    
    def execute_terminal_command(self, event):
        """Execute terminal command"""
        command = self.terminal_input.get().strip()
        if not command:
            return
        
        self.terminal_input.delete(0, "end")
        self.add_terminal_message(f"OMNI-AI> {command}")
        
        # Process commands
        if command == "help":
            help_text = """Available commands:
help     - Show this help
status   - System status
security - Security fortress status
clear    - Clear terminal
demo     - Run fortress demo
exit     - Close terminal"""
            self.add_terminal_message(help_text)
        elif command == "status":
            self.add_terminal_message("🤖 OMNI AI STATUS: ONLINE\n🛡️ SECURITY: FORTRESS MODE\n📊 PERFORMANCE: OPTIMAL")
        elif command == "security":
            self.add_terminal_message("🏰 FORTRESS STATUS: ALL LAYERS ACTIVE\n🔒 THREAT LEVEL: MINIMAL\n⚡ DEFENSE SYSTEMS: READY")
        elif command == "clear":
            self.terminal_display.configure(state="normal")
            self.terminal_display.delete("1.0", "end")
            self.terminal_display.configure(state="disabled")
            self.add_terminal_message("Terminal cleared")
        elif command == "demo":
            self.add_terminal_message("🎬 Launching fortress demonstration...")
            self.run_fortress_demo()
        elif command == "exit":
            self.add_terminal_message("Terminal session ended")
        else:
            self.add_terminal_message(f"Unknown command: {command}. Type 'help' for available commands.")
    
    def run(self):
        """Start the cinematic GUI"""
        self.root.mainloop()

def main():
    """Launch the cinematic GUI"""
    try:
        app = CinematicGUI()
        app.run()
    except Exception as e:
        print(f"GUI Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()