#!/usr/bin/env python3
"""
Demo Runner

Simple script to run the web chatbot demo.
"""

import subprocess
import sys
import os

def print_banner():
    """Print banner for demo."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════════╗
║                           🤖 Simple Chatbot Demo                                ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║ A minimal conversational AI assistant powered by LangGraph and Ollama           ║
║                                                                                  ║
║ Features:                                                                        ║
║ • Clean web interface                                                           ║
║ • Local LLM integration via Ollama                                             ║
║ • Simple conversation flow                                                      ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_ollama():
    """Basic check for Ollama."""
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/tags"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✅ Ollama server is running")
            return True
        else:
            print("❌ Ollama server not accessible at localhost:11434")
            print("   Please make sure Ollama is running: ollama serve")
            return False
    except:
        print("⚠️  Could not check Ollama connection")
        return True

def run_demo():
    """Run the web demo."""
    print("\n🚀 Starting web interface...")
    print("📍 Open your browser to: http://localhost:8000")
    print("   (Press Ctrl+C to stop)")
    
    try:
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Demo stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo failed: {e}")

def main():
    """Main function."""
    print_banner()
    
    if not check_ollama():
        print("\nPlease start Ollama and try again.")
        return
    
    input("\nPress Enter to start the demo (or Ctrl+C to cancel)...")
    run_demo()

if __name__ == "__main__":
    main()
