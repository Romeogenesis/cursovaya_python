"""
Main entry point for the Metrological Equipment Analysis System.
Run with: streamlit run main.py
"""
import streamlit as st
from ui.dashboard import main as dashboard_main

if __name__ == "__main__":
    dashboard_main()