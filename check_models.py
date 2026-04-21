import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyAvd1QbHIq17yJ3tkFgEX-xezPtt__sw1E")

print("Modelos disponibles para tu API Key:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"-> {m.name}")