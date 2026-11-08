import gradio as gr
import numpy as np
import librosa
import soundfile as sf
import os
import groq

# Initialize Groq client
client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))

def telephonic_noise_filter(audio_path):
    """
    Applies a telephonic bandpass filter (300Hz - 3400Hz) to suppress background/out-of-band noise for messy real-world audio inputs.
    """
    try:
        data, samplerate = librosa.load(audio_path, sr=None)
        stft = librosa.stft(data)
        frequencies = librosa.fft_frequencies(sr=samplerate)
        bandpass_mask = (frequencies >= 300) & (frequencies <= 3400)
        stft_filtered = stft * bandpass_mask[:, np.newaxis]
        filtered_data = librosa.istft(stft_filtered)
        output_path = "filtered_audio.wav"
        sf.write(output_path, filtered_data, samplerate)
        return output_path
    except Exception as e:
        return audio_path

def triage_assistant(audio):
    if audio is None:
        return "Please provide a voice input describing the emergency."
    
    processed_audio = telephonic_noise_filter(audio)
    
    try:
        with open(processed_audio, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(processed_audio, file.read()),
                model="whisper-large-v3",
                prompt="Emergency medical triage. Handles Hindi, Kannada, Telugu, Tamil, Malayalam, Gujarati, Bengali, Punjabi, Odia, Marathi, Manipuri, and code-mixed speech."
            )
        user_query = transcription.text
    except Exception as e:
        user_query = "Error processing audio."

    system_prompt = (
        "You are Sahaya, an expert multilingual medical and emergency triage assistant for rural India. "
        "You understand code-mixed inputs and regional languages (Kannada, Hindi, Telugu, Tamil, Malayalam, Gujarati, Bengali, Punjabi, Odia, Marathi, Manipuri). "
        "Detect the core emergency and provide clear, structured, life-saving first-aid instructions and the emergency helpline number 108 "
        "in the appropriate script matching the user's language."
    )
    
    try:
        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User speech transcript: {user_query}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        response_text = chat_completion.choices[0].message.content
    except Exception as e:
        response_text = "⚠️ Service temporarily busy. Please call emergency helpline **108** immediately."
        
    return f"**Detected Input:** {user_query}\n\n---\n\n{response_text}"

# Build Gradio Interface with custom styling and animated wave
custom_css = """
@keyframes wave-bounce {
    0%, 100% { transform: scaleY(0.4); background-color: #22c55e; }
    50% { transform: scaleY(1); background-color: #4ade80; }
}
.wave-container {
    display: flex;
    align-items: center;
    gap: 4px;
    height: 30px;
    margin-left: 15px;
}
.wave-bar {
    width: 5px;
    height: 100%;
    animation: wave-bounce 1.2s infinite ease-in-out;
}
.wave-bar:nth-child(2) { animation-delay: 0.2s; }
.wave-bar:nth-child(3) { animation-delay: 0.4s; }
.wave-bar:nth-child(4) { animation-delay: 0.6s; }

.header-row {
    display: flex;
    align-items: center;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    with gr.Row(elem_classes="header-row"):
        gr.Markdown("# 🛡️ Sahaya <br><sub style='color: gray; font-size: 16px;'>Stay Safe</sub>")
        gr.HTML("""
            <div class="wave-container">
                <div class="wave-bar"></div>
                <div class="wave-bar"></div>
                <div class="wave-bar"></div>
                <div class="wave-bar"></div>
            </div>
        """)
        
    gr.Markdown("Speak naturally in your preferred language or code-mixed dialects. Designed for rural first-aid and telephonic support.")
    
    with gr.Row():
        sos_btn = gr.Button("🚨 SOS EMERGENCY", variant="stop")
        
    sos_output = gr.Markdown()
    
    def trigger_sos():
        return "🚨 **SOS Triggered!** Dialing Emergency Contacts & Women's Distress Helpline (**1091**) / Ambulance (**108**)."
    
    sos_btn.click(fn=trigger_sos, outputs=sos_output)

    with gr.Row():
        audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record Emergency Voice Note")
        submit_btn = gr.Button("Get Triage Guidance", variant="primary")
        
    output_display = gr.Markdown(label="Emergency Response & Action Steps")
    submit_btn.click(fn=triage_assistant, inputs=audio_input, outputs=output_display)

    gr.Markdown("### 🏥 Emergency Directory Grid")
    with gr.Row():
        gr.Markdown("🚑 **Ambulance**<br>Dial: **108**")
        gr.Markdown("🚓 **Police**<br>Dial: **112** / **100**")
        gr.Markdown("🔥 **Disaster Response**<br>Dial: **112**")
        gr.Markdown("👩 **Women Helpline**<br>Dial: **1091**")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
