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
    Applies a telephonic bandpass filter (300Hz - 3400Hz) to suppress 
    background/out-of-band noise for messy real-world audio inputs.
    """
    try:
        data, samplerate = librosa.load(audio_path, sr=None)
        
        # Apply a simple bandpass filter via FFT masking or librosa effects
        # Standard telephonic frequency range: 300 Hz to 3400 Hz
        stft = librosa.stft(data)
        frequencies = librosa.fft_frequencies(sr=samplerate)
        
        # Zero out frequencies outside telephonic range
        bandpass_mask = (frequencies >= 300) & (frequencies <= 3400)
        # Extend mask across time frames
        stft_filtered = stft * bandpass_mask[:, np.newaxis]
        
        filtered_data = librosa.istft(stft_filtered)
        
        output_path = "filtered_audio.wav"
        sf.write(output_path, filtered_data, samplerate)
        return output_path
    except Exception as e:
        # Fallback to original audio if filtering encounters an issue
        return audio_path

def triage_assistant(audio):
    if audio is None:
        return "Please provide a voice input describing the emergency."
    
    # Preprocess audio with the noise filter
    processed_audio = telephonic_noise_filter(audio)
    
    # Transcribe audio using Whisper on Groq
    try:
        with open(processed_audio, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(processed_audio, file.read()),
                model="whisper-large-v3",
                prompt="Emergency medical triage. Handles Hindi, Kannada, Telugu, Marathi, and code-mixed speech."
            )
        user_query = transcription.text
    except Exception as e:
        user_query = "Error processing audio."

    # Triage reasoning prompt supporting code-mixed inputs and native scripts
    system_prompt = (
        "You are Sahaya, an expert multilingual medical triage assistant for rural India "
        "(designed for farmers, kirana owners, and healthcare workers). "
        "You understand code-mixed inputs (e.g., Hindi mixed with English, Kannada mixed with Hindi) "
        "and regional languages. "
        "Detect the core emergency (e.g., snakebite, chest pain, pesticide poisoning, trauma) "
        "and provide clear, structured, life-saving first-aid instructions and the emergency helpline number 108 "
        "in the appropriate script matching the user's language."
    )

    try:
        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User speech transcript (code-mixed/regional): {user_query}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        response_text = chat_completion.choices[0].message.content
    except Exception as e:
        response_text = "⚠️ Service temporarily busy. Please call emergency helpline **108** immediately."

    return f"**Detected Input:** {user_query}\n\n--- muito ---\n\n{response_text}"

# Build Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Sahaya: Multilingual Emergency Triage Assistant")
    gr.Markdown("Speak naturally in your preferred language or code-mixed dialects (e.g., Kannada, Hindi, Telugu, Marathi). Designed for rural first-aid and telephonic support.")
    
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record Emergency Voice Note")
    
    submit_btn = gr.Button("Get Triage Guidance", variant="primary")
    output_display = gr.Markdown(label="Emergency Response & Action Steps")
    
    submit_btn.click(fn=triage_assistant, inputs=audio_input, outputs=output_display)

if __name__ == "__main__":
    demo.launch()
