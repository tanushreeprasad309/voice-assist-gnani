import gradio as gr
import os
import groq

# Initialize Groq client
groq_api_key = os.environ.get("GROQ_API_KEY", "")
client = groq.Groq(api_key=groq_api_key) if groq_api_key else None

LANGUAGE_MAPPING = {
    "Auto-Detect / Code-Mixed": None,
    "Kannada (ಕನ್ನಡ)": "kn",
    "Hindi (हिंदी)": "hi",
    "Telugu (తెలుగు)": "te",
    "Tamil (தமிழ்)": "ta",
    "Malayalam (മലയാളം)": "ml",
    "Marathi (मराठी)": "mr",
    "Bengali (বাংলা)": "bn",
    "Gujarati (ગુજરાતી)": "gu",
    "Punjabi (ਪੰਜਾਬੀ)": "pa",
    "Odia (ଓଡ଼ିଆ)": "or",
    "English (India)": "en"
}

def triage_assistant(audio, text_input, selected_language):
    user_query = ""
    lang_code = LANGUAGE_MAPPING.get(selected_language, None)

    # 1. Handle Voice Input
    if audio is not None:
        if not client:
            return "⚠️ **Configuration Error**: `GROQ_API_KEY` is missing in Render Environment Variables."
        try:
            with open(audio, "rb") as file:
                transcription_kwargs = {
                    "file": (os.path.basename(audio), file.read()),
                    "model": "whisper-large-v3",
                    "prompt": "Emergency first aid triage: snake bite, accident, bleeding, heart attack, injury. Languages: Kannada (ಹಾವು ಕಡಿತ), Hindi (सांप का काटना), English."
                }
                if lang_code:
                    transcription_kwargs["language"] = lang_code

                transcription = client.audio.transcriptions.create(**transcription_kwargs)
                user_query = transcription.text.strip()
        except Exception as e:
            user_query = f"[Audio Transcription Failed: {str(e)}]"

    # Fallback / manual input
    if text_input and text_input.strip():
        if user_query:
            user_query += f" | Additional note: {text_input.strip()}"
        else:
            user_query = text_input.strip()

    if not user_query:
        return "⚠️ Please record your voice or type your emergency details above."

    if not client:
        return f"**Recognized Alert:** {user_query}\n\n⚠️ Groq API Key missing. Please dial **108** or **112** immediately."

    # 2. Generate Triage Response via Groq
    target_lang = selected_language if selected_language != "Auto-Detect / Code-Mixed" else "the user's spoken language (e.g., Kannada if spoken in Kannada)"
    system_prompt = (
        "You are Sahaya, an instant emergency response and first-aid triage assistant. "
        f"Respond in {target_lang} with simple, life-saving steps: "
        "1. Reassure the user briefly. "
        "2. Provide 3-4 direct DOs and DONTs (e.g., for snake bite: immobilize limb, do NOT tourniquet, do NOT cut/suck wound). "
        "3. Emphasize calling emergency 108 or 112 immediately."
    )

    try:
        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Emergency triage request: {user_query}"}
            ],
            temperature=0.2,
            max_tokens=600
        )
        response_text = chat_completion.choices[0].message.content
    except Exception as e:
        response_text = f"⚠️ Service connection error: {str(e)}. Please call **108** immediately."

    return f"### 📝 Detected Emergency Input\n> {user_query}\n\n---\n\n### 🩺 First-Aid & Emergency Guidance\n{response_text}"

# Client-side JS Web Audio Siren (15s sweep) + Auto-dialer trigger
sos_js_action = """
() => {
    try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'sawtooth';
        
        const now = audioCtx.currentTime;
        gain.gain.setValueAtTime(0.3, now);
        
        // 15-second dual-tone emergency police/ambulance sweep
        for (let i = 0; i < 15; i++) {
            osc.frequency.setValueAtTime(700, now + i);
            osc.frequency.exponentialRampToValueAtTime(1200, now + i + 0.5);
            osc.frequency.exponentialRampToValueAtTime(700, now + i + 1.0);
        }
        
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(now + 15);
    } catch(e) {
        console.error("AudioContext error: ", e);
    }
    
    // Trigger Phone Dialer for Helpline 1091
    window.open('tel:1091', '_self');
    return "🚨 **SOS Activated!** 15-second siren ringing & dialing 1091...";
}
"""

custom_css = """
/* Clean Light Theme Matching Reference Video */
body, .gradio-container {
    background-color: #fdfbf7 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    max-width: 720px !important;
    margin: 0 auto !important;
}

/* Header */
.sahaya-header {
    text-align: center;
    margin-top: 15px;
    margin-bottom: 5px;
}
.sahaya-title-wrap {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
}
.sahaya-title {
    font-size: 38px;
    font-weight: 800;
    color: #1e3a8a;
    margin: 0;
}
.sahaya-wave {
    display: flex;
    align-items: flex-end;
    gap: 4px;
    height: 28px;
}
.sahaya-wave-bar {
    width: 4px;
    background: #0284c7;
    border-radius: 3px;
    animation: waveAnim 1s ease-in-out infinite alternate;
}
.sahaya-wave-bar:nth-child(1) { height: 40%; animation-delay: 0.1s; }
.sahaya-wave-bar:nth-child(2) { height: 75%; animation-delay: 0.3s; }
.sahaya-wave-bar:nth-child(3) { height: 100%; animation-delay: 0.2s; }
.sahaya-wave-bar:nth-child(4) { height: 50%; animation-delay: 0.4s; }

@keyframes waveAnim {
    0% { transform: scaleY(0.4); background: #0284c7; }
    100% { transform: scaleY(1); background: #16a34a; }
}

.sahaya-sub {
    font-size: 15px;
    color: #475569;
    font-weight: 500;
    margin-top: 4px;
}

/* SOS Button */
.sos-btn {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
    color: #ffffff !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 18px !important;
    box-shadow: 0 4px 12px rgba(220, 38, 38, 0.25) !important;
    margin: 15px 0 !important;
}

/* Alert Submit Button */
.send-alert-btn {
    background: #0284c7 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 10px !important;
    margin-top: 8px !important;
}

/* Emergency Directory */
.directory-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.directory-info {
    display: flex;
    align-items: center;
    gap: 12px;
}
.directory-icon {
    font-size: 24px;
}
.directory-name {
    font-weight: 700;
    font-size: 14px;
    color: #1e293b;
}
.directory-num {
    font-size: 13px;
    font-weight: 700;
    color: #dc2626;
}
.call-badge {
    background: #f1f5f9;
    color: #0284c7;
    font-weight: 700;
    font-size: 13px;
    padding: 6px 14px;
    border-radius: 6px;
    text-decoration: none;
    border: 1px solid #cbd5e1;
}
"""

with gr.Blocks(css=custom_css, title="Sahaya Emergency Assistant") as demo:
    # Header
    gr.HTML("""
        <div class="sahaya-header">
            <div class="sahaya-title-wrap">
                <span class="sahaya-title">Sahaya</span>
                <div class="sahaya-wave">
                    <div class="sahaya-wave-bar"></div>
                    <div class="sahaya-wave-bar"></div>
                    <div class="sahaya-wave-bar"></div>
                    <div class="sahaya-wave-bar"></div>
                </div>
            </div>
            <div class="sahaya-sub">Stay Safe. 🛡️</div>
        </div>
    """)

    # SOS Button
    sos_btn = gr.Button("🚨 EMERGENCY SOS • LOUD ALARM + DIAL 1091", elem_classes=["sos-btn"])
    sos_status = gr.Markdown(visible=True)
    sos_btn.click(fn=None, js=sos_js_action, outputs=sos_status)

    # Language Dropdown
    lang_dropdown = gr.Dropdown(
        choices=list(LANGUAGE_MAPPING.keys()),
        value="Kannada (ಕನ್ನಡ)",
        label="Language Selection / ಭಾಷೆ / भाषा / ಭಾಷೆ"
    )

    # Voice & Text Inputs
    audio_mic = gr.Audio(sources=["microphone", "upload"], type="filepath", label="🎙️ Tap to Start Listening / Record")
    text_desc = gr.Textbox(placeholder="e.g., Need an ambulance, snake bite, road accident...", label="Or Type Emergency Details", lines=2)
    
    send_btn = gr.Button("Send Emergency Alert", elem_classes=["send-alert-btn"])
    
    # Response Output
    response_box = gr.Markdown(label="Sahaya Voice Response")
    send_btn.click(
        fn=triage_assistant, 
        inputs=[audio_mic, text_desc, lang_dropdown], 
        outputs=response_box
    )

    # Directory Grid with direct call links
    gr.HTML("""
        <div style="margin-top: 25px;">
            <h4 style="color: #334155; font-size: 16px; margin-bottom: 12px;">📁 Emergency Services Directory</h4>
            
            <div class="directory-card">
                <div class="directory-info">
                    <span class="directory-icon">🚑</span>
                    <div>
                        <div class="directory-name">Medical & Ambulance</div>
                        <div class="directory-num">108</div>
                    </div>
                </div>
                <a class="call-badge" href="tel:108">Call</a>
            </div>

            <div class="directory-card">
                <div class="directory-info">
                    <span class="directory-icon">👮</span>
                    <div>
                        <div class="directory-name">Police & Safety</div>
                        <div class="directory-num">112</div>
                    </div>
                </div>
                <a class="call-badge" href="tel:112">Call</a>
            </div>

            <div class="directory-card">
                <div class="directory-info">
                    <span class="directory-icon">👩</span>
                    <div>
                        <div class="directory-name">Women Distress Helpline</div>
                        <div class="directory-num">1091</div>
                    </div>
                </div>
                <a class="call-badge" href="tel:1091">Call</a>
            </div>

            <div class="directory-card">
                <div class="directory-info">
                    <span class="directory-icon">🚒</span>
                    <div>
                        <div class="directory-name">Fire & Rescue</div>
                        <div class="directory-num">101</div>
                    </div>
                </div>
                <a class="call-badge" href="tel:101">Call</a>
            </div>

            <div class="directory-card">
                <div class="directory-info">
                    <span class="directory-icon">🌊</span>
                    <div>
                        <div class="directory-name">Disaster Support</div>
                        <div class="directory-num">1078</div>
                    </div>
                </div>
                <a class="call-badge" href="tel:1078">Call</a>
            </div>
        </div>
    """)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
