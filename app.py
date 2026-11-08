import os
import json
import base64
import io
import time
import gradio as gr
import numpy as np

# Try importing soundfile / scipy for telephonic filtering; fallback to raw if not present
try:
    import soundfile as sf
    from scipy.signal import butter, lfilter
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False

# ==========================================
# 1. AUDIO PREPROCESSING (Noise Canceller)
# ==========================================
def telephonic_bandpass_filter(audio_bytes):
    """
    Simulates a 300Hz - 3400Hz bandpass filter to suppress low-frequency rumble
    and high-frequency hiss typical in noisy, telephonic audio streams.
    """
    if not AUDIO_PROCESSING_AVAILABLE:
        return audio_bytes
    try:
        data, samplerate = sf.read(io.BytesIO(audio_bytes))
        # Bandpass filter coefficients
        lowcut = 300.0
        highcut = 3400.0
        nyq = 0.5 * samplerate
        low = lowcut / nyq
        high = min(highcut / nyq, 0.99)
        b, a = butter(2, [low, high], btype='band')
        filtered = lfilter(b, a, data)
        
        out_buf = io.BytesIO()
        sf.write(out_buf, filtered, samplerate, format='WAV')
        return out_buf.getvalue()
    except Exception:
        return audio_bytes

# ==========================================
# 2. EMERGENCY & FIRST AID TRIAGE ENGINE
#    (Real Deal & Main Character Awards)
# ==========================================
TRIAGE_DATABASE = {
    "snakebite": {
        "en": "Keep the limb still below heart level. Do not tie tourniquet or cut wound. Get to hospital immediately for Anti-Snake Venom (ASV).",
        "kn": "ಕಚ್ಚಿದ ಜಾಗವನ್ನು ಅಲುಗಾಡಿಸಬೇಡಿ ಮತ್ತು ಹೃದಯದ ಮಟ್ಟಕ್ಕಿಂತ ಕೆಳಗೆ ಇರಿಸಿ. ಹಗ್ಗ ಕಟ್ಟಬೇಡಿ ಅಥವಾ ಗಾಯವನ್ನು ಕತ್ತರಿಸಬೇಡಿ. ತಕ್ಷಣವೇ ಆಸ್ಪತ್ರೆಗೆ ತೆರಳಿ ಆಂಟಿ-ಸ್ನೇಕ್ ವೆನಮ್ (ASV) ಪಡೆಯಿರಿ.",
        "hi": "काटे गए अंग को स्थिर रखें और दिल के स्तर से नीचे रखें। कसकर कपड़ा न बांधें और न ही चीरा लगाएं। तुरंत नजदीकी अस्पताल जाकर एंटी-वेनम लें।"
    },
    "pesticide": {
        "en": "Remove contaminated clothing immediately. Wash skin with copious clean water and soap. Administer Atropine under medical care. Dial 108.",
        "kn": "ವಿಷದ ಬಟ್ಟೆಗಳನ್ನು ತಕ್ಷಣವೇ ತೆಗೆಯಿರಿ. ಚರ್ಮವನ್ನು ಸಾಕಷ್ಟು ಶುದ್ಧ ನೀರು ಮತ್ತು ಸೋಪಿನಿಂದ ತೊಳೆಯಿರಿ. ಕೂಡಲೇ 108 ಗೆ ಕರೆ ಮಾಡಿ ಆಸ್ಪತ್ರೆಗೆ ಕರೆದೊಯ್ಯಿರಿ.",
        "hi": "कीटनाशक लगे कपड़ों को तुरंत उतारें। त्वचा को भरपूर साफ पानी और साबुन से धोएं। तुरंत 108 पर संपर्क करें।"
    },
    "cardiac": {
        "en": "Have patient sit down and rest. Loosen tight clothing. If unconscious and not breathing, begin chest compressions at 100-120 bpm. Call 108.",
        "kn": "ರೋಗಿಯನ್ನು ಕೂಡಲೇ ಕುಳ್ಳಿರಿಸಿ. ಬಿಗಿಯಾದ ಬಟ್ಟೆಗಳನ್ನು ಸಡಿಲಗೊಳಿಸಿ. ಉಸಿರಾಟವಿಲ್ಲದಿದ್ದರೆ, ಎದೆಯ ಮಧ್ಯಭಾಗದಲ್ಲಿ ಸಿಪಿಆರ್ (CPR) ಪ್ರಾರಂಭಿಸಿ. ಕೂಡಲೇ 108 ಗೆ ಕರೆ ಮಾಡಿ.",
        "hi": "मरीज को तुरंत बैठाएं। तंग कपड़े ढीले करें। यदि सांस नहीं चल रही है, तो तुरंत CPR शुरू करें (100-120 प्रति मिनट)। 108 डायल करें।"
    },
    "maternal": {
        "en": "Position the mother lying on her left side. Monitor contraction frequency. Keep warm and clean. Dispatch ambulance immediately via 108/102.",
        "kn": "ತಾಯಿಯನ್ನು ಎಡ ಮಗ್ಗುಲಿನಲ್ಲಿ ಮಲಗಿಸಿ. ಉಸಿರಾಟ ಸುಲಭವಾಗಿರಲಿ. ಬೆಚ್ಚಗಿನ ವಾತಾವರಣ ಇರಲಿ. ತಕ್ಷಣವೇ 108/102 ತುರ್ತು ಆಂಬ್ಯುಲೆನ್ಸ್‌ಗೆ ಕರೆ ಮಾಡಿ.",
        "hi": "गर्भवती को बाईं करवट लिटाएं। साफ और गर्म माहौल रखें। तुरंत 108 या 102 एम्बुलेंस को कॉल करें।"
    }
}

def analyze_intent(text: str):
    """
    Handles mixed-code utterances (Tanglish, Kanglish, Hinglish) and extracts triage actions.
    """
    text_lower = text.lower()
    
    # Snakebite triggers across scripts & code-mixed forms
    if any(k in text_lower for k in ["haavu", "havu", "snake", "saap", "paambu", "sarpa"]):
        return "snakebite"
    # Pesticide triggers
    if any(k in text_lower for k in ["keetnashaka", "pesticide", "poison", "visha", "medicine poison", "kheti dawa"]):
        return "pesticide"
    # Maternal triggers
    if any(k in text_lower for k in ["delivery", "garbhini", "prasava", "maternal", "labour", "pregnant"]):
        return "maternal"
    # Default to cardiac/collapse triage
    return "cardiac"

def resolve_language(text: str):
    # Detect Kannada unicode range
    if any('\u0C80' <= char <= '\u0CFF' for char in text):
        return "kn"
    # Detect Devanagari range
    if any('\u0900' <= char <= '\u097F' for char in text):
        return "hi"
    return "en"

# ==========================================
# 3. GNANI AI INTEGRATION STUB / SIMULATOR
#    (Prisma STT + Timbre TTS)
# ==========================================
def process_audio_pipeline(audio_base64: str):
    """
    Takes base64 recorded audio, strips headers, filters noise,
    transcribes via Indic STT, resolves medical emergency, and responds.
    """
    if not audio_base64:
        return "No audio received", "Please speak into the microphone.", "", ""

    try:
        # Decode base64 payload from browser recorder
        header, encoded = audio_base64.split(",", 1) if "," in audio_base64 else ("", audio_base64)
        raw_bytes = base64.b64decode(encoded)
        
        # Telephonic clean
        cleaned_audio = telephonic_bandpass_filter(raw_bytes)
        
        # In actual deployment, invoke Gnani Prisma API endpoint here:
        # headers = {'Authorization': f'Bearer {GNANI_API_KEY}', 'Content-Type': 'audio/wav'}
        # response = requests.post("https://api.gnani.ai/v1/stt", data=cleaned_audio, headers=headers)
        
        # Simulated transcription representing typical emergency input
        transcribed_text = "ತುರ್ತು ಪರಿಸ್ಥಿತಿ! ಹಾವಿನ ಕಡಿತ ಆಗಿದೆ ಬೇಗ ಸಹಾಯ ಮಾಡಿ (Snake bite emergency)"
        
        lang = resolve_language(transcribed_text)
        condition_key = analyze_intent(transcribed_text)
        protocol = TRIAGE_DATABASE.get(condition_key, TRIAGE_DATABASE["cardiac"])
        guidance = protocol.get(lang, protocol["en"])
        
        # Jugaad Feature: Offline SMS/USSD payload generation for remote locations
        ussd_payload = f"*108*1*{condition_key.upper()}# -> DISPATCHED"
        
        return transcribed_text, guidance, condition_key.upper(), ussd_payload
    except Exception as e:
        return f"Error: {str(e)}", "Please retry speech input.", "SYSTEM ERROR", ""

# ==========================================
# 4. CUSTOM HTML + CSS + JS INTERFACE
#    (Centered Button + Auto Voice Silence Detection)
# ==========================================
CUSTOM_HEAD = """
<style>
  .centered-hub {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin: 40px auto 20px auto;
    text-align: center;
  }
  .record-circle-btn {
    width: 140px;
    height: 140px;
    border-radius: 50%;
    background: linear-gradient(135deg, #e53e3e, #dd6b20);
    border: none;
    outline: none;
    color: #ffffff;
    font-size: 17px;
    font-weight: 700;
    cursor: pointer;
    box-shadow: 0 0 25px rgba(229, 62, 62, 0.45);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    display: flex;
    align-items: center;
    justify-content: center;
    user-select: none;
  }
  .record-circle-btn:hover {
    transform: scale(1.04);
    box-shadow: 0 0 35px rgba(229, 62, 62, 0.6);
  }
  .record-circle-btn.recording {
    background: linear-gradient(135deg, #9b2c2c, #c53030);
    animation: emergencyPulse 1.2s infinite alternate;
  }
  @keyframes emergencyPulse {
    0% { transform: scale(1); box-shadow: 0 0 20px rgba(229, 62, 62, 0.5); }
    100% { transform: scale(1.12); box-shadow: 0 0 45px rgba(229, 62, 62, 0.95); }
  }
  .status-label {
    margin-top: 18px;
    font-size: 15px;
    font-weight: 600;
    color: #4a5568;
    letter-spacing: 0.3px;
  }
</style>

<script>
let mediaRecorder = null;
let audioChunks = [];
let audioContext = null;
let analyserNode = null;
let silenceTimer = null;
const SILENCE_THRESHOLD_RMS = 0.015;
const SILENCE_DURATION_MS = 1500;

async function toggleRecording() {
  const btn = document.getElementById('sahaya-rec-btn');
  const label = document.getElementById('sahaya-status-label');

  if (mediaRecorder && mediaRecorder.state === "recording") {
    stopRecording();
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyserNode = audioContext.createAnalyser();
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyserNode);
    analyserNode.fftSize = 512;

    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      label.innerText = "Analyzing speech & triaging...";
      btn.classList.remove('recording');
      btn.innerText = "Tap to Record";
      
      const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
      const reader = new FileReader();
      reader.readAsDataURL(audioBlob);
      reader.onloadend = () => {
        const base64Audio = reader.result;
        // Inject into hidden Gradio input
        const hiddenInput = document.querySelector('#audio-data-payload textarea');
        if (hiddenInput) {
          hiddenInput.value = base64Audio;
          hiddenInput.dispatchEvent(new Event('input', { bubbles: true }));
          document.getElementById('sahaya-process-trigger').click();
        }
      };
      stream.getTracks().forEach(track => track.stop());
    };

    mediaRecorder.start(200);
    btn.classList.add('recording');
    btn.innerText = "Listening...";
    label.innerText = "Listening... Speak now. Stops automatically on silence.";

    monitorSilence();
  } catch (err) {
    console.error("Microphone access error:", err);
    label.innerText = "Microphone permission denied.";
  }
}

function monitorSilence() {
  const buffer = new Float32Array(analyserNode.fftSize);
  
  function check() {
    if (!mediaRecorder || mediaRecorder.state !== "recording") return;
    analyserNode.getFloatTimeDomainData(buffer);
    
    // Calculate Root Mean Square (RMS) volume
    let sum = 0;
    for (let i = 0; i < buffer.length; i++) {
      sum += buffer[i] * buffer[i];
    }
    const rms = Math.sqrt(sum / buffer.length);

    if (rms < SILENCE_THRESHOLD_RMS) {
      if (!silenceTimer) {
        silenceTimer = setTimeout(() => {
          stopRecording();
        }, SILENCE_DURATION_MS);
      }
    } else {
      if (silenceTimer) {
        clearTimeout(silenceTimer);
        silenceTimer = null;
      }
    }
    requestAnimationFrame(check);
  }
  requestAnimationFrame(check);
}

function stopRecording() {
  if (silenceTimer) {
    clearTimeout(silenceTimer);
    silenceTimer = null;
  }
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
  }
}
</script>
"""

# ==========================================
# 5. GRADIO APPLICATION LAYOUT
# ==========================================
with gr.Blocks(head=CUSTOM_HEAD, title="Sahaya - Indic Emergency First-Aid AI") as demo:
    gr.Markdown(
        """
        # 🚑 Sahaya (ಸಹಾಯ)
        ### Real-Time Multilingual First-Aid & Emergency Voice Triager
        *Built for rural health workers, farmers, and frontline emergency responders.*
        """
    )
    
    # Centered Custom Recording Area
    with gr.Column(elem_classes=["centered-hub"]):
        gr.HTML(
            """
            <div class="centered-hub">
              <button id="sahaya-rec-btn" class="record-circle-btn" onclick="toggleRecording()">
                Tap to Record
              </button>
              <div id="sahaya-status-label" class="status-label">
                Tap the button and state the emergency in your language
              </div>
            </div>
            """
        )
    
    # Hidden payload elements to pass recorded audio to Python
    with gr.Row(visible=False):
        raw_audio_input = gr.Textbox(elem_id="audio-data-payload")
        submit_btn = gr.Button(elem_id="sahaya-process-trigger")

    # Output Dashboard
    with gr.Row():
        with gr.Column(scale=1):
            severity_badge = gr.Textbox(label="Detected Triage Category", interactive=False)
            ussd_output = gr.Textbox(label="Jugaad Fallback (Offline SMS/USSD Payload)", interactive=False)
        with gr.Column(scale=2):
            transcript_box = gr.Textbox(label="Prisma Transcribed Speech (Native / Code-Mixed)", interactive=False)
            triage_guidance = gr.Textbox(label="Immediate First-Aid Protocol (Native Voice Output)", lines=4, interactive=False)

    # Wire event listener
    submit_btn.click(
        fn=process_audio_pipeline,
        inputs=[raw_audio_input],
        outputs=[transcript_box, triage_guidance, severity_badge, ussd_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
