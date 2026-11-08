"""
Sahaya Emergency Assistant (Gradio on Render)
Voice-first emergency triage and first aid for India across 12 languages.

Gnani AI stack:
Prisma  (speech-to-text)   -> https://api.vachana.ai/stt/v3
Timbre  (text-to-speech)   -> https://api.vachana.ai/api/v1/tts/inference
Groq (triage reasoning)    -> https://api.groq.com/openai/v1/chat/completions
"""

import html
import json
import os
import re
import tempfile
import gradio as gr
import requests
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False

STT_URL = "https://api.vachana.ai/stt/v3"
TTS_URL = "https://api.vachana.ai/api/v1/tts/inference"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

# ---------------------------------------------------------------------------
# Noise Reduction & Telephonic Audio Preprocessor
# ---------------------------------------------------------------------------
def preprocess_telephonic_audio(audio_path: str) -> str:
    """
    Applies Butterworth bandpass filtering (300 Hz - 3400 Hz) and stationary/non-stationary
    spectral gating to strip background noise from telephonic and noisy microphone recordings.
    """
    if not audio_path or not os.path.exists(audio_path):
        return audio_path

    try:
        data, sr = sf.read(audio_path)
        # Convert stereo to mono if needed
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # 1. 4th order Butterworth Bandpass Filter (300Hz - 3400Hz telephone bandwidth)
        lowcut = 300.0
        highcut = min(3400.0, (sr / 2.0) - 50.0)
        sos = butter(4, [lowcut, highcut], btype='bandpass', fs=sr, output='sos')
        filtered_audio = sosfilt(sos, data)

        # 2. Spectral Gating Noise Cancellation
        if NOISEREDUCE_AVAILABLE:
            cleaned_audio = nr.reduce_noise(
                y=filtered_audio,
                sr=sr,
                stationary=False,
                prop_decrease=0.75
            )
        else:
            cleaned_audio = filtered_audio

        # Normalize audio amplitude
        max_val = np.max(np.abs(cleaned_audio))
        if max_val > 0:
            cleaned_audio = cleaned_audio / max_val * 0.95

        tmp_out = tempfile.NamedTemporaryFile(suffix="_clean.wav", delete=False)
        sf.write(tmp_out.name, cleaned_audio, sr)
        return tmp_out.name
    except Exception:
        return audio_path


# ---------------------------------------------------------------------------
# Languages & UI Translations (12 Indian Languages)
# ---------------------------------------------------------------------------
LANGUAGES = {
    "en_IN": "English (India)",
    "kn_IN": "ಕನ್ನಡ (Kannada)",
    "hi_IN": "हिन्दी (Hindi)",
    "ta_IN": "தமிழ் (Tamil)",
    "te_IN": "తెలుగు (Telugu)",
    "ml_IN": "മലയാളം (Malayalam)",
    "mr_IN": "मराठी (Marathi)",
    "gu_IN": "ગુજરાતી (Gujarati)",
    "bn_IN": "বাংলা (Bengali)",
    "pa_IN": "ਪੰਜਾਬੀ (Punjabi)",
    "or_IN": "ଓଡ଼ିଆ (Odia)",
    "mni_IN": "ꯃꯤꯇꯩꯂꯣꯟ (Manipuri)",
}

LANG_NAMES = {
    "en_IN": "English",
    "kn_IN": "Kannada (ಕನ್ನಡ)",
    "hi_IN": "Hindi (हिन्दी)",
    "ta_IN": "Tamil (தமிழ்)",
    "te_IN": "Telugu (తెలుగు)",
    "ml_IN": "Malayalam (മലയാളം)",
    "mr_IN": "Marathi (मराठी)",
    "gu_IN": "Gujarati (ગુજરાતી)",
    "bn_IN": "Bengali (বাংলা)",
    "pa_IN": "Punjabi (ਪੰਜਾਬੀ)",
    "or_IN": "Odia (ଓଡ଼ିଆ)",
    "mni_IN": "Manipuri (Meitei Mayek)",
}

TIMBRE_LANGS = {
    "en_IN", "hi_IN", "ta_IN", "te_IN", "kn_IN",
    "ml_IN", "mr_IN", "pa_IN", "bn_IN", "gu_IN",
}

TRANSLATIONS = {
    "en_IN": {
        "title": "Sahaya",
        "subtitle": "Stay Safe. 🛡️",
        "sos_btn": "🚨 EMERGENCY SOS • LOUD ALARM + DIAL 1091",
        "mic_header": "🎙️ Tap to Start Listening",
        "type_header": "⌨️ Or Type Emergency Details",
        "type_placeholder": "e.g., Need an ambulance, snake bite, road accident...",
        "send": "Send Emergency Alert",
        "detected": "🗣️ Detected Emergency:",
        "action_header": "Recommended Action & First-Aid",
        "first_aid_label": "Immediate First-Aid Steps:",
        "call_btn": "Call {number} Immediately",
        "directory_header": "Emergency Services Directory",
        "helpline": "Helpline:",
        "audio_response": "🔊 Sahaya Voice Guidance:",
        "med_title": "Medical & Ambulance",
        "police_title": "Police & Safety",
        "women_title": "Women Distress Helpline",
        "fire_title": "Fire & Rescue",
        "disaster_title": "Disaster Support",
        "call_txt": "Call"
    },
    "kn_IN": {
        "title": "ಸಹಾಯ (Sahaya)",
        "subtitle": "ಸುರಕ್ಷಿತವಾಗಿರಿ. 🛡️",
        "sos_btn": "🚨 ತುರ್ತು SOS • ತೀವ್ರ ಸೈರನ್ + 1091 ಗೆ ಕರೆ",
        "mic_header": "🎙️ ಧ್ವನಿ ರೆಕಾರ್ಡ್ ಮಾಡಲು ಟ್ಯಾಪ್ ಮಾಡಿ / ಆಲಿಸಿ",
        "type_header": "⌨️ ಅಥವಾ ತುರ್ತು ವಿವರಗಳನ್ನು ಇಲ್ಲಿ ಟೈಪ್ ಮಾಡಿ",
        "type_placeholder": "ಉದಾ: ಹಾವು ಕಡಿತ, ಅಪಘಾತ, ರಕ್ತಸ್ರಾವ, ಆಂಬ್ಯುಲೆನ್ಸ್ ಬೇಕು...",
        "send": "ತುರ್ತು ಎಚ್ಚರಿಕೆ ಕಳುಹಿಸಿ",
        "detected": "🗣️ ಪತ್ತೆಯಾದ ತುರ್ತು ಪರಿಸ್ಥಿತಿ:",
        "action_header": "ಶಿಫಾರಸು ಮಾಡಿದ ಕ್ರಮ ಮತ್ತು ಪ್ರಥಮ ಚಿಕಿತ್ಸೆ",
        "first_aid_label": "ತಕ್ಷಣದ ಪ್ರಥಮ ಚಿಕಿತ್ಸಾ ಕ್ರಮಗಳು:",
        "call_btn": "ತಕ್ಷಣ {number} ಗೆ ಕರೆ ಮಾಡಿ",
        "directory_header": "ತುರ್ತು ಸೇವೆಗಳ ಡೈರೆಕ್ಟರಿ",
        "helpline": "ಸಹಾಯವಾಣಿ:",
        "audio_response": "🔊 ಸಹಾಯ ಧ್ವನಿ ಮಾರ್ಗದರ್ಶನ:",
        "med_title": "ವೈದ್ಯಕೀಯ ಮತ್ತು ಆಂಬ್ಯುಲೆನ್ಸ್",
        "police_title": "ಪೊಲೀಸ್ ಮತ್ತು ಸುರಕ್ಷತೆ",
        "women_title": "ಮಹಿಳಾ ಸಹಾಯವಾಣಿ",
        "fire_title": "ಅಗ್ನಿಶಾಮಕ ಮತ್ತು ರಕ್ಷಣೆ",
        "disaster_title": "ವಿಪತ್ತು ನೆರವು",
        "call_txt": "ಕರೆ ಮಾಡಿ"
    },
    "hi_IN": {
        "title": "सहायता (Sahaya)",
        "subtitle": "सुरक्षित रहें। 🛡️",
        "sos_btn": "🚨 आपातकालीन SOS • तेज सायरन + 1091 पर कॉल",
        "mic_header": "🎙️ बोलने के लिए टैप करें / रिकॉर्ड करें",
        "type_header": "⌨️ या आपातकालीन विवरण टाइप करें",
        "type_placeholder": "जैसे सांप का काटना, सड़क दुर्घटना, एम्बुलेंस चाहिए...",
        "send": "आपातकालीन अलर्ट भेजें",
        "detected": "🗣️ पता चली आपात स्थिति:",
        "action_header": "अनुशंसित कार्रवाई और प्राथमिक उपचार",
        "first_aid_label": "प्राथमिक उपचार दिशानिर्देश:",
        "call_btn": "तुरंत {number} पर कॉल करें",
        "directory_header": "आपातकालीन सेवा डायरेक्टरी",
        "helpline": "हेल्पलाइन:",
        "audio_response": "🔊 सहायता वॉयस मार्गदर्शन:",
        "med_title": "चिकित्सा एवं एम्बुलेंस",
        "police_title": "पुलिस एवं सुरक्षा",
        "women_title": "महिला हेल्पलाइन",
        "fire_title": "दमकल एवं बचाव",
        "disaster_title": "आपदा राहत सहायता",
        "call_txt": "कॉल करें"
    },
    "ta_IN": {
        "title": "சஹாயா (Sahaya)",
        "subtitle": "பாதுகாப்பாக இருங்கள். 🛡️",
        "sos_btn": "🚨 அவசர SOS • உரத்த சைரன் + 1091 ஐ அழைக்கவும்",
        "mic_header": "🎙️ அவசரத்தைப் பேச மைக் தொடவும்",
        "type_header": "⌨️ அவசர விவரங்களை தட்டச்சு செய்யவும்",
        "type_placeholder": "எ.கா. விபத்து, பாம்பு கடி, அதிக ரத்தப்போக்கு...",
        "send": "அவசர எச்சரிக்கை அனுப்பு",
        "detected": "🗣️ கண்டறியப்பட்ட அவசரம்:",
        "action_header": "பரிந்துரைக்கப்பட்ட முதலுதவி",
        "first_aid_label": "உடனடி முதலுதவி வழிகாட்டுதல்கள்:",
        "call_btn": "உடனே {number} ஐ அழைக்கவும்",
        "directory_header": "அவசர சேவை அடைவு",
        "helpline": "உதவி எண்:",
        "audio_response": "🔊 சஹாயா குரல் வழிகாட்டல்:",
        "med_title": "மருத்துவம் & ஆம்புலன்ஸ்",
        "police_title": "காவல்துறை & பாதுகாப்பு",
        "women_title": "பெண்கள் உதவி எண்",
        "fire_title": "தீயணைப்பு & மீட்பு",
        "disaster_title": "பேரிடர் மேலாண்மை",
        "call_txt": "அழைக்க"
    },
    "te_IN": {
        "title": "సహాయ (Sahaya)",
        "subtitle": "సురక్షితంగా ఉండండి. 🛡️",
        "sos_btn": "🚨 అత్యవసర SOS • పెద్ద సైరన్ + 1091 కి కాల్",
        "mic_header": "🎙️ మాట్లాడటానికి మైక్ నొక్కండి",
        "type_header": "⌨️ అత్యవసర వివరాలు టైప్ చేయండి",
        "type_placeholder": "ఉదా. పాము కాటు, ప్రమాదం, అంబులెన్స్ కావాలి...",
        "send": "అత్యవసర హెచ్చరిక పంపండి",
        "detected": "🗣️ గుర్తించబడిన అత్యవసరం:",
        "action_header": "సిఫార్సు చేయబడిన ప్రథమ చికిత్స",
        "first_aid_label": "తక్షణ ప్రథమ చికిత్స మార్గదర్శకాలు:",
        "call_btn": "ఇప్పుడే {number} కి కాల్ చేయండి",
        "directory_header": "అత్యవసర సేవల డైరెక్టరీ",
        "helpline": "హెల్ప్‌లైన్:",
        "audio_response": "🔊 సహాయ వాయిస్ మార్గదర్శకత్వం:",
        "med_title": "వైద్యం & అంబులెన్స్",
        "police_title": "పోలీస్ & భద్రత",
        "women_title": "మహిళా హెల్ప్‌లైన్",
        "fire_title": "అగ్నిమాపక & రక్షణ",
        "disaster_title": "విపత్తు సహాయం",
        "call_txt": "కాల్ చేయండి"
    },
    "ml_IN": {
        "title": "സഹായ (Sahaya)",
        "subtitle": "സുരക്ഷിതരായിരിക്കൂ. 🛡️",
        "sos_btn": "🚨 എമർജൻസി SOS • വലിയ സൈറൻ + 1091 ലേക്ക് വിളിക്കുക",
        "mic_header": "🎙️ സംസാരിക്കാൻ മൈക്ക് ടാപ്പ് ചെയ്യുക",
        "type_header": "⌨️ വിവരങ്ങൾ ടൈപ്പ് ചെയ്യുക",
        "type_placeholder": "ഉദാ. പാമ്പ് കടി, അപകടം, ആംബുലൻസ് വേണം...",
        "send": "അടിയന്തര മുന്നറിയിപ്പ് അയയ്ക്കുക",
        "detected": "🗣️ കണ്ടെത്തിയ അത്യാഹിതം:",
        "action_header": "ശുപാർശ ചെയ്യുന്ന പ്രഥമശുശ്രൂഷ",
        "first_aid_label": "ഉടൻ ചെയ്യേണ്ട പ്രഥമശുശ്രൂഷാ ഘട്ടങ്ങൾ:",
        "call_btn": "ഉടൻ {number}-ലേക്ക് വിളിക്കുക",
        "directory_header": "എമർജൻസി ഡയറക്ടറി",
        "helpline": "ഹെൽപ്പ്‌ലൈൻ:",
        "audio_response": "🔊 സഹായ വോയ്സ് പ്രതികരണം:",
        "med_title": "മെഡിക്കൽ & ആംബുലൻസ്",
        "police_title": "പോലീസ് & സുരക്ഷ",
        "women_title": "വനിതാ ഹെൽപ്പ് ലൈൻ",
        "fire_title": "ഫയർ & റെസ്ക്യൂ",
        "disaster_title": "ദുരന്ത നിവാരണം",
        "call_txt": "വിളിക്കുക"
    },
    "mr_IN": {
        "title": "साहाय्य (Sahaya)",
        "subtitle": "सुरक्षित राहा. 🛡️",
        "sos_btn": "🚨 आणीबाणी SOS • मोठा सायरन + 1091 वर कॉल",
        "mic_header": "🎙️ बोलण्यासाठी माइक टॅप करा",
        "type_header": "⌨️ किंवा टाइप करा",
        "type_placeholder": "उदा. साप चावला, अपघात, छातीत दुखणे...",
        "send": "आणीबाणी सूचना पाठवा",
        "detected": "🗣️ आढळलेली आणीबाणी:",
        "action_header": "शिफारस केलेले प्रथमोपचार",
        "first_aid_label": "तातडीच्या प्रथमोपचार सूचना:",
        "call_btn": "आता {number} वर कॉल करा",
        "directory_header": "आणीबाणी निर्देशिका",
        "helpline": "हेल्पलाइन:",
        "audio_response": "🔊 साहाय्य व्हॉइस मार्गदर्शन:",
        "med_title": "वैद्यकीय आणि रुग्णवाहिका",
        "police_title": "पोलीस आणि सुरक्षा",
        "women_title": "महिला हेल्पलाइन",
        "fire_title": "अग्निशामक आणि बचाव",
        "disaster_title": "आपत्ती मदत",
        "call_txt": "कॉल करा"
    },
    "gu_IN": {
        "title": "સહાય (Sahaya)",
        "subtitle": "સુરક્ષિત રહો. 🛡️",
        "sos_btn": "🚨 કટોકટી SOS • મોટો સાયરન + 1091 પર કૉલ",
        "mic_header": "🎙️ બોલવા માટે માઇક દબાવો",
        "type_header": "⌨️ અથવા વિગતો ટાઇપ કરો",
        "type_placeholder": "દા.ત. સાપ કરડ્યો, અકસ્માત, એમ્બ્યુલન્સ જોઈએ...",
        "send": "કટોકટી ચેતવણી મોકલો",
        "detected": "🗣️ મળેલી કટોકટી:",
        "action_header": "પ્રાથમિક સારવાર માર્ગદર્શિકા",
        "first_aid_label": "તાત્કાલિક પ્રાથમિક સારવારનાં પગલાં:",
        "call_btn": "હમણાં {number} પર કૉલ કરો",
        "directory_header": "ઇમરજન્સી ડિરેક્ટરી",
        "helpline": "હેલ્પલાઇન:",
        "audio_response": "🔊 સહાય વૉઇસ માર્ગદર્શન:",
        "med_title": "તબીબી અને એમ્બ્યુલન્સ",
        "police_title": "પોલીસ અને સુરક્ષા",
        "women_title": "મહિલા હેલ્પલાઇન",
        "fire_title": "અગ્નિશામક અને બચાવ",
        "disaster_title": "આપત્તિ સહાય",
        "call_txt": "કૉલ કરો"
    },
    "bn_IN": {
        "title": "সহায় (Sahaya)",
        "subtitle": "সুরক্ষিত থাকুন। 🛡️",
        "sos_btn": "🚨 জরুরি SOS • উচ্চ সাইরেন + 1091-এ কল",
        "mic_header": "🎙️ কথা বলতে মাইক ট্যাপ করুন",
        "type_header": "⌨️ জরুরি বিবরণ টাইপ করুন",
        "type_placeholder": "যেমন সাপে কাটা, দুর্ঘটনা, রক্তপাত...",
        "send": "জরুরি সতর্কতা পাঠান",
        "detected": "🗣️ শনাক্ত জরুরি অবস্থা:",
        "action_header": "প্রাথমিক চিকিৎসার নির্দেশিকা",
        "first_aid_label": "জরুরি প্রাথমিক চিকিৎসার পদক্ষেপ:",
        "call_btn": "এখনই {number}-এ কল করুন",
        "directory_header": "জরুরি ডিরেক্টরি",
        "helpline": "হেল্পলাইন:",
        "audio_response": "🔊 সহায় ভয়েস সহায়তা:",
        "med_title": "মেডিকেল ও অ্যাম্বুলেন্স",
        "police_title": "পুলিশ ও নিরাপত্তা",
        "women_title": "মহিলা হেল্পলাইন",
        "fire_title": "দমকল ও উদ্ধার",
        "disaster_title": "দুর্যোগ সহায়তা",
        "call_txt": "কল করুন"
    },
    "pa_IN": {
        "title": "ਸਹਾਇਤਾ (Sahaya)",
        "subtitle": "ਸੁਰੱਖਿਅਤ ਰਹੋ। 🛡️",
        "sos_btn": "🚨 ਐਮਰਜੈਂਸੀ SOS • ਉੱਚੀ ਸਾਇਰਨ + 1091 'ਤੇ ਕਾਲ ਕਰੋ",
        "mic_header": "🎙️ ਬੋਲਣ ਲਈ ਮਾਈਕ ਟੈਪ ਕਰੋ",
        "type_header": "⌨️ ਜਾਂ ਵੇਰਵੇ ਟਾਈਪ ਕਰੋ",
        "type_placeholder": "ਜਿਵੇਂ ਸੱਪ ਲੜਨਾ, ਹਾਦਸਾ, ਖੂਨ ਵਹਿਣਾ...",
        "send": "ਐਮਰਜੈਂਸੀ ਅਲਰਟ ਭੇਜੋ",
        "detected": "🗣️ ਪਤਾ ਲੱਗੀ ਐਮਰਜੈਂਸੀ:",
        "action_header": "ਮੁੱਢਲੀ ਸਹਾਇਤਾ ਦਿਸ਼ਾ-ਨਿਰਦੇਸ਼",
        "first_aid_label": "ਤੁਰੰਤ ਮੁੱਢਲੀ ਸਹਾਇਤਾ ਕਦਮ:",
        "call_btn": "ਹੁਣੇ {number} 'ਤੇ ਕਾਲ ਕਰੋ",
        "directory_header": "ਐਮਰਜੈਂਸੀ ਡਾਇਰੈਕਟਰੀ",
        "helpline": "ਹੈਲਪਲਾਈਨ:",
        "audio_response": "🔊 ਸਹਾਇਤਾ ਵੌਇਸ ਸੇਧ:",
        "med_title": "ਮੈਡੀਕਲ ਅਤੇ ਐਂਬੂਲੈਂਸ",
        "police_title": "ਪੁਲਿਸ ਅਤੇ ਸੁਰੱਖਿਆ",
        "women_title": "ਮਹਿਲਾ ਹੈਲਪਲਾਈਨ",
        "fire_title": "ਫਾਇਰ ਅਤੇ ਬਚਾਅ",
        "disaster_title": "ਆਫ਼ਤ ਸਹਾਇਤਾ",
        "call_txt": "ਕਾਲ ਕਰੋ"
    },
    "or_IN": {
        "title": "ସହାୟ (Sahaya)",
        "subtitle": "ସୁରକ୍ଷିତ ରୁହନ୍ତୁ। 🛡️",
        "sos_btn": "🚨 ଜରୁରୀକାଳୀନ SOS • ଉଚ୍ଚ ସାଇରନ + 1091 କୁ କଲ୍",
        "mic_header": "🎙️ କହିବାକୁ ମାଇକ୍ ଟ୍ୟାପ୍ କରନ୍ତୁ",
        "type_header": "⌨️ ଜରୁରୀ ବିବରଣୀ ଟାଇପ୍ କରନ୍ତୁ",
        "type_placeholder": "ଯେପରି ସାପ କାମୁଡ଼ା, ଦୁର୍ଘଟଣା, ରକ୍ତସ୍ରାବ...",
        "send": "ଜରୁରୀକାଳୀନ ଆଲର୍ଟ ପଠାନ୍ତୁ",
        "detected": "🗣️ ଚିହ୍ନଟ ଜରୁରୀକାଳୀନ ପରିସ୍ଥିତି:",
        "action_header": "ପ୍ରଥମ ଚିକିତ୍ସା ନିର୍ଦ୍ଦେଶାବଳୀ",
        "first_aid_label": "ତୁରନ୍ତ ପ୍ରଥମ ଚିକିତ୍ସା ପଦକ୍ଷେପ:",
        "call_btn": "ତୁରନ୍ତ {number} କୁ କଲ୍ କରନ୍ତୁ",
        "directory_header": "ଜରୁରୀକାଳୀନ ଡାଇରେକ୍ଟୋରୀ",
        "helpline": "ହେଲ୍ପଲାଇନ୍:",
        "audio_response": "🔊 ସହାୟ ଭଏସ୍ ମାର୍ଗଦର୍ଶନ:",
        "med_title": "ଡାକ୍ତରୀ ଏବଂ ଆମ୍ବୁଲାନ୍ସ",
        "police_title": "ପୋଲିସ୍ ଏବଂ ସୁରକ୍ଷା",
        "women_title": "ମହିଳା ହେଲ୍ପଲାଇନ",
        "fire_title": "ଅଗ୍ନିଶମ ଏବଂ ଉଦ୍ଧାର",
        "disaster_title": "ବିପର୍ଯ୍ୟୟ ସହାୟତା",
        "call_txt": "କଲ୍ କରନ୍ତୁ"
    },
    "mni_IN": {
        "title": "ꯁꯍꯥꯌ (Sahaya)",
        "subtitle": "ꯉꯥꯛꯊꯣꯛꯆꯕꯤꯌꯨ 🛡️",
        "sos_btn": "🚨 ꯈꯨꯗꯣꯡꯊꯤꯕ SOS • ꯑꯀꯟꯕ ꯁꯥꯏꯔꯦꯟ + 1091 ꯗ ꯀꯣꯜ",
        "mic_header": "🎙️ ꯋꯥ ꯉꯥꯡꯅꯕ ꯃꯥꯏꯛ ꯇꯦꯞ ꯇꯧꯕꯤꯌꯨ",
        "type_header": "⌨️ ꯈꯨꯗꯣꯡꯊꯤꯕ ꯏꯕꯤꯌꯨ",
        "type_placeholder": "ꯂꯤꯟ ꯆꯤꯛꯄ, ꯑꯦꯛꯁꯤꯗꯦꯟꯠ, ꯃꯩ ꯆꯥꯛꯄ...",
        "send": "ꯈꯨꯗꯣꯡꯊꯤꯕꯒꯤ ꯄꯥꯎ ꯊꯥꯕꯤꯌꯨ",
        "detected": "🗣️ ꯈꯪꯗꯣꯛꯂꯕ ꯈꯨꯗꯣꯡꯊꯤꯕ:",
        "action_header": "ꯑꯍꯥꯟꯕ ꯃꯇꯦꯡ (First Aid)",
        "first_aid_label": "ꯈꯨꯗꯛꯇ ꯇꯧꯒꯗꯕ ꯊꯕꯛꯁꯤꯡ:",
        "call_btn": "{number} ꯗ ꯍꯧꯖꯤꯛꯃꯛ ꯀꯣꯜ ꯇꯧꯕꯤꯌꯨ",
        "directory_header": "ꯑꯇꯣꯞꯄ ꯃꯇꯦꯡ ꯅꯝꯕꯔꯁꯤꯡ",
        "helpline": "ꯍꯦꯜꯞꯂꯥꯏꯟ:",
        "audio_response": "🔊 ꯁꯍꯥꯌꯒꯤ ꯄꯥꯎꯈꯨꯝ:",
        "med_title": "ꯃꯦꯗꯤꯀꯦꯜ ꯑꯃꯁꯨꯡ ꯑꯦꯝꯕꯨꯂꯦꯟꯁ",
        "police_title": "ꯄꯨꯂꯤꯁ ꯑꯃꯁꯨꯡ ꯉꯥꯛꯊꯣꯛꯄ",
        "women_title": "ꯅꯨꯄꯤꯒꯤ ꯃꯇꯦꯡ ꯅꯝꯕꯔ",
        "fire_title": "ꯃꯩ ꯊꯥꯛꯄ ꯑꯃꯁꯨꯡ ꯀꯟꯕ",
        "disaster_title": "ꯈꯨꯗꯣꯡꯊꯤꯕꯒꯤ ꯃꯇꯦꯡ",
        "call_txt": "ꯀꯣꯜ ꯇꯧꯕꯤꯌꯨ"
    },
}

# ---------------------------------------------------------------------------
# Emergency Directory
# ---------------------------------------------------------------------------
EMERGENCY_DIRECTORY = [
    {
        "key": "med_title",
        "icon": "🚑",
        "number": "108",
        "desc": "Medical emergencies, accidents, cardiac arrest, trauma & urgent patient dispatch.",
    },
    {
        "key": "police_title",
        "icon": "👮",
        "number": "112",
        "desc": "Immediate law enforcement, road crash clearance & accident site protection.",
    },
    {
        "key": "women_title",
        "icon": "🛡️",
        "number": "1091",
        "desc": "Direct protection, helpline assistance and intervention against harassment.",
    },
    {
        "key": "fire_title",
        "icon": "🚒",
        "number": "101",
        "desc": "Fire hazard suppression, severe burns mitigation and trapped vehicle extraction.",
    },
    {
        "key": "disaster_title",
        "icon": "🌊",
        "number": "1078",
        "desc": "Relief and evacuation assistance during floods, earthquakes and storms.",
    },
]

CATEGORY_INDEX = {"medical": 0, "police": 1, "assault": 2, "fire": 3, "disaster": 4}

# ---------------------------------------------------------------------------
# Script Ranges & Romanized Keywords for All 12 Languages
# ---------------------------------------------------------------------------
SCRIPT_RANGES = [
    (0x0C80, 0x0CFF, "kn_IN"),  # Kannada
    (0x0900, 0x097F, "hi_IN"),  # Devanagari (Hindi/Marathi)
    (0x0B80, 0x0BFF, "ta_IN"),  # Tamil
    (0x0C00, 0x0C7F, "te_IN"),  # Telugu
    (0x0D00, 0x0D7F, "ml_IN"),  # Malayalam
    (0x0980, 0x09FF, "bn_IN"),  # Bengali
    (0x0A80, 0x0AFF, "gu_IN"),  # Gujarati
    (0x0A00, 0x0A7F, "pa_IN"),  # Punjabi
    (0x0B00, 0x0B7F, "or_IN"),  # Odia
    (0xABC0, 0xABFF, "mni_IN"), # Meitei Mayek
]

ROMAN_LANG_KEYWORDS = {
    "kn_IN": {
        "nana", "nanna", "nanage", "namage", "tande", "taayi", "thande", "thaayi",
        "appa", "amma", "edeno", "edenoovu", "ede", "novu", "usiru", "husiradal",
        "husiru", "usiradalu", "tondare", "tandre", "aagide", "aagite", "shuruagite",
        "shuru", "agat", "aagatte", "bahla", "bahala", "haavu", "havu", "kacchide",
        "kachide", "beku", "sahaya", "madadi", "dayavittu", "rakta", "benki",
        "aspatre", "aaspatre", "othada", "kodi", "gundige", "sikakide", "biddare",
        "apaghata", "gaaya", "muridide", "churukide", "sudide"
    },
    "hi_IN": {
        "mujhe", "humein", "humko", "mere", "meri", "pita", "pitaji", "papa", "mataji",
        "seene", "dard", "saans", "takleef", "ruk", "chhati", "saanp", "saap",
        "bachao", "madad", "khoon", "chot", "ghayal", "aspatal", "jaldi", "kripya", "bhai",
        "hadsa", "jal", "gira", "toot", "haddi"
    },
    "ta_IN": {
        "enakku", "engalukku", "nenju", "vali", "swasam", "moochu", "thinaral",
        "appa", "thanthai", "amma", "pambu", "paambu", "kadichathu", "udavi",
        "vendum", "ratham", "thee", "kaapathunga", "maruthuvamanai", "seekiram",
        "vibathu", "kaayam", "murivu"
    },
    "te_IN": {
        "naaku", "maaku", "gunde", "noppi", "swasa", "kastam", "adatam", "nanna",
        "thandri", "amma", "paamu", "karichindi", "sahayam", "kavali", "raktam",
        "mantalu", "kapadandi", "aaspattri", "thondaraga", "pramadam", "gayamu"
    },
    "ml_IN": {
        "enikku", "njangalkku", "nenju", "vedana", "shwasam", "muttal", "achan",
        "amma", "paampu", "kadichu", "sahayam", "venam", "raktham", "thee",
        "rakshikku", "aashupathri", "vegam", "apadakam", "murivu"
    },
    "mr_IN": {
        "mala", "aamhaala", "majhya", "chhati", "dukhate", "dukhne", "shwas",
        "trass", "vadil", "baba", "saap", "chavla", "madat", "havi", "rakta",
        "vachva", "davakhana", "lavkar", "apghat", "jhaala"
    },
    "bn_IN": {
        "amar", "amader", "buker", "byatha", "batha", "shwas", "kosto", "baba",
        "shap", "kamreche", "sahajjo", "dorkar", "rokto", "aagun", "bachan",
        "haspatal", "taratari", "durghotona"
    },
    "gu_IN": {
        "mane", "amne", "maru", "chhati", "dard", "shwas", "takleef", "bapa",
        "saap", "dankhyo", "madad", "joiye", "lohi", "aag", "bachavo", "davakhana", "jaldi", "aksmata"
    },
    "pa_IN": {
        "meri", "mere", "sanu", "chhati", "peerh", "saah", "aukh", "bapu",
        "sapp", "dangea", "madad", "chahidi", "khoon", "agg", "bachao", "aspataal", "cheti", "haadsa"
    },
    "or_IN": {
        "mora", "aamara", "chhati", "bitha", "shwasa", "kasta", "bapa", "sapa",
        "kamudichi", "sahajya", "darkar", "rakta", "nia", "bachantu", "sighra", "durghatana"
    },
    "mni_IN": {
        "eigi", "eikhoi", "thamoigi", "chikpa", "masha", "ithak", "lin",
        "chikre", "mateng", "pibiyu", "ee", "mei", "kanbiyu", "thuna"
    },
    "en_IN": {
        "accident", "crash", "bleeding", "blood", "fracture", "bone", "burn", "burns",
        "fall", "head", "injury", "choking", "drowning", "electric", "shock", "chest",
        "heart", "pain", "breathe", "ambulance", "hospital"
    }
}

# ---------------------------------------------------------------------------
# Comprehensive Minor & Major Accidents Triage
# ---------------------------------------------------------------------------
def identify_condition(text: str) -> str:
    t = text.lower()
    
    # 1. Road Accident / Severe Trauma / Bleeding / Fractures
    if any(k in t for k in [
        "accident", "crash", "bike", "car", "hit", "run over", "bleeding", "blood",
        "fracture", "bone broke", "bone broken", "cut", "wound", "injury", "head injury",
        "apaghata", "gaaya", "rakta", "muridide", "rakthasrava",
        "haadsa", "khoon", "chot", "toot", "haddi", "ghayal",
        "vibathu", "ratham", "kaayam", "murivu",
        "pramadam", "raktam", "gayamu", "virigindi"
    ]):
        return "accident_trauma"

    # 2. Cardiac & Respiratory Emergencies
    if any(k in t for k in [
        "heart", "chest", "breath", "breathing", "cardiac", "stroke", "attack",
        "ede", "edeno", "usir", "husir", "gundige", "seene", "chhati", "saans",
        "nenju", "moochu", "swasam", "gunde", "swasa", "buker", "shwas", "peerh", "saah"
    ]):
        return "cardiac"

    # 3. Snakebite & Envenomation
    if any(k in t for k in [
        "snake", "bite", "haavu", "havu", "kachide", "saanp", "saap",
        "paambu", "paamu", "lin", "chikpa", "sapp", "sapa", "dankh"
    ]):
        return "snakebite"

    # 4. Burns & Scalds (Minor to Major)
    if any(k in t for k in [
        "fire", "burn", "burns", "hot water", "scald", "cylinder", "blast",
        "benki", "sudide", "aag", "jal gaya", "thee", "agg", "nia", "mei"
    ]):
        return "burns"

    # 5. Electric Shock
    if any(k in t for k in [
        "electric", "shock", "current", "wire", "vidyut", "karent"
    ]):
        return "electric_shock"

    # 6. Choking / Asphyxiation
    if any(k in t for k in [
        "chok", "stuck in throat", "gantlu", "adagide", "gale mein", "thொண்டை"
    ]):
        return "choking"

    # 7. Poison / Pesticide Ingestion
    if any(k in t for k in [
        "poison", "pesticide", "chemical", "visha", "keetnashaka", "zeher", "marunthu"
    ]):
        return "pesticide"

    # 8. Maternal / Delivery
    if any(k in t for k in [
        "pregnant", "pregnancy", "delivery", "labor", "maternal", "garbhini", "prasava"
    ]):
        return "maternal"

    # 9. Violence / Crime / Police
    if any(k in t for k in [
        "police", "thief", "robbery", "attack", "fight", "assault", "harassment", "chor", "bachao"
    ]):
        return "police"

    return "general_medical"

CONDITION_FALLBACKS = {
    "accident_trauma": {
        "service_idx": 0,
        "urgency": "critical",
        "kn_IN": (
            "ಅಪಘಾತ ತುರ್ತು ಪರಿಸ್ಥಿತಿ! ದಯವಿಟ್ಟು ರೋಗಿಯನ್ನು ಅತಿಯಾಗಿ ಚಲಿಸಬೇಡಿ ಮತ್ತು ತಕ್ಷಣ 108 ಆಂಬ್ಯುಲೆನ್ಸ್‌ಗೆ ಕರೆ ಮಾಡಿ.",
            [
                "ತೀವ್ರ ರಕ್ತಸ್ರಾವವಿದ್ದರೆ, ಸ್ವಚ್ಛವಾದ ಬಟ್ಟೆಯಿಂದ ಗಾಯದ ಮೇಲೆ ನೇರವಾಗಿ ಬಲವಾದ ಒತ್ತಡ ಹಾಕಿ ಹಿಡಿಯಿರಿ.",
                "ಕುತ್ತಿಗೆ ಅಥವಾ ಬೆನ್ನುಮೂಳೆಗೆ ಪೆಟ್ಟಾಗಿದ್ದರೆ, ರೋಗಿಯನ್ನು ಎತ್ತಬೇಡಿ ಅಥವಾ ತಿರುಗಿಸಬೇಡಿ (ಬೆನ್ನಿನ ಸ್ಥಿರತೆ ಮುಖ್ಯ).",
                "ಮೂಳೆ ಮುರಿದಿದ್ದರೆ (Fracture), ಆ ಜಾಗವನ್ನು ಅಲುಗಾಡಿಸದೆ ಸ್ಪ್ಲಿಂಟ್ ಅಥವಾ ಮರದ ತುಂಡಿನಿಂದ ಸ್ಥಿರವಾಗಿರಿಸಿ.",
                "ಗಾಯಾಳುವಿಗೆ ಕುಡಿಯಲು ನೀರು ಅಥವಾ ಆಹಾರ ನೀಡಬೇಡಿ. ತಕ್ಷಣ 108 ಬರುವವರೆಗೆ ಜೊತೆಯಲ್ಲೇ ಇರಿ."
            ]
        ),
        "hi_IN": (
            "दुर्घटना / चोट की आपात स्थिति! मरीज को ज्यादा न हिलाएं और तुरंत 108 एम्बुलेंस को कॉल करें।",
            [
                "यदि खून बह रहा है, तो साफ कपड़े से घाव पर सीधा और मजबूत दबाव बनाए रखें।",
                "यदि रीढ़ या गर्दन में चोट की आशंका है, तो मरीज को बिल्कुल न हिलाएं।",
                "टूटी हुई हड्डी को हिलाने की कोशिश न करें, उसे सीधा सहारा देकर रखें।",
                "घायल व्यक्ति को कुछ भी खाने या पीने को न दें और 108 का इंतजार करें।"
            ]
        ),
        "en_IN": (
            "Accident & Trauma Emergency! Keep the patient still and dial 108 immediately.",
            [
                "For severe bleeding: Apply firm, direct pressure over the wound with a clean cloth or gauze.",
                "Do NOT move the patient if head, neck, or spine injuries are suspected.",
                "If a bone is fractured: Keep the injured limb completely still and supported.",
                "Do not give food or water to the injured person. Await 108 emergency paramedics."
            ]
        ),
    },
    "snakebite": {
        "service_idx": 0,
        "urgency": "critical",
        "kn_IN": (
            "ಹಾವು ಕಚ್ಚಿದ ತುರ್ತು ಪರಿಸ್ಥಿತಿ! ರೋಗಿಯನ್ನು ಶಾಂತವಾಗಿರಿಸಿ, ತಕ್ಷಣ 108 ಕರೆ ಮಾಡಿ.",
            [
                "ಕಚ್ಚಿದ ಅಂಗವನ್ನು ಅಲುಗಾಡಿಸದೆ ಹೃದಯದ ಮಟ್ಟಕ್ಕಿಂತ ಕೆಳಗೆ ಇರಿಸಿ.",
                "ಗಟ್ಟಿಯಾಗಿ ಹಗ್ಗ ಕಟ್ಟಬೇಡಿ, ಗಾಯವನ್ನು ಕತ್ತರಿಸಬೇಡಿ ಅಥವಾ ವಿಷ ಹೀರುವ ಪ್ರಯತ್ನ ಮಾಡಬೇಡಿ.",
                "ಆಂಟಿ-ವೆನಮ್ (ASV) ಲಭ್ಯವಿರುವ ಸಮೀಪದ ಆಸ್ಪತ್ರೆಗೆ ತಕ್ಷಣ ಕರೆದೊಯ್ಯಿರಿ."
            ]
        ),
        "hi_IN": (
            "सांप के काटने पर तुरंत 108 पर कॉल करें और शांत रहें।",
            [
                "काटे गए अंग को हिलाए बिना दिल के स्तर से नीचे रखें।",
                "कसकर पट्टी न बांधें, चीरा न लगाएं और जहर चूसने की कोशिश न करें।",
                "मरीज को तुरंत एंटी-स्नेक वेनम वाले अस्पताल ले जाएं।"
            ]
        ),
        "en_IN": (
            "Snakebite emergency! Keep the victim still and calm, dial 108 immediately.",
            [
                "Immobilize the bitten limb and keep it below heart level.",
                "DO NOT apply a tight tourniquet, do not cut the wound, and do not suck venom.",
                "Rush immediately to the nearest hospital equipped with Anti-Snake Venom (ASV)."
            ]
        ),
    },
    "general_medical": {
        "service_idx": 0,
        "urgency": "urgent",
        "kn_IN": (
            "ತುರ್ತು ಪರಿಸ್ಥಿತಿ! ಗಾಬರಿಯಾಗಬೇಡಿ, ತಕ್ಷಣ 108 ಆಂಬ್ಯುಲೆನ್ಸ್‌ಗೆ ಕರೆ ಮಾಡಿ.",
            [
                "ವ್ಯಕ್ತಿಯ ಉಸಿರಾಟ ಮತ್ತು ಪ್ರಜ್ಞೆಯನ್ನು ನಿರಂತರವಾಗಿ ಪರಿಶೀಲಿಸಿ.",
                "ಗಾಯ ಅಥವಾ ರಕ್ತಸ್ರಾವವಿದ್ದರೆ ಸ್ವಚ್ಛವಾದ ಬಟ್ಟೆಯಿಂದ ಗಾಯದ ಮೇಲೆ ನೇರ ಒತ್ತಡ ಹಾಕಿ.",
                "ಆಂಬ್ಯುಲೆನ್ಸ್ ಬರುವವರೆಗೆ ರೋಗಿಯೊಂದಿಗೆ ಶಾಂತವಾಗಿ ಇರಿ."
            ]
        ),
        "hi_IN": (
            "आपात स्थिति! शांत रहें और तुरंत 108 पर कॉल करें।",
            [
                "व्यक्ति की सांस और होश की जांच करें।",
                "घाव पर साफ कपड़े से सीधा दबाव बनाकर रखें।",
                "तुरंत एम्बुलेंस की प्रतीक्षा करें।"
            ]
        ),
        "en_IN": (
            "Medical Emergency! Please stay calm and immediately call 108.",
            [
                "Check airway, responsiveness, and normal breathing continuously.",
                "Apply clean, firm pressure to any actively bleeding wounds.",
                "Stay with the patient until the emergency medical team arrives."
            ]
        ),
    }
}

# ---------------------------------------------------------------------------
# Language & Response Detection
# ---------------------------------------------------------------------------
def get_env(name):
    return os.getenv(name, "").strip()

def to_hyphen(code):
    return code.replace("_", "-")

def detect_lang(text: str, fallback: str) -> str:
    if not text:
        return fallback

    counts = {}
    for ch in text:
        cp = ord(ch)
        for lo, hi, code in SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    if counts:
        best_script = max(counts, key=counts.get)
        if best_script == "hi_IN" and fallback == "mr_IN":
            return "mr_IN"
        return best_script

    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    scores = {}
    for code, kw_set in ROMAN_LANG_KEYWORDS.items():
        matched = words & kw_set
        if matched:
            scores[code] = len(matched)

    if scores:
        best_code = max(scores, key=scores.get)
        if fallback in scores and scores[fallback] >= scores[best_code] - 1:
            return fallback
        return best_code

    return fallback

def get_fallback_response(transcript: str, lang_code: str):
    cond = identify_condition(transcript)
    cond_data = CONDITION_FALLBACKS.get(cond, CONDITION_FALLBACKS["accident_trauma"])
    service = EMERGENCY_DIRECTORY[cond_data["service_idx"]]
    urgency = cond_data["urgency"]

    if lang_code in cond_data:
        reply, steps = cond_data[lang_code]
    elif "en_IN" in cond_data:
        reply, steps = cond_data["en_IN"]
    else:
        reply, steps = CONDITION_FALLBACKS["accident_trauma"]["en_IN"]

    return service, reply, steps, urgency

# ---------------------------------------------------------------------------
# AI Engine: STT, Groq Reasoning, TTS
# ---------------------------------------------------------------------------
def transcribe(audio_path, lang_code):
    api_key = get_env("GNANI_API_KEY")
    if not api_key:
        return "", "GNANI_API_KEY is missing."

    # Preprocess audio with bandpass and noise gating
    clean_audio_path = preprocess_telephonic_audio(audio_path)
    stt_lang = "en-IN" if lang_code == "mni_IN" else to_hyphen(lang_code)

    try:
        with open(clean_audio_path, "rb") as f:
            r = requests.post(
                STT_URL,
                headers={"X-API-Key-ID": api_key},
                files={"audio_file": ("voice.wav", f, "audio/wav")},
                data={"language_code": stt_lang, "format": "transcribe"},
                timeout=30,
            )
    except Exception as e:
        return "", f"Speech connection failed: {e}"

    if r.status_code != 200:
        return "", f"Speech recognition failed ({r.status_code})"

    try:
        data = r.json()
    except ValueError:
        return "", "Invalid response from speech server."

    transcript = str(data.get("transcript") or data.get("text") or "").strip()
    return transcript, ""

def groq_triage(transcript: str, detected_lang: str):
    groq_key = get_env("GROQ_API_KEY")
    if not groq_key:
        return None, "Groq key missing"

    system_prompt = f"""You are Sahaya, an empathetic AI emergency first-aid dispatcher in India.
Your mission is to handle all minor and major emergencies (road accidents, severe bleeding, fractures, burns, cardiac arrest, snakebites, electric shock, choking, etc.).

CRITICAL RULES:
1. DETECT INTENDED LANGUAGE:
   - Identify whether the caller used Kannada, Hindi, Tamil, Telugu, Malayalam, Marathi, Bengali, Gujarati, Punjabi, Odia, Manipuri, or English.
   - If Romanized Kannada is detected (e.g. 'nana tande...', 'apaghata...'), set "lang_code": "kn_IN".

2. RESPONSE MUST BE IN THE EXACT NATIVE SCRIPT:
   - Respond strictly in the native script of that detected language.
   - NEVER reply in Hindi or English if Kannada was spoken!

3. ACCURATE FIRST-AID:
   - For road accidents: direct pressure on bleeding, do not move neck/spine, immobilize fractures, call 108.
   - For burns: 20 mins clean cool running water, no ice/toothpaste.
   - For cardiac: loosen clothing, CPR if unresponsive, call 108.

4. RETURN ONLY STRICT RAW JSON:
{{
  "lang_code": "{detected_lang}",
  "category": "medical",
  "urgency": "critical",
  "reply": "Calm supportive response with call-to-action in native script",
  "steps": [
    "Step 1 in native script",
    "Step 2 in native script",
    "Step 3 in native script",
    "Step 4 in native script"
  ]
}}"""

    custom_model = get_env("GROQ_MODEL")
    models_to_try = ([custom_model] if custom_model else []) + [
        m for m in GROQ_MODELS if m != custom_model
    ]

    for model_name in models_to_try:
        try:
            r = requests.post(
                GROQ_CHAT_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {groq_key}",
                },
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Emergency: {transcript}"},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 550,
                },
                timeout=20,
            )
            if r.status_code != 200:
                continue

            content = r.json()["choices"][0]["message"]["content"]
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            clean_json = re.sub(r"```(json)?", "", content).strip()
            start, end = clean_json.find("{"), clean_json.rfind("}")
            if start == -1 or end <= start:
                continue

            data = json.loads(clean_json[start : end + 1])
            reply = str(data.get("reply", "")).strip()
            if not reply:
                continue

            steps = [
                re.sub(r"^\d+\.\s*", "", str(s).strip())
                for s in data.get("steps", [])
                if str(s).strip()
            ][:5]

            det_lang = str(data.get("lang_code", detected_lang)).strip()
            if det_lang not in TRANSLATIONS:
                det_lang = detected_lang

            return {
                "lang_code": det_lang,
                "category": str(data.get("category", "medical")).strip().lower(),
                "urgency": str(data.get("urgency", "critical")).strip().lower(),
                "reply": reply,
                "steps": steps,
            }, ""
        except Exception:
            continue

    return None, "Offline fallback active"

def timbre_speak(text: str, lang_code: str):
    if lang_code not in TIMBRE_LANGS:
        return None, "Voice not supported for this dialect"
    api_key = get_env("GNANI_API_KEY")
    if not api_key:
        return None, "API key missing"

    try:
        r = requests.post(
            TTS_URL,
            headers={"X-API-Key-ID": api_key, "Content-Type": "application/json"},
            json={
                "text": text,
                "voice": get_env("TIMBRE_VOICE") or "Shivani",
                "model": get_env("TIMBRE_MODEL") or "timbre-v3.0",
                "language": to_hyphen(lang_code),
                "speed": 1.0,
                "audio_config": {
                    "sample_rate": 24000,
                    "num_channels": 1,
                    "sample_width": 2,
                    "encoding": "linear_pcm",
                    "container": "wav",
                },
            },
            timeout=25,
        )
        if r.status_code != 200:
            return None, "TTS failed"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(r.content)
            return tmp.name, ""
    except Exception:
        return None, "TTS error"

# ---------------------------------------------------------------------------
# UI Construction Matching Video
# ---------------------------------------------------------------------------
def header_html(t):
    return f"""
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
        <div class="sahaya-sub">{html.escape(t['subtitle'])}</div>
    </div>
    """

def result_card(service, steps, urgency, rt):
    body = "<ol class='step-list'>" + "".join(f"<li>{html.escape(s)}</li>" for s in steps) + "</ol>"
    urgency_tag = f'<span class="badge badge-{urgency}">{urgency.upper()}</span>' if urgency else ""
    return (
        '<div class="emergency-banner card-box">'
        f' <div class="emergency-icon-wrap">{service["icon"]}</div>'
        '  <div class="emergency-info">'
        f'   <div class="emergency-title-row"><h2>{html.escape(rt.get(service["key"], service["key"]))}</h2>{urgency_tag}</div>'
        f'   <p class="service-desc">{html.escape(service["desc"])}</p>'
        '  </div>'
        '</div>'
        '<div class="content-card card-box">'
        f' <div class="card-heading">💊 {html.escape(rt["first_aid_label"])}</div>'
        f' {body}'
        f' <a class="call-action-btn" href="tel:{service["number"]}">'
        f'   📞 {html.escape(rt["call_btn"].format(number=service["number"]))}'
        '  </a>'
        '</div>'
    )

def directory_html(t):
    cards = []
    for srv in EMERGENCY_DIRECTORY:
        title_text = t.get(srv["key"], srv["key"])
        cards.append(f"""
        <div class="directory-card">
            <div class="directory-info">
                <span class="directory-icon">{srv['icon']}</span>
                <div>
                    <div class="directory-name">{html.escape(title_text)}</div>
                    <div class="directory-num">{srv['number']}</div>
                </div>
            </div>
            <a class="call-badge" href="tel:{srv['number']}">{t['call_txt']}</a>
        </div>
        """)
    return f"""
    <div style="margin-top: 25px;">
        <h4 style="color: #334155; font-size: 16px; margin-bottom: 12px; font-weight: 700;">📁 {html.escape(t['directory_header'])}</h4>
        {''.join(cards)}
    </div>
    """

# ---------------------------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------------------------
def run_pipeline(transcript: str, selected_lang: str):
    reply_lang = detect_lang(transcript, selected_lang)
    rt = TRANSLATIONS.get(reply_lang, TRANSLATIONS["en_IN"])
    transcript_bubble = f"**{rt['detected']}**\n\n💬 {transcript}"
    yield "", transcript_bubble, "", None, ""

    triage, _ = groq_triage(transcript, reply_lang)

    if triage:
        reply_lang = triage.get("lang_code", reply_lang)
        rt = TRANSLATIONS.get(reply_lang, TRANSLATIONS["en_IN"])
        transcript_bubble = f"**{rt['detected']}**\n\n💬 {transcript}"
        service = EMERGENCY_DIRECTORY[CATEGORY_INDEX.get(triage["category"], 0)]
        reply_text = triage["reply"]
        steps = triage["steps"]
        urgency = triage["urgency"]
    else:
        service, reply_text, steps, urgency = get_fallback_response(transcript, reply_lang)

    reply_bubble = f"**{rt['audio_response']}**\n\n{reply_text}"
    card = result_card(service, steps, urgency, rt)
    yield "", transcript_bubble, reply_bubble, None, card

    # Spoken narration in exact language via Timbre
    steps_narration = " ".join([f"{i+1}. {s}" for i, s in enumerate(steps)])
    spoken_text = f"{reply_text}. {steps_narration}"
    audio_path, _ = timbre_speak(spoken_text, reply_lang)
    yield "", transcript_bubble, reply_bubble, audio_path, card

def on_audio(audio_path, lang_code):
    if not audio_path:
        yield "⚠️ Please record your voice message first.", "", "", None, ""
        return
    yield "", "", "", None, ""
    transcript, err = transcribe(audio_path, lang_code)
    if err:
        yield f"⚠️ {err}", "", "", None, ""
        return
    yield from run_pipeline(transcript, lang_code)

def on_text(text, lang_code):
    text = (text or "").strip()
    if not text:
        yield "⚠️ Please enter your message first.", "", "", None, ""
        return
    yield from run_pipeline(text, lang_code)

def apply_language(lang_code):
    t = TRANSLATIONS.get(lang_code, TRANSLATIONS["en_IN"])
    return (
        header_html(t),
        t["sos_btn"],
        gr.update(label=t["mic_header"]),
        gr.update(label=t["type_header"], placeholder=t["type_placeholder"]),
        t["send"],
        directory_html(t),
    )

# ---------------------------------------------------------------------------
# Clean Video Layout Styling
# ---------------------------------------------------------------------------
CSS = """
body, .gradio-container {
    background-color: #fdfbf7 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    max-width: 680px !important;
    margin: 0 auto !important;
}

/* Header */
.sahaya-header {
    text-align: center;
    margin-top: 15px;
    margin-bottom: 8px;
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
    height: 26px;
}
.sahaya-wave-bar {
    width: 4px;
    background: #0284c7;
    border-radius: 3px;
    animation: waveAnim 1s ease-in-out infinite alternate;
}
.sahaya-wave-bar:nth-child(1) { height: 40%; animation-delay: 0.1s; }
.sahaya-wave-bar:nth-child(2) { height: 80%; animation-delay: 0.3s; }
.sahaya-wave-bar:nth-child(3) { height: 100%; animation-delay: 0.2s; }
.sahaya-wave-bar:nth-child(4) { height: 50%; animation-delay: 0.4s; }

@keyframes waveAnim {
    0% { transform: scaleY(0.4); background: #0284c7; }
    100% { transform: scaleY(1); background: #16a34a; }
}

.sahaya-sub {
    font-size: 15px;
    color: #475569;
    font-weight: 600;
    margin-top: 4px;
}

/* SOS Red Button */
.sos-panic-btn {
    width: 100%;
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
    color: #ffffff !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 14px 18px !important;
    box-shadow: 0 4px 12px rgba(220, 38, 38, 0.25) !important;
    margin: 12px 0 !important;
    cursor: pointer;
}

/* Send Alert Button */
.send-alert-btn {
    background: #0284c7 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 12px !important;
    margin-top: 6px !important;
}

/* Card Boxes */
.card-box {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
}

.emergency-banner {
    display: flex;
    align-items: center;
    gap: 16px;
    background: #FEF2F2 !important;
    border: 1.5px solid #FECACA !important;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-top: 1rem;
}
.emergency-icon-wrap { font-size: 2.2rem; }
.emergency-info { flex: 1; }
.emergency-title-row { display: flex; align-items: center; gap: 12px; }
.emergency-title-row h2 { font-size: 1.3rem; margin: 0; color: #DC2626 !important; font-weight: 800; }
.service-desc { color: #334155 !important; font-size: 0.95rem; font-weight: 600; margin: 4px 0 0; }
.badge { padding: 3px 12px; border-radius: 20px; font-size: 0.72rem; font-weight: 800; }
.badge-critical { background: #DC2626; color: #FFF !important; }
.badge-urgent { background: #D97706; color: #FFF !important; }

.content-card {
    padding: 1.4rem 1.6rem !important;
    margin-top: 0.8rem !important;
}
.card-heading {
    font-size: 1.2rem !important;
    font-weight: 800 !important;
    color: #0F172A !important;
}
.step-list {
    margin: 0.8rem 0 1.2rem !important;
    padding-left: 1.6rem !important;
}
.step-list li {
    color: #0F172A !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    line-height: 1.7 !important;
    margin-bottom: 0.5rem !important;
}

.call-action-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    background: #16A34A;
    color: #FFFFFF !important;
    text-decoration: none !important;
    font-size: 1.15rem;
    font-weight: 800;
    padding: 12px 18px;
    border-radius: 8px;
    margin-top: 1rem;
}

/* Directory Cards */
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

INIT_JS = """() => {
  window._sosCtx = null;
  window._sosOsc = null;
  window._sosPlaying = false;
  window.startSiren = function() {
    if (window._sosPlaying) { window.stopSiren(); return; }
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const gain = ctx.createGain();
    gain.gain.value = 0.85;
    gain.connect(ctx.destination);
    const osc = ctx.createOscillator();
    osc.type = 'sawtooth';
    osc.frequency.value = 800;
    osc.connect(gain);
    osc.start();
    function sweep() {
      if (!window._sosPlaying) return;
      const now = ctx.currentTime;
      osc.frequency.setValueAtTime(800, now);
      osc.frequency.linearRampToValueAtTime(1600, now + 0.5);
      osc.frequency.linearRampToValueAtTime(800, now + 1.0);
      setTimeout(sweep, 1000);
    }
    window._sosCtx = ctx; window._sosOsc = osc; window._sosPlaying = true;
    sweep();
    setTimeout(() => { window.stopSiren(); }, 15000);
  };
  window.stopSiren = function() {
    window._sosPlaying = false;
    try { window._sosOsc && window._sosOsc.stop(); } catch(e) {}
    try { window._sosCtx && window._sosCtx.close(); } catch(e) {}
    window._sosOsc = null; window._sosCtx = null;
  };
}"""

# ---------------------------------------------------------------------------
# UI Construction
# ---------------------------------------------------------------------------
t0 = TRANSLATIONS["kn_IN"]

with gr.Blocks(title="Sahaya Emergency", theme=gr.themes.Base(), css=CSS, js=INIT_JS) as demo:
    header = gr.HTML(header_html(t0))
    
    sos_btn = gr.Button(t0["sos_btn"], elem_classes=["sos-panic-btn"])
    sos_btn.click(
        fn=None,
        js="""() => {
            window.startSiren();
            setTimeout(() => { window.open('tel:1091', '_self'); }, 400);
        }"""
    )

    lang_dd = gr.Dropdown(
        choices=[(label, code) for code, label in LANGUAGES.items()],
        value="kn_IN",
        label="🌐 Language Selection / ಭಾಷೆ / भाषा / ꯃꯤꯇꯩꯂꯣꯟ",
    )

    mic = gr.Audio(
        sources=["microphone"],
        type="filepath",
        label=t0["mic_header"],
    )

    textbox = gr.Textbox(
        label=t0["type_header"],
        placeholder=t0["type_placeholder"],
        lines=2
    )
    
    send_btn = gr.Button(t0["send"], elem_classes=["send-alert-btn"])

    status = gr.Markdown("", visible=False)
    transcript_out = gr.Markdown("", elem_classes=["live-bubble"])
    reply_out = gr.Markdown("", elem_classes=["live-bubble"])
    reply_audio = gr.Audio(label="🔊 Sahaya Voice Response", type="filepath", autoplay=True, interactive=False)
    card_out = gr.HTML("")
    dir_html = gr.HTML(directory_html(t0))

    result_outputs = [status, transcript_out, reply_out, reply_audio, card_out]

    mic.stop_recording(on_audio, inputs=[mic, lang_dd], outputs=result_outputs)
    send_btn.click(on_text, inputs=[textbox, lang_dd], outputs=result_outputs)
    textbox.submit(on_text, inputs=[textbox, lang_dd], outputs=result_outputs)
    
    lang_dd.change(
        apply_language,
        inputs=lang_dd,
        outputs=[header, sos_btn, mic, textbox, send_btn, dir_html]
    )

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    demo.launch(server_name="0.0.0.0", server_port=port)
