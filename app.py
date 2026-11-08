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
    spectral gating to isolate voice from telephone line static, engine humming, and wind.
    """
    if not audio_path or not os.path.exists(audio_path):
        return audio_path

    try:
        data, sr = sf.read(audio_path)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # 4th order Butterworth Bandpass Filter (300Hz - 3400Hz speech/telephone range)
        lowcut = 300.0
        highcut = min(3400.0, (sr / 2.0) - 50.0)
        sos = butter(4, [lowcut, highcut], btype='bandpass', fs=sr, output='sos')
        filtered_audio = sosfilt(sos, data)

        # Spectral Gating Noise Cancellation
        if NOISEREDUCE_AVAILABLE:
            cleaned_audio = nr.reduce_noise(
                y=filtered_audio,
                sr=sr,
                stationary=False,
                prop_decrease=0.75
            )
        else:
            cleaned_audio = filtered_audio

        # Normalize audio levels
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
    "kn_IN": "ಕನ್ನಡ (Kannada)",
    "en_IN": "English (India)",
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

TIMBRE_LANGS = {
    "en_IN", "hi_IN", "ta_IN", "te_IN", "kn_IN",
    "ml_IN", "mr_IN", "pa_IN", "bn_IN", "gu_IN",
}

TRANSLATIONS = {
    "kn_IN": {
        "title": "Sahaya",
        "subtitle": "Stay Safe. 🛡️",
        "sos_btn": "🚨 EMERGENCY SOS • LOUD ALARM + DIAL 1091",
        "mic_header": "🎙️ Tap to Start Listening",
        "type_header": "⌨️ Or Type Emergency Details",
        "type_placeholder": "ಉದಾ: ಹಾವು ಕಡಿತ, ಅಪಘಾತ, ರಕ್ತಸ್ರಾವ, ಆಂಬ್ಯುಲೆನ್ಸ್ ಬೇಕು...",
        "send": "Send Emergency Alert",
        "detected": "🗣️ ಪತ್ತೆಯಾದ ತುರ್ತು ವಿವರ:",
        "first_aid_label": "ತಕ್ಷಣದ ಪ್ರಥಮ ಚಿಕಿತ್ಸಾ ಕ್ರಮಗಳು:",
        "call_btn": "ತಕ್ಷಣ {number} ಗೆ ಕರೆ ಮಾಡಿ",
        "directory_header": "Emergency Services Directory",
        "audio_response": "🔊 ಸಹಾಯ ತುರ್ತು ಮಾರ್ಗದರ್ಶನ:",
        "med_title": "Medical & Ambulance",
        "police_title": "Police & Safety",
        "women_title": "Women Distress Helpline",
        "fire_title": "Fire & Rescue",
        "disaster_title": "Disaster Support",
        "call_txt": "Call"
    },
    "en_IN": {
        "title": "Sahaya",
        "subtitle": "Stay Safe. 🛡️",
        "sos_btn": "🚨 EMERGENCY SOS • LOUD ALARM + DIAL 1091",
        "mic_header": "🎙️ Tap to Start Listening",
        "type_header": "⌨️ Or Type Emergency Details",
        "type_placeholder": "e.g., Need an ambulance, snake bite, road accident...",
        "send": "Send Emergency Alert",
        "detected": "🗣️ Detected Emergency:",
        "first_aid_label": "Immediate First-Aid Steps:",
        "call_btn": "Call {number} Immediately",
        "directory_header": "Emergency Services Directory",
        "audio_response": "🔊 Sahaya Voice Guidance:",
        "med_title": "Medical & Ambulance",
        "police_title": "Police & Safety",
        "women_title": "Women Distress Helpline",
        "fire_title": "Fire & Rescue",
        "disaster_title": "Disaster Support",
        "call_txt": "Call"
    },
    "hi_IN": {
        "title": "Sahaya",
        "subtitle": "Stay Safe. 🛡️",
        "sos_btn": "🚨 EMERGENCY SOS • LOUD ALARM + DIAL 1091",
        "mic_header": "🎙️ Tap to Start Listening",
        "type_header": "⌨️ Or Type Emergency Details",
        "type_placeholder": "जैसे सांप का काटना, सड़क दुर्घटना, एम्बुलेंस चाहिए...",
        "send": "Send Emergency Alert",
        "detected": "🗣️ पता चली आपात स्थिति:",
        "first_aid_label": "प्राथमिक उपचार दिशानिर्देश:",
        "call_btn": "तुरंत {number} पर कॉल करें",
        "directory_header": "Emergency Services Directory",
        "audio_response": "🔊 सहायता वॉयस मार्गदर्शन:",
        "med_title": "Medical & Ambulance",
        "police_title": "Police & Safety",
        "women_title": "Women Distress Helpline",
        "fire_title": "Fire & Rescue",
        "disaster_title": "Disaster Support",
        "call_txt": "Call"
    },
    "ta_IN": {
        "title": "Sahaya",
        "subtitle": "Stay Safe. 🛡️",
        "sos_btn": "🚨 EMERGENCY SOS • LOUD ALARM + DIAL 1091",
        "mic_header": "🎙️ Tap to Start Listening",
        "type_header": "⌨️ Or Type Emergency Details",
        "type_placeholder": "எ.கா. விபத்து, பாம்பு கடி, அதிக ரத்தப்போக்கு...",
        "send": "Send Emergency Alert",
        "detected": "🗣️ கண்டறியப்பட்ட அவசரம்:",
        "first_aid_label": "உடனடி முதலுதவி வழிகாட்டுதல்கள்:",
        "call_btn": "உடனே {number} ஐ அழைக்கவும்",
        "directory_header": "Emergency Services Directory",
        "audio_response": "🔊 சஹாயா குரல் வழிகாட்டல்:",
        "med_title": "Medical & Ambulance",
        "police_title": "Police & Safety",
        "women_title": "Women Distress Helpline",
        "fire_title": "Fire & Rescue",
        "disaster_title": "Disaster Support",
        "call_txt": "Call"
    },
    "te_IN": {
        "title": "Sahaya",
        "subtitle": "Stay Safe. 🛡️",
        "sos_btn": "🚨 EMERGENCY SOS • LOUD ALARM + DIAL 1091",
        "mic_header": "🎙️ Tap to Start Listening",
        "type_header": "⌨️ Or Type Emergency Details",
        "type_placeholder": "ఉదా. పాము కాటు, ప్రమాదం, అంబులెన్స్ కావాలి...",
        "send": "Send Emergency Alert",
        "detected": "🗣️ గుర్తించబడిన అత్యవసరం:",
        "first_aid_label": "తక్షణ ప్రథమ చికిత్స మార్గదర్శకాలు:",
        "call_btn": "ఇప్పుడే {number} కి కాల్ చేయండి",
        "directory_header": "Emergency Services Directory",
        "audio_response": "🔊 సహాయ వాయిస్ మార్గదర్శకత్వం:",
        "med_title": "Medical & Ambulance",
        "police_title": "Police & Safety",
        "women_title": "Women Distress Helpline",
        "fire_title": "Fire & Rescue",
        "disaster_title": "Disaster Support",
        "call_txt": "Call"
    },
    "ml_IN": {
        "title": "Sahaya",
        "subtitle": "Stay Safe. 🛡️",
        "sos_btn": "🚨 EMERGENCY SOS • LOUD ALARM + DIAL 1091",
        "mic_header": "🎙️ Tap to Start Listening",
        "type_header": "⌨️ Or Type Emergency Details",
        "type_placeholder": "ഉദാ. പാമ്പ് കടി, അപകടം, ആംബുലൻസ് വേണം...",
        "send": "Send Emergency Alert",
        "detected": "🗣️ കണ്ടെത്തിയ അത്യാഹിതം:",
        "first_aid_label": "ഉടൻ ചെയ്യേണ്ട പ്രഥമശുശ്രൂഷാ ഘട്ടങ്ങൾ:",
        "call_btn": "ഉടൻ {number}-ലേക്ക് വിളിക്കുക",
        "directory_header": "Emergency Services Directory",
        "audio_response": "🔊 സഹായ വോയ്സ് പ്രതികരണം:",
        "med_title": "Medical & Ambulance",
        "police_title": "Police & Safety",
        "women_title": "Women Distress Helpline",
        "fire_title": "Fire & Rescue",
        "disaster_title": "Disaster Support",
        "call_txt": "Call"
    },
    "mr_IN": {
        "title": "Sahaya",
        "subtitle": "Stay Safe. 🛡️",
        "sos_btn": "🚨 EMERGENCY SOS • LOUD ALARM + DIAL 1091",
        "mic_header": "🎙️ Tap to Start Listening",
        "type_header": "⌨️ Or Type Emergency Details",
        "type_placeholder": "उदा. साप चावला, अपघात, छातीत दुखणे...",
        "send": "Send Emergency Alert",
        "detected": "🗣️ आढळलेली आणीबाणी:",
        "first_aid_label": "तातडीच्या प्रथमोपचार सूचना:",
        "call_btn": "आता {number} वर कॉल करा",
        "directory_header": "Emergency Services Directory",
        "audio_response": "🔊 साहाय्य व्हॉइस मार्गदर्शन:",
        "med_title": "Medical & Ambulance",
        "police_title": "Police & Safety",
        "women_title": "Women Distress Helpline",
        "fire_title": "Fire & Rescue",
        "disaster_title": "Disaster Support",
        "call_txt": "Call"
    },
    "gu_IN": {
        "title": "Sahaya",
        "subtitle": "Stay Safe. 🛡️",
        "sos_btn": "🚨 EMERGENCY SOS • LOUD ALARM + DIAL 1091",
        "mic_header": "🎙️ Tap to Start Listening",
        "type_header": "⌨️ Or Type Emergency Details",
        "type_placeholder": "દા.ત. સાપ કરડ્યો, અકસ્માત, એમ્બ્યુલન્સ જોઈએ...",
        "send": "Send Emergency Alert",
        "detected": "🗣️ મળેલી કટોકટી:",
        "first_aid_label": "તાત્કાલિક પ્રાથમિક સારવારનાં પગલાં:",
        "call_btn": "હમણાં {number} પર કૉલ કરો",
        "directory_header": "Emergency Services Directory",
        "audio_response": "🔊 સહાય વૉઇસ માર્ગદર્શન:",
        "med_title": "Medical & Ambulance",
        "police_title": "Police & Safety",
        "women_title": "Women Distress Helpline",
        "fire_title": "Fire & Rescue",
        "disaster_title": "Disaster Support",
        "call_txt": "Call"
    },
    "bn_IN": {
        "title": "Sahaya",
        "subtitle": "Stay Safe. 🛡️",
        "sos_btn": "🚨 EMERGENCY SOS • LOUD ALARM + DIAL 1091",
        "mic_header": "🎙️ Tap to Start Listening",
        "type_header": "⌨️ Or Type Emergency Details",
        "type_placeholder": "যেমন সাপে কাটা, दुर्घटना, रक्तপাত...",
        "send": "Send Emergency Alert",
        "detected": "🗣️ শনাক্ত জরুরি অবস্থা:",
        "first_aid_label": "জরুরি প্রাথমিক চিকিৎসার পদক্ষেপ:",
        "call_btn": "এখনই {number}-এ কল করুন",
        "directory_header": "Emergency Services Directory",
        "audio_response": "🔊 সহায় ভয়েস সহায়তা:",
        "med_title": "Medical & Ambulance",
        "police_title": "Police & Safety",
        "women_title": "Women Distress Helpline",
        "fire_title": "Fire & Rescue",
        "disaster_title": "Disaster Support",
        "call_txt": "Call"
    },
    "pa_IN": {
        "title": "Sahaya",
        "subtitle": "Stay Safe. 🛡️",
        "sos_btn": "🚨 EMERGENCY SOS • LOUD ALARM + DIAL 1091",
        "mic_header": "🎙️ Tap to Start Listening",
        "type_header": "⌨️ Or Type Emergency Details",
        "type_placeholder": "ਜਿਵੇਂ ਸੱਪ ਲੜਨਾ, ਹਾਦਸਾ, ਖੂਨ ਵਹਿਣਾ...",
        "send": "Send Emergency Alert",
        "detected": "🗣️ ਪਤਾ ਲੱਗੀ ਐਮਰਜੈਂਸੀ:",
        "first_aid_label": "ਤੁਰੰਤ ਮੁੱਢਲੀ ਸਹਾਇਤਾ ਕਦਮ:",
        "call_btn": "ਹੁਣੇ {number} 'ਤੇ ਕਾਲ ਕਰੋ",
        "directory_header": "Emergency Services Directory",
        "audio_response": "🔊 ਸਹਾਇਤਾ ਵੌਇਸ ਸੇਧ:",
        "med_title": "Medical & Ambulance",
        "police_title": "Police & Safety",
        "women_title": "Women Distress Helpline",
        "fire_title": "Fire & Rescue",
        "disaster_title": "Disaster Support",
        "call_txt": "Call"
    },
    "or_IN": {
        "title": "Sahaya",
        "subtitle": "Stay Safe. 🛡️",
        "sos_btn": "🚨 EMERGENCY SOS • LOUD ALARM + DIAL 1091",
        "mic_header": "🎙️ Tap to Start Listening",
        "type_header": "⌨️ Or Type Emergency Details",
        "type_placeholder": "ଯେପରି ସାପ କାମୁଡ଼ା, ଦୁର୍ଘଟଣା, ରକ୍ତସ୍ରାବ...",
        "send": "Send Emergency Alert",
        "detected": "🗣️ ଚିହ୍ନଟ ଜରୁରୀକାଳୀନ ପରିସ୍ଥିତି:",
        "first_aid_label": "ତୁରନ୍ତ ପ୍ରଥମ ଚିକିତ୍ସା ପଦକ୍ଷେପ:",
        "call_btn": "ତୁରନ୍ତ {number} କୁ କଲ୍ କରନ୍ତୁ",
        "directory_header": "Emergency Services Directory",
        "audio_response": "🔊 ସହାୟ ଭଏସ୍ ମାର୍ଗଦର୍ଶନ:",
        "med_title": "Medical & Ambulance",
        "police_title": "Police & Safety",
        "women_title": "Women Distress Helpline",
        "fire_title": "Fire & Rescue",
        "disaster_title": "Disaster Support",
        "call_txt": "Call"
    },
    "mni_IN": {
        "title": "Sahaya",
        "subtitle": "Stay Safe. 🛡️",
        "sos_btn": "🚨 EMERGENCY SOS • LOUD ALARM + DIAL 1091",
        "mic_header": "🎙️ Tap to Start Listening",
        "type_header": "⌨️ Or Type Emergency Details",
        "type_placeholder": "ꯂꯤꯟ ꯆꯤꯛꯄ, ꯑꯦꯛꯁꯤꯗꯦꯟꯠ, ꯃꯩ ꯆꯥꯛꯄ...",
        "send": "Send Emergency Alert",
        "detected": "🗣️ ꯈꯪꯗꯣꯛꯂꯕ ꯈꯨꯗꯣꯡꯊꯤꯕ:",
        "first_aid_label": "ꯈꯨꯗꯛꯇ ꯇꯧꯒꯗꯕ ꯊꯕꯛꯁꯤꯡ:",
        "call_btn": "{number} ꯗ ꯍꯧꯖꯤꯛꯃꯛ ꯀꯣꯜ ꯇꯧꯕꯤꯌꯨ",
        "directory_header": "Emergency Services Directory",
        "audio_response": "🔊 ꯁꯍꯥꯌꯒꯤ ꯄꯥꯎꯈꯨꯝ:",
        "med_title": "Medical & Ambulance",
        "police_title": "Police & Safety",
        "women_title": "Women Distress Helpline",
        "fire_title": "Fire & Rescue",
        "disaster_title": "Disaster Support",
        "call_txt": "Call"
    },
}

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
# Fallback Medical Triage Dictionaries (All 12 Languages)
# ---------------------------------------------------------------------------
def identify_condition(text: str) -> str:
    t = text.lower()
    if any(k in t for k in [
        "snake", "bite", "haavu", "havu", "kachide", "saanp", "saap",
        "paambu", "paamu", "lin", "chikpa", "sapp", "sapa", "dankh"
    ]):
        return "snakebite"

    if any(k in t for k in [
        "accident", "crash", "bike", "car", "hit", "run over", "bleeding", "blood",
        "fracture", "bone broke", "bone broken", "cut", "wound", "injury", "head injury",
        "apaghata", "gaaya", "rakta", "muridide", "rakthasrava",
        "haadsa", "khoon", "chot", "toot", "haddi", "ghayal",
        "vibathu", "ratham", "kaayam", "murivu",
        "pramadam", "raktam", "gayamu", "virigindi"
    ]):
        return "accident_trauma"

    if any(k in t for k in [
        "heart", "chest", "breath", "breathing", "cardiac", "stroke", "attack",
        "ede", "edeno", "usir", "husir", "gundige", "seene", "chhati", "saans",
        "nenju", "moochu", "swasam", "gunde", "swasa", "buker", "shwas", "peerh", "saah"
    ]):
        return "cardiac"

    if any(k in t for k in [
        "fire", "burn", "burns", "hot water", "scald", "cylinder", "blast",
        "benki", "sudide", "aag", "jal gaya", "thee", "agg", "nia", "mei"
    ]):
        return "burns"

    return "general_medical"

CONDITION_FALLBACKS = {
    "snakebite": {
        "service_idx": 0,
        "urgency": "critical",
        "kn_IN": (
            "ಹಾವು ಕಚ್ಚಿದ ತುರ್ತು ಪರಿಸ್ಥಿತಿ! ರೋಗಿಯನ್ನು ಶಾಂತವಾಗಿರಿಸಿ, ತಕ್ಷಣ 108 ಕರೆ ಮಾಡಿ.",
            [
                "ಕಚ್ಚಿದ ಅಂಗವನ್ನು ಅಲುಗಾಡಿಸದೆ ಹೃದಯದ ಮಟ್ಟಕ್ಕಿಂತ ಕೆಳಗೆ ಇರಿಸಿ.",
                "ಗಟ್ಟಿಯಾಗಿ ಹಗ್ಗ ಕಟ್ಟಬೇಡಿ, ಗಾಯವನ್ನು ಕತ್ತರಿಸಬೇಡಿ ಅಥವಾ ವಿಷ ಹೀರುವ ಪ್ರಯತ್ನ ಮಾಡಬೇಡಿ.",
                "ರೋಗಿಯನ್ನು ಸಂಪೂರ್ಣವಾಗಿ ಮಲಗಿಸಿ, ಗಾಬರಿಯಾಗದಂತೆ ಧೈರ್ಯ ನೀಡಿ.",
                "ಆಂಟಿ-ವೆನಮ್ (ASV) ಲಭ್ಯವಿರುವ ಸಮೀಪದ ಆಸ್ಪತ್ರೆಗೆ ತಕ್ಷಣ ಕರೆದೊಯ್ಯಿರಿ."
            ]
        ),
        "hi_IN": (
            "सांप के काटने की आपात स्थिति! मरीज को शांत रखें और तुरंत 108 पर कॉल करें।",
            [
                "काटे गए अंग को हिलाए बिना दिल के स्तर से नीचे स्थिर रखें।",
                "कसकर पट्टी न बांधें, घाव को न काटें और जहर चूसने का प्रयास न करें।",
                "मरीज को शांत रखें और चलने-फिरने न दें।",
                "एंटी-स्नेक वेनम की सुविधा वाले नजदीकी अस्पताल में तुरंत ले जाएं।"
            ]
        ),
        "ta_IN": (
            "பாம்பு கடி அவசரநிலை! நோயாளியை அமைதிப்படுத்தி உடனடியாக 108-ஐ அழைக்கவும்.",
            [
                "கடிபட்ட உறுப்பை அசைக்காமல் இதய மட்டத்திற்கு கீழே வைக்கவும்.",
                "காயத்தை கத்தியால் வெட்டவோ அல்லது விஷத்தை உறிஞ்சவோ கூடாது.",
                "நோயாளியை நடக்க விடாமல் முழு ஓய்வில் வைக்கவும்.",
                "உடனடியாக விஷமுறிவு மருந்து உள்ள மருத்துவமனைக்கு கொண்டு செல்லவும்."
            ]
        ),
        "te_IN": (
            "పాము కాటు అత్యవసర పరిస్థితి! రోగిని ప్రశాంతంగా ఉంచి వెంటనే 108కి కాల్ చేయండి.",
            [
                "కాటు వేసిన భాగాన్ని కదల్చకుండా గుండె స్థాయి కంటే తక్కువగా ఉంచండి.",
                "గాటు పెట్టడం లేదా విషాన్ని నోటితో పీల్చే ప్రయత్నం చేయవద్దు.",
                "రోగిని కంగారు పడకుండా స్థిరంగా ఉంచండి.",
                "యాంటీ-స్నేక్ వెనమ్ ఉన్న ఆసుపత్రికి త్వరగా తీసుకెళ్లండి."
            ]
        ),
        "ml_IN": (
            "പാമ്പ് കടിയേറ്റ അടിയന്തരാവസ്ഥ! രോഗിയെ ശാന്തനാക്കി ഉടൻ 108 വിളിക്കുക.",
            [
                "കടിയേറ്റ ഭാഗം അനക്കാതെ ഹൃദയനിരപ്പിൽ താഴെയായി വെക്കുക.",
                "മുറിവുണ്ടാക്കാനോ വിഷം വലിച്ച് പുറത്തെടുക്കാനോ ശ്രമിക്കരുത്.",
                "രോഗി പരിഭ്രാന്തനാകാതെ നോക്കുക, ചലനം പൂർണ്ണമായി ഒഴിവാക്കുക.",
                "ആന്റി-വെനം ലഭിക്കുന്ന ഏറ്റവും അടുത്തുള്ള ആശുപത്രിയിലെത്തിക്കുക."
            ]
        ),
        "mr_IN": (
            "सर्पदंश आणीबाणी! रुग्णाला शांत ठेवा आणि त्वरित 108 वर कॉल करा.",
            [
                "दंश झालेला अवयव न हलवता हृदयाच्या पातळीपेक्षा खाली स्थिर ठेवा.",
                "जखमेवर चीरा मारू नका किंवा विष तोंडाने ओढण्याचा प्रयत्न करू नका.",
                "रुग्णाला चालणे किंवा हालचाल करू देऊ नका.",
                "अँटी-व्हेनम उपलब्ध असलेल्या जवळच्या रुग्णालयात त्वरित दाखल करा."
            ]
        ),
        "gu_IN": (
            "સાપ કરડવાની કટોકટી! દર્દીને શાંત રાખો અને તરત જ 108 પર કૉલ કરો.",
            [
                "કરડેલા અંગને હલાવ્યા વિના હૃદયના સ્તરથી નીચે સ્થિર રાખો.",
                "ઘા પર કાપો ન મુકવો કે ઝેર ચૂસવાનો પ્રયત્ન ન કરવો.",
                "દર્દીને સંપૂર્ણ આરામ આપો અને ચાલવા ન દો.",
                "એન્ટિ-સ્નેક વેનમ સુવિધા ધરાવતી હોસ્પિટલમાં તરત લઈ જાઓ."
            ]
        ),
        "bn_IN": (
            "সাপের কামড়ের জরুরি অবস্থা! রোগীকে শান্ত রাখুন এবং অবিলম্বে 108 এ কল করুন।",
            [
                "দংশিত অঙ্গটি না নাড়িয়ে হৃদপিণ্ডের স্তরের নিচে স্থির রাখুন।",
                "ক্ষতস্থানে ব্লেড বা ছুরি চালাবেন না এবং বিষ চোষার চেষ্টা করবেন না।",
                "রোগীকে হাঁটতে দেবেন না, সম্পূর্ণ বিশ্রামে রাখুন।",
                "অ্যান্টি-ভেনম সমৃদ্ধ নিকটস্থ হাসপাতালে দ্রুত নিয়ে যান।"
            ]
        ),
        "pa_IN": (
            "ਸੱਪ ਦੇ ਕੱਟਣ ਦੀ ਐਮਰਜੈਂਸੀ! ਮਰੀਜ਼ ਨੂੰ ਸ਼ਾਂਤ ਰੱਖੋ ਅਤੇ ਤੁਰੰਤ 108 'ਤੇ ਕਾਲ ਕਰੋ।",
            [
                "ਡੰਗੇ ਹੋਏ ਅੰਗ ਨੂੰ ਹਿਲਾਏ ਬਿਨਾਂ ਦਿਲ ਦੇ ਪੱਧਰ ਤੋਂ ਹੇਠਾਂ ਰੱਖੋ।",
                "ਜ਼ਖ਼ਮ 'ਤੇ ਕੱਟ ਨਾ ਲਗਾਓ ਅਤੇ ਜ਼ਹਿਰ ਚੂਸਣ ਦੀ ਕੋਸ਼ਿਸ਼ ਨਾ ਕਰੋ।",
                "ਮਰੀਜ਼ ਨੂੰ ਚੱਲਣ-ਫਿਰਨ ਨਾ ਦਿਓ ਅਤੇ ਸ਼ਾਂਤ ਰੱਖੋ।",
                "ਐਂਟੀ-ਵੈਨਮ ਵਾਲੇ ਨਜ਼ਦੀਕੀ ਹਸਪਤਾਲ ਵਿੱਚ ਤੁਰੰਤ ਪਹੁੰਚਾਓ।"
            ]
        ),
        "or_IN": (
            "ସାପ କାମୁଡ଼ା ଜରୁରୀ ପରିସ୍ଥିତି! ରୋଗୀଙ୍କୁ ଶାନ୍ତ ରଖନ୍ତୁ ଏବଂ ତୁରନ୍ତ 108 କୁ କଲ୍ କରନ୍ତୁ।",
            [
                "କାମୁଡ଼ା ସ୍ଥାନକୁ ସ୍ଥିର ରଖି ହୃଦୟର ସ୍ତରଠାରୁ ତଳେ ରଖନ୍ତୁ।",
                "କ୍ଷତ ସ୍ଥାନରେ ଚିରା ଲଗାନ୍ତୁ ନାହିଁ କିମ୍ବା ବିଷ ଶୋଷିବାକୁ ଚେଷ୍ଟା କରନ୍ତୁ ନାହିଁ।",
                "ରୋଗୀଙ୍କୁ ଭୟଭୀତ ନକରାଇ ଶାନ୍ତ ରଖନ୍ତୁ।",
                "ଆଣ୍ଟି-ସ୍ନେକ ଭେନମ ଥିବା ନିକଟସ୍ଥ ଡାକ୍ତରଖାନାକୁ ଶୀଘ୍ର ନିଅନ୍ତୁ।"
            ]
        ),
        "mni_IN": (
            "ꯂꯤꯟ ꯆꯤꯛꯄꯒꯤ ꯈꯨꯗꯣꯡꯊꯤꯕ! ꯑꯅꯥꯕ ꯃꯤꯑꯣꯏ ꯑꯗꯨ ꯊꯋꯥꯏ ꯌꯥꯝꯅ ꯇꯞꯅ ꯂꯩꯍꯟꯕꯤꯌꯨ ꯑꯃꯁꯨꯡ 108 ꯀꯣꯜ ꯇꯧꯕꯤꯌꯨ।",
            [
                "ꯆꯤꯛꯈ꯭ꯔꯕ ꯃꯐꯝ ꯑꯗꯨ ꯍꯥꯏꯗꯣꯛ-ꯍꯥꯏꯖꯤꯟ ꯇꯧꯍꯟꯒꯅꯨ, ꯊꯃꯣꯏꯗꯒꯤ ꯅꯦꯝꯅ ꯊꯝꯃꯨ।",
                "ꯃꯐꯝ ꯑꯗꯨ ꯌꯥꯟꯊꯠꯄ ꯑꯃꯁꯨꯡ ꯍꯨ ꯆꯨꯊꯣꯛꯅꯕ ꯍꯣꯠꯅꯒꯅꯨ।",
                "ꯑꯅꯥꯕ ꯑꯗꯨꯕꯨ ꯆꯠꯊꯣꯛ-ꯆꯠꯁꯤꯟ ꯇꯧꯍꯟꯒꯅꯨ।",
                "ꯑꯦꯟꯇꯤ-ꯚꯦꯅꯝ ꯂꯩꯕ ꯈ꯭ꯋꯥꯏꯗꯒꯤ ꯅꯛꯄ ꯍꯣꯁꯄꯤꯇꯥꯜꯗ ꯈꯨꯗꯛꯇ ꯄꯨꯈꯠꯂꯨ।"
            ]
        ),
        "en_IN": (
            "Snakebite emergency! Keep the victim still and calm, dial 108 immediately.",
            [
                "Immobilize the bitten limb and keep it below heart level.",
                "DO NOT apply a tight tourniquet, do not cut the wound, and do not suck venom.",
                "Keep the person completely still and prevent walking or exertion.",
                "Rush immediately to the nearest hospital equipped with Anti-Snake Venom (ASV)."
            ]
        ),
    },
    "accident_trauma": {
        "service_idx": 0,
        "urgency": "critical",
        "kn_IN": (
            "ಅಪಘಾತ ತುರ್ತು ಪರಿಸ್ಥಿತಿ! ದಯವಿಟ್ಟು ರೋಗಿಯನ್ನು ಅತಿಯಾಗಿ ಚಲಿಸಬೇಡಿ ಮತ್ತು ತಕ್ಷಣ 108 ಆಂಬ್ಯುಲೆನ್ಸ್‌ಗೆ ಕರೆ ಮಾಡಿ.",
            [
                "ತೀವ್ರ ರಕ್ತಸ್ರಾವವಿದ್ದರೆ, ಸ್ವಚ್ಛವಾದ ಬಟ್ಟೆಯಿಂದ ಗಾಯದ ಮೇಲೆ ನೇರವಾಗಿ ಬಲವಾದ ಒತ್ತಡ ಹಾಕಿ ಹಿಡಿಯಿರಿ.",
                "ಕುತ್ತಿಗೆ ಅಥವಾ ಬೆನ್ನುಮೂಳೆಗೆ ಪೆಟ್ಟಾಗಿದ್ದರೆ, ರೋಗಿಯನ್ನು ಎತ್ತಬೇಡಿ ಅಥವಾ ತಿರುಗಿಸಬೇಡಿ.",
                "ಮೂಳೆ ಮುರಿದಿದ್ದರೆ, ಆ ಜಾಗವನ್ನು ಅಲುಗಾಡಿಸದೆ ಸ್ಪ್ಲಿಂಟ್ ಅಥವಾ ಮರದ ತುಂಡಿನಿಂದ ಸ್ಥಿರವಾಗಿರಿಸಿ.",
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
        "ta_IN": (
            "விபத்து அவசரநிலை! நோயாளியை அசைக்காமல் உடனே 108 ஆம்புலன்ஸை அழைக்கவும்.",
            [
                "அதிக ரத்தப்போக்கு இருந்தால், சுத்தமான துணியால் காயத்தின் மீது அழுத்திப் பிடிக்கவும்.",
                "கழுத்து அல்லது முதுகெலும்பில் அடிபட்டிருந்தால் நோயாளியை நகர்த்த வேண்டாம்.",
                "எலும்பு முறிவு ஏற்பட்டால் அந்தப் பகுதியை அசைக்காமல் நிலைநிறுத்தவும்.",
                "உடனடியாக 108 மருத்துவக் குழு வரும் வரை உடனிருங்கள்."
            ]
        ),
        "te_IN": (
            "ప్రమాద అత్యవసర పరిస్థితి! రోగిని కదల్చకండి, వెంటనే 108 అంబులెన్స్‌కు కాల్ చేయండి.",
            [
                "తీవ్ర రక్తస్రావం ఉంటే శుభ్రమైన గుడ్డతో గాయంపై గట్టిగా ఒత్తిడి ఉంచండి.",
                "మెడ లేదా వెన్నుముక గాయాలుంటే రోగిని అస్సలు కదల్చవద్దు.",
                "ఎముక విరిగిన భాగాన్ని కదలకుండా స్థిరంగా ఉంచండి.",
                "గాయపడిన వారికి ఏమీ తినిపించవద్దు లేదా తాగించవద్దు. 108 కోసం వేచి ఉండండి."
            ]
        ),
        "ml_IN": (
            "അപകട അടിയന്തരാവസ്ഥ! രോഗിയെ അനക്കാതെ ഉടൻ 108 ആംബുലൻസ് വിളിക്കുക.",
            [
                "രക്തസ്രാവമുണ്ടെങ്കിൽ വൃത്തിയുള്ള തുണികൊണ്ട് മുറിവിൽ അമർത്തിപ്പിടിക്കുക.",
                "കഴുത്തിനോ നട്ടെല്ലിനോ പരിക്കേറ്റിട്ടുണ്ടെങ്കിൽ രോഗിയെ മാറ്റരുത്.",
                "അസ്ഥി ഒടിഞ്ഞ ഭാഗം അനങ്ങാതെ താങ്ങിനിർത്തുക.",
                "ഉടൻ തന്നെ 108 എത്തിച്ചേരുന്നതുവരെ ഒപ്പം തുടരുക."
            ]
        ),
        "mr_IN": (
            "अपघात आणीबाणी! रुग्णाला जास्त हलवू नका आणि त्वरित 108 रुग्णवाहिकेला कॉल करा.",
            [
                "रक्तस्त्राव होत असल्यास स्वच्छ कापडाने जखमेवर थेट दाब द्या.",
                "मान किंवा पाठीच्या कण्याला इजा झाली असल्यास रुग्णाला हलवू नका.",
                "हाड मोडले असल्यास ते स्थिर ठेवा.",
                "रुग्णाला पाणी किंवा अन्न देऊ नका, त्वरित 108 ची वाट पहा."
            ]
        ),
        "gu_IN": (
            "અકસ્માત કટોકટી! દર્દીને બિલકુલ હલાવશો નહીં અને તરત જ 108 પર કૉલ કરો.",
            [
                "જો લોહી વહેતું હોય, તો સાફ કપડાથી ઘા પર સીધું દબાણ આપો.",
                "ગરદન કે કરોડરજ્જુની ઇજા હોય તો દર્દીને જરાય ખસેડશો નહીં.",
                "હાડકું તૂટી ગયું હોય તો તે ભાગને સ્થિર રાખો.",
                "108 એમ્બ્યુલન્સ આવે ત્યાં સુધી દર્દી સાથે રહો."
            ]
        ),
        "bn_IN": (
            "দুর্ঘটনাজনিত জরুরি অবস্থা! রোগীকে বেশি নাড়াচাড়া করবেন না এবং অবিলম্বে 108-এ কল করুন।",
            [
                "মারাত্মক রক্তপাত হলে পরিষ্কার কাপড় দিয়ে ক্ষতের ওপর শক্ত করে চেপে ধরুন।",
                "ঘাড় বা মেরুদণ্ডে আঘাতের সম্ভাবনা থাকলে রোগীকে সরাবেন না।",
                "হাড় ভেঙে গেলে সেই অংশটি সম্পূর্ণ স্থির রাখুন।",
                "আহত ব্যক্তিকে খাবার বা জল দেবেন না। 108-এর জন্য অপেক্ষা করুন।"
            ]
        ),
        "pa_IN": (
            "ਹਾਦਸਾ ਐਮਰਜੈਂਸੀ! ਮਰੀਜ਼ ਨੂੰ ਨਾ ਹਿਲਾਓ ਅਤੇ ਤੁਰੰਤ 108 ਐਂਬੂਲੈਂਸ ਨੂੰ ਕਾਲ ਕਰੋ।",
            [
                "ਜੇ ਖੂਨ ਵਹਿ ਰਿਹਾ ਹੈ, ਤਾਂ ਸਾਫ਼ ਕੱਪੜੇ ਨਾਲ ਜ਼ਖ਼ਮ 'ਤੇ ਦਬਾਅ ਬਣਾਓ।",
                "ਧੌਣ ਜਾਂ ਰੀੜ੍ਹ ਦੀ ਹੱਡੀ ਦੀ ਸੱਟ ਲੱਗਣ 'ਤੇ ਮਰੀਜ਼ ਨੂੰ ਬਿਲਕੁਲ ਨਾ ਹਿਲਾਓ।",
                "ਟੁੱਟੀ ਹੋਈ ਹੱਡੀ ਨੂੰ ਸਹਾਰਾ ਦੇ ਕੇ ਸਥਿਰ ਰੱਖੋ।",
                "ਤੁਰੰਤ 108 ਐਂਬੂਲੈਂਸ ਦੀ ਉਡੀਕ ਕਰੋ।"
            ]
        ),
        "or_IN": (
            "ଦୁର୍ଘଟଣା ଜରୁରୀକାଳୀନ ପରିସ୍ଥିତି! ରୋଗୀଙ୍କୁ ନ ହଲାଇ ତୁରନ୍ତ 108 ଆମ୍ବୁଲାନ୍ସକୁ କଲ୍ କରନ୍ତୁ।",
            [
                "ରକ୍ତସ୍ରାବ ହେଉଥିଲେ ପରିଷ୍କାର କପଡ଼ାରେ କ୍ଷତ ସ୍ଥାନକୁ ଜୋରରେ ଚାପି ଧରନ୍ତୁ।",
                "ବେକ କିମ୍ବା ମେରୁଦଣ୍ଡରେ ଆଘାତ ଥିଲେ ରୋଗୀଙ୍କୁ ଆଦୌ ଘୁଞ୍ଚାନ୍ତୁ ନାହିଁ।",
                "ଭଙ୍ଗା ହାଡ଼କୁ ସ୍ଥିର ରଖନ୍ତୁ।",
                "ତୁରନ୍ତ 108 ର ସହାୟତା ନିଅନ୍ତୁ।"
            ]
        ),
        "mni_IN": (
            "ꯑꯦꯛꯁꯤꯗꯦꯟꯠ ꯑꯆꯧꯕ ꯈꯨꯗꯣꯡꯊꯤꯕ! ꯑꯅꯥꯕ ꯃꯤꯑꯣꯏ ꯑꯗꯨ ꯍꯥꯏꯗꯣꯛ-ꯍꯥꯏꯖꯤꯟ ꯇꯧꯒꯅꯨ, 108 ꯀꯣꯜ ꯇꯧꯕꯤꯌꯨ।",
            [
                "ꯏ ꯌꯥꯝꯅ ꯊꯣꯛꯂꯕꯗꯤ ꯂꯨꯅ ꯂꯩꯕ ꯐꯤꯅ ꯀꯟꯅ ꯅꯝꯁꯤꯟꯕꯤꯌꯨ।",
                "ꯀꯣꯛ ꯑꯃꯁꯨꯡ ꯉꯛ ꯁꯣꯛꯂꯕꯗꯤ ꯃꯤꯑꯣꯏ ꯑꯗꯨꯕꯨ ꯍꯤꯡꯁꯥꯠ ꯇꯧꯍꯟꯒꯅꯨ।",
                "ꯁꯔꯨ ꯊꯨꯒꯥꯏꯔꯕꯗꯤ ꯆꯠꯊꯣꯛ-ꯆꯠꯁꯤꯟ ꯇꯧꯍꯟꯒꯅꯨ।",
                "108 ꯂꯥꯛꯇ꯭ꯔꯤꯐꯥꯎ ꯃꯇꯦꯡ ꯄꯥꯡꯕꯤꯌꯨ।"
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
# Language & Response Utilities
# ---------------------------------------------------------------------------
def get_env(name):
    return os.getenv(name, "").strip()

def to_hyphen(code):
    return code.replace("_", "-")

def detect_lang(text: str, fallback: str) -> str:
    if not text:
        return fallback

    # 1. Native unicode script identification
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

    # 2. Romanized keywords detection
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

def groq_triage(transcript: str, target_lang: str):
    groq_key = get_env("GROQ_API_KEY")
    if not groq_key:
        return None, "Groq key missing"

    system_prompt = f"""You are Sahaya, an empathetic AI emergency first-aid dispatcher in India.
Your mission is to handle all emergencies (snakebites, accidents, severe bleeding, cardiac arrest, burns, etc.).

CRITICAL LANGUAGE ENFORCEMENT:
1. Target language code: '{target_lang}'.
2. You MUST produce the 'reply' and all 'steps' EXCLUSIVELY in the official native script of '{target_lang}'.
   - If '{target_lang}' is 'kn_IN': Reply ONLY in Kannada script (ಕನ್ನಡ).
   - If '{target_lang}' is 'hi_IN': Reply ONLY in Hindi Devanagari script (हिन्दी).
   - If '{target_lang}' is 'ta_IN': Reply ONLY in Tamil script (தமிழ்).
   - If '{target_lang}' is 'te_IN': Reply ONLY in Telugu script (తెలుగు).
   - If '{target_lang}' is 'ml_IN': Reply ONLY in Malayalam script (മലയാളം).
   - If '{target_lang}' is 'mr_IN': Reply ONLY in Marathi Devanagari script (मराठी).
   - If '{target_lang}' is 'gu_IN': Reply ONLY in Gujarati script (ગુજરાતી).
   - If '{target_lang}' is 'bn_IN': Reply ONLY in Bengali script (বাংলা).
   - If '{target_lang}' is 'pa_IN': Reply ONLY in Punjabi Gurmukhi script (ਪੰਜਾਬੀ).
   - If '{target_lang}' is 'or_IN': Reply ONLY in Odia script (ଓଡ଼ିଆ).
   - If '{target_lang}' is 'mni_IN': Reply ONLY in Meitei Mayek script (ꯃꯤꯇꯩꯂꯣꯟ).
   - If '{target_lang}' is 'en_IN': Reply in English.
3. NEVER switch to English or Hindi unless that is the exact target language requested.

ACCURATE EMERGENCY FIRST-AID:
- Snakebite: Immobilize bitten limb below heart level, do not cut/suck venom, do not tie tight tourniquets, rush for Anti-Snake Venom (ASV).
- Accidents/Bleeding: Firm pressure on wounds with clean cloth, do not move head/neck if spine injury suspected, immobilize fractures.
- Cardiac: Sit comfortably, loosen clothing, CPR if unresponsive.

STRICT JSON OUTPUT FORMAT ONLY:
{{
  "lang_code": "{target_lang}",
  "category": "medical",
  "urgency": "critical",
  "reply": "Empathetic instruction in the EXACT native script of {target_lang}",
  "steps": [
    "Step 1 in the EXACT native script of {target_lang}",
    "Step 2 in the EXACT native script of {target_lang}",
    "Step 3 in the EXACT native script of {target_lang}",
    "Step 4 in the EXACT native script of {target_lang}"
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
                    "max_tokens": 600,
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

            return {
                "lang_code": target_lang,
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
# UI Construction
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
        <h4 style="color: #0f172a; font-size: 16px; margin-bottom: 12px; font-weight: 700;">📁 {html.escape(t['directory_header'])}</h4>
        {''.join(cards)}
    </div>
    """

# ---------------------------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------------------------
def run_pipeline(transcript: str, selected_lang: str):
    reply_lang = detect_lang(transcript, selected_lang)
    rt = TRANSLATIONS.get(reply_lang, TRANSLATIONS["kn_IN"])
    transcript_bubble = f"**{rt['detected']}**\n\n💬 {transcript}"
    yield "", transcript_bubble, "", None, ""

    triage, _ = groq_triage(transcript, reply_lang)

    if triage:
        reply_lang = triage.get("lang_code", reply_lang)
        rt = TRANSLATIONS.get(reply_lang, TRANSLATIONS["kn_IN"])
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
    t = TRANSLATIONS.get(lang_code, TRANSLATIONS["kn_IN"])
    return (
        header_html(t),
        t["sos_btn"],
        gr.update(label=t["mic_header"]),
        gr.update(label=t["type_header"], placeholder=t["type_placeholder"]),
        t["send"],
        directory_html(t),
    )

# ---------------------------------------------------------------------------
# High-Contrast CSS: Removes all dark boxes and fixes unreadable text
# ---------------------------------------------------------------------------
CSS = """
:root {
    --body-background-fill: #FFFDF9 !important;
    --background-fill-primary: #FFFFFF !important;
    --background-fill-secondary: #F8FAFC !important;
    --block-background-fill: #FFFFFF !important;
    --block-border-color: #CBD5E1 !important;
    --input-background-fill: #FFFFFF !important;
    --input-border-color: #94A3B8 !important;
    --body-text-color: #0F172A !important;
    --block-label-text-color: #0F172A !important;
}

body, .gradio-container {
    background-color: #FFFDF9 !important;
    color: #0F172A !important;
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

/* REMOVE BLACK BACKGROUNDS: Force white/light fills for inputs and audio boxes */
.gradio-container .block,
.gradio-container .gr-box,
.gradio-container .gr-panel,
.gradio-container .gr-input,
.gradio-container textarea,
.gradio-container input,
.gradio-container .gr-audio {
    background-color: #FFFFFF !important;
    background: #FFFFFF !important;
    color: #0F172A !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 8px !important;
}

/* Explicitly target audio widgets to eliminate dark playback strips */
.gradio-container audio,
.gradio-container [data-testid="audio"] {
    background-color: #F8FAFC !important;
    border-radius: 8px !important;
}

/* Input labels and placeholders */
.gradio-container label,
.gradio-container .block-title,
.gradio-container span {
    color: #0F172A !important;
    font-weight: 600 !important;
}

.gradio-container textarea::placeholder,
.gradio-container input::placeholder {
    color: #64748B !important;
    font-weight: 500 !important;
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
    cursor: pointer;
}

/* Live Chat/Text Bubbles */
.live-bubble {
    background: #F1F5F9 !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    color: #0F172A !important;
    font-size: 15px !important;
    margin-top: 8px !important;
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
    font-weight: 700 !important;
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
# UI Assembly
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
