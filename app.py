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
# Audio Conditioning
# ---------------------------------------------------------------------------
def preprocess_telephonic_audio(audio_path: str) -> str:
    if not audio_path or not os.path.exists(audio_path):
        return audio_path

    try:
        data, sr = sf.read(audio_path)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        sos = butter(2, 70.0, btype='highpass', fs=sr, output='sos')
        filtered = sosfilt(sos, data)

        max_val = np.max(np.abs(filtered))
        if max_val > 0.01:
            filtered = (filtered / max_val) * 0.95

        tmp_out = tempfile.NamedTemporaryFile(suffix="_clean.wav", delete=False)
        sf.write(tmp_out.name, filtered.astype(np.float32), sr)
        return tmp_out.name
    except Exception:
        return audio_path

# ---------------------------------------------------------------------------
# Languages & Directory Info
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
        "type_placeholder": "ಉದಾ: ಎದೆನೋವು, ಉಸಿರಾಟದ ಸಮಸ್ಯೆ, ಕಟ್ಟಡ ಕುಸಿತ, ಅಪಘಾತ, ಹಾವು ಕಡಿತ...",
        "send": "Send Emergency Alert",
        "detected": "🗣️ ಪತ್ತೆಯಾದ ತುರ್ತು ವಿವರ",
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
        "type_placeholder": "e.g., Chest pain, breathing issue, building collapse, fall from height...",
        "send": "Send Emergency Alert",
        "detected": "🗣️ Detected Emergency",
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
        "type_placeholder": "जैसे सीने में दर्द, सांस लेने में तकलीफ, इमारत ढहना, ऊंचाई से गिरना...",
        "send": "Send Emergency Alert",
        "detected": "🗣️ पता चली आपात स्थिति",
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
        "type_placeholder": "எ.கா. நெஞ்சு வலி, மூச்சுத் திணறல், கட்டிட விபத்து, கீழே விழுந்தது...",
        "send": "Send Emergency Alert",
        "detected": "🗣️ கண்டறியப்பட்ட அவசரம்",
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
        "type_placeholder": "ఉదా. గుండె నొప్పి, శ్వాస ఆడకపోవడం, భవనం కూలడం, ప్రమాదం...",
        "send": "Send Emergency Alert",
        "detected": "🗣️ గుర్తించబడిన అత్యవసరం",
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
        "type_placeholder": "ഉദാ. നെഞ്ചുവേദന, ശ്വാസതടസ്സം, കെട്ടിടം തകരൽ, ഉയരത്തിൽ നിന്ന് വീഴൽ...",
        "send": "Send Emergency Alert",
        "detected": "🗣️ കണ്ടെത്തിയ അത്യാഹിതം",
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
        "type_placeholder": "उदा. छातीत दुखणे, श्वास घेण्यास त्रास, इमारत कोसळणे, पडणे...",
        "send": "Send Emergency Alert",
        "detected": "🗣️ आढळलेली आणीबाणी",
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
        "type_placeholder": "દા.ત. છાતીમાં દુખાવો, શ્વાસ લેવામાં તકલીફ, મકાન પડવું, પડી જવું...",
        "send": "Send Emergency Alert",
        "detected": "🗣️ મળેલી કટોકટી",
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
        "type_placeholder": "যেমন বুকে ব্যথা, শ্বাসকষ্ট, বাড়ি ধসে পড়া, উঁচু থেকে পড়ে যাওয়া...",
        "send": "Send Emergency Alert",
        "detected": "🗣️ শনাক্ত জরুরি অবস্থা",
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
        "type_placeholder": "ਜਿਵੇਂ ਛਾਤੀ ਵਿੱਚ ਦਰਦ, ਸਾਹ ਲੈਣ ਵਿੱਚ ਔਖ, ਇਮਾਰਤ ਡਿੱਗਣਾ, ਸੱਟ ਲੱਗਣਾ...",
        "send": "Send Emergency Alert",
        "detected": "🗣️ ਪਤਾ ਲੱਗੀ ਐਮਰਜੈਂਸੀ",
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
        "type_placeholder": "ଯେପରି ଛାତି ଯନ୍ତ୍ରଣା, ଶ୍ୱାସକଷ୍ଟ, କୋଠା ଭୁଶୁଡ଼ିବା, ଉପରୁ ପଡ଼ିଯିବା...",
        "send": "Send Emergency Alert",
        "detected": "🗣️ ଚିହ୍ନଟ ଜରୁରୀକାଳୀନ ପରିସ୍ଥିତି",
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
        "type_placeholder": "ꯊꯃꯣꯏ ꯆꯤꯛꯄ, ꯁ꯭ꯕꯥꯁ ꯍꯣꯟꯕ ꯋꯥꯕ, ꯌꯨꯝ ꯇꯨꯈꯥꯏꯕ, ꯃꯩ ꯆꯥꯛꯄ...",
        "send": "Send Emergency Alert",
        "detected": "🗣️ ꯈꯪꯗꯣꯛꯂꯕ ꯈꯨꯗꯣꯡꯊꯤꯕ",
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
        "desc": "Relief and evacuation assistance during floods, earthquakes and building collapses.",
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
        "husiru", "usiradalu", "tondare", "tandre", "aagide", "aagite", "haavu",
        "kachide", "beku", "sahaya", "madadi", "rakta", "aspatre", "kattada", "bidda"
    },
    "hi_IN": {
        "mujhe", "humein", "mere", "meri", "pita", "papa", "mataji", "seene", "dard",
        "saans", "takleef", "chhati", "saanp", "bachao", "madad", "khoon", "aspatal",
        "gira", "gir", "toot", "imarat"
    },
    "ta_IN": {
        "enakku", "nenju", "vali", "swasam", "moochu", "thinaral", "appa", "amma",
        "pambu", "udavi", "ratham", "maruthuvamanai", "katidom", "keezhe"
    },
    "te_IN": {
        "naaku", "gunde", "noppi", "swasa", "kastam", "nanna", "amma", "paamu",
        "sahayam", "raktam", "aaspattri", "bhavanam", "kinda"
    },
    "ml_IN": {
        "enikku", "nenju", "vedana", "shwasam", "muttal", "achan", "amma", "paampu",
        "sahayam", "raktham", "aashupathri", "kettidam", "veenu"
    },
    "mr_IN": {
        "mala", "majhya", "chhati", "dukhate", "shwas", "trass", "vadil", "baba",
        "madat", "rakta", "davakhana", "padla", "imarat"
    },
    "gu_IN": {
        "mane", "maru", "chhati", "dard", "shwas", "takleef", "bapa", "madad", "lohi", "padi"
    },
    "bn_IN": {
        "amar", "buker", "byatha", "batha", "shwas", "kosto", "baba", "sahajjo", "rokto", "pore"
    },
    "pa_IN": {
        "meri", "mere", "sanu", "chhati", "peerh", "saah", "aukh", "bapu", "madad", "diga"
    },
    "or_IN": {
        "mora", "chhati", "bitha", "shwasa", "kasta", "bapa", "sahajya", "rakta", "padigala"
    },
    "mni_IN": {
        "eigi", "thamoigi", "chikpa", "masha", "ithak", "lin", "mateng", "tarure"
    },
    "en_IN": {
        "accident", "crash", "bleeding", "blood", "fracture", "chest", "pain", "heart",
        "breathe", "breathing", "burn", "snake", "ambulance", "fall", "collapse", "disaster"
    }
}

# ---------------------------------------------------------------------------
# Comprehensive Scenario Triage Fallbacks
# ---------------------------------------------------------------------------
def identify_condition(text: str) -> str:
    t = text.lower()
    
    # 1. Construction Fall / Building Collapse
    if any(k in t for k in [
        "collapse", "construction", "fall", "fell", "height", "debris", "trapped",
        "rubble", "building", "kattada", "bidda", "biddare", "imarat", "gir gaya",
        "gira", "katidom", "keezhe", "veenu", "padla", "bhukamp", "earthquake", "flood"
    ]):
        return "collapse_fall"

    # 2. Cardiac & Respiratory Emergencies
    if any(k in t for k in [
        "chest", "heart", "breathe", "breathing", "breath", "cardiac", "stroke", "attack",
        "ede", "edeno", "usiru", "husiru", "gundige", "seene", "chhati", "saans",
        "nenju", "moochu", "swasam", "gunde", "swasa", "buker", "shwas", "peerh", "saah"
    ]):
        return "cardiac"

    # 3. Snakebite
    if any(k in t for k in [
        "snake", "bite", "haavu", "havu", "kachide", "saanp", "saap",
        "paambu", "paamu", "lin", "chikpa", "sapp", "sapa", "dankh"
    ]):
        return "snakebite"

    # 4. Fire / Scalds
    if any(k in t for k in [
        "fire", "burn", "burns", "hot water", "scald", "cylinder", "blast",
        "benki", "sudide", "aag", "thee", "agg", "nia", "mei"
    ]):
        return "burns"

    # 5. Road Accident / Severe Trauma / Bleeding
    if any(k in t for k in [
        "accident", "crash", "bike", "car", "hit", "run over", "bleeding", "blood",
        "fracture", "bone broke", "wound", "apaghata", "gaaya", "rakta",
        "haadsa", "khoon", "vibathu", "pramadam"
    ]):
        return "accident_trauma"

    return "cardiac" if any(w in t for w in ["pain", "hurt", "father", "tande", "appa"]) else "general_medical"

CONDITION_FALLBACKS = {
    "collapse_fall": {
        "service_idx": 4,
        "urgency": "critical",
        "kn_IN": (
            "ಕಟ್ಟಡ ಕುಸಿತ / ಎತ್ತರದಿಂದ ಬಿದ್ದ ತುರ್ತು ಪರಿಸ್ಥಿತಿ! ತಕ್ಷಣ 108 ಮತ್ತು 112/1078 ಗೆ ಕರೆ ಮಾಡಿ.",
            [
                "ರೋಗಿಯನ್ನು ಯಾವುದೇ ಕಾರಣಕ್ಕೂ ಎತ್ತಬೇಡಿ ಅಥವಾ ಕುತ್ತಿಗೆ-ಬೆನ್ನು ತಿರುಗಿಸಬೇಡಿ (ಬೆನ್ನುಮೂಳೆಯ ಗಾಯದ ಅಪಾಯ).",
                "ಅವಶೇಷಗಳ ಅಡಿಯಲ್ಲಿ ಸಿಲುಕಿದ್ದರೆ, ಅವರ ಮುಖ ಮತ್ತು ಉಸಿರಾಟದ ಮಾರ್ಗವನ್ನು ಧೂಳಿನಿಂದ ರಕ್ಷಿಸಿ.",
                "ರಕ್ತಸ್ರಾವವಾಗುತ್ತಿದ್ದರೆ, ಸ್ವಚ್ಛವಾದ ಬಟ್ಟೆಯಿಂದ ಗಾಯದ ಮೇಲೆ ನೇರವಾದ ಒತ್ತಡ ಹಾಕಿ ಹಿಡಿಯಿರಿ.",
                "ರಕ್ಷಕ ತಂಡಗಳು ಬರುವವರೆಗೂ ರೋಗಿಯನ್ನು ಶಾಂತವಾಗಿರಿಸಿ."
            ]
        ),
        "hi_IN": (
            "इमारत ढहने / ऊंचाई से गिरने की आपात स्थिति! तुरंत 108 और 1078/112 पर कॉल करें।",
            [
                "मरीज की रीढ़ या गर्दन में गंभीर चोट की आशंका रहती है, इसलिए उन्हें बिल्कुल न हिलाएं।",
                "यदि व्यक्ति मलबे में दबा है, तो सांस लेने का रास्ता साफ रखें और धूल से बचाएं।",
                "रक्तस्रಾವ वाले स्थान पर साफ कपड़े से सीधा और मजबूत दबाव बनाए रखें।",
                "राहत और बचाव दल आने तक मरीज को होश में रखें और ढांढस बंधाएं।"
            ]
        ),
        "en_IN": (
            "Building collapse / severe fall emergency! Dial 108 and 1078/112 immediately.",
            [
                "Do NOT move or drag the victim (risk of severe spinal damage).",
                "Keep the head, neck, and spine completely aligned and immobilized.",
                "Apply clean, firm, direct pressure to any actively bleeding wounds.",
                "Clear dust from the airway if safe to do so, and await disaster rescue teams."
            ]
        ),
        "ta_IN": (
            "கட்டிட விபத்து / உயரத்திலிருந்து விழுந்த அவசரநிலை! உடனே 108 மற்றும் 112-ஐ அழைக்கவும்.",
            [
                "நோயாளியின் கழுத்து அல்லது முதுகெலும்பை அசைக்கவோ தூக்கவோ வேண்டாம்.",
                "சுவாசப் பாதையில் தூசி சேராமல் பார்த்துக் கொள்ளுங்கள்.",
                "ரத்தப்போக்கு இருந்தால் சுத்தமான துணியால் அழுத்தவும்.",
                "மீட்புக் குழுவினர் வரும் வரை பாதிக்கப்பட்டவரை அமைதிப்படுத்துங்கள்."
            ]
        ),
        "te_IN": (
            "భవనం కూలడం / ఎత్తు నుండి పడటం అత్యవసర పరిస్థితి! వెంటనే 108 మరియు 112 కు కాల్ చేయండి.",
            [
                "రోగిని కదల్చవద్దు, వెన్నుపాము దెబ్బతినే ప్రమాదం ఉంది.",
                "ధూళి నుండి శ్వాస మార్గాన్ని రక్షించండి.",
                "రక్తస్రావం ఉంటే శుభ్రమైన గుడ్డతో ఒత్తిడి ఉంచండి.",
                "రెస్క్యూ బృందం వచ్చేవరకు రోగికి ధైర్యం చెప్పండి."
            ]
        ),
    },
    "cardiac": {
        "service_idx": 0,
        "urgency": "critical",
        "kn_IN": (
            "ಎದೆನೋವು / ಉಸಿರಾಟದ ತೊಂದರೆ ತುರ್ತು ಪರಿಸ್ಥಿತಿ! ತಕ್ಷಣ 108 ಆಂಬ್ಯುಲೆನ್ಸ್‌ಗೆ ಕರೆ ಮಾಡಿ.",
            [
                "ರೋಗಿಯನ್ನು ಆರಾಮವಾಗಿ ಕುಳ್ಳಿರಿಸಿ ಮತ್ತು ಕುತ್ತಿಗೆ ಹಾಗೂ ಎದೆಯ ಬಿಗಿಯಾದ ಬಟ್ಟೆಗಳನ್ನು ಸಡಿಲಗೊಳಿಸಿ.",
                "ರೋಗಿಯನ್ನು ಸಂಪೂರ್ಣ ಶಾಂತವಾಗಿರಲು ಹೇಳಿ, ಯಾವುದೇ ಶ್ರಮ ಅಥವಾ ನಡಿಗೆಗೆ ಬಿಡಬೇಡಿ.",
                "ಪ್ರಜ್ಞೆ ತಪ್ಪಿ ಉಸಿರಾಟ ನಿಂತಿದ್ದರೆ, ತಕ್ಷಣ ಎದೆಯ ಮಧ್ಯಭಾಗದಲ್ಲಿ ಸಿಪಿಆರ್ (CPR) ಒತ್ತಡ ಪ್ರಾರಂಭಿಸಿ (ನಿಮಿಷಕ್ಕೆ 100-120 ಬಾರಿ).",
                "ತಕ್ಷಣ 108 ಆಂಬ್ಯುಲೆನ್ಸ್ ಮೂಲಕ ಹೃದಯ ಚಿಕಿತ್ಸಾ ತುರ್ತು ವಿಭಾಗವಿರುವ ಆಸ್ಪತ್ರೆಗೆ ರವಾನಿಸಿ."
            ]
        ),
        "hi_IN": (
            "सीने में दर्द और सांस की आपात स्थिति! तुरंत 108 पर कॉल करें।",
            [
                "मरीज को आरामदायक स्थिति में बैठाएं और गले व छाती के कपड़े ढीले करें।",
                "मरीज को शांत रखें और बिल्कुल चलने या श्रम न करने दें।",
                "यदि सांस रुक जाए तो तुरंत सीपीआर (CPR) शुरू करें (100-120 प्रति मिनट)।",
                "तुरंत 108 एम्बुलेंस से नजदीकी हृदय रोग अस्पताल ले जाएं।"
            ]
        ),
        "en_IN": (
            "Chest pain & breathing emergency! Dial 108 immediately and follow these steps.",
            [
                "Have the person sit down comfortably in a semi-upright position and loosen tight clothing around neck and chest.",
                "Keep the patient calm and completely still. Do not let them walk or exert themselves.",
                "If the patient becomes unresponsive or stops breathing normally, begin CPR immediately (100-120 compressions/min).",
                "Ensure immediate transport to an emergency cardiac care hospital via 108."
            ]
        ),
        "ta_IN": (
            "நெஞ்சு வலி மற்றும் சுவாச அவசரநிலை! உடனடியாக 108-ஐ அழைக்கவும்.",
            [
                "நோயாளியை சாய்ந்த நிலையில் வசதியாக உட்கார வைக்கவும், இறுக்கமான ஆடைகளைத் தளர்த்தவும்.",
                "நோயாளியை அமைதியாக வைக்கவும், நடக்க அனுமதிக்காதீர்கள்.",
                "சுவாசம் நின்றால் உடனடியாக சிபிஆர் (CPR) மார்பு அழுத்தத்தைத் தொடங்கவும்.",
                "உடனடியாக 108 ஆம்புலன்ஸ் மூலம் மருத்துவமனைக்கு கொண்டு செல்லுங்கள்."
            ]
        ),
        "te_IN": (
            "గుండె నొప్పి మరియు శ్వాస అత్యవసర పరిస్థితి! వెంటనే 108కి కాల్ చేయండి.",
            [
                "రోగిని సౌకర్యవంతంగా కూర్చోబెట్టి, మెడ మరియు ఛాతీ వద్ద బట్టలను వదులు చేయండి.",
                "రోగిని ప్రశాంతంగా ఉంచండి, నడవడానికి అనుమతించవద్దు.",
                "శ్వాస ఆగిపోతే వెంటనే సిపిఆర్ (CPR) ఛాతీ ఒత్తిడిని ప్రారంభించండి.",
                "108 అంబులెన్స్ ద్వారా వెంటనే ఆసుపత్రికి తరలించండి."
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
    "general_medical": {
        "service_idx": 0,
        "urgency": "urgent",
        "kn_IN": (
            "ವೈದ್ಯಕೀಯ ತುರ್ತು ಪರಿಸ್ಥಿತಿ! ಗಾಬರಿಯಾಗಬೇಡಿ, ತಕ್ಷಣ 108 ಆಂಬ್ಯುಲೆನ್ಸ್‌ಗೆ ಕರೆ ಮಾಡಿ.",
            [
                "ವ್ಯಕ್ತಿಯ ಉಸಿರಾಟ ಮತ್ತು ಪ್ರಜ್ಞೆಯನ್ನು ನಿರಂತರವಾಗಿ ಪರಿಶೀಲಿಸಿ.",
                "ಆರಾಮದಾಯಕ ಭಂಗಿಯಲ್ಲಿ ಕುಳ್ಳಿರಿಸಿ ಧೈರ್ಯ ನೀಡಿ.",
                "108 ವೈದ್ಯಕೀಯ ತಂಡ ಬರುವವರೆಗೆ ರೋಗಿಯ ಜೊತೆಯಲ್ಲೇ ಇರಿ."
            ]
        ),
        "en_IN": (
            "Medical Emergency! Please stay calm and immediately call 108.",
            [
                "Check airway, responsiveness, and breathing continuously.",
                "Keep the patient resting comfortably and calm.",
                "Stay with the patient until emergency paramedics arrive."
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
    cond_data = CONDITION_FALLBACKS.get(cond, CONDITION_FALLBACKS["cardiac"])
    service = EMERGENCY_DIRECTORY[cond_data["service_idx"]]
    urgency = cond_data["urgency"]

    if lang_code in cond_data:
        reply, steps = cond_data[lang_code]
    elif "kn_IN" in cond_data and lang_code == "kn_IN":
        reply, steps = cond_data["kn_IN"]
    else:
        reply, steps = cond_data.get("en_IN", list(cond_data.values())[2])

    return service, reply, steps, urgency

# ---------------------------------------------------------------------------
# AI STT & Reasoning
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

    transcript = str(
        data.get("transcript") or
        data.get("text") or
        data.get("transcription") or
        (data.get("result") if isinstance(data.get("result"), str) else "") or
        ""
    ).strip()

    return transcript, ""

def groq_triage(transcript: str, target_lang: str):
    groq_key = get_env("GROQ_API_KEY")
    if not groq_key:
        return None, "Groq key missing"

    system_prompt = f"""You are Sahaya, an empathetic AI emergency dispatcher in India.
Current incident reported by user: "{transcript}"

CRITICAL INSTRUCTIONS:
1. TARGET SCRIPT: You MUST output all text strictly in the NATIVE SCRIPT of '{target_lang}'.
   - kn_IN -> Kannada script (ಕನ್ನಡ)
   - en_IN -> English
   - hi_IN -> Hindi Devanagari script (हिन्दी)
   - ta_IN -> Tamil (தமிழ்), te_IN -> Telugu (తెలుగు), ml_IN -> Malayalam (മലയാളം), mr_IN -> Marathi (मराठी), bn_IN -> Bengali (বাংলা), gu_IN -> Gujarati (ગુજરાતી), pa_IN -> Punjabi (ਪੰਜਾਬੀ), or_IN -> Odia (ଓଡ଼ିଆ), mni_IN -> Meitei Mayek (ꯃꯤꯇꯩꯂꯣꯟ).
2. ACCURATE FIRST-AID:
   - For building collapse / fall from height / disaster: Do NOT move victim (spinal injury risk), clear dust from airway, apply direct pressure on bleeding, call 108 & 112/1078.
   - For chest pain/heart/breathing issues: Sit down comfortably, loosen clothing, keep calm, CPR if unresponsive, call 108.
   - For snakebite: Keep limb below heart level, no cuts/suction, rush for Anti-Snake Venom (ASV).
   - For road accidents: Direct pressure on bleeding wounds, immobilize spine/neck.
3. Return STRICT JSON ONLY:
{{
  "lang_code": "{target_lang}",
  "category": "medical",
  "urgency": "critical",
  "reply": "Empathetic instruction in the exact script of {target_lang}",
  "steps": [
    "Step 1 in {target_lang}",
    "Step 2 in {target_lang}",
    "Step 3 in {target_lang}",
    "Step 4 in {target_lang}"
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
    transcript = (transcript or "").strip()
    if not transcript:
        msg = "⚠️ Could not hear audio clearly. Please hold the mic closer or type your emergency below."
        yield "", msg, "", None, ""
        return

    reply_lang = detect_lang(transcript, selected_lang)
    lang_label = LANGUAGES.get(reply_lang, "Detected")
    rt = TRANSLATIONS.get(reply_lang, TRANSLATIONS["kn_IN"])
    
    transcript_bubble = f"**{rt['detected']} ({lang_label}):**\n\n💬 {transcript}"
    yield "", transcript_bubble, "", None, ""

    triage, _ = groq_triage(transcript, reply_lang)

    if triage:
        reply_lang = triage.get("lang_code", reply_lang)
        rt = TRANSLATIONS.get(reply_lang, TRANSLATIONS["kn_IN"])
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
    yield "", "⏳ Processing emergency audio...", "", None, ""
    transcript, err = transcribe(audio_path, lang_code)
    if err or not transcript:
        yield f"⚠️ Could not transcribe clearly. Please type or repeat.", "", "", None, ""
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
# High-Contrast CSS: Removes All Dark Boxes & Fixes Gradio 5/6 Elements
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

/* Eliminate dark background shells in inputs, audio, and tools */
.gradio-container .block,
.gradio-container .gr-box,
.gradio-container .gr-panel,
.gradio-container .gr-input,
.gradio-container textarea,
.gradio-container input,
.gradio-container .gr-audio,
.gradio-container [data-testid="audio"],
.gradio-container .audio-container {
    background-color: #FFFFFF !important;
    background: #FFFFFF !important;
    color: #0F172A !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 8px !important;
}

/* Explicitly eliminate dark buttons and icons inside audio bars */
.gradio-container audio,
.gradio-container .record-button,
.gradio-container .record-icon,
.gradio-container .action,
.gradio-container button.icon,
.gradio-container [aria-label="Record"],
.gradio-container [aria-label="Clear"],
.gradio-container .playback-button {
    background-color: #F1F5F9 !important;
    background: #F1F5F9 !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
}

.gradio-container svg {
    fill: #0F172A !important;
    color: #0F172A !important;
}

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

.live-bubble {
    background: #F1F5F9 !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 8px !important;
    padding: 14px 18px !important;
    color: #0F172A !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
    margin-top: 8px !important;
}

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
# UI Assembly (Gradio 5/6 Compatible Constructor)
# ---------------------------------------------------------------------------
t0 = TRANSLATIONS["kn_IN"]

with gr.Blocks(title="Sahaya Emergency") as demo:
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
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=gr.themes.Base(),
        css=CSS,
        js=INIT_JS
    )
