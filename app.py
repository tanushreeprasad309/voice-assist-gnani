"""
Sahaya Emergency Assistant (Gradio on Render)
Voice-first emergency triage and first aid for India across 12 languages.
"""

import base64
from concurrent.futures import ThreadPoolExecutor
import html
import json
import math
import os
import re
import tempfile
import gradio as gr
import numpy as np
import requests
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, istft, resample_poly, sosfilt, stft

STT_URL = "https://api.vachana.ai/stt/v3"
TTS_URL = "https://api.vachana.ai/api/v1/tts/inference"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
]

TARGET_SR = 16000
MAX_SECONDS = 60

HTTP_SESSION = requests.Session()

# ---------------------------------------------------------------------------
# Supported Languages (12 Indic Languages)
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

SCRIPT_NAMES = {
    "kn_IN": "Kannada script (ಕನ್ನಡ)",
    "en_IN": "English",
    "hi_IN": "Hindi, Devanagari script (हिन्दी)",
    "ta_IN": "Tamil script (தமிழ்)",
    "te_IN": "Telugu script (తెలుగు)",
    "ml_IN": "Malayalam script (മലയാളം)",
    "mr_IN": "Marathi, Devanagari script (मराठी)",
    "gu_IN": "Gujarati script (ગુજરાતી)",
    "bn_IN": "Bengali script (বাংলা)",
    "pa_IN": "Gurmukhi script (ਪੰਜਾਬੀ)",
    "or_IN": "Odia script (ଓଡ଼ିଆ)",
    "mni_IN": "Meitei Mayek script (ꯃꯤꯇꯩꯂꯣꯟ)",
}

TIMBRE_LANGS = {
    "en_IN", "hi_IN", "ta_IN", "te_IN", "kn_IN",
    "ml_IN", "mr_IN", "pa_IN", "bn_IN", "gu_IN",
}

WHISPER_NAME_TO_CODE = {
    "english": "en_IN", "hindi": "hi_IN", "urdu": "hi_IN", "nepali": "hi_IN", "sanskrit": "hi_IN",
    "kannada": "kn_IN", "tamil": "ta_IN", "telugu": "te_IN", "malayalam": "ml_IN",
    "marathi": "mr_IN", "gujarati": "gu_IN", "bengali": "bn_IN", "assamese": "bn_IN",
    "punjabi": "pa_IN", "odia": "or_IN", "oriya": "or_IN", "manipuri": "mni_IN",
    "en": "en_IN", "hi": "hi_IN", "ur": "hi_IN", "kn": "kn_IN", "ta": "ta_IN",
    "te": "te_IN", "ml": "ml_IN", "mr": "mr_IN", "gu": "gu_IN", "bn": "bn_IN",
    "as": "bn_IN", "pa": "pa_IN", "or": "or_IN", "mni": "mni_IN",
}

SCRIPT_RANGES = [
    (0x0C80, 0x0CFF, "kn_IN"),
    (0x0900, 0x097F, "hi_IN"),
    (0x0B80, 0x0BFF, "ta_IN"),
    (0x0C00, 0x0C7F, "te_IN"),
    (0x0D00, 0x0D7F, "ml_IN"),
    (0x0980, 0x09FF, "bn_IN"),
    (0x0A80, 0x0AFF, "gu_IN"),
    (0x0A00, 0x0A7F, "pa_IN"),
    (0x0B00, 0x0B7F, "or_IN"),
    (0xABC0, 0xABFF, "mni_IN"),
]

MR_MARKERS = ["आहे", "आहेत", "मला", "माझ", "तुम्ही", "करा", "झाले", "झाला", "नाही", "छातीत", "दुखत", "येत नाही"]
HI_MARKERS = ["है", "हैं", "मुझे", "मेरे", "मेरी", "में", "रहा", "रही", "नहीं", "करो", "कृपया", "को ", "मदद"]

ROMAN_LANG_KEYWORDS = {
    "kn_IN": {"nanage", "nanna", "namma", "tande", "thande", "taayi", "thaayi", "ede", "edenovu", "edenoovu", "novu", "usiru", "husiru", "usiradalu", "huduga", "aagide", "aagtide", "aagta", "ide", "beku", "bega", "maadi", "sahaya", "rakta", "aspatre", "kattada", "bidda", "haavu", "kachide"},
    "hi_IN": {"madad", "chahiye", "bachao", "bachaao", "mujhe", "humein", "mere", "mera", "meri", "papa", "pitaji", "mataji", "seene", "seena", "dard", "saans", "takleef", "saanp", "khoon", "aspatal", "gira", "hai", "hain", "raha", "rahi", "nahi", "jaldi", "behosh"},
    "ta_IN": {"udavi", "enakku", "ennaku", "nenju", "vali", "swasam", "moochu", "thinaral", "pambu", "ratham", "maruthuvamanai", "vizhunthuttar", "irukku", "pannunga", "seekiram"},
    "te_IN": {"sahayam", "naaku", "naku", "gunde", "noppi", "swasa", "kastam", "paamu", "aaspatri", "kinda", "undi", "ledu", "cheyandi", "tvaraga", "padipoyaru"},
    "ml_IN": {"sahayam", "enikku", "vedana", "shwasam", "muttal", "paampu", "aashupathri", "veenu", "undu", "cheyyu", "vegam"},
    "mr_IN": {"madat", "mala", "majhya", "majha", "dukhate", "dukhat", "trass", "davakhana", "padla", "aahe", "aahet", "kara", "lavkar"},
    "gu_IN": {"madad", "mane", "maru", "mara", "lohi", "padi", "che", "nathi", "karo", "tarat"},
    "bn_IN": {"sahajjo", "amar", "amake", "buker", "byatha", "batha", "kosto", "rokto", "pore", "ache", "korun", "taratari"},
    "pa_IN": {"madad", "sanu", "mainu", "peerh", "saah", "aukh", "diga", "kro"},
    "or_IN": {"sahajya", "mora", "bitha", "shwasa", "kasta", "padigala", "achhi"},
    "mni_IN": {"mateng", "eigi", "thamoigi", "chikpa", "masha", "lin", "tarure"},
}

EN_STOP = {"the", "is", "my", "has", "have", "he", "she", "and", "please", "help", "not", "can", "are", "was", "his", "her", "with", "from", "at", "of", "we", "there", "someone", "need", "very", "cannot", "unable", "just", "now", "fell", "hurt", "i", "am"}

BASE_T = {
    "brand": "Sahaya",
    "ready_status": "Ready",
    "hero_title": "AI Voice Assistant",
    "subtitle": "Talk. Translate. Stay Safe.",
    "sos_btn": "🚨 EMERGENCY SOS • LOUD ALARM + DIAL 112",
    "mic_header": "Tap to Start Recording",
    "mic_recording": "Recording... (Speak now — auto-detects silence)",
    "mic_sub": "Speak naturally in any Indian language",
    "shield_text": "Real-time emergency triage & first aid",
    "auto_detect_badge": "Auto-Detect Active (12 Indian Languages)",
    "placeholder": "Or type emergency details (e.g., chest pain, accident)...",
    "send_btn": "Translate & Help",
    "clear_btn": "🗑️ Clear",
    "live_trans": "Live Emergency Triage",
    "notified_sub": "Emergency Services Notified",
    "first_aid_label": "Immediate First-Aid Steps:",
    "call_btn": "Call {number} Immediately",
    "directory_header": "Emergency Services Directory",
    "med_title": "Medical & Ambulance",
    "police_title": "Police & Safety",
    "women_title": "Women Distress Helpline",
    "fire_title": "Fire & Rescue",
    "disaster_title": "Disaster Support",
    "call_txt": "Call",
}

LOCAL_T = {
    "kn_IN": {
        "brand": "ಸಹಾಯ (Sahaya)",
        "ready_status": "ಸಿದ್ಧವಾಗಿದೆ",
        "hero_title": "AI ಧ್ವನಿ ಸಹಾಯಕ",
        "subtitle": "ಮಾತನಾಡಿ. ಅನುವಾದಿಸಿ. ಸುರಕ್ಷಿತವಾಗಿರಿ.",
        "sos_btn": "🚨 ತುರ್ತು SOS • ಜೋರು ಸೈರನ್ + 112 ಗೆ ಕರೆ",
        "mic_header": "ರೆಕಾರ್ಡಿಂಗ್ ಪ್ರಾರಂಭಿಸಲು ಟ್ಯಾಪ್ ಮಾಡಿ",
        "mic_recording": "ರೆಕಾರ್ಡ್ ಆಗುತ್ತಿದೆ... (ಈಗ ಮಾತನಾಡಿ)",
        "mic_sub": "ಯಾವುದೇ ಭಾರತೀಯ ಭಾಷೆಯಲ್ಲಿ ಸಹಜವಾಗಿ ಮಾತನಾಡಿ",
        "shield_text": "ನೈಜ-ಸಮಯದ ತುರ್ತು ಪ್ರಥಮ ಚಿಕಿತ್ಸೆ",
        "auto_detect_badge": "ಧ್ವನಿ ಸ್ವಯಂ ಪತ್ತೆ ಸಕ್ರಿಯವಾಗಿದೆ (12 ಭಾಷೆಗಳು)",
        "placeholder": "ಅಥವಾ ತುರ್ತು ವಿವರಗಳನ್ನು ಟೈಪ್ ಮಾಡಿ (ಉದಾ: ಎದೆ ನೋವು, ಅಪಘಾತ)...",
        "send_btn": "ಅನುವಾದಿಸಿ & ಸಹಾಯ",
        "clear_btn": "🗑️ ಅಳಿಸಿ (Clear)",
        "live_trans": "ನೇರ ತುರ್ತು ಮಾಹಿತಿ",
        "notified_sub": "ತುರ್ತು ಸೇವೆಗಳಿಗೆ ತಿಳಿಸಲಾಗಿದೆ",
        "first_aid_label": "ತಕ್ಷಣದ ಪ್ರಥಮ ಚಿಕಿತ್ಸಾ ಕ್ರಮಗಳು:",
        "call_btn": "ತಕ್ಷಣ {number} ಗೆ ಕರೆ ಮಾಡಿ",
        "directory_header": "ತುರ್ತು ಸೇವೆಗಳ ಡೈರೆಕ್ಟರಿ",
        "med_title": "ವೈದ್ಯಕೀಯ ಮತ್ತು ಆಂಬ್ಯುಲೆನ್ಸ್",
        "police_title": "ಪೊಲೀಸ್ ಮತ್ತು ಸುರಕ್ಷತೆ",
        "women_title": "ಮಹಿಳಾ ಸಹಾಯವಾಣಿ",
        "fire_title": "ಅಗ್ನಿಶಾಮಕ ಮತ್ತು ರಕ್ಷಣೆ",
        "disaster_title": "ವಿಪತ್ತು ನಿರ್ವಹಣಾ ಸಹಾಯ",
        "call_txt": "ಕರೆ ಮಾಡಿ",
    },
    "hi_IN": {
        "brand": "सहाय (Sahaya)",
        "ready_status": "तैयार",
        "hero_title": "AI वॉयस असिस्टेंट",
        "subtitle": "बोलें। अनुवाद करें। सुरक्षित रहें।",
        "sos_btn": "🚨 आपातकालीन SOS • तेज सायरन + 112 डायल करें",
        "mic_header": "रिकॉर्डिंग शुरू करने के लिए टैप करें",
        "mic_recording": "रिकॉर्डिंग हो रही है... (अब बोलें)",
        "mic_sub": "किसी भी भारतीय भाषा में स्वाभाविक रूप से बोलें",
        "shield_text": "रीयल-टाइम आपातकालीन प्राथमिक चिकित्सा",
        "auto_detect_badge": "वॉयस ऑटो-डिटेक्ट सक्रिय (12 भाषाएं)",
        "placeholder": "या आपातकालीन विवरण टाइप करें (जैसे: सीने में दर्द, दुर्घटना)...",
        "send_btn": "अनुवाद और सहायता",
        "clear_btn": "🗑️ साफ़ करें",
        "live_trans": "लाइव आपातकालीन ट्राइएज",
        "notified_sub": "आपातकालीन सेवाएं सूचित",
        "first_aid_label": "तत्काल प्राथमिक उपचार कदम:",
        "call_btn": "तुरंत {number} पर कॉल करें",
        "directory_header": "आपातकालीन सेवा निर्देशिका",
        "med_title": "चिकित्सा एवं एम्बुलेंस",
        "police_title": "पुलिस एवं सुरक्षा",
        "women_title": "महिला हेल्पलाइन",
        "fire_title": "अग्निशमन एवं बचाव",
        "disaster_title": "आपदा सहायता",
        "call_txt": "कॉल करें",
    },
    "ta_IN": {
        "brand": "சஹாயா (Sahaya)",
        "ready_status": "தயார்",
        "hero_title": "AI குரல் உதவியாளர்",
        "subtitle": "பேசுங்கள். மொழிபெயர்க்கவும். பாதுகாப்பாக இருங்கள்.",
        "sos_btn": "🚨 அவசர SOS • சைரன் ஒலி + 112 அழைக்கவும்",
        "mic_header": "பதிவைத் தொடங்க தட்டவும்",
        "mic_recording": "பதிவாகிறது... (இப்போது பேசுங்கள்)",
        "mic_sub": "இயல்பாகப் பேசுங்கள்",
        "shield_text": "நிகழ்நேர முதலுதவி மற்றும் அவசர சிகிச்சை",
        "auto_detect_badge": "குரல் தானியங்கி கண்டறிதல் (12 மொழிகள்)",
        "placeholder": "அல்லது அவசர விவரங்களை தட்டச்சு செய்யவும்...",
        "send_btn": "மொழிபெயர்த்து உதவு",
        "clear_btn": "🗑️ அழி",
        "live_trans": "நேரடி அவசர வழிகாட்டுதல்",
        "notified_sub": "அவசர சேவைகளுக்கு தகவல் அளிக்கப்பட்டது",
        "first_aid_label": "உடனடி முதலுதவி வழிகாட்டுதல்கள்:",
        "call_btn": "உடனே {number} ஐ அழைக்கவும்",
        "directory_header": "அவசர சேவைகள் பட்டியல்",
        "med_title": "மருத்துவம் & ஆம்புலன்ஸ்",
        "police_title": "காவல்துறை & பாதுகாப்பு",
        "women_title": "பெண்கள் உதவி எண்",
        "fire_title": "தீயணைப்பு & மீட்பு",
        "disaster_title": "பேரிடர் உதவி",
        "call_txt": "அழைக்க",
    },
    "te_IN": {
        "brand": "సహాయ (Sahaya)",
        "ready_status": "సిద్ధం",
        "hero_title": "AI వాయిస్ అసిస్టెంట్",
        "subtitle": "మాట్లాడండి. అనువదించండి. సురక్షితంగా ఉండండి.",
        "sos_btn": "🚨 అత్యవసర SOS • సైరన్ + 112 కు కాల్ చేయండి",
        "mic_header": "రికార్డింగ్ ప్రారంభించడానికి ట్యాప్ చేయండి",
        "mic_recording": "రికార్డింగ్ జరుగుతోంది... (ఇప్పుడు మాట్లాడండి)",
        "mic_sub": "సహజంగా మాట్లాడండి",
        "shield_text": "రియల్-టైమ్ అత్యవసర ప్రథమ చికిత్స",
        "auto_detect_badge": "వాయిస్ ఆటో-డిటెక్షన్ సక్రియం (12 భాషలు)",
        "placeholder": "లేదా అత్యవసర వివరాలను టైప్ చేయండి...",
        "send_btn": "అనువదించు & సహాయం",
        "clear_btn": "🗑️ క్లియర్",
        "live_trans": "ప్రత్యక్ష అత్యవసర సహాయం",
        "notified_sub": "అత్యవసర సేవలకు సమాచారం అందించబడింది",
        "first_aid_label": "తక్షణ ప్రథమ చికిత్స మార్గదర్శకాలు:",
        "call_btn": "ఇప్పుడే {number} కి కాల్ చేయండి",
        "directory_header": "అత్యవసర సేవల డైరెక్టరీ",
        "med_title": "వైద్యం & అంబులెన్స్",
        "police_title": "పోలీస్ & భద్రత",
        "women_title": "మహిళా హెల్ప్‌లైన్",
        "fire_title": "అగ్నిమాపక & రక్షణ",
        "disaster_title": "విపత్తు సహాయం",
        "call_txt": "కాల్",
    },
}

TRANSLATIONS = {c: {**BASE_T, **LOCAL_T.get(c, {})} for c in LANGUAGES}

EMERGENCY_DIRECTORY = [
    {"key": "med_title", "icon": "🚑", "number": "108"},
    {"key": "police_title", "icon": "👮", "number": "112"},
    {"key": "women_title", "icon": "🛡️", "number": "1091"},
    {"key": "fire_title", "icon": "🚒", "number": "101"},
    {"key": "disaster_title", "icon": "🌊", "number": "1078"},
]

CATEGORY_INDEX = {"medical": 0, "police": 1, "assault": 2, "fire": 3, "disaster": 4}

CONDITION_RULES = [
    ("snakebite", {"snake", "snakes", "snakebite", "bitten", "bite", "haavu", "havu", "haavina", "kachide", "saanp", "saap", "sanp", "paambu", "pambu", "paamu", "pamu", "paampu", "sap"}, ["ಹಾವು", "ಹಾವಿನ", "साँप", "सांप", "साप", "பாம்பு", "పాము", "പാമ്പ്", "সাপ", "સાપ", "ਸੱਪ", "ସାପ"]),
    ("burns", {"fire", "burn", "burns", "burnt", "burning", "scald", "scalded", "cylinder", "blast", "boiling", "benki", "aag", "thee", "neruppu", "mantalu", "agun", "aagun", "hot water"}, ["ಬೆಂಕಿ", "ಸುಟ್ಟ", "आग", "जला", "जल गया", "झुलस", "தீக்காய", "தீப்பிடி", "நெருப்பு", "మంట", "నిప్పు", "തീപിടി", "പൊള്ള", "আগুন", "পুড়ে", "পোড়া", "આગ", "દાઝ", "ਅੱਗ", "ਸੜ", "ନିଆଁ", "ପୋଡ଼"]),
    ("cardiac", {"chest", "heart", "breathe", "breathing", "breath", "cardiac", "stroke", "attack", "ede", "edeno", "edenovu", "usiru", "husiru", "usiradalu", "gundige", "seene", "seena", "chhati", "chati", "saans", "sans", "nenju", "moochu", "swasam", "gunde", "swasa", "buker", "shwas", "shwasa", "saah", "dil", "dhadkan"}, ["ಎದೆ", "ಉಸಿರ", "ಹೃದಯ", "सीने", "छाती", "सांस", "साँस", "दिल", "हृदय", "श्वास", "நெஞ்சு", "மூச்சு", "இதய", "గుండె", "ఛాతీ", "శ్వాస", "ఊపిరి", "നെഞ്ച", "ശ്വാസ", "ഹൃദയ", "বুক", "শ্বাস", "হৃদ", "છાતી", "શ્વાસ", "હૃદય", "ਛਾਤੀ", "ਸਾਹ", "ਦਿਲ", "ଛାତି", "ଶ୍ୱାସ", "ହୃଦ", "ꯊꯃꯣꯏ"]),
    ("collapse_fall", {"collapse", "collapsed", "construction", "fell", "fall", "falling", "height", "debris", "trapped", "rubble", "building", "kattada", "bidda", "biddare", "bidde", "imarat", "imaarat", "gir", "gira", "giri", "katidom", "keezhe", "veenu", "padla", "padli", "bhukamp", "earthquake", "flood", "landslide"}, ["ಕಟ್ಟಡ", "ಕುಸಿ", "ಬಿದ್ದ", "ಬೀಳ", "ಎತ್ತರ", "इमारत", "ढह", "गिर", "भूकंप", "बाढ़", "கட்டிட", "விழு", "கீழே", "நிலநடுக்க", "భవనం", "కూలి", "పడ్డ", "పడి", "భూకంప", "കെട്ടിടം", "തകർ", "വീണ", "ഭൂചലന", "দালান", "ভবন", "ধস", "পড়ে ગ", "ভূমিকম্প", "ઇમારત", "મકાન", "પડ", "ભૂકંપ", "ਇਮਾਰਤ", "ਡਿੱ", "ਭੂਚਾଳ", "କୋଠା", "ଭୁଶୁଡ଼", "ପଡ଼"]),
    ("accident_trauma", {"accident", "crash", "bike", "car", "hit", "bleeding", "bleed", "blood", "fracture", "broke", "broken", "wound", "injury", "injured", "run over", "apaghata", "apaghaata", "gaaya", "rakta", "haadsa", "hadsa", "khoon", "vibathu", "pramadam", "raktham", "roktho", "lohi"}, ["ಅಪಘಾತ", "ರಕ್ತ", "ಗಾಯ", "ಮುರಿ", "दुर्घटना", "हादसा", "खून", "चोट", "घाव", "फ्रैक्चर", "விபத்து", "ரத்த", "இரத்த", "காயம்", "எலும்பு", "ಪ್ರಮಾదం", "రక్త", "గాయం", "విరిగ", "അപകടം", "രക്ത", "മുറിവ്", "দুর্ঘটনা", "রক্ত", "আঘাত", "ক্ষত", "અકસ્માત", "લોહી", "ઈજા", "ਹਾਦਸਾ", "ਖੂਨ", "ਸੱਟ", "ਜ਼ਖ਼ਮ", "ଦୁର୍ଘଟଣା", "ରକ୍ତ", "ଆଘାତ"]),
]

FALLBACKS = {
    "collapse_fall": {
        "en_IN": ("Building collapse / severe fall emergency! Dial 108 now, and 112 / 1078 for rescue.", [
            "Do NOT move or drag the victim (risk of spinal injury).",
            "Keep the head, neck and back still and in line.",
            "Press firmly with a clean cloth on any bleeding wound.",
            "Keep dust away from the face and talk to the person until rescuers arrive."
        ]),
        "hi_IN": ("इमारत ढहने या ऊंचाई से गिरने की आपात स्थिति! तुरंत 108 और 112 पर कॉल करें।", [
            "मरीज को बिल्कुल न हिलाएं, रीढ़ या गर्दन में गंभीर चोट हो सकती है।",
            "सिर, गर्दन और पीठ को सीधा और स्थिर रखें।",
            "खून बह रहा हो तो साफ कपड़े से सीधा और मजबूत दबाव डालें।",
            "बचाव दल आने तक मरीज से बात करते रहें और ढांढस बंधाएं।"
        ]),
        "kn_IN": ("ಕಟ್ಟಡ ಕುಸಿತ / ಎತ್ತರದಿಂದ ಬಿದ್ದ ತುರ್ತು ಪರಿಸ್ಥಿತಿ! ತಕ್ಷಣ 108 ಗೆ ಕರೆ ಮಾಡಿ, ರಕ್ಷಣೆಗೆ 112.", [
            "ರೋಗಿಯನ್ನು ಎತ್ತಬೇಡಿ ಅಥವಾ ಎಳೆಯಬೇಡಿ (ಬೆನ್ನುಮೂಳೆಯ ಗಾಯದ ಅಪಾಯ).",
            "ತಲೆ, ಕುತ್ತಿಗೆ ಮತ್ತು ಬೆನ್ನನ್ನು ಅಲುಗಾಡಿಸದೆ ನೇರವಾಗಿ ಇರಿಸಿ.",
            "ರಕ್ತಸ್ರಾವವಿದ್ದರೆ ಸ್ವಚ್ಛ ಬಟ್ಟೆಯಿಂದ ಗಾಯದ ಮೇಲೆ ಗಟ್ಟಿಯಾಗಿ ಒತ್ತಿ ಹಿಡಿಯಿರಿ.",
            "ಮುಖಕ್ಕೆ ಧೂಳು ಬರದಂತೆ ನೋಡಿಕೊಳ್ಳಿ ಮತ್ತು ರಕ್ಷಕರು ಬರುವವರೆಗೆ ಧೈರ್ಯ ಹೇಳುತ್ತಿರಿ."
        ]),
    },
    "cardiac": {
        "en_IN": ("Chest pain / breathing emergency! Dial 108 immediately and follow these steps.", [
            "Sit the person down comfortably, semi-upright, and loosen tight clothing around neck and chest.",
            "Keep them calm and still. Do not let them walk or exert themselves.",
            "If they become unresponsive and are not breathing normally, start CPR at once (100-120 compressions per minute).",
            "Stay with them until the 108 ambulance arrives."
        ]),
        "hi_IN": ("सीने में दर्द और सांस की आपात स्थिति! तुरंत 108 एम्बुलेंस को कॉल करें।", [
            "मरीज को आराम से बैठाएं और गले व छाती के कपड़े ढीले करें।",
            "मरीज को शांत रखें और बिल्कुल चलने या मेहनत न करने दें।",
            "यदि वह बेहोश हो और सामान्य सांस न ले रहा हो तो तुरंत सीपीआर (CPR) शुरू करें।",
            "108 एम्बुलेंस आने तक उसके साथ रहें।"
        ]),
        "kn_IN": ("ಎದೆನೋವು / ಉಸಿರಾಟದ ತೊಂದರೆ ತುರ್ತು ಪರಿಸ್ಥಿತಿ! ತಕ್ಷಣ 108 ಗೆ ಕರೆ ಮಾಡಿ.", [
            "ರೋಗಿಯನ್ನು ಆರಾಮವಾಗಿ ಕುಳ್ಳಿರಿಸಿ ಮತ್ತು ಕುತ್ತಿಗೆ ಹಾಗೂ ಎದೆಯ ಬಿಗಿಯಾದ ಬಟ್ಟೆಗಳನ್ನು ಸಡಿಲಗೊಳಿಸಿ.",
            "ರೋಗಿಯನ್ನು ಶಾಂತವಾಗಿರಿಸಿ, ನಡೆಯಲು ಅಥವಾ ಶ್ರಮಪಡಲು ಬಿಡಬೇಡಿ.",
            "ಪ್ರಜ್ಞೆ ತಪ್ಪಿ ಸಾಮಾನ್ಯ ಉಸಿರಾಟ ಇಲ್ಲದಿದ್ದರೆ ತಕ್ಷಣ ಸಿಪಿಆರ್ ಪ್ರಾರಂಭಿಸಿ (ನಿಮಿಷಕ್ಕೆ 100-120 ಬಾರಿ).",
            "108 ಆಂಬ್ಯುಲೆನ್ಸ್ ಬರುವವರೆಗೆ ರೋಗಿಯ ಜೊತೆಯಲ್ಲೇ ಇರಿ."
        ]),
    },
    "snakebite": {
        "en_IN": ("Snakebite emergency! Keep the victim still and calm, dial 108 immediately.", [
            "Keep the bitten limb still and at or below heart level.",
            "Do NOT tie a tight band, cut the wound or try to suck out venom.",
            "Keep the person lying down and calm; do not let them walk.",
            "Rush to the nearest hospital that has Anti-Snake Venom (ASV)."
        ]),
        "hi_IN": ("सांप के काटने की आपात स्थिति! मरीज को शांत रखें और तुरंत 108 पर कॉल करें।", [
            "काटे गए अंग को हिलाए बिना दिल के स्तर से नीचे स्थिर रखें।",
            "कसकर पट्टी न बांधें, घाव को न काटें और जहर चूसने की कोशिश न करें।",
            "मरीज को लिटाकर शांत रखें और चलने न दें।",
            "एंटी-स्नेक वेनम वाले नजदीकी अस्पताल में तुरंत ले जाएं।"
        ]),
        "kn_IN": ("ಹಾವು ಕಚ್ಚಿದ ತುರ್ತು ಪರಿಸ್ಥಿತಿ! ರೋಗಿಯನ್ನು ಶಾಂತವಾಗಿರಿಸಿ, ತಕ್ಷಣ 108 ಗೆ ಕರೆ ಮಾಡಿ.", [
            "ಕಚ್ಚಿದ ಅಂಗವನ್ನು ಅಲುಗಾಡಿಸದೆ ಹೃದಯದ ಮಟ್ಟಕ್ಕಿಂತ ಕೆಳಗೆ ಇರಿಸಿ.",
            "ಗಟ್ಟಿಯಾಗಿ ಹಗ್ಗ ಕಟ್ಟಬೇಡಿ, ಗಾಯವನ್ನು ಕತ್ತರಿಸಬೇಡಿ ಅಥವಾ ವಿಷ ಹೀರಬೇಡಿ.",
            "ರೋಗಿಯನ್ನು ಮಲಗಿಸಿ ಧೈರ್ಯ ನೀಡಿ, ನಡೆಯಲು ಬಿಡಬೇಡಿ.",
            "ಆಂಟಿ-ವೆನಮ್ (ASV) ಇರುವ ಸಮೀಪದ ಆಸ್ಪತ್ರೆಗೆ ತಕ್ಷಣ ಕರೆದೊಯ್ಯಿರಿ."
        ]),
    },
    "accident_trauma": {
        "en_IN": ("Accident / heavy bleeding emergency! Dial 108 immediately.", [
            "Press firmly and directly on the wound with a clean cloth.",
            "If a neck or back injury is possible, do not move the person.",
            "Give nothing to eat or drink. Keep them warm and keep talking to them.",
            "If they are unresponsive and not breathing normally, start CPR."
        ]),
        "hi_IN": ("दुर्घटना और भारी रक्तस्राव की आपात स्थिति! तुरंत 108 पर कॉल करें।", [
            "साफ कपड़े से घाव पर सीधा और मजबूत दबाव डालें। कपड़ा भीग जाए तो हटाएं नहीं।",
            "गर्दन या पीठ में चोट की आशंका हो तो व्यक्ति को न हिलाएं।",
            "खाने या पीने को कुछ न दें। उन्हें गर्म रखें और बात करते रहें।",
            "यदि सांस न ले रहा हो तो तुरंत सीपीआर शुरू करें।"
        ]),
        "kn_IN": ("ಅಪಘಾತ / ತೀವ್ರ ರಕ್ತಸ್ರಾವದ ತುರ್ತು ಪರಿಸ್ಥಿತಿ! ತಕ್ಷಣ 108 ಗೆ ಕರೆ ಮಾಡಿ.", [
            "ಸ್ವಚ್ಛ ಬಟ್ಟೆಯಿಂದ ಗಾಯದ ಮೇಲೆ ನೇರವಾಗಿ ಗಟ್ಟಿಯಾಗಿ ಒತ್ತಿ ಹಿಡಿಯಿರಿ.",
            "ಕುತ್ತಿಗೆ ಅಥವಾ ಬೆನ್ನಿಗೆ ಪೆಟ್ಟಾಗಿರುವ ಸಂಶಯವಿದ್ದರೆ ವ್ಯಕ್ತಿಯನ್ನು ಅಲುಗಾಡಿಸಬೇಡಿ.",
            "ತಿನ್ನಲು ಅಥವಾ ಕುಡಿಯಲು ಏನನ್ನೂ ಕೊಡಬೇಡಿ; ಬೆಚ್ಚಗಿರಿಸಿ ಮತ್ತು ಮಾತನಾಡುತ್ತಿರಿ.",
            "ಉಸಿರಾಟ ನಿಂತಿದ್ದರೆ ತಕ್ಷಣ ಸಿಪಿಆರ್ ಪ್ರಾರಂಭಿಸಿ."
        ]),
    },
    "general_medical": {
        "en_IN": ("Medical emergency! Stay calm and call 108 now.", [
            "Check continuously that the person is awake and breathing.",
            "Keep them resting in a comfortable position and reassure them.",
            "If they are unresponsive and not breathing normally, start CPR.",
            "Stay with them until the 108 ambulance arrives."
        ]),
        "hi_IN": ("चिकित्सा आपात स्थिति! शांत रहें और तुरंत 108 पर कॉल करें।", [
            "लगातार देखें कि व्यक्ति होश में है और सांस ले रहा है।",
            "उन्हें आराम की स्थिति में रखें और ढांढस बंधाएं।",
            "यदि वह बेहोश हो और सांस न ले रहा हो तो सीपीआर शुरू करें।",
            "108 एम्बुलेंस आने तक उनके साथ रहें।"
        ]),
        "kn_IN": ("ವೈದ್ಯಕೀಯ ತುರ್ತು ಪರಿಸ್ಥಿತಿ! ಗಾಬರಿಯಾಗಬೇಡಿ, ತಕ್ಷಣ 108 ಗೆ ಕರೆ ಮಾಡಿ.", [
            "ವ್ಯಕ್ತಿಗೆ ಪ್ರಜ್ಞೆ ಮತ್ತು ಉಸಿರಾಟ ಇದೆಯೇ ಎಂದು ನಿರಂತರವಾಗಿ ಪರಿಶೀಲಿಸಿ.",
            "ಆರಾಮದಾಯಕ ಭಂಗಿಯಲ್ಲಿ ವಿಶ್ರಾಂತಿ ನೀಡಿ ಮತ್ತು ಧೈರ್ಯ ಹೇಳಿ.",
            "ಪ್ರಜ್ಞೆ ಇಲ್ಲದೆ ಸಾಮಾನ್ಯ ಉಸಿರಾಟ ಇಲ್ಲದಿದ್ದರೆ ಸಿಪಿಆರ್ ಪ್ರಾರಂಭಿಸಿ.",
            "108 ಆಂಬ್ಯುಲೆನ್ಸ್ ಬರುವವರೆಗೆ ಜೊತೆಯಲ್ಲೇ ಇರಿ."
        ]),
    },
}

def identify_condition(text: str) -> str:
    t = (text or "").lower()
    tokens = set(re.findall(r"[a-z]+", t))
    for cond, ascii_kw, native_kw in CONDITION_RULES:
        if any(k in t for k in native_kw):
            return cond
        for k in ascii_kw:
            if (" " in k and k in t) or (" " not in k and k in tokens):
                return cond
    return "general_medical"

def get_env(name, default=""):
    return os.getenv(name, default).strip()

def to_hyphen(code):
    return code.replace("_", "-")

# ---------------------------------------------------------------------------
# FUNCTION 1: Independent Auto-Detection Engine
# ---------------------------------------------------------------------------
def script_lang(text: str):
    counts = {}
    for ch in (text or ""):
        cp = ord(ch)
        for lo, hi, code in SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    if not counts:
        return None
    best = max(counts, key=counts.get)
    if counts[best] < 2:
        return None
    if best == "hi_IN":
        mr = sum(text.count(m) for m in MR_MARKERS)
        hi = sum(text.count(m) for m in HI_MARKERS)
        if mr > hi:
            return "mr_IN"
        return "hi_IN"
    return best

def roman_lang(text: str, audio_lang=None):
    words = re.findall(r"[a-z]+", (text or "").lower())
    if not words:
        return None, False
    wset = set(words)
    scores = {c: len(wset & kw) for c, kw in ROMAN_LANG_KEYWORDS.items()}
    scores = {c: s for c, s in scores.items() if s > 0}
    en_hits = len(wset & EN_STOP)
    if scores:
        top = max(scores.values())
        if en_hits >= 3 and top < 2:
            return "en_IN", True
        tied = [c for c, s in scores.items() if s == top]
        pick = audio_lang if audio_lang in tied else tied[0]
        return pick, top >= 1
    if en_hits >= 2:
        return "en_IN", True
    return None, False

def autonomous_spoken_language_detector(w_text: str, g_text: str, w_lang: str) -> str:
    s_lang = script_lang(w_text) or script_lang(g_text)
    if s_lang in LANGUAGES:
        return s_lang
    r_lang, _ = roman_lang(f"{w_text or ''} {g_text or ''}", audio_lang=w_lang)
    if r_lang in LANGUAGES:
        return r_lang
    if w_lang in LANGUAGES:
        return w_lang
    return "hi_IN" if ("madad" in (w_text or "").lower() or "bachao" in (w_text or "").lower()) else "en_IN"

# ---------------------------------------------------------------------------
# FUNCTION 2: Dedicated UI Localization Engine
# ---------------------------------------------------------------------------
def set_app_ui_language(lang_code: str):
    target = lang_code if lang_code in LANGUAGES else "en_IN"
    t = TRANSLATIONS.get(target, TRANSLATIONS["en_IN"])
    return (
        header_html(t),
        t["sos_btn"],
        custom_mic_card_html(t),
        gr.update(placeholder=t["placeholder"]),
        t["send_btn"],
        t["clear_btn"],
        directory_html(t),
    )

def spectral_denoise(x: np.ndarray, sr: int) -> np.ndarray:
    nper, nover = 512, 384
    _, _, Z = stft(x, fs=sr, nperseg=nper, noverlap=nover)
    mag = np.abs(Z)
    noise = np.percentile(mag, 15, axis=1, keepdims=True)
    gain = np.clip(1.0 - 1.2 * noise / (mag + 1e-8), 0.2, 1.0)
    gain = uniform_filter1d(gain, size=3, axis=1)
    _, y = istft(Z * gain, fs=sr, nperseg=nper, noverlap=nover)
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y[: len(x)].astype(np.float32)

def trim_silence(x: np.ndarray, sr: int) -> np.ndarray:
    frame = int(0.02 * sr)
    n = len(x) // frame
    if n < 5:
        return x
    rms = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1))
    thr = max(np.percentile(rms, 90) * 0.08, 1e-4)
    idx = np.where(rms > thr)[0]
    if len(idx) == 0:
        return x
    pad = int(0.20 * sr)
    start = max(idx[0] * frame - pad, 0)
    end = min((idx[-1] + 1) * frame + pad, len(x))
    return x[start:end]

def preprocess_audio(path):
    if not path or not os.path.exists(path):
        return None, None, "no_audio"
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        return path, path, None
    data = data.mean(axis=1)
    sr = int(sr)
    if len(data) < sr * 0.20:
        return None, None, "too_short"
    data = data[: sr * MAX_SECONDS]
    if sr != TARGET_SR:
        g = math.gcd(sr, TARGET_SR)
        data = resample_poly(data, TARGET_SR // g, sr // g).astype(np.float32)
    data = data - float(np.mean(data))
    raw_peak = float(np.max(np.abs(data)))
    if raw_peak < 1e-4:
        return None, None, "silent"
    raw_out = tempfile.NamedTemporaryFile(suffix="_raw.wav", delete=False).name
    sf.write(raw_out, (data / raw_peak * 0.95).astype(np.float32), TARGET_SR, subtype="PCM_16")
    try:
        sos = butter(4, [100.0, 3800.0], btype="bandpass", fs=TARGET_SR, output="sos")
        x = sosfilt(sos, data).astype(np.float32)
        if get_env("NOISE_REDUCE", "1") != "0":
            x = spectral_denoise(x, TARGET_SR)
        x = trim_silence(x, TARGET_SR)
        peak = float(np.max(np.abs(x)))
        if peak < 1e-4:
            return raw_out, raw_out, None
        x = x / peak * 0.95
        out = tempfile.NamedTemporaryFile(suffix="_clean.wav", delete=False).name
        sf.write(out, x.astype(np.float32), TARGET_SR, subtype="PCM_16")
        return out, raw_out, None
    except Exception:
        return raw_out, raw_out, None

def _find_text(obj):
    if isinstance(obj, str):
        return obj.strip()
    if isinstance(obj, dict):
        for key in ("transcript", "text", "transcription", "result", "data"):
            if key in obj:
                found = _find_text(obj[key])
                if found:
                    return found
    if isinstance(obj, list):
        parts = [_find_text(i) for i in obj]
        return " ".join(p for p in parts if p).strip()
    return ""

def groq_whisper(audio_path):
    key = get_env("GROQ_API_KEY")
    if not key:
        return "", None, "GROQ_API_KEY missing"
    data = {
        "model": get_env("GROQ_STT_MODEL", "whisper-large-v3-turbo"),
        "response_format": "verbose_json",
        "temperature": "0",
    }
    try:
        with open(audio_path, "rb") as f:
            r = HTTP_SESSION.post(
                GROQ_STT_URL,
                headers={"Authorization": f"Bearer {key}"},
                files={"file": ("audio.wav", f, "audio/wav")},
                data=data,
                timeout=3.8,
            )
        if r.status_code != 200:
            return "", None, f"whisper {r.status_code}"
        j = r.json()
        name = str(j.get("language", "")).strip().lower()
        detected = WHISPER_NAME_TO_CODE.get(name)
        return str(j.get("text", "")).strip(), detected, ""
    except Exception as e:
        return "", None, f"whisper error: {e}"

def gnani_stt(audio_path, lang_code="hi_IN"):
    api_key = get_env("GNANI_API_KEY")
    if not api_key:
        return "", "GNANI_API_KEY missing"
    stt_lang = "en-IN" if lang_code == "mni_IN" else to_hyphen(lang_code)
    try:
        with open(audio_path, "rb") as f:
            r = HTTP_SESSION.post(
                STT_URL,
                headers={"X-API-Key-ID": api_key},
                files={"audio_file": ("voice.wav", f, "audio/wav")},
                data={"language_code": stt_lang, "format": "transcribe"},
                timeout=3.8,
            )
        if r.status_code == 200:
            return _find_text(r.json()), ""
        return "", f"STT failed ({r.status_code})"
    except Exception as e:
        return "", f"STT error: {e}"

def listen_and_detect_spoken_language(audio_path):
    clean, raw, code = preprocess_audio(audio_path)
    if code == "no_audio":
        return "", None, "⚠️ Please tap the microphone and speak your emergency."
    if code == "too_short":
        return "", None, "⚠️ Recording too short. Please speak clearly for 2–3 seconds."
    if code == "silent":
        return "", None, "⚠️ No sound detected. Please speak closer to the microphone."

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_whisper = pool.submit(groq_whisper, clean)
        fut_gnani = pool.submit(gnani_stt, clean, "hi_IN")
        w_text, w_lang, _ = fut_whisper.result()
        g_text, _ = fut_gnani.result()

    detected_spoken_lang = autonomous_spoken_language_detector(w_text, g_text, w_lang)
    transcript = w_text or g_text

    if not transcript and raw and raw != clean:
        transcript, _ = gnani_stt(raw, detected_spoken_lang)

    if not transcript:
        return "", detected_spoken_lang, "⚠️ Could not hear audio clearly. Please try again."

    return transcript, detected_spoken_lang, ""

def run_groq_triage(transcript: str, detected_spoken_lang: str):
    key = get_env("GROQ_API_KEY")
    target_script = SCRIPT_NAMES.get(detected_spoken_lang, "English")

    system_prompt = f"""You are Sahaya, an Indian emergency response triage AI.
CRITICAL DIRECTIVE:
- The caller spoke in: {detected_spoken_lang} ({target_script}).
- You MUST answer in the EXACT SAME LANGUAGE: {detected_spoken_lang} ({target_script}).
- DO NOT answer in English unless the caller spoke English.
- Return ONLY valid JSON:
{{"category": "medical|police|assault|fire|disaster", "urgency": "critical|urgent", "reply": "short reassurance in {target_script}", "steps": ["step 1 in {target_script}", "step 2 in {target_script}", "step 3 in {target_script}"]}}"""

    custom = get_env("GROQ_MODEL")
    models = ([custom] if custom else []) + [m for m in GROQ_MODELS if m != custom]

    for model in models:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"EMERGENCY CALL: {transcript}"},
            ],
            "temperature": 0.1,
            "max_tokens": 400,
            "response_format": {"type": "json_object"},
        }
        try:
            r = HTTP_SESSION.post(GROQ_CHAT_URL, headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"}, json=body, timeout=3.5)
            if r.status_code != 200:
                continue
            content = r.json()["choices"][0]["message"]["content"]
            data = json.loads(content)
            reply = str(data.get("reply", "")).strip()
            steps = [str(x).strip() for x in data.get("steps", []) if str(x).strip()]
            if reply and steps:
                return {
                    "category": str(data.get("category", "medical")).lower(),
                    "urgency": str(data.get("urgency", "critical")).lower(),
                    "reply": reply,
                    "steps": steps,
                    "lang_code": detected_spoken_lang,
                }
        except Exception:
            continue

    cond = identify_condition(transcript)
    data = FALLBACKS.get(cond, FALLBACKS["general_medical"])
    reply, steps = data.get(detected_spoken_lang, data.get("hi_IN", data["en_IN"]))
    return {
        "category": "medical",
        "urgency": "critical",
        "reply": reply,
        "steps": steps,
        "lang_code": detected_spoken_lang,
    }

def timbre_speak(text: str, lang_code: str):
    if lang_code not in TIMBRE_LANGS:
        return None, ""
    api_key = get_env("GNANI_API_KEY")
    if not api_key:
        return None, ""

    payload = {
        "text": text[:600],
        "voice": get_env("TIMBRE_VOICE") or "Shivani",
        "model": get_env("TIMBRE_MODEL") or "timbre-v3.0",
        "language": to_hyphen(lang_code),
        "speed": 1.05,
        "audio_config": {
            "sample_rate": 24000,
            "num_channels": 1,
            "sample_width": 2,
            "encoding": "linear_pcm",
            "container": "wav",
        },
    }
    try:
        r = HTTP_SESSION.post(TTS_URL, headers={"X-API-Key-ID": api_key, "Content-Type": "application/json"}, json=payload, timeout=4.2)
        if r.status_code == 200 and r.content:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(r.content)
            return tmp.name, ""
    except Exception:
        pass
    return None, ""

# ---------------------------------------------------------------------------
# UI Renderers (Explicit Inline Colors & High-Contrast Fallbacks)
# ---------------------------------------------------------------------------
def header_html(t):
    return f"""
    <div class="app-top-bar">
      <div class="brand-left">
        <div class="brand-wave-icon" aria-label="Audio Wave">
          <span></span><span></span><span></span><span></span><span></span>
        </div>
        <span class="brand-name">{html.escape(t['brand'])}</span>
      </div>
      <div class="connected-pill">
        <span class="conn-dot"></span>
        <span class="conn-text-label">{html.escape(t['ready_status'])}</span>
      </div>
      <button type="button" class="gear-icon-btn" onclick="window.openSettingsModal()" title="App Settings">
        <svg class="gear-svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#000000" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="3"></circle>
          <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
        </svg>
      </button>
    </div>
    """

def custom_mic_card_html(t):
    return f"""
    <div class="assistant-hero-card">
      <h1 class="hero-title">{html.escape(t['hero_title'])}</h1>
      <p class="hero-subtitle">{html.escape(t['subtitle'])}</p>

      <div class="mic-stage">
        <div class="mic-ring-outer">
          <div class="mic-ring-inner">
            <button type="button" class="blue-mic-btn" id="customMicBtn" onclick="window.toggleSahayaMic()">
              <svg id="svgMicIcon" width="38" height="38" viewBox="0 0 24 24" fill="white">
                <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
                <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
              </svg>
              <svg id="svgStopIcon" width="28" height="28" viewBox="0 0 24 24" fill="white" style="display:none;">
                <rect x="6" y="6" width="12" height="12" rx="2" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      <div class="mic-prompt-main" id="micPromptMain">{html.escape(t['mic_header'])}</div>

      <div class="lang-pill-wrap">
        <div class="autonomous-detect-badge">
          ⚡ <span>{html.escape(t['auto_detect_badge'])}</span>
        </div>
      </div>

      <div class="shield-banner">
        <div class="shield-text">🛡️ {html.escape(t['shield_text'])}</div>
      </div>
    </div>
    """

def result_card(service, steps, urgency, rt, reply_text, caller_text="", detected_speech=""):
    body = "<ol class='step-list'>" + "".join(f"<li style='color:inherit !important;'>{html.escape(s)}</li>" for s in steps) + "</ol>"
    caller_bubble = ""
    if caller_text:
        tag = f" <span style='font-size:11px; opacity:0.8;'>[{detected_speech}]</span>" if detected_speech else ""
        caller_bubble = f"""
        <div class="trans-bubble-blue" style="color: #1E3A8A !important;">
          <span class="bubble-vol">🗣️</span>
          <div class="bubble-txt" style="color: #1E3A8A !important;">{html.escape(caller_text)}{tag}</div>
        </div>
        """
    return f"""
    <div class="accident-result-card">
      <div class="accident-alert-top">
        <div class="alert-triangle-icon">⚠️</div>
        <h2 class="accident-main-title">{html.escape(rt.get(service['key'], 'Emergency Alert'))}</h2>
        <p class="accident-sub-note">{html.escape(rt.get('notified_sub', 'Emergency Services Notified'))} • {html.escape(urgency.upper())}</p>
      </div>

      <div class="live-trans-surface">
        <div class="live-trans-header-row">
          <div class="live-trans-header">
            <span>🚨 {html.escape(rt.get('live_trans', 'Emergency Response'))}</span>
          </div>
          <button type="button" class="inline-clear-btn" onclick="window.triggerClearEmergency()">
            {html.escape(rt.get('clear_btn', 'Clear'))}
          </button>
        </div>
        {caller_bubble}
        <div class="trans-bubble-gray">
          <span class="bubble-vol">🔊</span>
          <div class="bubble-txt">{html.escape(reply_text)}</div>
        </div>

        <div class="first-aid-section">
          <div class="first-aid-title">🩺 {html.escape(rt.get('first_aid_label', 'Immediate Steps:'))}</div>
          {body}
        </div>

        <a class="dial-now-btn" href="tel:{service['number']}">
          📞 {html.escape(rt.get('call_btn', 'Call {number}').format(number=service['number']))}
        </a>
      </div>
    </div>
    """

def directory_html(t):
    cards = []
    for srv in EMERGENCY_DIRECTORY:
        cards.append(f"""
        <div class="directory-card">
          <div class="directory-info">
            <span class="directory-icon">{srv['icon']}</span>
            <div>
              <div class="directory-name">{html.escape(t.get(srv['key'], srv['key']))}</div>
              <div class="directory-num">{srv['number']}</div>
            </div>
          </div>
          <a class="call-badge" href="tel:{srv['number']}">{html.escape(t.get('call_txt', 'Call'))}</a>
        </div>""")
    return f"""
    <div class="directory-section">
      <h4 class="directory-heading">📁 {html.escape(t['directory_header'])}</h4>
      <div class="directory-grid">{''.join(cards)}</div>
    </div>"""

# ---------------------------------------------------------------------------
# Execution Workflows
# ---------------------------------------------------------------------------
def process_emergency_flow(transcript: str, detected_spoken_lang: str):
    transcript = (transcript or "").strip()
    if not transcript:
        yield gr.update(value="⚠️ No voice or input received. Please speak clearly.", visible=True), gr.update(value=None, visible=False), gr.update(value="", visible=False)
        return

    triage = run_groq_triage(transcript, detected_spoken_lang)
    svc = EMERGENCY_DIRECTORY[CATEGORY_INDEX.get(triage["category"], 0)]
    rt = TRANSLATIONS.get(detected_spoken_lang, TRANSLATIONS["en_IN"])

    card = result_card(svc, triage["steps"], triage["urgency"], rt, triage["reply"], caller_text=transcript, detected_speech=detected_spoken_lang)
    yield gr.update(value="", visible=False), gr.update(value=None, visible=False), gr.update(value=card, visible=True)

    narration = " ".join(f"{i + 1}. {s}" for i, s in enumerate(triage["steps"]))
    audio_path, _ = timbre_speak(f"{triage['reply']} {narration}", detected_spoken_lang)
    yield gr.update(value="", visible=False), gr.update(value=audio_path, visible=bool(audio_path)), gr.update(value=card, visible=True)

def on_b64_audio(b64_data):
    if not b64_data or "," not in b64_data:
        yield gr.update(value="⚠️ Tap microphone and speak your emergency.", visible=True), gr.update(value=None, visible=False), gr.update(value="", visible=False)
        return
    try:
        raw_bytes = base64.b64decode(b64_data.split(",", 1)[1])
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        with open(tmp, "wb") as f:
            f.write(raw_bytes)
    except Exception as e:
        yield gr.update(value=f"⚠️ Audio error: {e}", visible=True), gr.update(value=None, visible=False), gr.update(value="", visible=False)
        return

    transcript, detected_spoken_lang, err = listen_and_detect_spoken_language(tmp)
    if err or not transcript:
        yield gr.update(value=err or "⚠️ Could not understand audio. Please speak again.", visible=True), gr.update(value=None, visible=False), gr.update(value="", visible=False)
        return

    yield from process_emergency_flow(transcript, detected_spoken_lang)

def on_text_submit(text_in):
    cleaned = (text_in or "").strip()
    if not cleaned:
        yield gr.update(value="⚠️ Please enter emergency details.", visible=True), gr.update(value=None, visible=False), gr.update(value="", visible=False)
        return
    detected_spoken_lang = autonomous_spoken_language_detector(cleaned, "", None)
    yield from process_emergency_flow(cleaned, detected_spoken_lang)

def on_clear():
    return "", gr.update(value="", visible=False), gr.update(value=None, visible=False), gr.update(value="", visible=False), ""

# ---------------------------------------------------------------------------
# CSS & JS Front-End
# ---------------------------------------------------------------------------
CSS = """
:root, body, .gradio-container {
    --bg-main: linear-gradient(135deg, #FDF8F2 0%, #F5EBE0 50%, #EBE1D2 100%);
    --card-bg: #FFFFFF;
    --card-border: #E2E8F0;
    --text-primary: #0F172A;
    --text-secondary: #475569;
    --input-bg: #FFFFFF;
    --input-text: #0F172A;
    --input-border: #CBD5E1;
    --topbar-bg: #FFFFFF;
}

body.dark-mode, .dark-mode .gradio-container {
    --bg-main: #18181B !important;
    --card-bg: #27272A !important;
    --card-border: #3F3F46 !important;
    --text-primary: #F8FAFC !important;
    --text-secondary: #CBD5E1 !important;
    --input-bg: #27272A !important;
    --input-text: #F8FAFC !important;
    --input-border: #52525B !important;
    --topbar-bg: #27272A !important;
}

body, .gradio-container {
    background: var(--bg-main) !important;
    color: var(--text-primary) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    max-width: 500px !important;
    margin: 0 auto !important;
    padding: 6px 10px 14px 10px !important;
}

footer { display: none !important; }
.js-bridge-hidden { display: none !important; }

/* Bouncy Sound Wave Bars Animation */
.brand-wave-icon {
    display: flex;
    align-items: center;
    gap: 3px;
    height: 22px;
}
.brand-wave-icon span {
    display: inline-block;
    width: 4px;
    height: 100%;
    background: linear-gradient(180deg, #F97316 0%, #2563EB 100%);
    border-radius: 4px;
    animation: waveBounce 1.2s ease-in-out infinite alternate;
}
.brand-wave-icon span:nth-child(1) { height: 8px; animation-delay: 0.1s; }
.brand-wave-icon span:nth-child(2) { height: 18px; animation-delay: 0.3s; }
.brand-wave-icon span:nth-child(3) { height: 12px; animation-delay: 0.15s; }
.brand-wave-icon span:nth-child(4) { height: 20px; animation-delay: 0.4s; }
.brand-wave-icon span:nth-child(5) { height: 10px; animation-delay: 0.2s; }

@keyframes waveBounce {
    0% { transform: scaleY(0.35); }
    100% { transform: scaleY(1.0); }
}

/* App Top Bar */
.app-top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    background: var(--topbar-bg) !important;
    border-radius: 16px;
    border: 1px solid var(--card-border);
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05);
    margin-bottom: 8px;
}
.brand-left { display: flex; align-items: center; gap: 8px; }
.brand-name { font-size: 22px; font-weight: 800; color: #2563EB !important; }

/* Connected Pill (Ready status) */
.connected-pill {
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
    background: #DCFCE7 !important;
    border: 1px solid #86EFAC !important;
    padding: 5px 12px !important;
    border-radius: 999px !important;
}
.conn-dot {
    width: 8px; height: 8px;
    background: #16A34A !important;
    border-radius: 50%;
    display: inline-block;
}
.conn-text-label {
    font-size: 13px !important;
    font-weight: 800 !important;
    color: #15803D !important;
    display: inline-block !important;
}

.gear-icon-btn {
    background: #FFFFFF !important;
    border: 1.5px solid #000000 !important;
    padding: 6px;
    border-radius: 50%;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* SOS Panic Button */
.sos-panic-btn {
    width: 100% !important;
    background: #DC2626 !important;
    color: #FFFFFF !important;
    font-size: 14px !important;
    font-weight: 800 !important;
    border-radius: 14px !important;
    margin-bottom: 8px !important;
    border: none !important;
}

/* Hero Mic Card */
.assistant-hero-card {
    background: var(--card-bg) !important;
    border-radius: 20px;
    padding: 18px 16px;
    text-align: center;
    border: 1px solid var(--card-border);
    margin-bottom: 8px;
}
.hero-title {
    font-size: 22px !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    margin: 0 !important;
}
.hero-subtitle {
    font-size: 14px !important;
    color: #2563EB !important;
    font-weight: 700 !important;
    margin: 3px 0 12px 0 !important;
}

.mic-stage {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin: 8px 0;
}
.mic-ring-outer {
    width: 120px; height: 120px; border-radius: 50%;
    background: rgba(37, 99, 235, 0.12);
    display: flex; align-items: center; justify-content: center;
}
.mic-ring-inner {
    width: 90px; height: 90px; border-radius: 50%;
    background: rgba(37, 99, 235, 0.2);
    display: flex; align-items: center; justify-content: center;
}
.blue-mic-btn {
    width: 68px; height: 68px; border-radius: 50%;
    background: #2563EB !important; border: none; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
}
.blue-mic-btn.recording-active { background: #DC2626 !important; }

/* Mic Prompt Main */
.mic-prompt-main {
    font-size: 15px !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    margin-top: 10px !important;
}

/* Language Badge */
.lang-pill-wrap {
    margin-top: 10px;
    display: flex;
    justify-content: center;
}
.autonomous-detect-badge {
    background: #F1F5F9 !important;
    border: 1.5px solid #94A3B8 !important;
    padding: 6px 14px !important;
    border-radius: 999px !important;
    font-size: 12px !important;
    font-weight: 800 !important;
    color: #0F172A !important;
}
.dark-mode .autonomous-detect-badge {
    background: #334155 !important;
    border: 1.5px solid #64748B !important;
    color: #F8FAFC !important;
}
.autonomous-detect-badge span {
    color: inherit !important;
    font-weight: inherit !important;
}

/* Shield Banner */
.shield-banner {
    background: rgba(22, 163, 74, 0.12) !important;
    border: 1px solid rgba(22, 163, 74, 0.25) !important;
    border-radius: 10px;
    padding: 8px 12px;
    margin-top: 12px;
}
.shield-text {
    font-size: 12px !important;
    font-weight: 800 !important;
    color: #15803D !important;
}
.dark-mode .shield-text {
    color: #4ADE80 !important;
}

/* Quick Text Row & Textbox overrides */
.quick-text-row {
    background: var(--card-bg) !important;
    border-radius: 14px !important;
    padding: 8px !important;
    display: flex !important;
    gap: 6px !important;
    margin-bottom: 8px !important;
    border: 1px solid var(--card-border) !important;
}
.quick-text-row textarea,
.quick-text-row input {
    background: var(--input-bg) !important;
    color: var(--input-text) !important;
    border: 1.5px solid var(--input-border) !important;
    border-radius: 10px !important;
    padding: 8px 12px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}
.quick-text-row textarea::placeholder,
.quick-text-row input::placeholder {
    color: #64748B !important;
    opacity: 1 !important;
}
.send-text-btn {
    background: #2563EB !important;
    color: #FFFFFF !important;
    font-weight: 800 !important;
    border-radius: 10px !important;
    border: none !important;
}
.clear-text-btn {
    background: #F1F5F9 !important;
    color: #0F172A !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    border: 1px solid #CBD5E1 !important;
}

/* Directory Section & Cards */
.directory-section { margin-top: 6px; }
.directory-heading {
    font-size: 14px !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    margin: 0 0 8px 2px !important;
}
.directory-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
}
.directory-card {
    background: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 12px;
    padding: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.directory-name {
    font-size: 12px !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
}
.directory-num {
    font-size: 13px !important;
    font-weight: 800 !important;
    color: #DC2626 !important;
}
.call-badge {
    background: #EFF6FF !important;
    color: #2563EB !important;
    border: 1px solid #BFDBFE !important;
    padding: 4px 8px;
    border-radius: 8px;
    text-decoration: none !important;
    font-weight: 800;
    font-size: 12px;
}

/* Result Card & Bubbles */
.accident-result-card {
    background: var(--card-bg) !important;
    border-radius: 18px;
    border: 1px solid #FECACA !important;
    overflow: hidden;
    margin-bottom: 8px;
}
.accident-alert-top {
    background: #FEE2E2 !important;
    padding: 12px;
    text-align: center;
}
.dark-mode .accident-alert-top {
    background: #450A0A !important;
}
.accident-main-title {
    font-size: 18px !important;
    font-weight: 800 !important;
    color: #DC2626 !important;
    margin: 0;
}
.accident-sub-note {
    font-size: 12px !important;
    font-weight: 700 !important;
    color: #991B1B !important;
    margin: 4px 0 0 0;
}
.dark-mode .accident-sub-note {
    color: #FCA5A5 !important;
}
.live-trans-surface {
    padding: 12px;
    background: var(--card-bg) !important;
    color: var(--text-primary) !important;
}
.live-trans-header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.live-trans-header {
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    font-size: 13px;
}
.inline-clear-btn {
    background: #FEE2E2 !important;
    color: #DC2626 !important;
    border: 1px solid #FCA5A5 !important;
    border-radius: 6px;
    padding: 3px 8px;
    cursor: pointer;
    font-weight: 700;
}
.trans-bubble-blue {
    background: #EFF6FF !important;
    color: #1E3A8A !important;
    border: 1px solid #BFDBFE !important;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 8px;
    font-weight: 700;
    font-size: 13px;
}
.dark-mode .trans-bubble-blue {
    background: #1E293B !important;
    color: #93C5FD !important;
    border-color: #3B82F6 !important;
}
.trans-bubble-gray {
    background: #F8FAFC !important;
    color: #0F172A !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
    font-weight: 700;
    font-size: 13px;
}
.dark-mode .trans-bubble-gray {
    background: #334155 !important;
    color: #F8FAFC !important;
    border-color: #475569 !important;
}
.bubble-txt {
    color: inherit !important;
}
.first-aid-section {
    color: var(--text-primary) !important;
}
.first-aid-title {
    font-size: 13px !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    margin-bottom: 6px;
}
.step-list {
    margin: 0 0 12px 0;
    padding-left: 20px;
    font-size: 13px;
    color: var(--text-primary) !important;
}
.step-list li {
    margin-bottom: 4px;
    font-weight: 600;
    color: var(--text-primary) !important;
}
.dial-now-btn {
    display: block;
    text-align: center;
    background: #16A34A !important;
    color: #FFFFFF !important;
    padding: 10px;
    border-radius: 10px;
    text-decoration: none !important;
    font-weight: 800;
    font-size: 14px;
}

/* Modal Settings Overlay */
.settings-modal-overlay {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(15, 23, 42, 0.55); backdrop-filter: blur(4px);
    display: none; justify-content: center; align-items: center; z-index: 9999;
}
.settings-modal-content {
    background: #FFFFFF !important; color: #0F172A !important; padding: 20px;
    border-radius: 18px; width: 90%; max-width: 380px; box-shadow: 0 10px 25px rgba(0,0,0,0.2);
}
.dark-mode .settings-modal-content {
    background: #27272A !important;
    color: #F8FAFC !important;
}
.settings-modal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
.settings-close-btn { border: none; background: #E2E8F0; border-radius: 50%; width: 28px; height: 28px; cursor: pointer; font-size: 16px; font-weight: 800; }
.setting-item { margin-bottom: 14px; }
.setting-item label { font-size: 12px; font-weight: 800; color: #475569; display: block; margin-bottom: 6px; }
.dark-mode .setting-item label { color: #CBD5E1 !important; }
.setting-item select { width: 100%; padding: 8px 10px; border-radius: 8px; border: 1.5px solid #CBD5E1; font-weight: 700; color: #0F172A; background: #FFFFFF; }
.dark-mode .setting-item select { background: #18181B !important; color: #F8FAFC !important; border-color: #52525B !important; }
"""

INIT_JS = """() => {
    window._sahayaRecording = false;
    window._audioCtx = null;
    window._mediaStream = null;
    window._scriptNode = null;
    window._pcmChunks = [];

    window.pushToGradio = function(boxId, btnId, val) {
        const el = document.querySelector('#' + boxId + ' textarea, #' + boxId + ' input');
        if (!el) return;
        const proto = el.tagName === 'TEXTAREA' ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
        const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
        if (setter) setter.call(el, val); else el.value = val;
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
        setTimeout(function() {
            const btn = document.getElementById(btnId);
            if (btn) btn.click();
        }, 30);
    };

    window.triggerClearEmergency = function() {
        const btn = document.getElementById('visible-clear-btn');
        if (btn) btn.click();
    };

    window.encodeWAVBase64 = function(buffers, inputSampleRate) {
        let totalLength = 0;
        for (let i = 0; i < buffers.length; i++) {
            totalLength += buffers[i].length;
        }
        const merged = new Float32Array(totalLength);
        let offset = 0;
        for (let i = 0; i < buffers.length; i++) {
            merged.set(buffers[i], offset);
            offset += buffers[i].length;
        }
        const targetRate = 16000;
        const ratio = inputSampleRate / targetRate;
        const samples = new Float32Array(Math.round(merged.length / ratio));
        for (let i = 0; i < samples.length; i++) {
            samples[i] = merged[Math.min(merged.length - 1, Math.floor(i * ratio))];
        }
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);
        const writeString = function(off, str) {
            for (let i = 0; i < str.length; i++) {
                view.setUint8(off + i, str.charCodeAt(i));
            }
        };
        writeString(0, 'RIFF');
        view.setUint32(4, 36 + samples.length * 2, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, targetRate, true);
        view.setUint32(28, targetRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(36, 'data');
        view.setUint32(40, samples.length * 2, true);

        let idx = 44;
        for (let i = 0; i < samples.length; i++, idx += 2) {
            let s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(idx, s < 0 ? s * 32768 : s * 32767, true);
        }

        const bytes = new Uint8Array(buffer);
        let binary = '';
        const chunkSize = 32768;
        for (let i = 0; i < bytes.length; i += chunkSize) {
            binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
        }
        return 'data:audio/wav;base64,' + btoa(binary);
    };

    window.toggleSahayaMic = async function() {
        const btn = document.getElementById('customMicBtn');
        const micSvg = document.getElementById('svgMicIcon');
        const stopSvg = document.getElementById('svgStopIcon');
        const promptMain = document.getElementById('micPromptMain');

        if (window._sahayaRecording) {
            window._sahayaRecording = false;
            try { if (window._scriptNode) window._scriptNode.disconnect(); } catch(e) {}
            try { if (window._mediaStream) window._mediaStream.getTracks().forEach(function(t) { t.stop(); }); } catch(e) {}
            try { if (window._audioCtx) window._audioCtx.close(); } catch(e) {}

            if (btn) btn.classList.remove('recording-active');
            if (micSvg) micSvg.style.display = 'block';
            if (stopSvg) stopSvg.style.display = 'none';
            if (promptMain) promptMain.textContent = 'Processing emergency speech...';

            if (window._pcmChunks.length > 0) {
                const b64 = window.encodeWAVBase64(window._pcmChunks, 44100);
                window._pcmChunks = [];
                window.pushToGradio('hidden-b64-audio', 'hidden-audio-submit-btn', b64);
            }
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            window._mediaStream = stream;
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            const ctx = new AudioCtx();
            window._audioCtx = ctx;
            const source = ctx.createMediaStreamSource(stream);
            const processor = ctx.createScriptProcessor(4096, 1, 1);
            window._scriptNode = processor;
            window._pcmChunks = [];
            window._sahayaRecording = true;

            processor.onaudioprocess = function(e) {
                if (!window._sahayaRecording) return;
                window._pcmChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
            };

            source.connect(processor);
            processor.connect(ctx.destination);

            if (btn) btn.classList.add('recording-active');
            if (micSvg) micSvg.style.display = 'none';
            if (stopSvg) stopSvg.style.display = 'block';
            if (promptMain) promptMain.textContent = 'Recording... (Tap button to stop)';
        } catch (err) {
            alert('Please allow microphone permissions to record.');
        }
    };

    window.openSettingsModal = function() {
        let modal = document.getElementById('sahayaSettingsModal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'sahayaSettingsModal';
            modal.className = 'settings-modal-overlay';
            modal.innerHTML = `
                <div class="settings-modal-content">
                    <div class="settings-modal-header">
                        <h3 style="margin:0;">⚙️ App Settings</h3>
                        <button class="settings-close-btn" onclick="window.closeSettingsModal()">&times;</button>
                    </div>
                    <div class="setting-item">
                        <label>🌐 Preferred Language (Translates Entire App)</label>
                        <select id="modalLangSelect" onchange="window.syncLanguageSelection(this.value)">
                            <option value="en_IN">English (India)</option>
                            <option value="kn_IN">ಕನ್ನಡ (Kannada)</option>
                            <option value="hi_IN">हिन्दी (Hindi)</option>
                            <option value="ta_IN">தமிழ் (Tamil)</option>
                            <option value="te_IN">తెలుగు (Telugu)</option>
                            <option value="ml_IN">മലയാളം (Malayalam)</option>
                            <option value="mr_IN">मराठी (Marathi)</option>
                            <option value="gu_IN">ગુજરાતી (Gujarati)</option>
                            <option value="bn_IN">বাংলা (Bengali)</option>
                            <option value="pa_IN">ਪੰਜਾਬੀ (Punjabi)</option>
                            <option value="or_IN">ଓଡ଼ିଆ (Odia)</option>
                            <option value="mni_IN">ꯃꯤꯇꯩꯂꯣꯟ (Manipuri)</option>
                        </select>
                    </div>
                    <div class="setting-item">
                        <label>🎨 Theme Mode</label>
                        <select onchange="document.body.classList.toggle('dark-mode', this.value === 'dark')">
                            <option value="light">☀️ Light Theme</option>
                            <option value="dark">🌙 Dark Theme</option>
                        </select>
                    </div>
                </div>
            `;
            document.body.appendChild(modal);
        }
        modal.style.display = 'flex';
    };

    window.closeSettingsModal = function() {
        const modal = document.getElementById('sahayaSettingsModal');
        if (modal) modal.style.display = 'none';
    };

    window.syncLanguageSelection = function(langCode) {
        window.pushToGradio('hidden-lang-box', 'hidden-lang-submit-btn', langCode);
        window.closeSettingsModal();
    };
}"""

# ---------------------------------------------------------------------------
# Gradio Application
# ---------------------------------------------------------------------------
t0 = TRANSLATIONS["en_IN"]

with gr.Blocks(title="Sahaya Emergency") as demo:
    header = gr.HTML(header_html(t0))

    sos_btn = gr.Button(t0["sos_btn"], elem_classes=["sos-panic-btn"])
    sos_btn.click(
        fn=None,
        js="""() => { window.open('tel:112', '_self'); }"""
    )

    mic_card = gr.HTML(custom_mic_card_html(t0))

    with gr.Row(elem_classes=["quick-text-row"]):
        text_in = gr.Textbox(placeholder=t0["placeholder"], show_label=False, scale=5, container=False)
        send_btn = gr.Button(t0["send_btn"], elem_classes=["send-text-btn"], scale=2)
        clear_btn = gr.Button(t0["clear_btn"], elem_id="visible-clear-btn", elem_classes=["clear-text-btn"], scale=1)

    status = gr.Markdown("", visible=False)
    reply_audio = gr.Audio(label="🔊 Voice Guidance", type="filepath", autoplay=True, interactive=False, visible=False)
    card_out = gr.HTML("", visible=False)

    dir_html = gr.HTML(directory_html(t0))

    # Hidden bridge elements for JS interop
    with gr.Row(elem_classes=["js-bridge-hidden"]):
        lang_state = gr.Textbox(value="en_IN", elem_id="hidden-lang-box")
        lang_btn = gr.Button("ApplyLang", elem_id="hidden-lang-submit-btn")
        b64_audio_in = gr.Textbox(value="", elem_id="hidden-b64-audio")
        b64_audio_btn = gr.Button("SubmitAudio", elem_id="hidden-audio-submit-btn")

    # Autonomous speech detection flow
    b64_audio_btn.click(
        on_b64_audio,
        inputs=[b64_audio_in],
        outputs=[status, reply_audio, card_out],
    )

    # Text submission flow
    send_btn.click(
        on_text_submit,
        inputs=[text_in],
        outputs=[status, reply_audio, card_out],
    )
    text_in.submit(
        on_text_submit,
        inputs=[text_in],
        outputs=[status, reply_audio, card_out],
    )

    # Clear state
    clear_btn.click(
        on_clear,
        inputs=None,
        outputs=[text_in, status, reply_audio, card_out, b64_audio_in],
    )

    # Explicit UI translation flow (decoupled from caller voice detection)
    lang_btn.click(
        set_app_ui_language,
        inputs=lang_state,
        outputs=[header, sos_btn, mic_card, text_in, send_btn, clear_btn, dir_html],
    )

    demo.queue(default_concurrency_limit=6, max_size=40)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=gr.themes.Base(),
        css=CSS,
        js=INIT_JS,
    )

