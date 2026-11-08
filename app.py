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
        "mic_header": "🎙️ Tap to Start Listening",
        "type_header": "⌨️ Or Type Emergency Details",
        "type_placeholder": "e.g., Need an ambulance, chest pain, snake bite...",
        "send": "Send Emergency Alert",
        "detected": "🗣️ Detected Emergency:",
        "action_header": "Recommended Action & First-Aid",
        "first_aid_label": "Immediate First-Aid Steps:",
        "call_btn": "Call {number} Immediately",
        "directory_header": "Emergency Services Directory",
        "helpline": "Helpline:",
        "audio_response": "🔊 Sahaya Voice Guidance:",
    },
    "kn_IN": {
        "title": "ಸಹಾಯ (Sahaya)",
        "subtitle": "ಸುರಕ್ಷಿತವಾಗಿರಿ. 🛡️",
        "mic_header": "🎙️ ಮಾತನಾಡಲು ಟ್ಯಾಪ್ ಮಾಡಿ",
        "type_header": "⌨️ ಅಥವಾ ತುರ್ತು ವಿವರಗಳನ್ನು ಇಲ್ಲಿ ಟೈಪ್ ಮಾಡಿ",
        "type_placeholder": "ಉದಾ. ಎದೆನೋವು, ಉಸಿರಾಟದ ತೊಂದರೆ, ಹಾವು ಕಚ್ಚಿದೆ, ಆಂಬ್ಯುಲೆನ್ಸ್ ಬೇಕು...",
        "send": "ತುರ್ತು ಎಚ್ಚರಿಕೆ ಕಳುಹಿಸಿ",
        "detected": "🗣️ ಪತ್ತೆಯಾದ ತುರ್ತು ಪರಿಸ್ಥಿತಿ:",
        "action_header": "ಶಿಫಾರಸು ಮಾಡಿದ ಕ್ರಮ ಮತ್ತು ಪ್ರಥಮ ಚಿಕಿತ್ಸೆ",
        "first_aid_label": "ತಕ್ಷಣದ ಪ್ರಥಮ ಚಿಕಿತ್ಸಾ ಕ್ರಮಗಳು:",
        "call_btn": "ತಕ್ಷಣ {number} ಗೆ ಕರೆ ಮಾಡಿ",
        "directory_header": "ತುರ್ತು ಸಹಾಯವಾಣಿ ಡೈರೆಕ್ಟರಿ",
        "helpline": "ಸಹಾಯವಾಣಿ:",
        "audio_response": "🔊 ಸಹಾಯ ಧ್ವನಿ ಮಾರ್ಗದರ್ಶನ:",
    },
    "hi_IN": {
        "title": "सहायता (Sahaya)",
        "subtitle": "सुरक्षित रहें। 🛡️",
        "mic_header": "🎙️ बोलने के लिए टैप करें",
        "type_header": "⌨️ या आपातकालीन विवरण टाइप करें",
        "type_placeholder": "जैसे सीने में दर्द, सांस लेने में तकलीफ, सांप ने काटा है...",
        "send": "आपातकालीन अलर्ट भेजें",
        "detected": "🗣️ पता चली आपात स्थिति:",
        "action_header": "अनुशंसित कार्रवाई और प्राथमिक उपचार",
        "first_aid_label": "प्राथमिक उपचार दिशानिर्देश:",
        "call_btn": "तुरंत {number} पर कॉल करें",
        "directory_header": "आपातकालीन सेवा डायरेक्टरी",
        "helpline": "हेल्पलाइन:",
        "audio_response": "🔊 सहायता वॉयस मार्गदर्शन:",
    },
    "ta_IN": {
        "title": "சஹாயா (Sahaya)",
        "subtitle": "பாதுகாப்பாக இருங்கள். 🛡️",
        "mic_header": "🎙️ பேச தட்டவும்",
        "type_header": "⌨️ அவசர விவரங்களை தட்டச்சு செய்யவும்",
        "type_placeholder": "எ.கா. நெஞ்சு வலி, மூச்சு திணறல், ஆம்புலன்ஸ் தேவை...",
        "send": "அவசர எச்சரிக்கை அனுப்பு",
        "detected": "🗣️ கண்டறியப்பட்ட அவசரம்:",
        "action_header": "பரிந்துரைக்கப்பட்ட முதலுதவி",
        "first_aid_label": "உடனடி முதலுதவி வழிகாட்டுதல்கள்:",
        "call_btn": "உடனே {number} ஐ அழைக்கவும்",
        "directory_header": "அவசர சேவை அடைவு",
        "helpline": "உதவி எண்:",
        "audio_response": "🔊 சஹாயா குரல் வழிகாட்டல்:",
    },
    "te_IN": {
        "title": "సహాయ (Sahaya)",
        "subtitle": "సురక్షితంగా ఉండండి. 🛡️",
        "mic_header": "🎙️ మాట్లాడటానికి నొక్కండి",
        "type_header": "⌨️ అత్యవసర వివరాలు టైప్ చేయండి",
        "type_placeholder": "ఉదా. గుండె నొప్పి, శ్వాస తీసుకోవడంలో ఇబ్బంది, పాము కరిచింది...",
        "send": "అత్యవసర హెచ్చరిక పంపండి",
        "detected": "🗣️ గుర్తించబడిన అత్యవసరం:",
        "action_header": "సిఫార్సు చేయబడిన ప్రథమ చికిత్స",
        "first_aid_label": "తక్షణ ప్రథమ చికిత్స మార్గదర్శకాలు:",
        "call_btn": "ఇప్పుడే {number} కి కాల్ చేయండి",
        "directory_header": "అత్యవసర సేవల డైరెక్టరీ",
        "helpline": "హెల్ప్‌లైన్:",
        "audio_response": "🔊 సహాయ వాయిస్ మార్గదర్శకత్వం:",
    },
    "ml_IN": {
        "title": "സഹായ (Sahaya)",
        "subtitle": "സുരക്ഷിതരായിരിക്കൂ. 🛡️",
        "mic_header": "🎙️ സംസാരിക്കാൻ ടാപ്പ് ചെയ്യുക",
        "type_header": "⌨️ വിവരങ്ങൾ ടൈപ്പ് ചെയ്യുക",
        "type_placeholder": "ഉദാ. നെഞ്ചുവേദന, ശ്വാസതടസ്സം, ആംബുലൻസ് വേണം...",
        "send": "അടിയന്തര മുന്നറിയിപ്പ് അയയ്ക്കുക",
        "detected": "🗣️ കണ്ടെത്തിയ അത്യാഹിതം:",
        "action_header": "ശുപാർശ ചെയ്യുന്ന പ്രഥമശുശ്രൂഷ",
        "first_aid_label": "ഉടൻ ചെയ്യേണ്ട പ്രഥമശുശ്രൂഷാ ഘട്ടങ്ങൾ:",
        "call_btn": "ഉടൻ {number}-ലേക്ക് വിളിക്കുക",
        "directory_header": "എമർജൻസി ഡയറക്ടറി",
        "helpline": "ഹെൽപ്പ്‌ലൈൻ:",
        "audio_response": "🔊 സഹായ വോയ്സ് പ്രതികരണം:",
    },
    "mr_IN": {
        "title": "साहाय्य (Sahaya)",
        "subtitle": "सुरक्षित राहा. 🛡️",
        "mic_header": "🎙️ बोलण्यासाठी टॅप करा",
        "type_header": "⌨️ किंवा टाइप करा",
        "type_placeholder": "उदा. छातीत दुखणे, श्वास घेण्यास त्रास, रुग्णवाहिका हवी...",
        "send": "आणीबाणी सूचना पाठवा",
        "detected": "🗣️ आढळलेली आणीबाणी:",
        "action_header": "शिफारस केलेले प्रथमोपचार",
        "first_aid_label": "तातडीच्या प्रथमोपचार सूचना:",
        "call_btn": "आता {number} वर कॉल करा",
        "directory_header": "आणीबाणी निर्देशिका",
        "helpline": "हेल्पलाइन:",
        "audio_response": "🔊 साहाय्य व्हॉइस मार्गदर्शन:",
    },
    "gu_IN": {
        "title": "સહાય (Sahaya)",
        "subtitle": "સુરક્ષિત રહો. 🛡️",
        "mic_header": "🎙️ બોલવા માટે ટેપ કરો",
        "type_header": "⌨️ અથવા વિગતો ટાઇપ કરો",
        "type_placeholder": "દા.ત. છાતીમાં દુખાવો, શ્વાસ લેવામાં તકલીફ, એમ્બ્યુલન્સ...",
        "send": "કટોકટી ચેતવણી મોકલો",
        "detected": "🗣️ મળેલી કટોકટી:",
        "action_header": "પ્રાથમિક સારવાર માર્ગદર્શિકા",
        "first_aid_label": "તાત્કાલિક પ્રાથમિક સારવારનાં પગલાં:",
        "call_btn": "હમણાં {number} પર કૉલ કરો",
        "directory_header": "ઇમરજન્સી ડિરેક્ટરી",
        "helpline": "હેલ્પલાઇન:",
        "audio_response": "🔊 સહાય વૉઇસ માર્ગદર્શન:",
    },
    "bn_IN": {
        "title": "সহায় (Sahaya)",
        "subtitle": "সুরক্ষিত থাকুন। 🛡️",
        "mic_header": "🎙️ কথা বলতে ট্যাপ করুন",
        "type_header": "⌨️ জরুরি বিবরণ টাইপ করুন",
        "type_placeholder": "যেমন বুকে ব্যথা, শ্বাসকষ্ট, অ্যাম্বুলেন্স দরকার...",
        "send": "জরুরি সতর্কতা পাঠান",
        "detected": "🗣️ শনাক্ত জরুরি অবস্থা:",
        "action_header": "প্রাথমিক চিকিৎসার নির্দেশিকা",
        "first_aid_label": "জরুরি প্রাথমিক চিকিৎসার পদক্ষেপ:",
        "call_btn": "এখনই {number}-এ কল করুন",
        "directory_header": "জরুরি ডিরেক্টরি",
        "helpline": "হেল্পলাইন:",
        "audio_response": "🔊 সহায় ভয়েস সহায়তা:",
    },
    "pa_IN": {
        "title": "ਸਹਾਇਤਾ (Sahaya)",
        "subtitle": "ਸੁਰੱਖਿਅਤ ਰਹੋ। 🛡️",
        "mic_header": "🎙️ ਬੋਲਣ ਲਈ ਟੈਪ ਕਰੋ",
        "type_header": "⌨️ ਜਾਂ ਵੇਰਵੇ ਟਾਈਪ ਕਰੋ",
        "type_placeholder": "ਜਿਵੇਂ ਛਾਤੀ ਵਿੱਚ ਦਰਦ, ਸਾਹ ਲੈਣ ਵਿੱਚ ਤਕਲੀਫ, ਐਂਬੂਲੈਂਸ...",
        "send": "ਐਮਰਜੈਂਸੀ ਅਲਰਟ ਭੇਜੋ",
        "detected": "🗣️ ਪਤਾ ਲੱਗੀ ਐਮਰਜੈਂਸੀ:",
        "action_header": "ਮੁੱਢਲੀ ਸਹਾਇਤਾ ਦਿਸ਼ਾ-ਨਿਰਦੇਸ਼",
        "first_aid_label": "ਤੁਰੰਤ ਮੁੱਢਲੀ ਸਹਾਇਤਾ ਕਦਮ:",
        "call_btn": "ਹੁਣੇ {number} 'ਤੇ ਕਾਲ ਕਰੋ",
        "directory_header": "ਐਮਰਜੈਂਸੀ ਡਾਇਰੈਕਟਰੀ",
        "helpline": "ਹੈਲਪਲਾਈਨ:",
        "audio_response": "🔊 ਸਹਾਇਤਾ ਵੌਇਸ ਸੇਧ:",
    },
    "or_IN": {
        "title": "ସହାୟ (Sahaya)",
        "subtitle": "ସୁରକ୍ଷିତ ରୁହନ୍ତୁ। 🛡️",
        "mic_header": "🎙️ କହିବାକୁ ଟ୍ୟାପ୍ କରନ୍ତୁ",
        "type_header": "⌨️ ଜରୁରୀ ବିବରଣୀ ଟାଇପ୍ କରନ୍ତୁ",
        "type_placeholder": "ଯେପରି ଛାତି ଯନ୍ତ୍ରଣା, ନିଶ୍ୱାସ ପ୍ରଶ୍ୱାସରେ କଷ୍ଟ, ଆମ୍ବୁଲାନ୍ସ...",
        "send": "ଜରୁରୀକାଳୀନ ଆଲର୍ଟ ପଠାନ୍ତୁ",
        "detected": "🗣️ ଚିହ୍ନଟ ଜରୁରୀକାଳୀନ ପରିସ୍ଥିତି:",
        "action_header": "ପ୍ରଥମ ଚିକିତ୍ସା ନିର୍ଦ୍ଦେଶାବଳୀ",
        "first_aid_label": "ତୁରନ୍ତ ପ୍ରଥମ ଚିକିତ୍ସା ପଦକ୍ଷେପ:",
        "call_btn": "ତୁରନ୍ତ {number} କୁ କଲ୍ କରନ୍ତୁ",
        "directory_header": "ଜରୁରୀକାଳୀନ ଡାଇରେକ୍ଟୋରୀ",
        "helpline": "ହେଲ୍ପଲାଇନ୍:",
        "audio_response": "🔊 ସହାୟ ଭଏସ୍ ମାର୍ଗଦର୍ଶନ:",
    },
    "mni_IN": {
        "title": "ꯁꯍꯥꯌ (Sahaya)",
        "subtitle": "ꯉꯥꯛꯊꯣꯛꯆꯕꯤꯌꯨ 🛡️",
        "mic_header": "🎙️ ꯋꯥ ꯉꯥꯡꯅꯕ ꯇꯦꯞ ꯇꯧꯕꯤꯌꯨ",
        "type_header": "⌨️ ꯈꯨꯗꯣꯡꯊꯤꯕ ꯏꯕꯤꯌꯨ",
        "type_placeholder": "ꯊꯃꯣꯏ ꯆꯤꯛꯄ, ꯁ꯭ꯕꯥꯁ ꯍꯣꯟꯕ ꯋꯥꯕ, ꯑꯦꯝꯕꯨꯂꯦꯟꯁ...",
        "send": "ꯈꯨꯗꯣꯡꯊꯤꯕꯒꯤ ꯄꯥꯎ ꯊꯥꯕꯤꯌꯨ",
        "detected": "🗣️ ꯈꯪꯗꯣꯛꯂꯕ ꯈꯨꯗꯣꯡꯊꯤꯕ:",
        "action_header": "ꯑꯍꯥꯟꯕ ꯃꯇꯦꯡ (First Aid)",
        "first_aid_label": "ꯈꯨꯗꯛꯇ ꯇꯧꯒꯗꯕ ꯊꯕꯛꯁꯤꯡ:",
        "call_btn": "{number} ꯗ ꯍꯧꯖꯤꯛꯃꯛ ꯀꯣꯜ ꯇꯧꯕꯤꯌꯨ",
        "directory_header": "ꯑꯇꯣꯞꯄ ꯃꯇꯦꯡ ꯅꯝꯕꯔꯁꯤꯡ",
        "helpline": "ꯍꯦꯜꯞꯂꯥꯏꯟ:",
        "audio_response": "🔊 ꯁꯍꯥꯌꯒꯤ ꯄꯥꯎꯈꯨꯝ:",
    },
}

# ---------------------------------------------------------------------------
# Emergency Directory
# ---------------------------------------------------------------------------
EMERGENCY_DIRECTORY = [
    {
        "title": "Medical & Ambulance",
        "icon": "🚑",
        "number": "108",
        "desc": "Medical emergencies, cardiac arrest, trauma, snake bites & urgent patient dispatch.",
    },
    {
        "title": "Police & Safety",
        "icon": "👮",
        "number": "112",
        "desc": "Immediate law enforcement intervention and urgent safety protection.",
    },
    {
        "title": "Women Distress Helpline",
        "icon": "🛡️",
        "number": "1091",
        "desc": "Direct protection, helpline assistance and intervention against harassment.",
    },
    {
        "title": "Fire & Rescue",
        "icon": "🚒",
        "number": "101",
        "desc": "Fire hazard suppression, burn mitigation and urgent structure rescue.",
    },
    {
        "title": "Disaster Support",
        "icon": "🌊",
        "number": "1078",
        "desc": "Relief and evacuation assistance during floods, earthquakes and storms.",
    },
]

CATEGORY_INDEX = {"medical": 0, "police": 1, "assault": 2, "fire": 3, "disaster": 4}

# ---------------------------------------------------------------------------
# Native Scripts & Robust Romanized Keywords for All 12 Languages
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
        "aspatre", "aaspatre", "othada", "kodi", "gundige", "sikakide", "biddare"
    },
    "hi_IN": {
        "mujhe", "humein", "humko", "mere", "meri", "pita", "pitaji", "papa", "mataji",
        "seene", "dard", "saans", "takleef", "ruk", "chhati", "saanp", "saap",
        "bachao", "madad", "khoon", "chot", "ghayal", "aspatal", "jaldi", "kripya", "bhai"
    },
    "ta_IN": {
        "enakku", "engalukku", "nenju", "vali", "swasam", "moochu", "thinaral",
        "appa", "thanthai", "amma", "pambu", "paambu", "kadichathu", "udavi",
        "vendum", "ratham", "thee", "kaapathunga", "maruthuvamanai", "seekiram"
    },
    "te_IN": {
        "naaku", "maaku", "gunde", "noppi", "swasa", "kastam", "adatam", "nanna",
        "thandri", "amma", "paamu", "karichindi", "sahayam", "kavali", "raktam",
        "mantalu", "kapadandi", "aaspattri", "thondaraga"
    },
    "ml_IN": {
        "enikku", "njangalkku", "nenju", "vedana", "shwasam", "muttal", "achan",
        "amma", "paampu", "kadichu", "sahayam", "venam", "raktham", "thee",
        "rakshikku", "aashupathri", "vegam"
    },
    "mr_IN": {
        "mala", "aamhaala", "majhya", "chhati", "dukhate", "dukhne", "shwas",
        "trass", "vadil", "baba", "saap", "chavla", "madat", "havi", "rakta",
        "vachva", "davakhana", "lavkar"
    },
    "bn_IN": {
        "amar", "amader", "buker", "byatha", "batha", "shwas", "kosto", "baba",
        "shap", "kamreche", "sahajjo", "dorkar", "rokto", "aagun", "bachan",
        "haspatal", "taratari"
    },
    "gu_IN": {
        "mane", "amne", "maru", "chhati", "dard", "shwas", "takleef", "bapa",
        "saap", "dankhyo", "madad", "joiye", "lohi", "aag", "bachavo", "davakhana", "jaldi"
    },
    "pa_IN": {
        "meri", "mere", "sanu", "chhati", "peerh", "saah", "aukh", "bapu",
        "sapp", "dangea", "madad", "chahidi", "khoon", "agg", "bachao", "aspataal", "cheti"
    },
    "or_IN": {
        "mora", "aamara", "chhati", "bitha", "shwasa", "kasta", "bapa", "sapa",
        "kamudichi", "sahajya", "darkar", "rakta", "nia", "bachantu", "sighra"
    },
    "mni_IN": {
        "eigi", "eikhoi", "thamoigi", "chikpa", "masha", "ithak", "lin",
        "chikre", "mateng", "pibiyu", "ee", "mei", "kanbiyu", "thuna"
    },
    "en_IN": {
        "heart", "chest", "pain", "breathe", "breathing", "father", "mother",
        "snake", "bite", "accident", "bleeding", "ambulance", "hospital", "urgent"
    }
}

# ---------------------------------------------------------------------------
# Condition Classifier & Rich 12-Language Offline Protocols
# ---------------------------------------------------------------------------
def identify_condition(text: str) -> str:
    t = text.lower()
    if any(k in t for k in [
        "heart", "chest", "breath", "breathing", "cardiac", "stroke", "attack",
        "ede", "edeno", "usir", "husir", "gundige", "seene", "chhati", "saans",
        "nenju", "moochu", "swasam", "gunde", "swasa", "buker", "shwas", "peerh", "saah"
    ]):
        return "cardiac"
    if any(k in t for k in [
        "snake", "bite", "haavu", "havu", "kachide", "saanp", "saap",
        "paambu", "paamu", "lin", "chikpa", "sapp", "sapa", "dankh"
    ]):
        return "snakebite"
    if any(k in t for k in [
        "poison", "pesticide", "chemical", "visha", "keetnashaka", "zeher", "marunthu"
    ]):
        return "pesticide"
    if any(k in t for k in [
        "fire", "burn", "smoke", "cylinder", "blast", "benki", "aag", "thee", "agg", "nia", "mei"
    ]):
        return "fire"
    if any(k in t for k in [
        "pregnant", "pregnancy", "delivery", "labor", "maternal", "garbhini", "prasava"
    ]):
        return "maternal"
    if any(k in t for k in [
        "police", "thief", "robbery", "attack", "fight", "assault", "harassment", "chor", "bachao"
    ]):
        return "police"
    return "general_medical"

CONDITION_FALLBACKS = {
    "cardiac": {
        "service_idx": 0,
        "urgency": "critical",
        "kn_IN": (
            "ದಯವಿಟ್ಟು ಗಾಬರಿಯಾಗಬೇಡಿ. ತಕ್ಷಣ 108 ಆಂಬ್ಯುಲೆನ್ಸ್‌ಗೆ ಕರೆ ಮಾಡಿ ಮತ್ತು ಈ ಕೆಳಗಿನ ತುರ್ತು ಕ್ರಮಗಳನ್ನು ಅನುಸರಿಸಿ.",
            [
                "ವ್ಯಕ್ತಿಯನ್ನು ಆರಾಮವಾಗಿ ಕುಳ್ಳಿರಿಸಿ ಮತ್ತು ಕುತ್ತಿಗೆ, ಎದೆಯ ಬಿಗಿಯಾದ ಬಟ್ಟೆಗಳನ್ನು ಸಡಿಲಗೊಳಿಸಿ.",
                "ರೋಗಿಯನ್ನು ಸಂಪೂರ್ಣ ಶಾಂತವಾಗಿರಲು ಹೇಳಿ, ಯಾವುದೇ ಆಯಾಸ ಅಥವಾ ನಡಿಗೆಗೆ ಬಿಡಬೇಡಿ.",
                "ಪ್ರಜ್ಞೆ ತಪ್ಪಿ ಉಸಿರಾಟ ನಿಂತಿದ್ದರೆ, ತಕ್ಷಣ ಎದೆಯ ಮಧ್ಯಭಾಗದಲ್ಲಿ ಸಿಪಿಆರ್ (CPR) ಒತ್ತಡ ಪ್ರಾರಂಭಿಸಿ (ನಿಮಿಷಕ್ಕೆ 100-120 ಬಾರಿ).",
                "ತಕ್ಷಣವೇ ಸಮೀಪದ ತುರ್ತು ಹೃದಯ ಚಿಕಿತ್ಸಾ ಆಸ್ಪತ್ರೆಗೆ ರವಾನಿಸಿ."
            ]
        ),
        "hi_IN": (
            "कृपया घबराएं नहीं। तुरंत 108 एम्बुलेंस को कॉल करें और प्राथमिक उपचार के इन चरणों का पालन करें।",
            [
                "मरीज को आराम से बैठाएं और गले व कमर के कपड़े ढीले करें।",
                "मरीज को शांत रखें और कोई भी शारीरिक श्रम न करने दें।",
                "यदि सांस रुक जाए या बेहोश हो जाएं, तो तुरंत सीपीआर (CPR) शुरू करें (100-120 प्रति मिनट)।",
                "तुरंत 108 पर संपर्क कर नजदीकी अस्पताल ले जाएं।"
            ]
        ),
        "en_IN": (
            "Please stay calm. Immediately call 108 for an ambulance and follow these cardiac emergency steps.",
            [
                "Sit the person down comfortably and loosen tight clothing around the neck and waist.",
                "Keep the patient calm and still. Do not let them walk or exert themselves.",
                "If the person becomes unresponsive and stops breathing normally, begin CPR immediately (100-120 chest compressions/min).",
                "Transport immediately to the nearest hospital with emergency cardiac facilities."
            ]
        ),
        "ta_IN": (
            "தயவுசெய்து பயப்பட வேண்டாம். உடனே 108 ஆம்புலன்ஸை அழைக்கவும், இந்த முதலுதவி வழிமுறைகளைப் பின்பற்றவும்.",
            [
                "நோயாளியை உட்கார வைத்து, கழுத்து மற்றும் இடுப்பில் உள்ள இறுக்கமான ஆடைகளை தளர்த்தவும்.",
                "நோயாளியை எந்த உடல் அசைவும் செய்ய விடாமல் அமைதியாக வைத்திருக்கவும்.",
                "சுவாசம் நின்றால் உடனே மார்பில் CPR அழுத்தத்தை தொடங்கவும் (நிமிடத்திற்கு 100-120 முறை).",
                "உடனடியாக அருகிலுள்ள அவசர சிகிச்சை மருத்துவமனைக்கு அழைத்துச் செல்லவும்."
            ]
        ),
        "te_IN": (
            "దయచేసి కంగారు పడవద్దు. వెంటనే 108 అంబులెన్స్‌కు కాల్ చేయండి మరియు ఈ ప్రథమ చికిత్స చర్యలు పాటించండి.",
            [
                "రోగిని విశ్రాంతిగా కూర్చోబెట్టి, మెడ మరియు నడుము వద్ద దుస్తులను వదులు చేయండి.",
                "రోగిని ప్రశాంతంగా ఉంచండి, ఎటువంటి శ్రమ చేయనివ్వవద్దు.",
                "శ్వాస ఆడకపోతే లేదా స్పృహ తప్పితే వెంటనే గుండెపై CPR ప్రారంభించండి (నిమిషానికి 100-120 సార్లు).",
                "వెంటనే సమీప ఆసుపత్రికి తరలించండి."
            ]
        ),
        "ml_IN": (
            "ദയവായി പരിഭ്രാന്തരാകരുത്. ഉടൻ 108 ആംബുലൻസ് വിളിക്കുക, ഈ പ്രഥമശുശ്രൂഷാ ഘട്ടങ്ങൾ പാലിക്കുക.",
            [
                "രോഗിയെ ശാന്തമായി ഇരുത്തുക, ഇറുകിയ വസ്ത്രങ്ങൾ അയക്കുക.",
                "രോഗി നടക്കുകയോ അധ്വാനിക്കുകയോ ചെയ്യാതിരിക്കാൻ ശ്രദ്ധിക്കുക.",
                "ശ്വാസമെടുക്കുന്നില്ലെങ്കിൽ ഉടൻ നെഞ്ചിൽ CPR അമർത്താൻ തുടങ്ങുക (മിനിറ്റിൽ 100-120 തവണ).",
                "ഉടൻ തന്നെ അടുത്തുള്ള ആശുപത്രിയിലേക്ക് കൊണ്ടുപോകുക."
            ]
        ),
        "mr_IN": (
            "कृपया घाबरू नका. त्वरित 108 रुग्णवाहिकेला कॉल करा आणि हे प्रथमोपचार करा.",
            [
                "रुग्णाला शांत बसवा आणि गळ्याभोवतीचे घट्ट कपडे सैल करा.",
                "रुग्णाला अजिबात हालचाल किंवा श्रम करू देऊ नका.",
                "श्वास थांबल्यास त्वरित छातीवर CPR सुरू करा (प्रति मिनिट 100-120 वेळा).",
                "लवकरात लवकर जवळच्या रुग्णालयात दाखल करा."
            ]
        ),
        "gu_IN": (
            "કૃપા કરીને ગભરાશો નહીં. તરત જ 108 એમ્બ્યુલન્સને કૉલ કરો અને આ પગલાં અનુસરો.",
            [
                "દર્દીને આરામથી બેસાડો અને ગળા તથા કમરના કપડાં ઢીલા કરો.",
                "દર્દીને શાંત રાખો અને બિલકુલ હલનચલન ન કરવા દો.",
                "જો શ્વાસ બંધ થઈ જાય તો તરત જ છાતી પર CPR શરૂ કરો (પ્રતિ મિનિટ 100-120 વાર).",
                "તરત જ નજીકની ઇમરજન્સી હોસ્પિટલમાં લઈ જાઓ."
            ]
        ),
        "bn_IN": (
            "অনুগ্রহ করে আতঙ্কিত হবেন না। অবিলম্বে 108 অ্যাম্বুলেন্সে কল করুন এবং এই পদক্ষেপগুলি অনুসরণ করুন।",
            [
                "রোগীকে শান্তভাবে বসিয়ে রাখুন এবং গলার ও কোমরের পোশাক ঢিলে করুন।",
                "রোগীকে কোনো শারীরিক পরিশ্রম বা হাঁটাচলা করতে দেবেন না।",
                "যদি শ্বাস বন্ধ হয়ে যায়, তবে অবিলম্বে বুকে চাপ দিয়ে CPR শুরু করুন (প্রতি মিনিটে 100-120 বার)।",
                "দেরি না করে নিকটস্থ হাসপাতালে নিয়ে যান।"
            ]
        ),
        "pa_IN": (
            "ਕਿਰਪਾ ਕਰਕੇ ਘਬਰਾਓ ਨਾ। ਤੁਰੰਤ 108 ਐਂਬੂਲੈਂਸ ਨੂੰ ਕਾਲ ਕਰੋ ਅਤੇ ਇਹ ਕਦਮ ਚੁੱਕੋ।",
            [
                "ਮਰੀਜ਼ ਨੂੰ ਅਰਾਮ ਨਾਲ ਬਿਠਾਓ ਅਤੇ ਤੰਗ ਕੱਪੜੇ ਢਿੱਲੇ ਕਰੋ।",
                "ਮਰੀਜ਼ ਨੂੰ ਸ਼ਾਂਤ ਰੱਖੋ ਅਤੇ ਕੋਈ ਵੀ ਹਰਕਤ ਨਾ ਕਰਨ ਦਿਓ।",
                "ਜੇਕਰ ਸਾਹ ਬੰਦ ਹੋ ਜਾਵੇ ਤਾਂ ਤੁਰੰਤ ਛਾਤੀ 'ਤੇ CPR ਸ਼ੁਰੂ ਕਰੋ (100-120 ਪ੍ਰਤੀ ਮਿੰਟ)।",
                "ਤੁਰੰਤ ਨਜ਼ਦੀਕੀ ਹਸਪਤਾਲ ਪਹੁੰਚਾਓ।"
            ]
        ),
        "or_IN": (
            "ଦୟାକରି ଭୟଭୀତ ହୁଅନ୍ତୁ ନାହିଁ। ତୁରନ୍ତ 108 ଆମ୍ବୁଲାନ୍ସକୁ କଲ୍ କରନ୍ତୁ ଏବଂ ଏହି ପଦକ୍ଷେପଗୁଡ଼ିକ ଅନୁସରଣ କରନ୍ତୁ।",
            [
                "ରୋଗୀଙ୍କୁ ଶାନ୍ତ ଭାବରେ ବସାନ୍ତୁ ଏବଂ ବେକ ଓ ଅଣ୍ଟା ପାଖ ପୋଷାକ ଢିଲା କରନ୍ତୁ।",
                "ରୋଗୀଙ୍କୁ ଶାନ୍ତ ରଖନ୍ତୁ ଏବଂ କୌଣସି ପରିଶ୍ରମ କରିବାକୁ ଦିଅନ୍ତୁ ନାହିଁ।",
                "ଯଦି ନିଶ୍ୱାସ ବନ୍ଦ ଥାଏ ତେବେ ତୁରନ୍ତ ଛାତି ଉପରେ CPR ଆରମ୍ଭ କରନ୍ତୁ।",
                "ଶୀଘ୍ର ନିକଟସ୍ଥ ଡାକ୍ତରଖାନାକୁ ନିଅନ୍ତୁ।"
            ]
        ),
        "mni_IN": (
            "ꯆꯥꯅꯕꯤꯗꯨꯅ ꯑꯀꯤꯕ ꯄꯣꯛꯂꯣꯏꯗꯕꯅꯤ। ꯈꯨꯗꯛꯇ 108 ꯑꯦꯝꯕꯨꯂꯦꯟꯁ ꯀꯣꯜ ꯇꯧꯕꯤꯌꯨ।",
            [
                "ꯑꯅꯥꯕ ꯃꯤꯑꯣꯏ ꯑꯗꯨꯕꯨ ꯅꯤꯡꯊꯤꯅ ꯐꯝꯍꯜꯂꯨ ꯑꯃꯁꯨꯡ ꯐꯤ ꯇꯦꯛꯍꯜꯂꯨ।",
                "ꯃꯤꯑꯣꯏ ꯑꯗꯨꯕꯨ ꯆꯠꯍꯟꯒꯅꯨ, ꯅꯤꯡꯊꯤꯅ ꯂꯩꯍꯟꯕꯤꯌꯨ।",
                "ꯁ꯭ꯕꯥꯁ ꯍꯣꯟꯕ ꯌꯥꯗ꯭ꯔꯕꯗꯤ CPR ꯊꯨꯅ ꯍꯧꯕꯤꯌꯨ।",
                "ꯈꯨꯗꯛꯇ ꯍꯣꯁꯄꯤꯇꯥꯜꯗ ꯄꯨꯈꯠꯄꯤꯌꯨ।"
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
                "ಆಂಟಿ-ವೆನಮ್ (ASV) ಲಭ್ಯವಿರುವ ಸಮೀಪದ ಸರ್ಕಾರಿ ಆಸ್ಪತ್ರೆಗೆ ತಕ್ಷಣ ಕರೆದೊಯ್ಯಿರಿ."
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
            "ದಯವಿಟ್ಟು ಗಾಬರಿಯಾಗಬೇಡಿ. ತಕ್ಷಣ 108 ಆಂಬ್ಯುಲೆನ್ಸ್‌ಗೆ ಕರೆ ಮಾಡಿ.",
            [
                "ವ್ಯಕ್ತಿಯ ಉಸಿರಾಟ ಮತ್ತು ಪ್ರಜ್ಞೆಯನ್ನು ನಿರಂತರವಾಗಿ ಪರಿಶೀಲಿಸಿ.",
                "ರಕ್ತಸ್ರಾವವಿದ್ದರೆ ಸ್ವಚ್ಛವಾದ ಬಟ್ಟೆಯಿಂದ ಗಾಯದ ಮೇಲೆ ನೇರ ಒತ್ತಡ ಹಾಕಿ.",
                "ಆಂಬ್ಯುಲೆನ್ಸ್ ಬರುವವರೆಗೆ ರೋಗಿಯೊಂದಿಗೆ ಇರಿ."
            ]
        ),
        "hi_IN": (
            "कृपया घबराएं नहीं। तुरंत 108 पर कॉल करें।",
            [
                "व्यक्ति की सांस और होश की जांच करें।",
                "घाव पर साफ कपड़े से सीधा दबाव बनाकर रखें।",
                "तुरंत एम्बुलेंस की प्रतीक्षा करें।"
            ]
        ),
        "en_IN": (
            "Please stay calm. Immediately call 108 for medical assistance.",
            [
                "Check airway, responsiveness, and normal breathing continuously.",
                "Apply clean, firm pressure to any actively bleeding wounds.",
                "Keep the patient reassured until the ambulance arrives."
            ]
        ),
    }
}

# ---------------------------------------------------------------------------
# Detection & Helpers
# ---------------------------------------------------------------------------
def get_env(name):
    return os.getenv(name, "").strip()

def to_hyphen(code):
    return code.replace("_", "-")

def detect_lang(text: str, fallback: str) -> str:
    """Detects user language from native script, romanized words, or falls back to dropdown."""
    if not text:
        return fallback

    # 1. Native script detection
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

    # 2. Romanized transliterated detection
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    scores = {}
    for code, kw_set in ROMAN_LANG_KEYWORDS.items():
        matched = words & kw_set
        if matched:
            scores[code] = len(matched)

    if scores:
        best_code = max(scores, key=scores.get)
        # If fallback language has matched keywords or ties, give preference to user's selected dropdown
        if fallback in scores and scores[fallback] >= scores[best_code] - 1:
            return fallback
        return best_code

    return fallback

def get_fallback_response(transcript: str, lang_code: str):
    cond = identify_condition(transcript)
    cond_data = CONDITION_FALLBACKS.get(cond, CONDITION_FALLBACKS["general_medical"])
    service = EMERGENCY_DIRECTORY[cond_data["service_idx"]]
    urgency = cond_data["urgency"]

    if lang_code in cond_data:
        reply, steps = cond_data[lang_code]
    elif "en_IN" in cond_data:
        reply, steps = cond_data["en_IN"]
    else:
        reply, steps = CONDITION_FALLBACKS["general_medical"]["en_IN"]

    return service, reply, steps, urgency

# ---------------------------------------------------------------------------
# Gnani Prisma (STT), Groq Triage, & Timbre (TTS)
# ---------------------------------------------------------------------------
def transcribe(audio_path, lang_code):
    api_key = get_env("GNANI_API_KEY")
    if not api_key:
        return "", "GNANI_API_KEY is missing."

    stt_lang = "en-IN" if lang_code == "mni_IN" else to_hyphen(lang_code)
    try:
        with open(audio_path, "rb") as f:
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

    target_lang_name = LANG_NAMES.get(detected_lang, "Kannada")

    system_prompt = f"""You are Sahaya, an empathetic AI emergency first-aid dispatcher in India.
Your mission is to save lives by diagnosing emergency severity, identifying the caller's language, and providing immediate first-aid instructions.

CRITICAL INSTRUCTIONS:
1. DETECT INTENDED LANGUAGE:
   - The user's input might be in native script OR Romanized/transliterated Indian speech (e.g., 'nana tande ke ideno shuruagite...' is KANNADA).
   - If Kannada is detected, set "lang_code": "kn_IN".
   - Supported codes: en_IN, kn_IN, hi_IN, ta_IN, te_IN, ml_IN, mr_IN, gu_IN, bn_IN, pa_IN, or_IN, mni_IN.

2. NATIVE SCRIPT RESPONSE:
   - Write 'reply' and all 'steps' ENTIRELY in the NATIVE SCRIPT of that language (Kannada script for kn_IN, Devanagari for hi_IN, etc.).
   - NEVER answer in Hindi or English if the user spoke Kannada!

3. EMERGENCY FIRST-AID ACCURACY:
   - For chest pain/cardiac: Instruct to sit upright, loosen tight clothes, call 108, and begin CPR if unresponsive.
   - For snakebite: Keep limb still below heart level, do not tourniquet, rush for Anti-Snake Venom.

4. OUTPUT ONLY RAW JSON (no markdown):
{{
  "lang_code": "{detected_lang}",
  "category": "medical",
  "urgency": "critical",
  "reply": "Calming 1-2 sentence response with urgent dispatch instruction in native script",
  "steps": [
    "Step 1 in native script",
    "Step 2 in native script",
    "Step 3 in native script"
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
                        {"role": "user", "content": f"User emergency message: {transcript}"},
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
# HTML Builders
# ---------------------------------------------------------------------------
def header_html(t):
    return (
        '<div class="brand-top">'
        '  <div class="brand-title-box">'
        f'   <h1 class="animate-drop">{html.escape(t["title"])}</h1>'
        '    <div class="audio-wave"><span></span><span></span><span></span><span></span></div>'
        '  </div>'
        f' <p class="brand-sub animate-fade-in">{html.escape(t["subtitle"])}</p>'
        '</div>'
    )

def result_card(service, steps, urgency, rt):
    body = "<ol class='step-list'>" + "".join(f"<li>{html.escape(s)}</li>" for s in steps) + "</ol>"
    urgency_tag = f'<span class="badge badge-{urgency}">{urgency.upper()}</span>' if urgency else ""
    return (
        '<div class="emergency-banner card-box animate-scale-in">'
        f' <div class="emergency-icon-wrap">{service["icon"]}</div>'
        '  <div class="emergency-info">'
        f'   <div class="emergency-title-row"><h2>{html.escape(service["title"])}</h2>{urgency_tag}</div>'
        f'   <p class="service-desc">{html.escape(service["desc"])}</p>'
        '  </div>'
        '</div>'
        '<div class="content-card card-box animate-fade-in">'
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
        cards.append(
            '<div class="dir-item card-box">'
            f' <div class="dir-icon">{srv["icon"]}</div>'
            '  <div class="dir-meta">'
            f'   <h4>{html.escape(srv["title"])}</h4>'
            f'   <span class="dir-number">{srv["number"]}</span>'
            '  </div>'
            f' <a class="quick-call-btn" href="tel:{srv["number"]}">Call</a>'
            '</div>'
        )
    return (
        f'<div class="dir-container"><h3 class="dir-title">📋 {html.escape(t["directory_header"])}</h3>'
        f'<div class="dir-list">{"".join(cards)}</div></div>'
    )

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

    # Spoken narration in the exact language
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
        gr.update(label=t["mic_header"]),
        gr.update(label=t["type_header"], placeholder=t["type_placeholder"]),
        gr.update(value=t["send"]),
        directory_html(t),
    )

# ---------------------------------------------------------------------------
# CSS & JS Styling
# ---------------------------------------------------------------------------
CSS = """
:root {
  --body-background-fill: #FBF9F5 !important;
  --background-fill-primary: #FFFFFF !important;
  --background-fill-secondary: #F6F3EC !important;
  --block-background-fill: #FFFFFF !important;
  --block-border-color: #E6E1D6 !important;
  --input-background-fill: #FFFFFF !important;
  --input-border-color: #D3CDC0 !important;
  --body-text-color: #0F172A !important;
  --block-label-text-color: #0F172A !important;
}
body, .gradio-container, .dark, body.dark {
  background-color: #FBF9F5 !important;
  color: #0F172A !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
  max-width: 920px !important;
  margin: 0 auto !important;
  padding: 1.5rem 1rem !important;
}
footer { display: none !important; }

@keyframes drop-in {
  from { opacity: 0; transform: translateY(-16px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes fade-in {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes scale-in {
  from { opacity: 0; transform: scale(0.96); }
  to { opacity: 1; transform: scale(1); }
}

.brand-top {
  text-align: center;
  padding: 2.2rem 1.6rem 1.8rem;
  border-radius: 22px !important;
  margin-bottom: 1.2rem;
  border: 1.5px solid #BAE6FD !important;
  box-shadow: 0 10px 28px rgba(14, 165, 233, 0.15) !important;
  background: linear-gradient(135deg, rgba(240, 249, 255, 0.95) 0%, rgba(186, 230, 253, 0.85) 100%) !important;
}
.brand-title-box { display: flex; align-items: center; justify-content: center; gap: 12px; }
.brand-top h1 { font-size: 2.6rem; font-weight: 800; margin: 0; color: #0369A1 !important; }
.brand-sub { font-size: 1.15rem; font-weight: 700; color: #0F172A !important; margin-top: 6px; }

.sos-row { text-align: center; margin: 0.5rem 0 1.2rem; }
.sos-panic-btn {
  background: linear-gradient(135deg, #DC2626 0%, #B91C1C 100%) !important;
  color: #FFFFFF !important;
  font-size: 1.05rem;
  font-weight: 700;
  padding: 14px 34px;
  border-radius: 50px;
  border: none !important;
  box-shadow: 0 8px 22px rgba(220, 38, 38, 0.32);
  cursor: pointer;
}

.gradio-container .block, .gradio-container .gr-panel, .gradio-container .gr-box, .card-box {
  background: #FFFFFF !important;
  border: 1px solid #E6E1D6 !important;
  border-radius: 16px !important;
  box-shadow: 0 4px 18px rgba(0, 0, 0, 0.04) !important;
}

.emergency-banner {
  display: flex;
  align-items: center;
  gap: 16px;
  background: #FEF2F2 !important;
  border: 1.5px solid #FECACA !important;
  border-radius: 16px;
  padding: 1.2rem 1.6rem;
  margin-top: 1rem;
}
.emergency-icon-wrap { font-size: 2.2rem; }
.emergency-info { flex: 1; }
.emergency-title-row { display: flex; align-items: center; gap: 12px; }
.emergency-title-row h2 { font-size: 1.35rem; margin: 0; color: #DC2626 !important; font-weight: 800; }
.badge { padding: 3px 12px; border-radius: 20px; font-size: 0.72rem; font-weight: 800; }
.badge-critical { background: #DC2626; color: #FFF !important; }
.badge-urgent { background: #D97706; color: #FFF !important; }

.content-card { padding: 1.6rem 2rem !important; margin-top: 1rem !important; }
.card-heading { font-size: 1.25rem !important; font-weight: 800 !important; color: #0F172A !important; margin-bottom: 1rem !important; }
.step-list { margin: 0.5rem 0 1.2rem !important; padding-left: 1.6rem !important; }
.step-list li { color: #0F172A !important; font-size: 1.05rem !important; font-weight: 600 !important; line-height: 1.75 !important; margin-bottom: 0.5rem !important; }

.call-action-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  background: #16A34A;
  color: #FFFFFF !important;
  text-decoration: none !important;
  font-size: 1.15rem;
  font-weight: 700;
  padding: 14px 20px;
  border-radius: 12px;
  margin-top: 1.3rem;
}

.dir-container { margin-top: 1.8rem; }
.dir-title { font-size: 1.25rem; font-weight: 800; margin-bottom: 1rem; color: #0F172A; }
.dir-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }
.dir-item { padding: 1rem 1.3rem; display: flex; align-items: center; gap: 14px; }
.dir-icon { font-size: 1.8rem; }
.dir-meta { flex: 1; }
.dir-meta h4 { margin: 0; font-size: 1rem; font-weight: 700; }
.dir-number { color: #0284C7 !important; font-weight: 800; }
.quick-call-btn {
  background: rgba(2, 132, 199, 0.08);
  color: #0284C7 !important;
  padding: 8px 16px;
  border-radius: 10px;
  text-decoration: none !important;
  font-weight: 700;
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
    gain.gain.value = 0.9;
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
# Gradio UI Layout
# ---------------------------------------------------------------------------
t0 = TRANSLATIONS["en_IN"]

with gr.Blocks(title="Sahaya Emergency", theme=gr.themes.Base(), css=CSS, js=INIT_JS) as demo:
    header = gr.HTML(header_html(t0))
    gr.HTML(
        '<div class="sos-row">'
        '<button class="sos-panic-btn" onclick="startSiren(); setTimeout(()=>window.open(\'tel:1091\'), 500);">'
        '🚨 EMERGENCY SOS &bull; LOUD ALARM + DIAL 1091'
        '</button>'
        '</div>'
    )
    lang_dd = gr.Dropdown(
        choices=[(label, code) for code, label in LANGUAGES.items()],
        value="kn_IN",
        label="🌐 Language Selection / ಭಾಷೆ / भाषा / ꯃꯤꯇꯩꯂꯣꯟ",
    )

    with gr.Row(equal_height=True):
        with gr.Column():
            mic = gr.Audio(sources=["microphone"], type="filepath", format="wav", label=t0["mic_header"])
        with gr.Column():
            textbox = gr.Textbox(label=t0["type_header"], placeholder=t0["type_placeholder"], lines=3)
            send_btn = gr.Button(t0["send"], variant="primary")

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
    lang_dd.change(apply_language, inputs=lang_dd, outputs=[header, mic, textbox, send_btn, dir_html])

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    demo.launch(server_name="0.0.0.0", server_port=port)
