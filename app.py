from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import os
import re

app = FastAPI(title="SaHaya Multilingual Emergency Assistant")

class EmergencyRequest(BaseModel):
    text: str
    lang_override: str = "auto"

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    elif os.path.exists("templates/index.html"):
        return FileResponse("templates/index.html")
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)


# ================= LANGUAGE & CODE-MIXING DETECTOR =================
LANG_NAMES = {
    "en-IN": "English",
    "hi-IN": "Hindi (हिन्दी)",
    "bn-IN": "Bengali (বাংলা)",
    "gu-IN": "Gujarati (ગુજરાતી)",
    "kn-IN": "Kannada (ಕನ್ನಡ)",
    "ml-IN": "Malayalam (മലയാളം)",
    "mr-IN": "Marathi (मराठी)",
    "pa-IN": "Punjabi (ਪੰਜਾਬੀ)",
    "ta-IN": "Tamil (தமிழ்)",
    "te-IN": "Telugu (తెలుగు)"
}

LANGUAGE_MARKERS = {
    "kn-IN": [
        "haavu", "havu", "kachide", "kacchide", "sahaya", "maadi", "madi", "nodi", "agide", "aagide",
        "benki", "hattide", "apaghata", "ede", "novu", "noovu", "rakta", "barthide", "kattada",
        "kusidide", "beku", "bega", "banni", "kapadi", "ambulance", "alli", "illi", "nanage", "avarige"
    ],
    "mr-IN": [
        "madat", "kara", "vachva", "saap", "chavla", "dukhata", "dukhtay", "apghat", "zhala", "zala",
        "mala", "tyala", "bhajle", "aag", "lagli", "imarat", "kosalli", "rakta", "yetay", "lavkar",
        "ya", "bolva", "chati", "chatit", "aahe", "ahe", "nahi", "krupa", "आपण", "आहे", "झाला", "मदत",
        "वाचवा", "साप", "चावला", "दुखत", "अपघात", "मला", "भाजले", "इमारत", "कोसळली", "लागली"
    ],
    "hi-IN": [
        "bachao", "madad", "karo", "saanp", "saap", "kaat", "kata", "liya", "dard", "ho", "raha",
        "rahi", "hai", "hain", "mujhe", "meri", "mere", "seene", "chhati", "mein", "aag", "lag", "gayi",
        "jal", "gaya", "durghatna", "accident", "hua", "khoon", "bah", "nikal", "imarat", "gir", "bhukamp",
        "jaldi", "bulao", "bhai", "please"
    ],
    "bn-IN": [
        "bachao", "sahajyo", "korun", "koro", "shaap", "sap", "kamreche", "kamdache", "buk", "buke",
        "byatha", "betha", "hocche", "agun", "legeche", "jole", "geche", "durghotona", "ghoteche",
        "rokto", "porche", "bhenge", "poddche", "amar", "ekhane", "taratari", "ashun"
    ],
    "gu-IN": [
        "bachavo", "madad", "karo", "saap", "kardyo", "kerdiyo", "chhati", "ma", "dukhavo", "thay",
        "chhe", "aag", "lagi", "dazhi", "gaya", "akasmaat", "thayo", "lohi", "nikle", "makan", "padi",
        "gayu", "mane", "jali", "lavkar", "aavo"
    ],
    "ml-IN": [
        "rakshikku", "sahayikku", "sahayam", "venam", "paambu", "pambu", "kadichu", "nenju", "nenjil",
        "vedana", "edukkunnu", "thee", "pidichu", "polli", "apagadam", "undayi", "chora", "varunnu",
        "kettidam", "idinjhu", "veenu", "enikku", "vegam", "varu"
    ],
    "pa-IN": [
        "bachao", "madad", "karo", "sapp", "sap", "ladeya", "katteya", "chhati", "vich", "dard", "ho",
        "riha", "hai", "agg", "lag", "gayi", "sadh", "gaya", "accident", "hoya", "khoon", "nikal",
        "imarat", "digg", "menu", "sadi", "cheti", "bulao"
    ],
    "ta-IN": [
        "udavi", "ulavi", "pannunga", "seyunga", "pambu", "paambu", "kadichiduchu", "kadithathu",
        "nenju", "vali", "valikuthu", "thee", "pidichiruchu", "kaayam", "vipathu", "nadanthuchu",
        "ratham", "varuthu", "kattidam", "idinthu", "vilunthathu", "enakku", "seekiram", "vaanga"
    ],
    "te-IN": [
        "sahayam", "cheyandi", "kapadandi", "paamu", "pamu", "karichindi", "gunde", "noppi", "vastundi",
        "manta", "mantalu", "antukundi", "kalindi", "pramadam", "jarigindi", "raktam", "vastondi",
        "bhavanam", "kulipoyindi", "naaku", "twaraga", "randi"
    ]
}

def detect_language(text: str, lang_override: str = "auto") -> str:
    if lang_override and lang_override != "auto":
        return lang_override

    if re.search(r"[\u0980-\u09FF]", text): return "bn-IN"
    if re.search(r"[\u0A80-\u0AFF]", text): return "gu-IN"
    if re.search(r"[\u0C80-\u0CFF]", text): return "kn-IN"
    if re.search(r"[\u0D00-\u0D7F]", text): return "ml-IN"
    if re.search(r"[\u0A00-\u0A7F]", text): return "pa-IN"
    if re.search(r"[\u0B80-\u0BFF]", text): return "ta-IN"
    if re.search(r"[\u0C00-\u0C7F]", text): return "te-IN"
    if re.search(r"[\u0900-\u097F]", text):
        if re.search(r"[\u0933]", text) or any(w in text for w in ["मदत", "वाचवा", "साप", "चावला", "दुखत", "अपघात", "मला", "भाजले", "इमारत", "कोसळली", "आहे", "झाला", "पाणी"]):
            return "mr-IN"
        return "hi-IN"

    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    scores = {lang: 0 for lang in LANGUAGE_MARKERS}
    for token in tokens:
        for lang, markers in LANGUAGE_MARKERS.items():
            if token in markers:
                scores[lang] += 2
            else:
                for m in markers:
                    if len(m) >= 4 and (token.startswith(m) or m in token):
                        scores[lang] += 1

    best_lang = max(scores, key=scores.get)
    if scores[best_lang] > 0:
        return best_lang

    return "en-IN"


SCENARIO_PATTERNS = {
    "snake": [
        "snake", "snakebite", "cobra", "viper", "krait", "venom", "poisonous", "bitten", "fang",
        "ಹಾವು", "ಕಚ್ಚಿದೆ", "ಕಚ್ಚಿ", "haavu", "havu", "kachide", "kacchide",
        "साप", "चावला", "दंश", "saap", "chavla",
        "सांप", "काटा", "काट लिया", "saanp", "kaat",
        "সাপ", "কামড়েছে", "কামড়", "shaap", "kamreche",
        "સાપ", "કરડ્યો", "ડંખ", "kardyo",
        "പാമ്പ്", "കടിച്ചു", "paambu", "pambu", "kadichu",
        "ਸੱਪ", "ਲੜਿਆ", "ਕੱਟਿਆ", "sapp", "ladeya",
        "பாம்பு", "கடித்தது", "கடி", "kadichiduchu",
        "పాము", "కరిచింది", "కాటు", "paamu", "karichindi"
    ],
    "chest": [
        "chest", "heart", "cardiac", "attack", "stroke", "breathless", "breathing", "cant breathe",
        "cannot breathe", "left arm pain", "tightness", "palpitation", "unconscious", "fainted",
        "ಎದೆ", "ನೋವು", "ಹೃದಯ", "ಉಸಿರು", "ede", "novu", "hrudaya", "usiru",
        "छाती", "छातीत", "हृदय", "श्वास", "chati", "chatit", "dukhtay",
        "सीने", "दर्द", "दिल", "सांस", "seene", "chhati", "saans",
        "বুক", "বুকে", "ব্যথা", "হার্ট", "শ্বাস", "buke", "byatha",
        "છાતી", "દુખાવો", "હૃદય", "શ્વાસ", "dukhavo",
        "നെഞ്ച്", "വേദന", "ഹൃദയം", "ശ്വാസം", "nenju", "vedana",
        "ਛਾਤੀ", "ਦਰਦ", "ਦਿਲ", "ਸਾਹ",
        "நெஞ்சு", "வலி", "இதயம்", "மூச்சு", "valikuthu",
        "గుండె", "నొప్పి", "ఛాతీ", "శ్వాస", "gunde", "noppi"
    ],
    "accident": [
        "accident", "crash", "crashed", "collision", "collided", "hit by", "run over", "road",
        "bike", "car", "truck", "bus", "bleeding", "blood", "fracture", "broken bone", "injured",
        "injury", "wound", "head injury", "fell down",
        "ಅಪಘಾತ", "ರಕ್ತ", "ಗಾಯ", "ವಾಹನ", "ಡಿಕ್ಕಿ", "apaghata", "rakta", "gaya",
        "अपघात", "धडक", "रक्त", "जखम", "गाडी", "apghat", "dhadak", "jakham",
        "दुर्घटना", "एक्सीडेंट", "खून", "चोट", "गाड़ी", "टक्कर", "durghatna", "khoon", "chot", "takkar",
        "দুর্ঘটনা", "রক্ত", "আঘাত", "গাড়ি", "ধাক্কা", "durghotona", "rokto",
        "અકસ્માત", "લોહી", "ઈજા", "ટક્કર", "akasmaat", "lohi",
        "അപകടം", "രക്തം", "ചോര", "മുറിവ്", "വണ്ടി", "apagadam", "chora",
        "ਐਕਸੀਡੈਂਟ", "ਹਾਦਸਾ", "ਖੂਨ", "ਸੱਟ", "ਟੱਕਰ", "hadsa", "satt",
        "விபத்து", "இரத்தம்", "காயம்", "வண்டி", "vipathu", "ratham",
        "ప్రమాదం", "రక్తం", "గాయం", "బండి", "pramadam", "raktam"
    ],
    "fire": [
        "fire", "burn", "burned", "burnt", "burning", "flame", "flames", "smoke", "cylinder blast",
        "gas leak", "short circuit", "scald", "hot water", "hot oil", "electric shock", "explosion",
        "ಬೆಂಕಿ", "ಸುಟ್ಟ", "ಹೊಗೆ", "ಸಿಲಿಂಡರ್", "benki", "sutta", "hoge",
        "आग", "भाजले", "भाजलं", "धूर", "aag", "bhajle",
        "जल", "जलना", "जला", "धुआं", "आग लग", "jalna", "jala",
        "আগুন", "পুড়ে", "পোড়া", "ধোঁয়া", "agun", "jole", "pure",
        "આગ", "દાઝી", "ધુમાડો", "dazhi",
        "തീ", "പൊള്ളി", "പുക", "thee", "polli",
        "ਅੱਗ", "ਸੜ", "ਧੂੰਆਂ", "agg", "sadh",
        "தீ", "தீக்காயம்", "புகை", "வெடிப்பு", "theekayam",
        "మంట", "మంటలు", "కాలింది", "పొగ", "manta", "mantalu", "kalindi"
    ],
    "disaster": [
        "disaster", "collapse", "collapsed", "building fell", "roof fell", "wall fell", "earthquake",
        "tremor", "flood", "flooding", "water rising", "landslide", "cyclone", "storm", "trapped",
        "stuck under", "debris", "rubble",
        "ಕಟ್ಟಡ", "ಕುಸಿತ", "ಕುಸಿದಿದೆ", "ಭೂಕಂಪ", "ಪ್ರವಾಹ", "ಮಳೆ", "ಸಿಲುಕಿ", "kattada", "kusita", "bhukampa",
        "इमारत", "कोसळली", "भूकंप", "पूर", "आपत्ती", "ढिगारा", "अडकलो", "imarat", "kosalli", "bhukamp",
        "इमारत गिर", "भूकंप", "बाढ़", "मलबा", "दब गए", "आपदा", "तूफान", "baadh", "malba",
        "ভবন", "ধসে", "ভূমিকম্প", "বন্যা", "আটকে", "bhumi", "bonna",
        "મકાન", "પડી", "ભૂકંપ", "પૂર", "ફસાયા", "makan",
        "കെട്ടിടം", "ഇടിഞ്ഞു", "ഭൂകമ്പം", "വെള്ളപ്പൊക്കം", "കുടുങ്ങി", "kettidam", "idinjhu",
        "ਇਮਾਰਤ", "ਡਿੱਗ", "ਭੂਚਾਲ", "ਹੜ੍ਹ", "ਫਸ", "bhuchal",
        "கட்டிடம்", "இடிந்து", "நிலநடுக்கம்", "வெள்ளம்", "சிக்கி", "kattidam", "vellam",
        "భవనం", "కూలిపోయింది", "భూకంపం", "వరద", "చిక్కుకున్నాం", "bhavanam", "kulipoyindi", "varada"
    ]
}

def detect_scenario(text: str) -> str:
    lower = text.lower()
    scores = {k: 0 for k in SCENARIO_PATTERNS}
    for scenario, keywords in SCENARIO_PATTERNS.items():
        for kw in keywords:
            if kw in lower:
                scores[scenario] += 2
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


GUIDANCE_DB = {
    "en-IN": {
        "snake": "🐍 First Aid (Snakebite):\n1. Stay calm and immobilize the bitten limb completely.\n2. Keep the bitten area below heart level.\n3. Do NOT tie a tight tourniquet, cut the wound, or suck the venom.\n4. Call 108 Ambulance immediately for anti-venom treatment.",
        "chest": "❤️ First Aid (Chest Pain / Cardiac Emergency):\n1. Make the person sit down immediately and rest calmly.\n2. Loosen any tight clothing and ensure fresh air.\n3. If unresponsive and not breathing, begin hands-only CPR.\n4. Call 108 Ambulance immediately.",
        "accident": "🚗 First Aid (Road Accident / Trauma):\n1. Do NOT move the injured person unnecessarily to protect their neck and spine.\n2. Apply firm, direct pressure with a clean cloth to stop any bleeding.\n3. Keep the airway clear and call 108 (Ambulance) & 112 (Police) immediately.",
        "fire": "🔥 First Aid (Fire & Burns):\n1. Move away from the fire and smoke immediately.\n2. Run cool (not ice-cold) water over the burn for 15–20 minutes.\n3. Do NOT peel off stuck clothing or pop blisters.\n4. Call 101 (Fire & Rescue) and 108 immediately.",
        "disaster": "🏢 First Aid (Building Collapse / Natural Disaster):\n1. Drop, Cover, and Hold On under sturdy furniture or near a load-bearing pillar.\n2. Cover your nose and mouth with cloth to avoid inhaling dust.\n3. If trapped under debris, tap on a pipe or wall instead of shouting. Call 112 immediately.",
        "general": "🚨 Emergency First-Aid Guidance:\n1. Stay calm and move to a safe, open location.\n2. Check responsiveness and breathing; apply firm pressure if there is bleeding.\n3. Tap 108 (Ambulance) or 112 (National SOS) below immediately."
    },
    "hi-IN": {
        "snake": "🐍 प्राथमिक उपचार (सांप काटने पर):\n1. घबराएं नहीं, शांत रहें और काटे गए अंग को बिल्कुल न हिलाएं।\n2. काटे गए हिस्से को दिल के स्तर से नीचे रखें।\n3. घाव पर कसकर पट्टी न बांधें और न ही चीरा लगाएं।\n4. तुरंत 108 एम्बुलेंस को कॉल करें।",
        "chest": "❤️ प्राथमिक उपचार (सीने में दर्द / हार्ट अटैक):\n1. व्यक्ति को तुरंत आरामदायक स्थिति में बैठाएं।\n2. तंग कपड़ों को ढीला करें और खुली हवा आने दें।\n3. तुरंत 108 एम्बुलेंस को कॉल करें।",
        "accident": "🚗 प्राथमिक उपचार (सड़क दुर्घटना / चोट):\n1. घायल व्यक्ति को बिना वजह न हिलाएं (रीढ़ या गर्दन में चोट हो सकती है)।\n2. खून बहने पर साफ कपड़े से घाव को जोर से दबाकर रखें।\n3. तुरंत 108 (एम्बुलेंस) और 112 (पुलिस) को कॉल करें।",
        "fire": "🔥 प्राथमिक उपचार (आग और जलना):\n1. तुरंत आग से दूर हटें। जले हुए हिस्से पर 15-20 मिनट तक सादा ठंडा पानी डालें।\n2. घाव से चिपके कपड़े न खींचें और छालों को न फोड़ें।\n3. तुरंत 101 (फायर ब्रिगेड) और 108 पर कॉल करें।",
        "disaster": "🏢 प्राथमिक उपचार (इमारत गिरना / प्राकृतिक आपदा):\n1. किसी मजबूत मेज या पिलर के नीचे छिप जाएं और सिर व गर्दन को बचाएं।\n2. धूल से बचने के लिए कपड़े से नाक और मुंह ढकें।\n3. मलबे में फंसने पर चिल्लाने के बजाय पाइप या दीवार थपथपाएं। तुरंत 112 पर कॉल करें।",
        "general": "🚨 आपातकालीन प्राथमिक उपचार:\n1. कृपया शांत रहें और सुरक्षित स्थान पर जाएं।\n2. यदि खून बह रहा है, तो साफ कपड़े से दबाकर रखें।\n3. तुरंत नीचे दिए गए 108 या 112 नंबर पर कॉल करें।"
    },
    "bn-IN": {
        "snake": "🐍 প্রাথমিক চিকিৎসা (সাপে কামড়ালে):\n১. আতঙ্কিত হবেন না, শান্ত থাকুন এবং কামড়ানো স্থানটি নাড়াচাড়া করবেন না।\n২. আক্রান্ত অংশটি হৃদপিণ্ডের নিচে রাখুন এবং শক্ত করে বাঁধবেন না।\n৩. অবিলম্বে 108 অ্যাম্বুলেন্সে কল করুন।",
        "chest": "❤️ প্রাথমিক চিকিৎসা (বুকে ব্যথা / হার্ট অ্যাটাক):\n১. রোগীকে অবিলম্বে বসিয়ে বিশ্রাম দিন।\n২. টাইট পোশাক ঢিলে করে দিন এবং বাতাস চলাচলের ব্যবস্থা করুন।\n৩. অবিলম্বে 108 অ্যাম্বুলেন্সে কল করুন।",
        "accident": "🚗 প্রাথমিক চিকিৎসা (সড়ক দুর্ঘটনা):\n১. আহত ব্যক্তিকে অহেতুক নাড়াচাড়া করবেন না।\n২. রক্তপাত হলে পরিষ্কার কাপড় দিয়ে ক্ষতস্থান চেপে ধরুন।\n৩. দ্রুত 108 (অ্যাম্বুলেন্স) এবং 112 (পুলিশ)-এ কল করুন।",
        "fire": "🔥 প্রাথমিক চিকিৎসা (আগুন ও পোড়া):\n১. আগুন থেকে দূরে সরুন এবং পোড়া স্থানে ১৫-২০ মিনিট ঠান্ডা জল ঢালুন।\n২. ক্ষতস্থানে আটকে থাকা কাপড় টানবেন না বা ফোসকা ফাটাবেন না।\n৩. দ্রুত 101 (ফায়ার ব্রিগেড) এবং 108-এ কল করুন।",
        "disaster": "🏢 প্রাথমিক চিকিৎসা (ভবন ধস / প্রাকৃতিক দুর্যোগ):\n১. মজবুত টেবিল বা পিলারের নিচে আশ্রয় নিন এবং মাথা বাঁচান।\n২. ধুলোবালি থেকে বাঁচতে নাক ও মুখ কাপড় দিয়ে ঢেকে রাখুন।\n৩. আটকে পড়লে দেওয়ালে বা পাইপে শব্দ করুন এবং 112-এ কল করুন।",
        "general": "🚨 জরুরি প্রাথমিক চিকিৎসা:\n১. শান্ত থাকুন এবং নিরাপদ স্থানে সরে যান।\n২. রক্তপাত হলে পরিষ্কার কাপড় দিয়ে চেপে ধরুন এবং দ্রুত 108 বা 112-এ কল করুন।"
    },
    "gu-IN": {
        "snake": "🐍 પ્રાથમિક સારવાર (સાપ કરડવા પર):\n૧. ગભરાશો નહીં, શાંત રહો અને કરડેલા ભાગને હલાવશો નહીં.\n૨. કરડેલા ભાગને હૃદયના સ્તરથી નીચે રાખો અને કસીને પટ્ટી ન બાંધો.\n૩. તાત્કાલિક 108 એમ્બ્યુલન્સને કોલ કરો.",
        "chest": "❤️ પ્રાથમિક સારવાર (છાતીમાં દુખાવો):\n૧. દર્દીને તરત જ આરામથી બેસાડો.\n૨. ચુસ્ત કપડાં ઢીલા કરો અને તાત્કાલિક 108 એમ્બ્યુલન્સને કોલ કરો.",
        "accident": "🚗 પ્રાથમિક સારવાર (માર્ગ અકસ્માત):\n૧. ઈજાગ્રસ્ત વ્યક્તિને બિનજરૂરી રીતે હલાવશો નહીં.\n૨. લોહી નીકળતું હોય તો સ્વચ્છ કપડાથી દબાવી રાખો.\n૩. તાત્કાલિક 108 અને 112 પર કોલ કરો.",
        "fire": "🔥 પ્રાથમિક સારવાર (આગ અને દાઝવું):\n૧. આગથી દૂર ખસો અને દાઝેલા ભાગ પર 15-20 મિનિટ ઠંડુ પાણી રેડો.\n૨. ચોંટેલા કપડાં ન ખેંચો. તાત્કાલિક 101 (ફાયર) અને 108 પર કોલ કરો.",
        "disaster": "🏢 પ્રાથમિક સારવાર (મકાન પડવું / આપત્તિ):\n૧. મજબૂત ટેબલ નીચે આશ્રય લો અને માથાનું રક્ષણ કરો.\n૨. ધૂળથી બચવા નાક અને મોં કપડાથી ઢાંકો. તરત જ 112 પર કોલ કરો.",
        "general": "🚨 કટોકટી પ્રાથમિક સારવાર:\n૧. શાંત રહો અને સુરક્ષિત સ્થળે ખસી જાઓ.\n૨. તાત્કાલિક નીચે આપેલા 108 અથવા 112 નંબર પર કોલ કરો."
    },
    "kn-IN": {
        "snake": "🐍 ಪ್ರಥಮ ಚಿಕಿತ್ಸೆ (ಹಾವು ಕಚ್ಚಿದಾಗ):\n1. ಗಾಬರಿಯಾಗಬೇಡಿ, ಶಾಂತವಾಗಿರಿ ಮತ್ತು ಕಚ್ಚಿದ ಭಾಗವನ್ನು ಅಲುಗಾಡಿಸಬೇಡಿ.\n2. ಕಚ್ಚಿದ ಭಾಗವನ್ನು ಹೃದಯದ ಮಟ್ಟಕ್ಕಿಂತ ಕೆಳಗಿರಿಸಿ.\n3. ಗಾಯವನ್ನು ಕೊರೆಯಬೇಡಿ ಅಥವಾ ಬಿಗಿಯಾಗಿ ಕಟ್ಟಬೇಡಿ.\n4. ತಕ್ಷಣ 108 ಆಂಬ್ಯುಲೆನ್ಸ್‌ಗೆ ಕರೆ ಮಾಡಿ.",
        "chest": "❤️ ಪ್ರಥಮ ಚಿಕಿತ್ಸೆ (ಎದೆ ನೋವು / ಹೃದಯಾಘಾತ):\n1. ತಕ್ಷಣ ಕುಳಿತುಕೊಳ್ಳಿ ಮತ್ತು ವಿಶ್ರಾಂತಿ ಪಡೆಯಿರಿ.\n2. ಬಿಗಿಯಾದ ಬಟ್ಟೆಗಳನ್ನು ಸಡಿಲಗೊಳಿಸಿ, ಸರಾಗವಾಗಿ ಉಸಿರಾಡಿ.\n3. ತಕ್ಷಣ 108 ಆಂಬ್ಯುಲೆನ್ಸ್‌ಗೆ ಕರೆ ಮಾಡಿ.",
        "accident": "🚗 ಪ್ರಥಮ ಚಿಕಿತ್ಸೆ (ರಸ್ತೆ ಅಪಘಾತ / ಗಾಯ):\n1. ಗಾಯಾಳುವಿನ ಕುತ್ತಿಗೆ ಅಥವಾ ಬೆನ್ನಿಗೆ ಪೆಟ್ಟಾಗಿದ್ದರೆ ಅವರನ್ನು ಅಲುಗಾಡಿಸಬೇಡಿ.\n2. ರಕ್ತಸ್ರಾವವಿದ್ದರೆ ಸ್ವಚ್ಛವಾದ ಬಟ್ಟೆಯಿಂದ ಗಾಯದ ಮೇಲೆ ಒತ್ತಿ ಹಿಡಿಯಿರಿ.\n3. ತಕ್ಷಣ 108 (ಆಂಬ್ಯುಲೆನ್ಸ್) ಮತ್ತು 112 (ಪೊಲೀಸ್) ಗೆ ಕರೆ ಮಾಡಿ.",
        "fire": "🔥 ಪ್ರಥಮ ಚಿಕಿತ್ಸೆ (ಬೆಂಕಿ ಅವಘಡ / ಸುಟ್ಟ ಗಾಯ):\n1. ತಕ್ಷಣ ಬೆಂಕಿಯಿಂದ ದೂರ ಸರಿಯಿರಿ. ಸುಟ್ಟ ಜಾಗಕ್ಕೆ 15-20 ನಿಮಿಷ ತಣ್ಣನೆಯ ನೀರನ್ನು ಸುರಿಯಿರಿ.\n2. ಗಾಯಕ್ಕೆ ಅಂಟಿಕೊಂಡಿರುವ ಬಟ್ಟೆಯನ್ನು ಎಳೆಯಬೇಡಿ ಮತ್ತು ಗುಳ್ಳೆಗಳನ್ನು ಒಡೆಯಬೇಡಿ.\n3. ತಕ್ಷಣ 101 (ಅಗ್ನಿಶಾಮಕ ದಳ) ಮತ್ತು 108 ಗೆ ಕರೆ ಮಾಡಿ.",
        "disaster": "🏢 ಪ್ರಥಮ ಚಿಕಿತ್ಸೆ (ಕಟ್ಟಡ ಕುಸಿತ / ಪ್ರಕೃತಿ ವಿಕೋಪ):\n1. ಗಟ್ಟಿಮುಟ್ಟಾದ ಮೇಜು ಅಥವಾ ಪಿಲ್ಲರ್ ಕೆಳಗೆ ಆಶ್ರಯ ಪಡೆಯಿರಿ; ತಲೆಯನ್ನು ರಕ್ಷಿಸಿಕೊಳ್ಳಿ.\n2. ಧೂಳಿನಿಂದ ರಕ್ಷಿಸಲು ಬಟ್ಟೆಯಿಂದ ಮೂಗು ಮತ್ತು ಬಾಯಿಯನ್ನು ಮುಚ್ಚಿಕೊಳ್ಳಿ.\n3. ಅವಶೇಷಗಳಡಿ ಸಿಲುಕಿದ್ದರೆ ಪೈಪ್ ಅಥವಾ ಗೋಡೆ ತಟ್ಟಿ ಶಬ್ದ ಮಾಡಿ. ತಕ್ಷಣ 112 ಗೆ ಕರೆ ಮಾಡಿ.",
        "general": "🚨 ತುರ್ತು ಸಹಾಯ ಮಾರ್ಗದರ್ಶನ:\n1. ದಯವಿಟ್ಟು ಧೈರ್ಯವಾಗಿರಿ, ಸುರಕ್ಷಿತ ಸ್ಥಳಕ್ಕೆ ಸರಿಯಿರಿ.\n2. ರಕ್ತಸ್ರಾವವಿದ್ದರೆ ಸ್ವಚ್ಛವಾದ ಬಟ್ಟೆಯಿಂದ ಒತ್ತಿ ಹಿಡಿಯಿರಿ.\n3. ಕೆಳಗಿನ 108 ಅಥವಾ 112 ಬಟನ್ ಒತ್ತಿ ತಕ್ಷಣ ಕರೆ ಮಾಡಿ."
    },
    "ml-IN": {
        "snake": "🐍 പ്രഥമശുശ്രൂഷ (പാമ്പുകടി):\n1. പരിഭ്രാന്തരാകാതെ ശാന്തമായിരിക്കുക, കടിയേറ്റ ഭാഗം അനക്കരുത്.\n2. കടിയേറ്റ ഭാഗം ഹൃദയത്തിന് താഴെയായി വെക്കുക, മുറുക്കി കെട്ടരുത്.\n3. ഉടൻ തന്നെ 108 ആംബുലൻസിൽ വിളിക്കുക.",
        "chest": "❤️ പ്രഥമശുശ്രൂഷ (നെഞ്ചുവേദന):\n1. ഉടൻ തന്നെ ഇരുന്ന് വിശ്രമിക്കുക.\n2. ഇറുകിയ വസ്ത്രങ്ങൾ അയച്ചിടുക. ഉടൻ 108 ആംബുലൻസിൽ വിളിക്കുക.",
        "accident": "🚗 പ്രഥമശുശ്രൂഷ (റോഡ് അപകടം):\n1. പരിക്കേറ്റയാളെ അനാവശ്യമായി മാറ്റരുത്.\n2. രക്തസ്രാവമുണ്ടെങ്കിൽ വൃത്തിയുള്ള തുണി ഉപയോഗിച്ച് അമർത്തിപ്പിടിക്കുക.\n3. ഉടൻ 108, 112 എന്നീ നമ്പറുകളിൽ വിളിക്കുക.",
        "fire": "🔥 പ്രഥമശുശ്രൂഷ (തീപിടുത്തം / പൊള്ളൽ):\n1. പൊള്ളലേറ്റ ഭാഗത്ത് 15-20 മിനിറ്റ് തണുത്ത വെള്ളം ഒഴിക്കുക.\n2. കുമിളകൾ പൊട്ടിക്കരുത്. ഉടൻ 101 (ഫയർ ഫോഴ്സ്), 108 എന്നിവയിൽ വിളിക്കുക.",
        "disaster": "🏢 പ്രഥമശുശ്രൂഷ (കെട്ടിടം തകരൽ / ദുരന്തം):\n1. ബലമുള്ള മേശയ്ക്കടിയിൽ അഭയം തേടുക, തല സംരക്ഷിക്കുക.\n2. പൊടി ശ്വസിക്കാതിരിക്കാൻ മൂക്കും വായും തുണികൊണ്ട് മൂടുക. ഉടൻ 112-ൽ വിളിക്കുക.",
        "general": "🚨 അടിയന്തര പ്രഥമശുശ്രൂഷ:\n1. ശാന്തമായി സുരക്ഷിത സ്ഥാനത്തേക്ക് മാറുക.\n2. താഴെ കാണുന്ന 108 അല്ലെങ്കിൽ 112 എന്ന നമ്പറിൽ ഉടൻ വിളിക്കുക."
    },
    "mr-IN": {
        "snake": "🐍 प्रथमोपचार (साप चावल्यास):\n1. घाबरू नका, शांत राहा आणि हालचाल टाळा.\n2. चावलेला भाग हृदयाच्या पातळीपेक्षा खाली ठेवा.\n3. जखमेवर घट्ट पट्टी बांधू नका किंवा कापू नका.\n4. तात्काळ 108 रुग्णवाहिकेला कॉल करा.",
        "chest": "❤️ प्रथमोपचार (छातीत दुखणे):\n1. लगेच शांत बसा आणि विश्रांती घ्या.\n2. घट्ट कपडे सैल करा आणि शांत श्वास घ्या.\n3. तात्काळ 108 वर कॉल करून वैद्यकीय मदत घ्या.",
        "accident": "🚗 प्रथमोपचार (रस्ते अपघात):\n1. जखमी व्यक्तीला अनावश्यक हलवू नका.\n2. रक्तस्त्राव होत असल्यास स्वच्छ कपड्याने जखम दाबून धरा.\n3. तात्काळ 108 (रुग्णवाहिका) आणि 112 (पोलीस) वर कॉल करा.",
        "fire": "🔥 प्रथमोपचार (आग आणि भाजणे):\n1. आगीपासून त्वरित दूर जा. भाजलेल्या भागावर 15-20 मिनिटे थंड पाणी ओता.\n2. जखमेला चिकटलेले कपडे काढू नका किंवा फोड फोडू नका.\n3. तात्काळ 101 (अग्निशमन दल) आणि 108 वर कॉल करा.",
        "disaster": "🏢 प्रथमोपचार (इमारत कोसळणे / नैसर्गिक आपत्ती):\n1. मजबूत टेबलखाली आश्रय घ्या आणि डोके झाकून घ्या.\n2. धूळ आणि धुरापासून वाचण्यासाठी नाक आणि तोंड कपड्याने झाका.\n3. ढिगाऱ्याखाली अडकल्यास पाईप किंवा भिंतीवर वाजवून आवाज करा. 112 वर कॉल करा.",
        "general": "🚨 आपत्कालीन प्रथमोपचार मदत:\n1. घाबरू नका, शांत राहा आणि सुरक्षित ठिकाणी थांबा.\n2. जखम असल्यास स्वच्छ कपड्याने दाबून ठेवा आणि 108 किंवा 112 वर कॉल करा."
    },
    "pa-IN": {
        "snake": "🐍 ਮੁੱਢਲੀ ਸਹਾਇਤਾ (ਸੱਪ ਲੜਨ 'ਤੇ):\n1. ਘਬਰਾਓ ਨਾ, ਸ਼ਾਂਤ ਰਹੋ ਅਤੇ ਕੱਟੇ ਹੋਏ ਅੰਗ ਨੂੰ ਨਾ ਹਿਲਾਓ।\n2. ਕੱਟੇ ਹੋਏ ਹਿੱਸੇ ਨੂੰ ਦਿਲ ਦੇ ਪੱਧਰ ਤੋਂ ਹੇਠਾਂ ਰੱਖੋ ਅਤੇ ਕੱਸ ਕੇ ਪੱਟੀ ਨਾ ਬੰਨ੍ਹੋ।\n3. ਤੁਰੰਤ 108 ਐਂਬੂਲੈਂਸ ਨੂੰ ਕਾਲ ਕਰੋ।",
        "chest": "❤️ ਮੁੱਢਲੀ ਸਹਾਇਤਾ (ਛਾਤੀ ਵਿੱਚ ਦਰਦ):\n1. ਮਰੀਜ਼ ਨੂੰ ਤੁਰੰਤ ਆਰਾਮ ਨਾਲ ਬਿਠਾਓ।\n2. ਤੰਗ ਕੱਪੜੇ ਢਿੱਲੇ ਕਰੋ ਅਤੇ ਤੁਰੰਤ 108 ਐਂਬੂਲੈਂਸ ਨੂੰ ਕਾਲ ਕਰੋ।",
        "accident": "🚗 ਮੁੱਢਲੀ ਸਹਾਇਤਾ (ਸੜਕ ਹਾਦਸਾ):\n1. ਜ਼ਖਮੀ ਵਿਅਕਤੀ ਨੂੰ ਬਿਨਾਂ ਵਜ੍ਹਾ ਨਾ ਹਿਲਾਓ।\n2. ਖੂਨ ਵਗਣ 'ਤੇ ਸਾਫ਼ ਕੱਪੜੇ ਨਾਲ ਜ਼ਖਮ ਨੂੰ ਦਬਾ ਕੇ ਰੱਖੋ।\n3. ਤੁਰੰਤ 108 ਅਤੇ 112 'ਤੇ ਕਾਲ ਕਰੋ।",
        "fire": "🔥 ਮੁੱਢਲੀ ਸਹਾਇਤਾ (ਅੱਗ ਅਤੇ ਸੜਨਾ):\n1. ਅੱਗ ਤੋਂ ਦੂਰ ਹਟੋ ਅਤੇ ਸੜੇ ਹੋਏ ਹਿੱਸੇ 'ਤੇ 15-20 ਮਿੰਟ ਠੰਡਾ ਪਾਣੀ ਪਾਓ।\n2. ਚਿਪਕੇ ਹੋਏ ਕੱਪੜੇ ਨਾ ਖਿੱਚੋ। ਤੁਰੰਤ 101 ਅਤੇ 108 'ਤੇ ਕਾਲ ਕਰੋ।",
        "disaster": "🏢 ਮੁੱਢਲੀ ਸਹਾਇਤਾ (ਇਮਾਰਤ ਡਿੱਗਣਾ / ਆਫ਼ਤ):\n1. ਕਿਸੇ ਮਜ਼ਬੂਤ ਮੇਜ਼ ਹੇਠਾਂ ਛੁਪ ਜਾਓ ਅਤੇ ਸਿਰ ਨੂੰ ਬਚਾਓ।\n2. ਧੂੜ ਤੋਂ ਬਚਣ ਲਈ ਨੱਕ ਅਤੇ ਮੂੰਹ ਕੱਪੜੇ ਨਾਲ ਢੱਕੋ। ਤੁਰੰਤ 112 'ਤੇ ਕਾਲ ਕਰੋ।",
        "general": "🚨 ਐਮਰਜੈਂਸੀ ਮੁੱਢਲੀ ਸਹਾਇਤਾ:\n1. ਕਿਰਪਾ ਕਰਕੇ ਸ਼ਾਂਤ ਰਹੋ ਅਤੇ ਸੁਰੱਖਿਅਤ ਥਾਂ 'ਤੇ ਜਾਓ।\n2. ਤੁਰੰਤ ਹੇਠਾਂ ਦਿੱਤੇ 108 ਜਾਂ 112 ਨੰਬਰ 'ਤੇ ਕਾਲ ਕਰੋ।"
    },
    "ta-IN": {
        "snake": "🐍 முதலுதவி (பாம்பு கடி):\n1. பதற்றப்படாமல் அமைதியாக இருங்கள், கடிபட்ட பகுதியை அசைக்க வேண்டாம்.\n2. கடிபட்ட இடத்தை இதய மட்டத்திற்கு கீழே வைக்கவும், இறுக்கமாக கட்டு போடாதீர்கள்.\n3. உடனே 108 ஆம்புலன்ஸை அழைக்கவும்.",
        "chest": "❤️ முதலுதவி (நெஞ்சு வலி):\n1. உடனடியாக உட்கார்ந்து ஓய்வெடுங்கள்.\n2. இறுக்கமான ஆடைகளை தளர்த்தவும். உடனே 108 ஆம்புலன்ஸை அழைக்கவும்.",
        "accident": "🚗 முதலுதவி (சாலை விபத்து):\n1. காயமடைந்தவரை தேவையில்லாமல் நகர்த்த வேண்டாம்.\n2. இரத்தப்போக்கு இருந்தால் சுத்தமான துணியால் அழுத்திப் பிடிக்கவும்.\n3. உடனே 108 மற்றும் 112 எண்ணிற்கு அழைக்கவும்.",
        "fire": "🔥 முதலுதவி (தீ மற்றும் தீக்காயம்):\n1. தீக்காயம் பட்ட இடத்தில் 15-20 நிமிடம் குளிர்ந்த நீரை ஊற்றவும்.\n2. கொப்புளங்களை உடைக்க வேண்டாம். உடனே 101 மற்றும் 108-ஐ அழைக்கவும்.",
        "disaster": "🏢 முதலுதவி (கட்டிடம் இடிந்து விழுதல் / பேரிடர்):\n1. உறுதியான மேஜைக்கு அடியில் பாதுகாப்பாக இருக்கவும்; தலையைப் பாதுகாக்கவும்.\n2. தூசியிலிருந்து தப்பிக்க மூக்கு மற்றும் வாயை துணியால் மூடவும். உடனே 112-ஐ அழைக்கவும்.",
        "general": "🚨 அவசர முதலுதவி:\n1. அமைதியாக பாதுகாப்பான இடத்திற்குச் செல்லவும்.\n2. உடனடியாக கீழே உள்ள 108 அல்லது 112 எண்ணிற்கு அழைக்கவும்."
    },
    "te-IN": {
        "snake": "🐍 ప్రథమ చికిత్స (పాము కాటు):\n1. భయపడకండి, ప్రశాంతంగా ఉండండి మరియు కాటు వేసిన భాగాన్ని కదిలించవద్దు.\n2. కాటు వేసిన భాగాన్ని గుండె స్థాయి కంటే దిగువన ఉంచండి, గట్టిగా కట్టు కట్టవద్దు.\n3. వెంటనే 108 అంబులెన్స్‌కు కాల్ చేయండి.",
        "chest": "❤️ ప్రథమ చికిత్స (గుండె నొప్పి):\n1. వెంటనే కూర్చుని విశ్రాంతి తీసుకోండి.\n2. బిగుతుగా ఉన్న దుస్తులను వదులు చేయండి. వెంటనే 108 కు కాల్ చేయండి.",
        "accident": "🚗 ప్రథమ చికిత్స (రోడ్డు ప్రమాదం):\n1. గాయపడిన వ్యక్తిని అనవసరంగా కదిలించవద్దు.\n2. రక్తస్రావం అవుతుంటే శుభ్రమైన గుడ్డతో గట్టిగా నొక్కి ఉంచండి.\n3. వెంటనే 108 మరియు 112 కు కాల్ చేయండి.",
        "fire": "🔥 ప్రథమ చికిత్స (అగ్ని ప్రమాదం / కాలిన గాయాలు):\n1. కాలిన గాయాలపై 15-20 నిమిషాలు చల్లని నీరు పోయండి.\n2. బొబ్బలను చిదమకండి. వెంటనే 101 మరియు 108 కు కాల్ చేయండి.",
        "disaster": "🏢 ప్రథమ చికిత్స (భవనం కూలిపోవడం / విపత్తు):\n1. దృఢమైన టేబుల్ కింద తలదాచుకోండి మరియు తలను రక్షించుకోండి.\n2. దుమ్ము లోపలికి వెళ్లకుండా ముక్కు, నోటిని గుడ్డతో కప్పుకోండి. వెంటనే 112 కు కాల్ చేయండి.",
        "general": "🚨 అత్యవసర ప్రథమ చికిత్స:\n1. ప్రశాంతంగా ఉండండి మరియు సురక్షిత ప్రాంతానికి వెళ్లండి.\n2. వెంటనే 108 లేదా 112 కు కాల్ చేయండి."
    }
}

SCENARIO_LABELS = {
    "snake": "Snakebite",
    "chest": "Chest Pain",
    "accident": "Road Accident",
    "fire": "Fire & Burns",
    "disaster": "Disaster / Collapse",
    "general": "General Emergency"
}

@app.post("/api/first-aid")
async def get_first_aid(req: EmergencyRequest):
    text = req.text.strip()
    lang_code = detect_language(text, req.lang_override)
    scenario_key = detect_scenario(text)

    pack = GUIDANCE_DB.get(lang_code, GUIDANCE_DB["en-IN"])
    reply = pack.get(scenario_key, pack["general"])

    return {
        "lang_code": lang_code,
        "language_name": LANG_NAMES.get(lang_code, "English"),
        "scenario": SCENARIO_LABELS.get(scenario_key, "General Emergency"),
        "first_aid": reply
    }
