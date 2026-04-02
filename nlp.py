import os

USE_LIGHTWEIGHT = os.environ.get('NLP_LIGHTWEIGHT', 'true').lower() == 'true'
MODELS_AVAILABLE = False
if not USE_LIGHTWEIGHT:
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer
        import torch; MODELS_AVAILABLE = True
    except ImportError:
        USE_LIGHTWEIGHT = True
print("Running in lightweight mode" if USE_LIGHTWEIGHT else "NLP models available")

_models = {}

CROP_TA = {"pepper":"மிளகு","bell":"குடமிளகாய்","tomato":"தக்காளி","potato":"உருளைக்கிழங்கு",
           "corn":"சோளம்","grape":"திராட்சை","apple":"ஆப்பிள்","cherry":"செர்ரி",
           "peach":"பீச்","strawberry":"ஸ்ட்ராபெரி","orange":"ஆரஞ்சு","squash":"பூசணி",
           "blueberry":"ப்ளூபெர்ரி","raspberry":"ராஸ்பெர்ரி","soybean":"சோயாபீன்"}

# disease_key -> {cause, medicine, treatment, ta_cause, ta_medicine, ta_treatment}
DISEASE_DB = {
    "bacterial spot": {
        "cause":      "Caused by Xanthomonas bacteria spread through rain and wind.",
        "medicine":   "Medicines: Copper Oxychloride (Blitox), Copper Hydroxide (Kocide), Streptomycin sulfate.",
        "treatment":  "Treatment: Remove infected leaves. Spray copper bactericide every 7-10 days. Avoid overhead watering. Ensure plant spacing. Rotate crops for 2-3 years.",
        "ta_cause":   "காரணம்: மழை மற்றும் காற்று மூலம் பரவும் சாந்தோமோனாஸ் பாக்டீரியா.",
        "ta_medicine":"மருந்துகள்: காப்பர் ஆக்சிகுளோரைடு (பிளிடாக்ஸ்), காப்பர் ஹைட்ராக்சைடு (கோசைடு), ஸ்ட்ரெப்டோமைசின் சல்பேட்.",
        "ta_treatment":"சிகிச்சை: பாதிக்கப்பட்ட இலைகளை அகற்றவும். 7-10 நாட்களுக்கு ஒருமுறை செம்பு மருந்து தெளிக்கவும். மேல்நிலை நீர்ப்பாசனம் தவிர்க்கவும். பயிர் சுழற்சி பின்பற்றவும்.",
    },
    "early blight": {
        "cause":      "Caused by Alternaria solani fungus, favored by warm humid conditions.",
        "medicine":   "Medicines: Mancozeb (Dithane M-45), Chlorothalonil (Kavach), Azoxystrobin (Amistar).",
        "treatment":  "Treatment: Remove lower infected leaves. Apply fungicide every 7 days. Mulch soil. Water at base only. Rotate crops annually.",
        "ta_cause":   "காரணம்: வெப்பமான ஈரமான சூழலில் வளரும் ஆல்டர்னேரியா சோலானி பூஞ்சை.",
        "ta_medicine":"மருந்துகள்: மேன்கோசெப் (டைதேன் M-45), குளோரோத்தாலோனில் (கவச்), அசாக்ஸிஸ்ட்ரோபின் (அமிஸ்டார்).",
        "ta_treatment":"சிகிச்சை: கீழ் இலைகளை அகற்றவும். 7 நாட்களுக்கு ஒருமுறை பூஞ்சைநாசினி தெளிக்கவும். மண்ணில் மூல்சிங் செய்யவும். அடிப்பகுதியில் நீர் ஊற்றவும்.",
    },
    "late blight": {
        "cause":      "Caused by Phytophthora infestans, spreads rapidly in cool wet weather.",
        "medicine":   "Medicines: Metalaxyl (Ridomil), Cymoxanil (Curzate), Dimethomorph (Acrobat).",
        "treatment":  "Treatment: Apply fungicide before rain. Remove all infected plants. Avoid evening watering. Use certified disease-free seeds. Improve air circulation.",
        "ta_cause":   "காரணம்: குளிர்ந்த ஈரமான காலநிலையில் வேகமாக பரவும் பைட்டோப்தோரா இன்பெஸ்டன்ஸ்.",
        "ta_medicine":"மருந்துகள்: மெட்டாலாக்சில் (ரிடோமில்), சைமோக்சானில் (கர்சேட்), டைமெத்தோமார்ப் (அக்ரோபேட்).",
        "ta_treatment":"சிகிச்சை: மழைக்கு முன் பூஞ்சைநாசினி தெளிக்கவும். பாதிக்கப்பட்ட செடிகளை அழிக்கவும். மாலையில் நீர் ஊற்றுவதை தவிர்க்கவும். நோயற்ற விதைகளை பயன்படுத்தவும்.",
    },
    "leaf spot": {
        "cause":      "Caused by various fungal pathogens, spread by water splash and wind.",
        "medicine":   "Medicines: Mancozeb (Dithane), Copper Oxychloride, Propiconazole (Tilt).",
        "treatment":  "Treatment: Remove infected leaves. Spray fungicide. Water early morning. Avoid overhead irrigation. Maintain good sanitation.",
        "ta_cause":   "காரணம்: நீர் மற்றும் காற்று மூலம் பரவும் பூஞ்சை நோய்க்கிருமிகள்.",
        "ta_medicine":"மருந்துகள்: மேன்கோசெப் (டைதேன்), காப்பர் ஆக்சிகுளோரைடு, ப்ரோபிகோனசோல் (டில்ட்).",
        "ta_treatment":"சிகிச்சை: பாதிக்கப்பட்ட இலைகளை அகற்றவும். பூஞ்சைநாசினி தெளிக்கவும். காலையில் நீர் ஊற்றவும். சுகாதாரம் பராமரிக்கவும்.",
    },
    "powdery mildew": {
        "cause":      "Caused by Erysiphe or Podosphaera fungi, thrives in dry warm conditions.",
        "medicine":   "Medicines: Sulfur dust, Potassium bicarbonate, Myclobutanil (Rally), Trifloxystrobin.",
        "treatment":  "Treatment: Spray sulfur or bicarbonate solution. Improve air circulation. Remove infected leaves. Avoid excess nitrogen fertilizer. Plant in sunny areas.",
        "ta_cause":   "காரணம்: வறண்ட வெப்பமான சூழலில் வளரும் எரிசிபே அல்லது போடோஸ்பேரா பூஞ்சை.",
        "ta_medicine":"மருந்துகள்: சல்பர் தூள், பொட்டாசியம் பைக்கார்பனேட், மைக்லோபுட்டானில் (ரேலி), ட்ரைஃப்ளாக்ஸிஸ்ட்ரோபின்.",
        "ta_treatment":"சிகிச்சை: சல்பர் கரைசல் தெளிக்கவும். காற்றோட்டம் ஏற்படுத்தவும். பாதிக்கப்பட்ட இலைகளை அகற்றவும். அதிக நைட்ரஜன் உரம் தவிர்க்கவும்.",
    },
    "septoria": {
        "cause":      "Caused by Septoria lycopersici fungus, spreads through rain splash.",
        "medicine":   "Medicines: Chlorothalonil (Bravo), Mancozeb, Copper-based fungicides.",
        "treatment":  "Treatment: Remove lower leaves with symptoms. Apply fungicide. Mulch soil. Space plants properly. Rotate crops for 2-3 years.",
        "ta_cause":   "காரணம்: மழை நீர் மூலம் பரவும் செப்டோரியா லைகோபெர்சிசி பூஞ்சை.",
        "ta_medicine":"மருந்துகள்: குளோரோத்தாலோனில் (பிரேவோ), மேன்கோசெப், செம்பு அடிப்படையிலான பூஞ்சைநாசினிகள்.",
        "ta_treatment":"சிகிச்சை: கீழ் இலைகளை அகற்றவும். பூஞ்சைநாசினி தெளிக்கவும். மூல்சிங் செய்யவும். 2-3 ஆண்டுகள் பயிர் சுழற்சி செய்யவும்.",
    },
    "target spo": {
        "cause":      "Caused by Corynespora cassiicola fungus in warm humid conditions.",
        "medicine":   "Medicines: Azoxystrobin (Amistar), Fluxapyroxad, Boscalid (Endura).",
        "treatment":  "Treatment: Apply fungicide at first sign. Remove infected debris. Improve air circulation. Avoid overhead irrigation. Use disease-free seeds.",
        "ta_cause":   "காரணம்: வெப்பமான ஈரமான சூழலில் வளரும் கோரினெஸ்போரா காசிகோலா பூஞ்சை.",
        "ta_medicine":"மருந்துகள்: அசாக்ஸிஸ்ட்ரோபின் (அமிஸ்டார்), ஃப்ளக்ஸாபைராக்ஸாட், போஸ்காலிட் (என்டுரா).",
        "ta_treatment":"சிகிச்சை: முதல் அறிகுறியிலேயே பூஞ்சைநாசினி தெளிக்கவும். பாதிக்கப்பட்ட பாகங்களை அகற்றவும். நோயற்ற விதைகளை பயன்படுத்தவும்.",
    },
    "target spot": {
        "cause":      "Caused by Corynespora cassiicola fungus in warm humid conditions.",
        "medicine":   "Medicines: Azoxystrobin (Amistar), Fluxapyroxad, Boscalid (Endura).",
        "treatment":  "Treatment: Apply fungicide at first sign. Remove infected debris. Improve air circulation. Avoid overhead irrigation. Use disease-free seeds.",
        "ta_cause":   "காரணம்: வெப்பமான ஈரமான சூழலில் வளரும் கோரினெஸ்போரா காசிகோலா பூஞ்சை.",
        "ta_medicine":"மருந்துகள்: அசாக்ஸிஸ்ட்ரோபின் (அமிஸ்டார்), ஃப்ளக்ஸாபைராக்ஸாட், போஸ்காலிட் (என்டுரா).",
        "ta_treatment":"சிகிச்சை: முதல் அறிகுறியிலேயே பூஞ்சைநாசினி தெளிக்கவும். பாதிக்கப்பட்ட பாகங்களை அகற்றவும். நோயற்ற விதைகளை பயன்படுத்தவும்.",
    },
    "leaf mold": {
        "cause":      "Caused by Passalora fulva fungus. Favored by high humidity and poor air circulation.",
        "medicine":   "Medicines: Chlorothalonil, Mancozeb, Copper-based fungicides, Azoxystrobin.",
        "treatment":  "Treatment: Use drip irrigation, avoid wetting foliage. Space plants for good airflow. Stake and prune for air circulation. Sterilize tools. Remove crop residue after season.",
        "ta_cause":   "காரணம்: அதிக ஈரப்பதம் மற்றும் மோசமான காற்றோட்டத்தில் வளரும் பாசலோரா ஃபுல்வா பூஞ்சை.",
        "ta_medicine":"மருந்துகள்: குளோரோத்தாலோனில், மேன்கோசெப், செம்பு அடிப்படையிலான பூஞ்சைநாசினிகள், அசாக்ஸிஸ்ட்ரோபின்.",
        "ta_treatment":"சிகிச்சை: சொட்டு நீர்ப்பாசனம் பயன்படுத்தவும். இலைகளை நனைக்காதீர்கள். காற்றோட்டத்திற்கு செடிகளை கட்டவும். கருவிகளை சுத்தம் செய்யவும். அறுவடைக்கு பிறகு செடி எச்சங்களை அகற்றவும்.",
    },
    "spider mite": {
        "cause":      "Caused by Two-spotted spider mite (Tetranychus urticae). Favored by hot dry conditions and excess nitrogen.",
        "medicine":   "Medicines: Abamectin (Agrimek), Bifenazate (Floramite), Spiromesifen (Oberon), Neem oil.",
        "treatment":  "Treatment: Spray water forcefully to dislodge mites. Apply miticide or neem oil. Avoid excess nitrogen. Introduce natural predators. Remove heavily infested leaves.",
        "ta_cause":   "காரணம்: வெப்பமான வறண்ட சூழலில் வளரும் இரண்டு புள்ளி சிலந்தி பூச்சி (டெட்ரானைகஸ் யூர்டிகே).",
        "ta_medicine":"மருந்துகள்: அபாமெக்டின் (அக்ரிமெக்), பைஃபெனசேட் (ஃப்ளோரமைட்), ஸ்பைரோமெசிஃபென் (ஒபெரான்), வேப்பெண்ணெய்.",
        "ta_treatment":"சிகிச்சை: தண்ணீரை வலுவாக தெளித்து பூச்சிகளை அகற்றவும். மைட்டிசைட் அல்லது வேப்பெண்ணெய் தெளிக்கவும். அதிக நைட்ரஜன் தவிர்க்கவும். அதிகமாக பாதிக்கப்பட்ட இலைகளை அகற்றவும்.",
    },
    "mosaic virus": {
        "cause":      "Caused by Tobacco Mosaic Virus or Cucumber Mosaic Virus, spread by aphids.",
        "medicine":   "Medicines: No direct cure. Use Imidacloprid (Confidor) to control aphid vectors. Neem oil spray.",
        "treatment":  "Treatment: Remove and destroy infected plants immediately. Control aphids with insecticide. Disinfect tools with bleach. Use virus-resistant varieties. Remove weeds.",
        "ta_cause":   "காரணம்: அஃபிட் பூச்சிகள் மூலம் பரவும் டொபாக்கோ மொசைக் வைரஸ் அல்லது குக்கம்பர் மொசைக் வைரஸ்.",
        "ta_medicine":"மருந்துகள்: நேரடி மருந்து இல்லை. அஃபிட் கட்டுப்பாட்டிற்கு இமிடாக்லோப்ரிட் (கான்பிடோர்). வேப்பெண்ணெய் தெளிப்பு.",
        "ta_treatment":"சிகிச்சை: பாதிக்கப்பட்ட செடிகளை உடனே அகற்றி அழிக்கவும். அஃபிட்களை கட்டுப்படுத்தவும். கருவிகளை சுத்தம் செய்யவும். நோய் எதிர்ப்பு வகைகளை பயிரிடவும்.",
    },
    "leaf curl": {
        "cause":      "Caused by Tomato Yellow Leaf Curl Virus, transmitted by whiteflies.",
        "medicine":   "Medicines: Thiamethoxam (Actara), Imidacloprid for whitefly control. Neem oil.",
        "treatment":  "Treatment: Remove affected leaves. Control whiteflies with insecticide. Use yellow sticky traps. Apply neem oil. Plant resistant varieties.",
        "ta_cause":   "காரணம்: வெள்ளை ஈ மூலம் பரவும் டொமேட்டோ யெல்லோ லீஃப் கர்ல் வைரஸ்.",
        "ta_medicine":"மருந்துகள்: தியாமெத்தோக்சம் (அக்டாரா), இமிடாக்லோப்ரிட் வெள்ளை ஈ கட்டுப்பாட்டிற்கு. வேப்பெண்ணெய்.",
        "ta_treatment":"சிகிச்சை: பாதிக்கப்பட்ட இலைகளை அகற்றவும். வெள்ளை ஈ கட்டுப்படுத்தவும். மஞ்சள் பசை பொறி வைக்கவும். வேப்பெண்ணெய் தெளிக்கவும்.",
    },
    "black rot": {
        "cause":      "Caused by Guignardia bidwellii fungus in grapes, or Xanthomonas in apples.",
        "medicine":   "Medicines: Mancozeb, Captan, Myclobutanil (Rally), Ziram.",
        "treatment":  "Treatment: Remove mummified fruits and infected canes. Apply fungicide from bud break. Improve canopy airflow. Prune infected wood. Rotate crops.",
        "ta_cause":   "காரணம்: திராட்சையில் குய்க்னார்டியா பிட்வெல்லி பூஞ்சை, ஆப்பிளில் சாந்தோமோனாஸ் பாக்டீரியா.",
        "ta_medicine":"மருந்துகள்: மேன்கோசெப், கேப்டன், மைக்லோபுட்டானில் (ரேலி), ஜிரம்.",
        "ta_treatment":"சிகிச்சை: பாதிக்கப்பட்ட கனிகள் மற்றும் கிளைகளை அகற்றவும். முளை வெடிக்கும் போது பூஞ்சைநாசினி தெளிக்கவும். கிளைகளை கத்தரிக்கவும்.",
    },
    "apple scab": {
        "cause":      "Caused by Venturia inaequalis fungus, spreads in wet spring conditions.",
        "medicine":   "Medicines: Captan, Myclobutanil, Dodine, Mancozeb.",
        "treatment":  "Treatment: Apply fungicide from green tip stage. Remove fallen leaves. Prune for air circulation. Use scab-resistant apple varieties.",
        "ta_cause":   "காரணம்: ஈரமான வசந்த காலத்தில் பரவும் வென்டூரியா இனேக்வாலிஸ் பூஞ்சை.",
        "ta_medicine":"மருந்துகள்: கேப்டன், மைக்லோபுட்டானில், டோடின், மேன்கோசெப்.",
        "ta_treatment":"சிகிச்சை: பச்சை நுனி நிலையிலிருந்து பூஞ்சைநாசினி தெளிக்கவும். விழுந்த இலைகளை அகற்றவும். காற்றோட்டத்திற்கு கிளைகளை கத்தரிக்கவும்.",
    },
    "cedar apple rust": {
        "cause":      "Caused by Gymnosporangium juniperi-virginianae, requires both apple and cedar trees.",
        "medicine":   "Medicines: Myclobutanil (Rally), Propiconazole, Triadimefon.",
        "treatment":  "Treatment: Apply fungicide during wet spring. Remove nearby cedar trees if possible. Use rust-resistant apple varieties. Spray at pink bud stage.",
        "ta_cause":   "காரணம்: ஆப்பிள் மற்றும் சீடார் மரங்கள் இரண்டையும் தேவைப்படும் ஜிம்னோஸ்போரான்ஜியம் பூஞ்சை.",
        "ta_medicine":"மருந்துகள்: மைக்லோபுட்டானில் (ரேலி), ப்ரோபிகோனசோல், ட்ரையடிமெஃபான்.",
        "ta_treatment":"சிகிச்சை: ஈரமான வசந்த காலத்தில் பூஞ்சைநாசினி தெளிக்கவும். அருகில் உள்ள சீடார் மரங்களை அகற்றவும். நோய் எதிர்ப்பு வகைகளை பயிரிடவும்.",
    },
    "northern leaf blight": {
        "cause":      "Caused by Exserohilum turcicum fungus in corn, favored by moderate temperatures.",
        "medicine":   "Medicines: Azoxystrobin, Propiconazole, Pyraclostrobin (Headline).",
        "treatment":  "Treatment: Apply fungicide at tasseling stage. Use resistant hybrids. Rotate corn with non-host crops. Till infected residue after harvest.",
        "ta_cause":   "காரணம்: மிதமான வெப்பநிலையில் சோளத்தில் வளரும் எக்ஸ்செரோஹிலம் டர்சிகம் பூஞ்சை.",
        "ta_medicine":"மருந்துகள்: அசாக்ஸிஸ்ட்ரோபின், ப்ரோபிகோனசோல், பைராக்லோஸ்ட்ரோபின் (ஹெட்லைன்).",
        "ta_treatment":"சிகிச்சை: தாசல் நிலையில் பூஞ்சைநாசினி தெளிக்கவும். நோய் எதிர்ப்பு ஹைப்ரிட் வகைகளை பயன்படுத்தவும். பயிர் சுழற்சி செய்யவும்.",
    },
    "common rust": {
        "cause":      "Caused by Puccinia sorghi fungus in corn, spreads through wind-borne spores.",
        "medicine":   "Medicines: Azoxystrobin (Amistar), Propiconazole (Tilt), Mancozeb.",
        "treatment":  "Treatment: Apply fungicide early. Use rust-resistant corn varieties. Plant early to avoid peak rust season. Monitor fields regularly.",
        "ta_cause":   "காரணம்: காற்று மூலம் பரவும் புக்சினியா சோர்கி பூஞ்சை.",
        "ta_medicine":"மருந்துகள்: அசாக்ஸிஸ்ட்ரோபின் (அமிஸ்டார்), ப்ரோபிகோனசோல் (டில்ட்), மேன்கோசெப்.",
        "ta_treatment":"சிகிச்சை: ஆரம்பத்திலேயே பூஞ்சைநாசினி தெளிக்கவும். நோய் எதிர்ப்பு வகைகளை பயிரிடவும். வயல்களை தொடர்ந்து கண்காணிக்கவும்.",
    },
    "leaf scorch": {
        "cause":      "Caused by Diplocarpon earlianum fungus in strawberry.",
        "medicine":   "Medicines: Captan, Myclobutanil, Azoxystrobin.",
        "treatment":  "Treatment: Remove infected leaves. Apply fungicide in spring. Use resistant varieties. Avoid overhead irrigation. Renovate beds after harvest.",
        "ta_cause":   "காரணம்: ஸ்ட்ராபெர்ரியில் டிப்லோகார்பான் எர்லியானம் பூஞ்சை.",
        "ta_medicine":"மருந்துகள்: கேப்டன், மைக்லோபுட்டானில், அசாக்ஸிஸ்ட்ரோபின்.",
        "ta_treatment":"சிகிச்சை: பாதிக்கப்பட்ட இலைகளை அகற்றவும். வசந்த காலத்தில் பூஞ்சைநாசினி தெளிக்கவும். மேல்நிலை நீர்ப்பாசனம் தவிர்க்கவும்.",
    },
    "haunglongbing": {
        "cause":      "Caused by Candidatus Liberibacter bacteria, spread by Asian citrus psyllid insect.",
        "medicine":   "Medicines: No cure available. Control psyllid with Imidacloprid, Thiamethoxam. Nutritional sprays to manage symptoms.",
        "treatment":  "Treatment: Remove and destroy infected trees. Control psyllid insects. Use certified disease-free nursery plants. Apply nutritional sprays. Monitor regularly.",
        "ta_cause":   "காரணம்: ஆசிய சிட்ரஸ் சில்லிட் பூச்சி மூலம் பரவும் கேண்டிடேட்டஸ் லிபெரிபேக்டர் பாக்டீரியா.",
        "ta_medicine":"மருந்துகள்: நேரடி மருந்து இல்லை. சில்லிட் கட்டுப்பாட்டிற்கு இமிடாக்லோப்ரிட், தியாமெத்தோக்சம். ஊட்டச்சத்து தெளிப்பு.",
        "ta_treatment":"சிகிச்சை: பாதிக்கப்பட்ட மரங்களை அகற்றி அழிக்கவும். சில்லிட் பூச்சிகளை கட்டுப்படுத்தவும். நோயற்ற நாற்றுகளை பயன்படுத்தவும்.",
    },
    "esca": {
        "cause":      "Caused by complex of fungi including Phaeomoniella and Phaeoacremonium in grapevines.",
        "medicine":   "Medicines: No effective chemical cure. Sodium arsenite (restricted). Trichoderma-based biocontrol.",
        "treatment":  "Treatment: Prune infected wood. Protect pruning wounds with fungicide paste. Remove severely infected vines. Avoid pruning in wet weather.",
        "ta_cause":   "காரணம்: திராட்சை கொடியில் பேயோமோனியெல்லா மற்றும் பேயோக்ரெமோனியம் பூஞ்சைகள்.",
        "ta_medicine":"மருந்துகள்: பயனுள்ள வேதி மருந்து இல்லை. ட்ரைக்கோடெர்மா அடிப்படையிலான உயிரியல் கட்டுப்பாடு.",
        "ta_treatment":"சிகிச்சை: பாதிக்கப்பட்ட மரக்கட்டைகளை கத்தரிக்கவும். கத்தரிக்கும் இடங்களை பூஞ்சைநாசினி பேஸ்ட்டால் பாதுகாக்கவும். ஈரமான காலத்தில் கத்தரிக்க வேண்டாம்.",
    },
    "healthy": {
        "cause":      "No disease detected. Your plant is healthy.",
        "medicine":   "No medicine needed. Maintain regular care.",
        "treatment":  "Continue regular monitoring, proper watering, balanced fertilization, good air circulation, and preventive care.",
        "ta_cause":   "நோய் இல்லை. உங்கள் செடி ஆரோக்கியமாக உள்ளது.",
        "ta_medicine":"மருந்து தேவையில்லை. வழக்கமான பராமரிப்பை தொடரவும்.",
        "ta_treatment":"தொடர்ந்து கண்காணிக்கவும், சரியான நீர் ஊற்றவும், சமநிலையான உரம் இடவும், காற்றோட்டம் உறுதி செய்யவும்.",
    },
}

DISEASE_TA = {
    "bacterial spot": "பாக்டீரியா புள்ளி நோய்",
    "early blight": "ஆரம்ப கருகல் நோய்",
    "late blight": "தாமத கருகல் நோய்",
    "leaf spot": "இலை புள்ளி நோய்",
    "powdery mildew": "பொடி பூஞ்சை நோய்",
    "septoria": "செப்டோரியா இலை புள்ளி நோய்",
    "target spot": "இலக்கு புள்ளி நோய்",
    "mosaic virus": "மொசைக் வைரஸ் நோய்",
    "leaf curl": "இலை சுருள் நோய்",
    "black rot": "கருப்பு அழுகல் நோய்",
    "apple scab": "ஆப்பிள் சொறி நோய்",
    "cedar apple rust": "சீடார் ஆப்பிள் துரு நோய்",
    "northern leaf blight": "வடக்கு இலை கருகல் நோய்",
    "common rust": "பொதுவான துரு நோய்",
    "leaf scorch": "இலை வெந்து நோய்",
    "haunglongbing": "சிட்ரஸ் பசுமை நோய்",
    "esca": "எஸ்கா நோய்",
    "healthy": "ஆரோக்கியமான செடி",
    "gray leaf spot": "சாம்பல் இலை புள்ளி நோய்",
    "cercospora": "சர்கோஸ்போரா இலை புள்ளி நோய்",
    "leaf mold": "இலை அச்சு நோய்",
    "spider mite": "சிலந்தி பூச்சி தாக்குதல்",
    "two-spotted": "இரண்டு புள்ளி சிலந்தி பூச்சி",
    "yellow leaf curl": "மஞ்சள் இலை சுருள் நோய்",
}

def _disease_ta(disease):
    d = disease.lower()
    for key, val in DISEASE_TA.items():
        if key in d:
            return val
    return disease

def _match(disease):
    d = disease.lower()
    for key in DISEASE_DB:
        if key in d:
            return DISEASE_DB[key]
    return None
    d = disease.lower()
    for key in DISEASE_DB:
        if key in d:
            return DISEASE_DB[key]
    return None

def extract_crop_from_prediction(pred):
    return pred.split('___')[0].replace('_', ' ') if '___' in pred else 'Plant'

def extract_disease_from_prediction(pred):
    return pred.split('___')[1].replace('_', ' ') if '___' in pred else pred.replace('_', ' ')

def get_nlp_recommendation(crop, disease, language='en'):
    data = _match(disease)
    crop_ta = CROP_TA.get(crop.lower(), crop)
    if language == 'ta':
        if data:
            return f"{crop_ta} செடியில் {_disease_ta(disease)} கண்டறியப்பட்டது. {data['ta_treatment']}"
        return f"{crop_ta} பயிரில் {_disease_ta(disease)}: வேளாண் நிபுணர்களை அணுகவும்."
    else:
        if data:
            return f"{disease} detected in {crop}. {data['treatment']}"
        return f"For {disease} in {crop}: Remove infected parts, apply appropriate treatment, ensure good air circulation, practice crop rotation, and consult an agricultural expert."

def get_full_voice_text(crop, disease, language='en'):
    """Returns full voice text: disease name + cause + medicine + treatment"""
    data = _match(disease)
    crop_ta = CROP_TA.get(crop.lower(), crop)
    if language == 'ta':
        name = f"{crop_ta} செடியில் {_disease_ta(disease)} கண்டறியப்பட்டது."
        if data:
            return f"{name} {data['ta_cause']} {data['ta_medicine']} {data['ta_treatment']}"
        return f"{name} வேளாண் நிபுணர்களை அணுகவும்."
    else:
        name = f"{disease} detected in {crop} plant."
        if data:
            return f"{name} {data['cause']} {data['medicine']} {data['treatment']}"
        return f"{name} Consult an agricultural expert for proper treatment."

def get_disease_info_tamil(crop, disease):
    data = _match(disease)
    crop_ta = CROP_TA.get(crop.lower(), crop)
    disease_ta = _disease_ta(disease)
    if data:
        return (f"<b>பயிர்:</b> {crop_ta} | <b>நோய்:</b> {disease_ta}<br/><br/>"
                f"<b>நோயின் காரணம்:</b><br/>{data['ta_cause']}<br/><br/>"
                f"<b>மருந்துகள்:</b><br/>{data['ta_medicine']}<br/><br/>"
                f"<b>சிகிச்சை முறை:</b><br/>{data['ta_treatment']}")
    return f"<b>பயிர்:</b> {crop_ta} | <b>நோய்:</b> {disease_ta}<br/><br/>வேளாண் நிபுணர்களை அணுகவும்."
