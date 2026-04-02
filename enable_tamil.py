"""
Script to enable and test Tamil NLP translation
Uses AI4Bharat IndicTrans2 model for English to Tamil translation
"""

import os
os.environ['NLP_LIGHTWEIGHT'] = 'false'
os.environ['ENABLE_TAMIL_NLP'] = 'true'

print("\n" + "="*70)
print("🌿 TAMIL NLP TRANSLATION TEST")
print("="*70)
print("Model: ai4bharat/indictrans2-en-indic-1B")
print("Target Language: Tamil (தமிழ்)")
print("="*70)

from nlp import get_nlp_recommendation

# Test cases
test_cases = [
    ("Pepper", "Bacterial spot"),
    ("Tomato", "Early blight"),
]

for crop, disease in test_cases:
    print(f"\n{'─'*70}")
    print(f"TEST: {crop} - {disease}")
    print('─'*70)
    
    # English recommendation
    print("\n📝 ENGLISH RECOMMENDATION:")
    print("-" * 70)
    english_rec = get_nlp_recommendation(crop, disease, language="en")
    print(english_rec)
    
    # Tamil recommendation
    print("\n🇮🇳 TAMIL RECOMMENDATION (தமிழ்):")
    print("-" * 70)
    print("Note: First run downloads model (~1.5GB, takes 2-5 minutes)")
    print("-" * 70)
    tamil_rec = get_nlp_recommendation(crop, disease, language="ta")
    print(tamil_rec)
    
    print("\n" + "─"*70)

print("\n" + "="*70)
print("✅ TAMIL TRANSLATION TEST COMPLETE!")
print("="*70)
print("\n📌 Next Steps:")
print("  1. Run app: run_with_tamil.bat")
print("  2. Or: set ENABLE_TAMIL_NLP=true && python app.py")
print("  3. Upload image and select 'Tamil (தமிழ்)' language")
print("  4. Click 'Listen' button for voice output")
print("\n" + "="*70)
