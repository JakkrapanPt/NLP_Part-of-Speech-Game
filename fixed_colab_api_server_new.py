# Google Colab API Server for Parts of Speech Game
# This file should be run in Google Colab to serve the LLM API

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import time
from pyngrok import ngrok
import openai
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import random

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global variables for models
llm_pipeline = None
tokenizer = None
model = None

# Configuration
CONFIG = {
    'model_name': 'gpt2', 
    'max_length': 100,
    'temperature': 0.7,
    'use_openai': False,  
    'openai_api_key': None  
}

def initialize_model():
    """Initialize the language model"""
    global llm_pipeline, tokenizer, model
    
    print("Initializing language model...")
    
    if CONFIG['use_openai'] and CONFIG['openai_api_key']:
        # Use OpenAI API
        openai.api_key = CONFIG['openai_api_key']
        print("Using OpenAI API")
    else:
        # Use Hugging Face transformers
        try:
            device = 0 if torch.cuda.is_available() else -1
            print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
            model = AutoModelForCausalLM.from_pretrained(CONFIG['model_name'])
            
            # Set pad token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Create text generation pipeline with better parameters
            llm_pipeline = pipeline(
                'text-generation',
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_length=CONFIG['max_length'],
                temperature=CONFIG['temperature'],
                do_sample=True,
                top_k=50,
                top_p=0.95,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.pad_token_id
            )
            print(f"Model {CONFIG['model_name']} loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to a smaller model
            try:
                print("Attempting to load fallback model...")
                tokenizer = AutoTokenizer.from_pretrained('gpt2')
                model = AutoModelForCausalLM.from_pretrained('gpt2')
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    
                llm_pipeline = pipeline(
                    'text-generation',
                    model=model,
                    tokenizer=tokenizer,
                    device=device if device == -1 else 0,
                    max_length=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
                print("Fallback to GPT-2 model successful")
            except Exception as e2:
                print(f"Error loading fallback model: {e2}")
                llm_pipeline = None

# Sample sentences as fallback
FALLBACK_SENTENCES = {
    'en': {
        'easy': [
            "The cat sleeps peacefully.",
            "She runs quickly today.",
            "Birds fly high above.",
            "We eat delicious food.",
            "He plays guitar well.",
            "They dance beautifully together.",
            "I read books daily.",
            "Children laugh happily outside.",
            "Dogs bark loudly sometimes.",
            "Teachers explain concepts clearly."
        ],
        'medium': [
            "The beautiful flowers bloom in spring garden.",
            "Students study hard for their important exams.",
            "My grandmother tells interesting stories every evening.",
            "The chef prepared delicious meals for everyone yesterday.",
            "Our team won the championship after intense practice.",
            "The museum exhibits ancient artifacts from various civilizations.",
            "She completed her project before the strict deadline.",
            "The musician composed beautiful melodies on his piano.",
            "The detective solved the mysterious case last week.",
            "Many tourists visit famous landmarks during summer vacation."
        ],
        'hard': [
            "The magnificent orchestra performed brilliantly at the prestigious concert hall last night.",
            "Scientists carefully analyze complex data to understand mysterious phenomena in deep space.",
            "The experienced journalist documented the extraordinary political developments throughout the turbulent decade.",
            "Environmental researchers discovered unprecedented changes in the ecosystem after extensive fieldwork.",
            "The innovative technology company unveiled revolutionary products during their annual conference yesterday.",
            "Professional athletes demonstrate remarkable discipline and dedication throughout their challenging careers.",
            "The renowned professor published groundbreaking research that transformed theoretical physics fundamentally.",
            "International diplomats negotiated complex agreements between multiple countries during the summit.",
            "Talented architects designed sustainable buildings that harmonize perfectly with natural surroundings.",
            "Medical researchers developed effective treatments for previously incurable diseases after decades of work."
        ]
    },
    'th': {
        'easy': [
            "แมวนอนหลับสบาย",
            "เขาวิ่งเร็วมาก",
            "นกบินสูงขึ้น",
            "เรากินข้าวอร่อย",
            "เธออ่านหนังสือเก่ง",
            "ฉันชอบดูหนัง",
            "พวกเขาเล่นดนตรี",
            "น้องร้องเพลงไพเราะ",
            "พ่อขับรถเร็ว",
            "แม่ทำอาหารอร่อย",
            "เด็กๆวิ่งเล่นสนุก",
            "ครูสอนหนังสือดี",
            "หมาเห่าเสียงดัง",
            "แดดร้อนมากวันนี้",
            "ฝนตกหนักมาก"
        ],
        'medium': [
            "นักเรียนขยันอ่านหนังสือเพื่อสอบ",
            "ดอกไม้สวยบานในสวนหลังบ้าน",
            "คุณยายเล่านิทานให้ฟังทุกคืน",
            "พวกเราไปเที่ยวทะเลในวันหยุด",
            "เด็กๆชอบกินไอศกรีมในวันร้อน",
            "นักกีฬาฝึกซ้อมหนักเพื่อการแข่งขัน",
            "คุณครูสอนวิชาคณิตศาสตร์อย่างสนุก",
            "นักดนตรีเล่นเพลงไพเราะบนเวที",
            "ชาวนาปลูกข้าวในฤดูฝนทุกปี",
            "นักเขียนแต่งนิยายสนุกหลายเล่ม",
            "ช่างภาพถ่ายรูปสวยในงานแต่งงาน",
            "หมอรักษาคนไข้อย่างเอาใจใส่",
            "พ่อค้าขายของในตลาดตั้งแต่เช้า",
            "เด็กนักเรียนทำการบ้านเสร็จก่อนนอน",
            "พนักงานทำงานหนักตลอดทั้งสัปดาห์"
        ],
        'hard': [
            "นักวิทยาศาสตร์วิเคราะห์ข้อมูลซับซ้อนเพื่อทำความเข้าใจปรากฏการณ์ลึกลับในอวกาศ",
            "วงดุริยางค์ชื่อดังแสดงอย่างยอดเยี่ยมในหอประชุมใหญ่เมื่อคืนที่ผ่านมา",
            "นักเขียนมีชื่อเสียงเปิดตัวนวนิยายเรื่องใหม่ที่ได้รับความนิยมอย่างล้นหลามในงานสัปดาห์หนังสือ",
            "นักกีฬาทีมชาติฝึกซ้อมอย่างหนักเพื่อเตรียมความพร้อมสำหรับการแข่งขันระดับนานาชาติในเดือนหน้า",
            "ผู้เชี่ยวชาญด้านสิ่งแวดล้อมเสนอแนวทางแก้ไขปัญหามลพิษทางอากาศที่กำลังส่งผลกระทบต่อเมืองใหญ่",
            "นักธุรกิจรุ่นใหม่พัฒนาแอปพลิเคชันที่ช่วยให้ผู้คนจัดการเวลาได้อย่างมีประสิทธิภาพมากขึ้น",
            "อาจารย์มหาวิทยาลัยนำเสนอผลงานวิจัยที่ได้รับการตีพิมพ์ในวารสารวิชาการระดับนานาชาติ",
            "ศิลปินชื่อดังจัดนิทรรศการแสดงผลงานศิลปะร่วมสมัยที่สะท้อนปัญหาสังคมในปัจจุบัน",
            "นักการทูตเจรจาข้อตกลงทางการค้าระหว่างประเทศที่จะส่งผลดีต่อเศรษฐกิจในภูมิภาค",
            "สถาปนิกออกแบบอาคารประหยัดพลังงานที่ใช้เทคโนโลยีทันสมัยและเป็นมิตรกับสิ่งแวดล้อม",
            "แพทย์ผู้เชี่ยวชาญค้นพบวิธีการรักษาโรคที่ซับซ้อนด้วยนวัตกรรมทางการแพทย์แบบใหม่",
            "นักวิจัยด้านปัญญาประดิษฐ์พัฒนาระบบที่สามารถวิเคราะห์และทำนายพฤติกรรมของผู้บริโภคได้อย่างแม่นยำ",
            "ผู้กำกับภาพยนตร์มีชื่อเสียงสร้างผลงานที่ได้รับการยกย่องจากนักวิจารณ์ทั่วโลก",
            "นักประวัติศาสตร์ค้นพบหลักฐานสำคัญที่เปลี่ยนความเข้าใจเกี่ยวกับอารยธรรมโบราณ",
            "นักเศรษฐศาสตร์วิเคราะห์แนวโน้มตลาดการเงินโลกที่กำลังเปลี่ยนแปลงอย่างรวดเร็ว"
        ]
    }
}

# Track used sentences to avoid repetition
# Using lists instead of sets for better control
USED_SENTENCES = {
    'en': {'easy': [], 'medium': [], 'hard': []},
    'th': {'easy': [], 'medium': [], 'hard': []}
}

# Track last used sentence to avoid immediate repetition
LAST_SENTENCE = {
    'en': {'easy': None, 'medium': None, 'hard': None},
    'th': {'easy': None, 'medium': None, 'hard': None}
}

# English-Thai translation pairs for common words (for simple translations)
TRANSLATION_PAIRS = {
    "cat": "แมว",
    "dog": "หมา",
    "bird": "นก",
    "run": "วิ่ง",
    "eat": "กิน",
    "sleep": "นอน",
    "read": "อ่าน",
    "write": "เขียน",
    "book": "หนังสือ",
    "food": "อาหาร",
    "house": "บ้าน",
    "school": "โรงเรียน",
    "teacher": "ครู",
    "student": "นักเรียน",
    "friend": "เพื่อน",
    "family": "ครอบครัว",
    "water": "น้ำ",
    "sun": "พระอาทิตย์",
    "moon": "พระจันทร์",
    "star": "ดาว",
    "tree": "ต้นไม้",
    "flower": "ดอกไม้",
    "beautiful": "สวย",
    "happy": "มีความสุข",
    "sad": "เศร้า",
    "big": "ใหญ่",
    "small": "เล็ก",
    "fast": "เร็ว",
    "slow": "ช้า",
    "good": "ดี",
    "bad": "แย่",
    "hot": "ร้อน",
    "cold": "หนาว"
}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': llm_pipeline is not None or CONFIG['use_openai'],
        'timestamp': time.time()
    })

def is_valid_thai_sentence(text):
    """Check if a text is a valid Thai sentence"""
    if not text or len(text) < 3:  # Too short to be valid
        return False
        
    # Simple check for Thai characters
    thai_chars = 0
    for char in text:
        if '\u0e00' <= char <= '\u0e7f':  # Thai Unicode range
            thai_chars += 1
    return thai_chars > len(text) * 0.5  # At least 50% Thai characters

def simple_translate_to_thai(english_text):
    """Very simple word-by-word translation for testing"""
    words = english_text.lower().translate(str.maketrans('', '', ',.!?;:')).split()
    thai_words = []
    
    for word in words:
        if word in TRANSLATION_PAIRS:
            thai_words.append(TRANSLATION_PAIRS[word])
        else:
            # Skip words we don't know
            continue
    
    # If we couldn't translate anything, return None
    if not thai_words:
        return None
        
    return ' '.join(thai_words)

@app.route('/generate_sentence', methods=['POST'])
def generate_sentence_api():
    """API endpoint to generate sentences"""
    try:
        # Add error handling for JSON parsing
        try:
            data = request.get_json()
            if data is None:
                return jsonify({
                    'success': False,
                    'error': 'Invalid JSON data in request'
                }), 400
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Error parsing request data: {str(e)}'
            }), 400
        
        # Extract parameters - only language is needed now
        language = data.get('language', 'en')
        
        # Always generate English sentences first
        english_sentence = None
        
        # Try to generate an English sentence
        if CONFIG['use_openai'] and CONFIG['openai_api_key']:
            try:
                # Use OpenAI API to generate an English sentence
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt="Generate an interesting English sentence:",
                    max_tokens=50,
                    temperature=0.7
                )
                english_sentence = response.choices[0].text.strip()
                method = "openai"
            except Exception as e:
                print(f"OpenAI API error: {e}")
                english_sentence = None
        
        # If OpenAI failed or not available, use Hugging Face
        if english_sentence is None and llm_pipeline is not None:
            try:
                # Generate an English sentence with varying complexity
                prompt = "Write an interesting English sentence:"
                outputs = llm_pipeline(prompt, max_length=100, num_return_sequences=1)
                
                # Extract the generated text
                generated_text = outputs[0]['generated_text']
                
                # Remove the prompt
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                
                # Clean up the text
                sentences = generated_text.split('.')
                if sentences and len(sentences) > 0:
                    # Get first non-empty sentence
                    for s in sentences:
                        if s.strip():
                            english_sentence = s.strip() + '.'
                            break
                else:
                    english_sentence = generated_text.strip()
                    if not english_sentence.endswith('.'):
                        english_sentence += '.'
                        
                method = "huggingface"
            except Exception as e:
                print(f"Hugging Face model error: {e}")
                english_sentence = None
        
        # If we still don't have a sentence, use a fallback
        if not english_sentence:
            # Get a random English sentence from our collection
            all_english = []
            for diff in ['easy', 'medium', 'hard']:
                all_english.extend(FALLBACK_SENTENCES['en'][diff])
            
            english_sentence = random.choice(all_english)
            method = "english_fallback"
            
        # Now we have an English sentence, determine its difficulty based on word count
        word_count = len(english_sentence.split())
        
        if word_count <= 6:
            difficulty = 'easy'
        elif word_count >= 15:
            difficulty = 'hard'
        else:
            difficulty = 'medium'
            
        # If the user wants English, we're done
        if language == 'en':
            sentence = english_sentence
        else:
            # For Thai, we need to translate the English sentence
            thai_sentence = None
            
            # Try to translate using the model
            try:
                if llm_pipeline is not None:
                    prompt = f"Translate this English sentence to Thai: '{english_sentence}'"
                    outputs = llm_pipeline(prompt, max_length=150, num_return_sequences=1)
                    
                    # Extract the generated text
                    generated_text = outputs[0]['generated_text']
                    
                    # Remove the prompt
                    if generated_text.startswith(prompt):
                        generated_text = generated_text[len(prompt):].strip()
                    
                    # Extract Thai text - look for Thai characters
                    thai_text = ""
                    in_thai = False
                    for char in generated_text:
                        if '\u0e00' <= char <= '\u0e7f':
                            thai_text += char
                            in_thai = True
                        elif in_thai and (char.isspace() or char in ',.;:!?'):
                            thai_text += char
                    
                    if thai_text.strip():
                        thai_sentence = thai_text.strip()
            except Exception as e:
                print(f"Translation error: {e}")
                thai_sentence = None
                
            # Try simple word-by-word translation as backup
            if not thai_sentence or not is_valid_thai_sentence(thai_sentence):
                simple_thai = simple_translate_to_thai(english_sentence)
                if simple_thai and is_valid_thai_sentence(simple_thai):
                    thai_sentence = simple_thai
                    method = "simple_translation"
            
            # If translation failed, use a fallback Thai sentence
            if not thai_sentence or not is_valid_thai_sentence(thai_sentence):
                # Get a Thai sentence of similar difficulty
                thai_sentences = FALLBACK_SENTENCES['th'][difficulty]
                
                # Get one that hasn't been used recently
                used_list = USED_SENTENCES['th'][difficulty]
                last_used = LAST_SENTENCE['th'][difficulty]
                
                available = [s for s in thai_sentences if s != last_used and s not in used_list]
                if not available:
                    available = [s for s in thai_sentences if s != last_used]
                    if not available:
                        available = thai_sentences
                
                thai_sentence = random.choice(available)
                used_list.append(thai_sentence)
                LAST_SENTENCE['th'][difficulty] = thai_sentence
                
                method = "thai_fallback"
            else:
                method = "translated"
            
            sentence = thai_sentence
        
        # Ensure we have a valid sentence
        if not sentence:
            sentence = "The quick brown fox jumps." if language == 'en' else "แมวดำวิ่งเร็ว"
            method = "emergency_fallback"
            difficulty = "easy"
        
        # For debugging
        print(f"Generated {language} sentence ({difficulty}): {sentence}")
        print(f"Method: {method}")
            
        return jsonify({
            'success': True,
            'sentence': sentence,
            'language': language,
            'difficulty': difficulty,
            'method': method
        })
        
    except Exception as e:
        print(f"Error in generate_sentence_api: {e}")
        # Return a fallback sentence even on error
        language = request.get_json().get('language', 'en') if request.get_json() else 'en'
        fallback = "The quick brown fox jumps." if language == 'en' else "แมวดำวิ่งเร็ว"
        
        return jsonify({
            'success': True,  # Return success to prevent app from breaking
            'sentence': fallback,
            'language': language,
            'difficulty': 'easy',
            'method': 'error_fallback',
            'error_info': str(e)
        })

def run_server():
    """Run the Flask server"""
    # Initialize the model
    initialize_model()
    
    # Start the server
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

def setup_ngrok():
    """Setup ngrok tunnel for external access"""
    try:
        # Create tunnel
        public_url = ngrok.connect(5000)
        print(f"\n🌐 Public URL: {public_url}")
        print(f"📱 Use this URL in your Streamlit app: {public_url}")
        print(f"🔗 API endpoint: {public_url}/generate_sentence")
        
        return public_url
        
    except Exception as e:
        print(f"Error setting up ngrok: {e}")
        return None

if __name__ == '__main__':
    print("🚀 Starting Parts of Speech Game API Server")
    
    print("🌐 Setting up ngrok tunnel...")
    public_url = setup_ngrok()
    
    print("\n🎯 API Endpoints:")
    print("  - POST /generate_sentence - Generate sentences")
    print("  - GET /health - Health check")
    
    print("\n🔥 Starting Flask server...")
    
    # Start the server
    try:
        run_server()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
        ngrok.disconnect(public_url)
        ngrok.kill()
