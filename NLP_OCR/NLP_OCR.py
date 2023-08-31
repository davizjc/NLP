import os
import re
import cv2
import nltk
from gensim import corpora, models
from nltk.corpus import stopwords, words
from nltk.stem import PorterStemmer
from pdf2image import convert_from_path
import pytesseract
from pytesseract import Output, image_to_string, image_to_data
from pythainlp import word_tokenize
from pythainlp.corpus import thai_words
from pythainlp.corpus.common import thai_stopwords
from spellchecker import SpellChecker

POPPLER_PATH = 'C:\\Users\\cdgs\\OneDrive\\Desktop\\poppler-23.07.0\\Library\\bin'
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\tesseract.exe'

# Download stopwords for English
nltk.download('stopwords')
nltk.download('words')

english_stopwords = set(stopwords.words('english'))
thai_stopword_set = set(thai_stopwords())
ps = PorterStemmer()
english_word_list = words.words()
thai_word_list = set(thai_words())
spell = SpellChecker()



# def is_word(word):
#     """Check if the word exists in the English or Thai dictionary."""
#     if any(char.isascii() for char in word):  # If word contains any ASCII (likely English)
#         return word in spell  # Check using PySpellChecker
#     else:
#         return word in thai_word_list  # Check against Thai word list from pythainlp
    
def is_word(word):
    """Check if the word exists in the English or Thai dictionary."""
    if any(char.isascii() for char in word):  # If word contains any ASCII (likely English)
        is_legitimate = word in spell  # Check using PySpellChecker
        return (is_legitimate, None if is_legitimate else word)
    else:
        is_legitimate = word in thai_word_list  # Check against Thai word list from pythainlp
        return (is_legitimate, None if is_legitimate else word)

def preprocess(text): #3
    """Tokenizes, stems, lowercases, removes short words, removes all numbers (both Arabic and Thai), and removes symbols from the text."""
    
    # This pattern matches symbols, Arabic numbers, and Thai numbers.
    pattern = r'[!@#$%^&*()_+={}[\]:;<>,.?~\-\/"\'|\\0-9\u0E50-\u0E59]'
    text_without_unwanted_chars = re.sub(pattern, '', text)
    
    words = []
    for segment in text_without_unwanted_chars.split():  # First split by spaces
        # If the segment contains Thai characters, tokenize it with pythainlp
        if any(u'\u0e00' <= ch <= u'\u0e7f' for ch in segment):
            words.extend(word_tokenize(segment, engine='newmm')) #pythainlp to split sentence into word
        else:
            words.append(segment)

    stemmed_words = []
    removed_words = []  # List to capture words that were removed
    for word in words:
        is_legit, removed_word = is_word(word)  # Check if word is legitimate and get removed word (if any)

        # Check if English word exists in the dictionary
        if any(char.isascii() for char in word) and is_legit:
            stemmed_words.append(ps.stem(word.lower()))
        # Keep Thai words as is (no stemming)
        elif not any(char.isascii() for char in word) and is_legit:
            stemmed_words.append(word)
        
        if removed_word:  # If a word was removed, append it to the list
            removed_words.append(removed_word)

    print(f"Words removed by is_word function: {', '.join(removed_words)}")

    return [word for word in stemmed_words if len(word) > 2]


def remove_stop_words_and_numbers(text): #5
    """Removes English and Thai stop words, standalone numbers, and preprocesses the text."""
    # preprocess the text first
    words = preprocess(text)
    # print(f'words {words}')
        
    # remove stop words
    my_stopword = ["กล่าว","กว่า","กัน","กับ","การ","ก็","ก่อน","ขณะ","ขอ","ของ","eee","ขึ้น","คง","ครั้ง","ความ","คือ","จะ","จัด",
                   "จาก","จึง","ช่วง","ซึ่ง","ดัง","ด้วย","ด้าน","ตั้ง","ตั้งแต่","ตาม","ต่อ","ต่าง","ต่างๆ","ต้อง","ถึง","ถูก","ถ้า",
                   "ทั้ง","ทั้งนี้","ทาง","ทำ","ทำให้","ที่","ที่สุด","ทุก","นอกจาก","นัก","นั้น","นำ","นี้","น่า","บาง","ผล","ผ่าน",
                   "พบ","พร้อม","มา","มาก","มี","ยัง","รวม","ระหว่าง","รับ","ราย","ร่วม","ลง","วัน","ว่า","สำหรับ","สุด","ส่ง",
                   "ส่วน","หนึ่ง","หรือ","หลัง","หลังจาก","หลาย","หาก","อยาก","อยู่","อย่าง","ออก","อะไร","อาจ","อีก","เขา",
                   "เข้า","เคย","เฉพาะ","เช่น","เดียว","เดียวกัน","เนื่องจาก","เปิด","เปิดเผย","เป็น","เป็นการ","เพราะ","เพื่อ","เมื่อ",
                   "เรา","เริ่ม","เลย","เห็น","เอง","แต่","แบบ","แรก","และ","แล้ว","แห่ง","โดย","ใน","ให้","ได้","ไป","ไม่","ไว้",
                   "ข้อ","มู่ล","สูต","โมง","ําหนด","ดังนี้","เนิน","ต้อ","ารถ","gor","ร่วมกับ","น่ง","ean","alo","ala"]    
    all_stopwords = set(my_stopword).union(english_stopwords).union(thai_stopword_set)
    filtered_words = [word for word in words if word not in all_stopwords]
    # print(f'filtered word {filtered_words}' )

    removed_words = [word for word in words if word in all_stopwords]
    
    return ' '.join(filtered_words), removed_words


all_cleaned_texts = []

def process_image(img_path):  # 2   OCR
    img = cv2.imread(img_path)
    ocr_result = pytesseract.image_to_string(img, lang='tha+eng')
    cleaned_ocr_result, removed_words = remove_stop_words_and_numbers(ocr_result)
    
    # print(cleaned_ocr_result)

    all_cleaned_texts.append(cleaned_ocr_result.split())  # split the cleaned text into words

    # print("Removed Words:", ", ".join(removed_words))
    return ocr_result, cleaned_ocr_result



def process_pdf(pdf_path):   # 1 READ PDF
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)

    for i, image in enumerate(images):
        print(f"Processing page {i+1} of PDF...")
        img_path = f"temp_page_{i+1}.png"
        image.save(img_path, 'PNG')
        _, cleaned_text = process_image(img_path)
            
    

def extract_dominant_words(topic):
    """Extract words and the dominant word for the given topic."""
    _, terms = topic
    words_with_scores = terms.split(" + ")
    
    words = [word.split('"')[1] for word in words_with_scores]
    dominant_word_score, dominant_word = max(
        [(float(word.split('*')[0]), word.split('*')[1].strip('"')) for word in words_with_scores],
        key=lambda x: x[0]
    )
    
    return words, dominant_word

def main():
    path = "C:\\Users\\cdgs\\OneDrive\\Desktop\\"
    file = path + "file1.pdf"
    process_pdf(file)

    # Create a dictionary from the tokenized documents
    dictionary = corpora.Dictionary(all_cleaned_texts)

    # Convert tokenized documents into a term-document matrix (Bag-of-Words)
    #split word into token
    corpus = [dictionary.doc2bow(text) for text in all_cleaned_texts]

    # Initialize a TF-IDF model and transform the corpus
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    # Train the LDA model
    lda_model = models.LdaModel(corpus_tfidf, num_topics=5, id2word=dictionary, passes=15)

    topics = lda_model.print_topics(num_words=200)

    with open('result1.txt', 'w', encoding='utf-8') as file:
        for topic_num, terms in topics:
            words, dominant_word = extract_dominant_words((topic_num, terms))
            for word in words:
                file.write(f"{word} {topic_num}\n")
            print(f"Topic {topic_num}: Dominant Word - {dominant_word}")

if __name__ == '__main__':
    main()
