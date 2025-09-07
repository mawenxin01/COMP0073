import pandas as pd
import time
import re
import os
from pathlib import Path
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as Meta
from ibm_watsonx_ai import Credentials, APIClient

# === Watsonx Configuration ===
API_KEY    = "7wRH61jLTGouQhEefLlJB6wQ7cQTm6Ymk56Ha1NsPpY3"
PROJECT_ID = "b3be303c-cd3a-4dc3-b4f5-ef51ebc72504"
REGION     = "eu-gb"
URL        = f"https://{REGION}.ml.cloud.ibm.com"

# 获取当前文件所在目录的路径
current_dir = Path(__file__).parent
project_root = current_dir.parent

# 定义输入输出路径

input_path = project_root / "data_processing" / "processed_data" / "triage_final.csv"
output_path = project_root / "data_processing" / "processed_data" / "triage_with_keywords.csv"

# 确保输出目录存在
output_path.parent.mkdir(parents=True, exist_ok=True)

credentials = Credentials(api_key=API_KEY, url=URL)
client = APIClient(credentials)

model = ModelInference(
    model_id="ibm/granite-13b-instruct-v2",
    credentials=credentials,
    project_id=PROJECT_ID,
    params={Meta.MAX_NEW_TOKENS: 50, Meta.TEMPERATURE: 0.2},
)

# === Abbreviation Mapping Table ===
abbreviation_map = {
    "n/v/d": "nausea, vomiting, diarrhea",
    "n/v": "nausea, vomiting",
    "lethagic": "lethargy",
    "s/p": "status post",
    "b pedal edema": "bilateral pedal edema",
    "abd": "abdominal"
}

def normalize_location(text):
    replacements = {
        " L ": " left ",
        " R ": " right ",
        " L.": " left",
        " R.": " right",
        "L ": "left ",
        "R ": "right ",
        " LUQ ": " left upper quadrant ",
        " LLQ ": " left lower quadrant ",
        " RUQ ": " right upper quadrant ",
        " RLQ ": " right lower quadrant "
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

# === 1. Text Preprocessing: Expand Abbreviations ===
def expand_abbreviations(text):
    text = text.lower()
    for abbr, full in abbreviation_map.items():
        text = text.replace(abbr, full)
    return text

# === 2. Keyword Extraction using Watsonx ===
def extract_keywords(text):
    prompt = f"""
You are a medical triage assistant. Given a patient's chief complaint, extract 1–5 symptom keywords that appear **explicitly** in the text.

Rules:
- Do NOT generate or guess symptoms not mentioned in the text.
- Fix common abbreviations only if they exist in the input.
- Output only a comma-separated list.

Chief complaint: {text}

Output:
"""
    try:
        response = model.generate_text(prompt=prompt)
        return response.strip()  # Just remove leading/trailing whitespace
    except Exception as e:
        print(f"❌ Watsonx processing failed: {e}")
        return ""

# === 3. Keyword Cleaning Function ===
def clean_keywords(raw):
    if not raw or raw.strip() == "":
        return ""
    
    raw = raw.lower().replace("，", ",").strip()
    tokens = [w.strip() for w in raw.split(",") if len(w.strip()) > 1]

    def remove_repeated_subwords(word):
        for base in ["abdominal", "pain", "distention", "vomiting", "nausea", "lethargy"]:
            if base in word:
                return base
        return word

    final = []
    for token in tokens:
        token = remove_repeated_subwords(token)
        token = re.sub(r"[^\w\s-]", "", token)
        if token not in final:
            final.append(token)

    return ", ".join(final[:5])

# === 4. Main processing function ===
def process_triage_data():
    print(f"📁 Input file: {input_path}")
    print(f"📁 Output file: {output_path}")
    
    # 检查输入文件是否存在
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return
    
    # 1. Read the existing progress or initialize
    if output_path.exists():
        df = pd.read_csv(output_path)
        print(f"🔁 Restore the processed progress files：{output_path}")
    else:
        df = pd.read_csv(input_path)
        df["complaint_keywords"] = ""
        print(f"📥 Load the original file：{input_path}")

    print(f"📊 Total records: {len(df)}")
    
    # 2. skip already processed records, process and save incrementally
    processed_count = 0
    for i in range(len(df)):
        if pd.notna(df.loc[i, "complaint_keywords"]) and df.loc[i, "complaint_keywords"].strip() != "":
            processed_count += 1
            continue # if not empty then skip

        complaint_raw = str(df.loc[i, "chiefcomplaint"])
        print(f"\n🔄 Processing record {i+1}/{len(df)}：{complaint_raw}")

        try:
            expanded_text = expand_abbreviations(complaint_raw)
            complaint_normalized = normalize_location(expanded_text)
            raw_keywords = extract_keywords(complaint_normalized)
            cleaned_keywords = clean_keywords(raw_keywords)

            df.loc[i, "complaint_keywords"] = cleaned_keywords
            print(f"✅ Extracted keywords (cleaned)：{cleaned_keywords}")

            # 每处理10条记录保存一次
            if (i + 1) % 10 == 0:
                df.to_csv(output_path, index=False)
                print(f"💾 Progress saved at record {i+1}")

        except Exception as e:
            print(f"❌ Error processing record {i+1}: {e}")
            df.loc[i, "complaint_keywords"] = ""
            continue

        time.sleep(1)  # 避免API限制

    # 最终保存
    df.to_csv(output_path, index=False)
    print(f"✅ Final file saved: {output_path}")
    print(f"📊 Processing completed. Total records: {len(df)}, Processed: {processed_count}")

# === 5. Main execution ===
if __name__ == "__main__":
    try:
        process_triage_data()
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        # === 6. Close model connection ===
        try:
            model.close_persistent_connection()
            print("🔌 Model connection closed")
        except:
            pass




