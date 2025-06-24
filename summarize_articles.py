import re
import time
import os
import pandas as pd
from dateutil.parser import isoparse
import argparse
import glob

# Các thư viện LLM
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))
# Initialize models globally once
model = ChatGoogleGenerativeAI(model = "gemini-2.0-flash-lite", temperature= 0.02)
time.sleep(1)
model_more_temperature = ChatGoogleGenerativeAI(model = "gemini-2.0-flash-lite", temperature= 0.1)
time.sleep(1)
model2 = ChatGoogleGenerativeAI(model = "gemini-2.5-pro-exp-03-25", temperature= 0.1)
time.sleep(1)

# ---------------------------
# Parameters and Setup
# ---------------------------
PORTFOLIO_STOCKS = ["FPT", "SSI", "VCB", "VHM", "HPG", "GAS", "MSN", "MWG", "GVR", "VCG"]
PORTFOLIO_SECTOR = ["Công nghệ", "Chứng khoán", "Ngân hàng", "Bất động sản", "Vật liệu cơ bản", 
                    "Dịch vụ Hạ tầng", "Tiêu dùng cơ bản", "Bán lẻ", "Chế biến", "Công nghiệp"]
# Maximum retries and base delay for exponential backoff
MAX_RETRIES = 5
BASE_DELAY = 30
# ---------------------------
# Prompt Templates
# ---------------------------
news_summarize_template = PromptTemplate.from_template("""
Bạn là một chuyên gia tóm tắt tin tức kinh tế thị trường. 
Dữ liệu đầu vào gồm các bài báo trong ngày, mỗi bài báo có tiêu đề, mô tả và chủ đề. 
Nhiệm vụ của bạn là tóm tắt, kết hợp và cô đọng nội dung của các tin tức đó thành 20 tin chính, 
sao cho mỗi tin tóm tắt phản ánh đầy đủ những điểm quan trọng của bài báo gốc, một cách ngắn gọn và súc tích, trên cùng một dòng.

Danh sách bài báo:
{articles_list}

Hãy xuất đầu ra theo định dạng sau:
[Chủ đề bài báo 1]: [Tiêu đề] | [Nội dung tóm tắt]
[Chủ đề bài báo 2]: [Tiêu đề] | [Nội dung tóm tắt]
...
[Chủ đề bài báo N]: [Tiêu đề] | [Nội dung tóm tắt]

Ví dụ:
(BĂT ĐẦU VÍ DỤ)
Chủ đề Thị trường:
Một trong những "cá mập" lớn nhất trên TTCK liên tục hút thêm vài nghìn tỷ vốn ngoại trong 4 tháng qua, mua hàng chục triệu cổ phiếu ngân hàng, HPG, DXG...","Đi ngược với xu hướng rút ròng của phần lớn các quỹ trên thị trường, VFMVSF ngày càng thu hẹp khoảng cách với quỹ có quy mô tài sản đứng đầu thị trường là VN DIAMOND ETF. Cả 2 đều thuộc quản lý của Dragon Capital.

Chủ đề Thế giới:
Chính phủ Cuba áp giá trần tạm thời với nông sản,Cuba áp giá trần tạm thời đối với các sản phẩm nông sản thiết yếu nhằm kiềm chế lạm phát và khủng hoảng kinh tế nghiêm trọng.

Chủ đề Thế giới:
Các ngân hàng Mỹ lo ngại về nguy cơ suy thoái nền kinh tế," Các ngân hàng Mỹ đang lo ngại về nguy cơ suy thoái kinh tế, theo hãng tin Bloomberg. Nỗi lo này xuất phát từ các mức thuế quan mới và chỉ số kinh tế không mấy khả quan.

Chủ đề Thế giới:
Chiến lược cạnh tranh của châu Âu trong ngành công nghiệp xanh,Báo cáo về khả năng cạnh tranh cho rằng châu Âu nên tập trung đầu tư vào các công nghệ mới nổi, chẳng hạn như hydro và pin, thay vì cố gắng cạnh tranh với Trung Quốc trong sản xuất tấm pin Mặt trời.

Chủ đề Thế giới:
Cơ hội cho Ấn Độ," Báo Deccan Herald vừa đăng bài phân tích của chuyên gia kinh tế Ajit Ranade, đánh giá về tác động và cơ hội đối với Ấn Độ từ những quyết định mới nhất của Tổng thống Mỹ Donald Trump về thuế quan."

Chủ đề Hàng hóa:
Giá vàng hôm nay (9-3): Quay đầu giảm,"Giá vàng hôm nay (9-3): Giá vàng trong nước hôm nay giảm nhẹ, vàng miếng SJC nhiều thương hiệu giảm 200.000 đồng/lượng."

Chủ đề Hàng hóa:
Giá xăng dầu hôm nay (9-3): Tuần giảm mạnh, có thời điểm bỏ mốc 70 USD/thùng,"Giá xăng dầu thế giới lập hat-trick giảm tuần. Đáng chú ý là trong tuần, giá dầu có thời điểm trượt xa mốc 70 USD/thùng."

Chủ đề Tài chính:
Tỷ giá USD hôm nay (9-3): Đồng USD lao dốc kỷ lục,"Tỷ giá USD hôm nay: Rạng sáng 9-3, Ngân hàng Nhà nước công bố tỷ giá trung tâm của đồng Việt Nam với USD tăng tuần 12 đồng, hiện ở mức 24.738 đồng."

Chủ đề Hàng hóa:
Bản tin nông sản hôm nay (9-3): Giá hồ tiêu ổn định mức cao,Bản tin nông sản hôm nay (9-3) ghi nhận giá hồ tiêu ổn định mức cao; giá cà phê tiếp tục giảm nhẹ.

Chủ đề Tài chính:
Standard Chartered điều chỉnh dự báo tỷ giá USD/VND, nâng mức giữa năm lên 26.000 đồng/USD và cuối năm 2025 lên 25.700 đồng/USD, phản ánh sức ép từ biến động kinh tế toàn cầu và khu vực.

Chủ đề Tài chính:
Ngân hàng ACB Việt Nam dự kiến góp thêm 1.000 tỷ đồng để tăng vốn điều lệ cho ACBS lên mức 11.000 tỷ đồng.

Chủ đề Tài chính:
Bất ổn thuế quan tiếp tục thúc đẩy nhu cầu trú ẩn an toàn bằng vàng, Giá vàng tăng nhẹ tại châu Á do bất ổn về chính sách thuế quan của Mỹ tiếp tục thúc đẩy nhu cầu trú ẩn an toàn trước những lo ngại về nguy cơ kinh tế Mỹ suy yếu và lạm phát gia tăng.

Chủ đề Bất động sản:
Cả nước dự kiến giảm từ 10.500 đơn vị cấp xã xuống khoảng 2.500 đơn vị,"Theo Phó thủ tướng Nguyễn Hòa Bình, số lượng đơn vị hành chính cấp xã trên toàn quốc dự kiến sẽ giảm từ hơn 10.500 xuống còn 2.500 sau khi sáp nhập.


Tóm tắt tin tức trong ngày:
Tài chính: "Cá mập" lớn trên TTCK liên tục hút thêm vài nghìn tỷ vốn ngoại 4 tháng qua | Một trong những "cá mập" lớn trên TTCK liên tục hút thêm vài nghìn tỷ vốn ngoại trong 4 tháng qua và mua hàng chục triệu cổ phiếu ngân hàng, HPG, DXG. Đồng thời, VFMVSF ngày càng thu hẹp khoảng cách với VN DIAMOND ETF - cả hai đều do Dragon Capital quản lý, phản ánh xu hướng đầu tư mạnh mẽ của các quỹ chiến lược.

Thế giới: Nguy cơ suy thoái đe dọa Mỹ, và cơ hội mở ra cho Ấn Độ | Các ngân hàng Mỹ đang lo ngại nguy cơ suy thoái kinh tế khi đối mặt với các mức thuế quan mới và chỉ số kinh tế không khả quan, theo thông tin từ Bloomberg, cho thấy sự bất ổn có thể lan rộng trong nền kinh tế Hoa Kỳ, nhưng là cơ hội cho Ấn Độ khi các quyết định mới của Tổng thống Mỹ Donald Trump về thuế quan có thể tạo ra những tác động tích cực đối với nền kinh tế Ấn Độ,

Thế giới: Cuba áp giá trần nông sản để hạ nhiệt lạm phát | Chính phủ Cuba áp giá trần tạm thời cho các sản phẩm nông sản thiết yếu nhằm kiềm chế lạm phát và ngăn chặn khủng hoảng kinh tế, tạo ra một biện pháp tạm thời để ổn định thị trường nội địa trong bối cảnh kinh tế khó khăn.

Thế giới: EU khuyến nghị đầu tư vào công nghệ xanh thay vì cạnh tranh pin mặt trời với Trung Quốc | Báo cáo cạnh tranh của châu Âu khuyến nghị nên tập trung đầu tư vào các công nghệ mới nổi như hydro và pin thay vì cố gắng cạnh tranh trực tiếp với Trung Quốc trong sản xuất tấm pin mặt trời, nhằm xây dựng nền công nghiệp xanh bền vững.

Hàng hóa: Giá vàng và xăng dầu biến động mạnh trước sức ép thuế quan | Giá vàng trong nước giảm nhẹ trong phiên giao dịch ngày 9-3, với vàng miếng SJC nhiều thương hiệu giảm khoảng 200.000 đồng/lượng, do xuất hiện nhu cầu trú ẩn an toàn bằng vàng tại Châu Á trước sức ép thuế quan. Giá xăng dầu thế giới có đợt giảm mạnh trong tuần qua, trượt xuống dưới mốc 70 USD/thùng, cho thấy sự biến động mạnh mẽ của thị trường kim loại quý.

Tài chính: Tỷ giá USD/VND biến động mạnh, dự báo tăng đến cuối năm 2025 | Tỷ giá USD lao dốc kỷ lục trong phiên giao dịch sáng 9-3 khi Ngân hàng Nhà nước tăng 12 đồng, USD/VND được định giá ở mức 24.738 đồng. Tuy nhiên Standard Chartered đã điều chỉnh dự báo tỷ giá USD/VND giữa năm lên 26.000 đồng/USD và cuối năm 2025 lên 25.700 đồng/USD, phản ánh sức ép từ biến động kinh tế toàn cầu và khu vực.

Hàng hóa: Giá nông sản Việt Nam ổn định | Giá nông sản ổn định, trong đó hồ tiêu giữ mức cao, giá cà phê giảm nhẹ.

Bất động sản: Sáp nhập hành chính, tinh giản hệ thống quản lý | Theo Phó thủ tướng Nguyễn Hòa Bình, sau quá trình sáp nhập hành chính, số lượng đơn vị cấp xã trên cả nước dự kiến sẽ giảm từ hơn 10.500 xuống còn khoảng 2.500, tạo nên cơ cấu hành chính hợp lý hơn và ảnh hưởng đến chính sách quản lý địa phương.

(KẾT THÚC VÍ DỤ)

Lưu ý:
- Ưu tiên những bài báo có chung chủ đề, hoặc có liên quan, gộp vào thành một bài báo duy nhất.
- Mỗi bài tóm tắt phải gồm những điểm chính về tiêu đề và mô tả. Mỗi bài báo đều được ghi trên một dòng. Chú ý trong bài tóm tắt không được có dấu xuống dòng hay dấu hai chấm :.
- Tổng số tin sau khi tóm tắt là 20 tin.
- Luôn chứa toàn bộ số liệu được ghi trong bài báo.
- Không được tự tạo thêm tin tức mới. Không dùng số liệu ngoài bài báo.
- Nếu bài báo nào thuộc danh mục cổ phiếu sau: {portfolio}, hãy ghi rõ tên cổ phiếu, và dành riêng một tin cho nó.
- Không ưu tiên những bài báo chỉ nói về một công ty cụ thể ở Việt Nam, và công ty đó không thuộc danh mục cổ phiếu.

Tóm tắt tin tức trong ngày:
""")

# Create separate chains for each prompt
chain_summary = news_summarize_template | model
time.sleep(1)  # Delay sau khi khởi tạo chain
chain_summary_more_temperature = news_summarize_template | model_more_temperature
time.sleep(1)
chain_summary_pro = news_summarize_template | model2
time.sleep(1)

def read_news_data(csv_path="cleaned_posts.csv"):
    """
    Reads the CSV file and converts the ISO date strings using dateutil.
    """
    df = pd.read_csv(csv_path)
    df['parsed_date'] = df['date'].apply(isoparse)
    return df

def build_article_text(row):
    """
    Build the text for an article based on its columns.
    """
    return (f"Ngày đăng: {row['date']}\n"
            f"Loại chủ đề: {row['group']}\n"
            f"Tựa đề: {row['title']}\n\n"
            f"Mô tả: {row['description']}")

def combine_articles(df: pd.DataFrame) -> str:
    """
    Kết hợp các bài báo trong ngày thành một chuỗi văn bản.
    """
    articles = [f"Chủ đề {row['group']}:\n{row['title']}, {row['description']}\n" 
                for idx, row in df.iterrows()]
    return "\n".join(articles)

def parse_summary_response(response, date_str, starting_index=1): # Đổi 'date' thành 'date_str' để nhất quán với cách gọi
    """
    Phân tích đầu ra của LLM theo định dạng '[Chủ đề]: Nội dung tóm tắt'.
    """
    response_text = str(response.content)
    # print(f"\n--- DEBUG: Raw LLM Response for date {date_str} ---")
    # print(response_text)
    # print(f"--- END DEBUG: Raw LLM Response ---\n")

    # Regex mới để khớp '[Chủ đề]: Nội dung tóm tắt'
    # Group 1: Chủ đề (ví dụ: "Chủ đề Thế giới")
    # Group 2: Nội dung tóm tắt
    pattern = r'^\[(.+?)\]:\s*(.+)$' 
    lines = response_text.splitlines()
    articles = []

    # print(f"DEBUG: Lines after splitlines for date {date_str}:")
    # for idx, line in enumerate(lines):
    #     print(f"  Line {idx}: '{line.strip()}'")
    # print(f"--- END DEBUG: Lines ---\n")

    current_idx_for_date = 0 # Counter for articles within this date to keep postID unique per run for this date
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        match = re.match(pattern, line)
        
        if match:
            topic_group, summary_content = match.groups()
            
            # Xóa các ký tự [ ] xung quanh group nếu có (đã được regex xử lý với (.+?))
            cleaned_group = topic_group.strip()
            
            # Tạo title từ group
            generated_title = f"Tóm tắt tin tức {cleaned_group}"
            
            # print(f"DEBUG: Match found for line '{line}':")
            # print(f"  Group: '{cleaned_group}'")
            # print(f"  Summary Content: '{summary_content.strip()}'")
            
            articles.append({
                "postID": starting_index + current_idx_for_date, # Dùng starting_index truyền vào + counter riêng cho ngày
                "title": generated_title,
                "description": summary_content.strip(), # Nội dung tóm tắt chính là description
                "date": date_str, # Giữ nguyên định dạng date truyền vào
                "group": cleaned_group # Nhóm chủ đề
            })
            current_idx_for_date += 1 # Tăng index cho bài tóm tắt tiếp theo trong cùng ngày
        else:
            print(f"DEBUG: No match found for line: '{line}'")
    
    # print(f"\n--- DEBUG: Final articles list for date {date_str} ---")
    # print(articles)
    # print(f"--- END DEBUG: Final articles list ---\n")

    return pd.DataFrame(articles)

def invoke_chain_with_retry(chain, prompt, max_retries=MAX_RETRIES, base_delay=BASE_DELAY):
    """
    Invokes a chain with exponential backoff retry logic
    """
    retry_count = 0
    while True:
        try:
            response = chain.invoke(prompt)
            return response
        except Exception as e:
            retry_count += 1
            if retry_count > max_retries:
                print(f"Maximum retries reached. Error: {e}")
                return None
                
            delay = base_delay * (2 ** (retry_count - 1))
            print(f"Error: {e}. Retrying in {delay} seconds... (Attempt {retry_count}/{max_retries})")
            time.sleep(delay)

# --- MAIN SUMMARIZATION LOGIC ---
def make_summarized_news(df: pd.DataFrame, batch_size=5):
    """
    Creates a summarized version of the news articles with batched processing
    """
    # Extract date part only for grouping
    df['only_date'] = pd.to_datetime(df['parsed_date']).dt.date
    df_summarized = pd.DataFrame(columns=["postID", "title", "description", "date", "group"])
    
    # Build portfolio string once
    portfolio_str = ", ".join(PORTFOLIO_STOCKS)
    
    # Group by date for processing
    date_groups = df.groupby("only_date")
    total_groups = len(date_groups)
    
    print(f"Processing {total_groups} dates for summarization")
    
    # Process in batches to allow checkpoints
    current_idx = 1
    for batch_idx, batch in enumerate(range(0, total_groups, batch_size)):
        batch_summaries = []
        
        # Process each date in this batch
        for i, (date, group) in enumerate(list(date_groups)[batch:batch+batch_size]):
            # check if date is before 22/03/2025
            # if date <= pd.to_datetime("2025-01-31").date():
            #     print(f"Skipping date {date}")
            #     continue
            # Combine articles for this date
            combined_articles = combine_articles(group)
            
            # Create prompt for summary
            summary_prompt = {
                "articles_list": combined_articles,
                "portfolio": portfolio_str,
            }
            
            print(f"Processing date {batch*batch_size + i + 1}/{total_groups}: {date}, articles: {len(group)}")
            
            # Get summary with retry logic for empty results
            max_summary_retries = 10
            summary_retry_count = 0
            summary_df = pd.DataFrame(columns=["postID", "title", "description", "date", "group"])
            
            while summary_retry_count < max_summary_retries:
                # Get summary with retry logic
                if summary_retry_count < 3:
                    summary_response = invoke_chain_with_retry(chain_summary, summary_prompt)
                    time.sleep(2)
                elif summary_retry_count < 5:
                    summary_response = invoke_chain_with_retry(chain_summary_more_temperature, summary_prompt)
                    time.sleep(2)
                else:
                    summary_response = invoke_chain_with_retry(chain_summary_pro, summary_prompt)
                    time.sleep(2)
                
                time.sleep(1)  # Rate limiting
                
                if summary_response is None:
                    print(f"Failed to summarize articles for date {date}")
                    break
                    
                # Format date for output
                date_str = f"{date}T16:00:00+07:00"
                
                # Parse summary response
                summary_df = parse_summary_response(summary_response, date_str, starting_index=current_idx)
                
                # Check if we got any articles
                if len(summary_df) > 0:
                    break
                    
                summary_retry_count += 1
                print(f"Summary for date {date} returned 0 articles. Retry {summary_retry_count}/{max_summary_retries}")
                time.sleep(BASE_DELAY)  # Wait before retrying
            
            # Only proceed if we have articles
            if len(summary_df) > 0:
                current_idx += len(summary_df)
                # Add to batch results
                batch_summaries.append(summary_df)
            else:
                print(f"⚠ Failed to get any summarized articles for date {date} after {max_summary_retries} attempts")
        
        # Combine all summaries in this batch
        if batch_summaries:
            batch_df = pd.concat(batch_summaries, ignore_index=True)
            df_summarized = pd.concat([df_summarized, batch_df], ignore_index=True)
            
            # Save checkpoint after each batch
            checkpoint_file = f"summarized_articles_checkpoint_{batch_idx}.csv"
            df_summarized.to_csv(checkpoint_file, index=False)
            print(f"Saved checkpoint with {len(df_summarized)} articles to {checkpoint_file}")
    
    # Save final result
    df_summarized.to_csv("summarized_articles.csv", index=False)
    print(f"Saved {len(df_summarized)} summarized articles to CSV")
    
    return df_summarized



def get_next_batch_index(checkpoint_dir="checkpoints"):
    import re
    existing_files = os.listdir(checkpoint_dir) if os.path.exists(checkpoint_dir) else []
    pattern = re.compile(r"summarized_articles_checkpoint_batch_(\d+)\.csv")
    indices = [int(match.group(1)) for f in existing_files if (match := pattern.match(f))]
    return max(indices) + 1 if indices else 0


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--input_file", type=str, default="cleaned_posts.csv")
    parser.add_argument("--output_file", type=str, default="summarized_articles.csv")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    df_raw = read_news_data(args.input_file)
    if df_raw.empty:
        print("No raw news data.")
        return

    if 'group' in df_raw.columns:
        df_raw = df_raw[df_raw['group'] != "Doanh nghiệp"]

    if 'parsed_date' in df_raw.columns:
        df_raw['parsed_date'] = pd.to_datetime(df_raw['parsed_date'], errors='coerce')
    df_raw.fillna("", inplace=True)
    df_raw = df_raw.sort_values(by='parsed_date').reset_index(drop=True)

    batch_number = get_next_batch_index(args.checkpoint_dir)
    start_idx = batch_number * args.batch_size
    end_idx = start_idx + args.batch_size

    df_batch = df_raw.iloc[start_idx:end_idx].copy()
    if df_batch.empty:
        print("No articles found for this batch range. Exiting.")
        return

    print(f"Starting summarization for batch {batch_number} (articles {start_idx + 1} to {end_idx})...")

    current_idx = 1
    if os.path.exists(args.output_file):
        existing_df = pd.read_csv(args.output_file)
        if 'postID' in existing_df.columns and pd.api.types.is_numeric_dtype(existing_df['postID']):
            current_idx = int(existing_df['postID'].max()) + 1

    summary_df = make_summarized_news(
        df_batch,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        batch_number=batch_number,
        current_idx=current_idx
    )

    if summary_df.empty:
        print("No summaries were generated in this batch.")
        return

    if os.path.exists(args.output_file):
        existing_df = pd.read_csv(args.output_file)
        combined = pd.concat([existing_df, summary_df], ignore_index=True)
        combined.drop_duplicates(subset=['postID'], inplace=True)
        combined.to_csv(args.output_file, index=False)
    else:
        summary_df.to_csv(args.output_file, index=False)

    print(f"Summarization complete. Output saved to {args.output_file}")


if __name__ == "__main__":
    main()
