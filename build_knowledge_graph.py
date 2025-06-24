import time
import os
import random
import pickle
import argparse
from concurrent.futures import ThreadPoolExecutor

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

import pandas as pd
import networkx as nx
from dateutil.parser import isoparse

# ---------------------------
# Parameters and Setup
# ---------------------------
MAX_ITER = 3
PORTFOLIO_STOCKS = ["FPT", "SSI", "VCB", "VHM", "HPG", "GAS", "MSN", "MWG", "GVR", "VCG"]
PORTFOLIO_SECTOR = ["Công nghệ", "Chứng khoán", "Ngân hàng", "Bất động sản", "Vật liệu cơ bản", "Dịch vụ Hạ tầng", "Tiêu dùng cơ bản", "Bán lẻ", "Chế biến", "Công nghiệp"]
# Maximum retries and base delay for exponential backoff
MAX_RETRIES = 5
BASE_DELAY = 30

entity_extraction_template = PromptTemplate.from_template("""Bạn đang làm việc dưới bối cảnh phân tích kinh tế. 
Bạn được cho một hoặc nhiều bài báo, bao gồm tựa đề và mô tả ngắn gọn về bài báo đó, ngoài ra bạn có
thông tin về ngày xuất bản của bài báo, và loại chủ đề mà bài báo đang đề cập tới.

Hạn chế tạo mới một thực thể, chỉ tạo liên kết tới 5 thực thể. Luôn ưu tiên liên kết với các thực thể đã có: {existing_entities}

Bạn cần phân tích bài báo, đưa ra tên của những thực thể (ví dụ như cổ phiếu, ngành nghề, công ty, quốc gia, tỉnh thành...)
sẽ bị ảnh hưởng trực tiếp bởi thông tin của bài báo, theo hướng tích cực hoặc tiêu cực.

Với mỗi thực thể, ở phần Tên thực thể, hạn chế dùng dấu chấm, gạch ngang, dấu và &, dấu chấm phẩy ;. Và cần ghi thêm quốc gia, địa phương cụ thể và ngành nghề của nó (nếu có).
Tên chỉ nói tới một thực thể duy nhất. Phần Tên không được quá phức tạp, đơn giản nhất có thể.
Nếu thực thể nào thuộc danh mục cổ phiếu sau: {portfolio}, hãy ghi rõ tên cổ phiếu.
Ví dụ: SSI-Chứng khoán; Ngành công nghiệp Việt Nam; Người dùng Mỹ; Ngành thép Châu Á; Ngành du lịch Hạ Long, ...

Ghi nhớ, Hạn chế tạo mới một thực thể, chỉ tạo liên kết tới 5 thực thể. Luôn cố liên kết với các thực thể đã có.

Phần giải thích mỗi thực thể, bắt buộc đánh giá số liệu được ghi, nhiều hoặc ít, tăng hoặc giảm, gấp bao nhiêu lần, ...
Cần cố gắng liên kết với nhiều thực thể khác. Tuy nhiên không suy ngoài phạm vi bài báo. Không tự chèn số liệu ngoài bài báo.
Không dùng dấu hai chấm trong phần giải thích, chỉ dùng hai chấm : để tách giữa Tên thực thể và phần giải thích.
                                                          
Đưa ra theo định dạng sau:
[[POSITIVE]]
[Entity 1]: [Explanation]
...
[Entity N]: [Explanation]

[[NEGATIVE]]
[Entity A]: [Explanation]
..
[Entity Z]: [Explanation]
                                                          
Một ví dụ cho bài báo:

(BẮT ĐẦU VÍ DỤ)

Ngày đăng: 2025-04-07T22:51:00+07:00
Loại chủ đề: Kinh tế
Tựa đề: Nỗ lực hiện thực hóa mục tiêu thông tuyến cao tốc từ Cao Bằng đến Cà Mau 

Mô tả: Nhằm hoàn thành mục tiêu đến năm 2025 cả nước có trên 3.000 km đường cao tốc, Bộ Xây dựng, các địa phương và doanh nghiệp đang triển khai thi công 28 dự án/dự án thành phần với tổng chiều dài khoảng 1.188 km. 
Đến nay, tiến độ đa số các dự án bám sát kế hoạch, nhiều dự án đăng ký hoàn thành thông tuyến trong năm 2025. Có thể nói ngành giao thông vận tải đang cố gắng hết sức.

Danh sách thực thể sẽ bị ảnh hưởng:

[[POSITIVE]]
Bộ Xây dựng Việt Nam: Áp lực quản lý 28 dự án với tổng chiều dài 1188 km, nhằm hiện thực hóa mục tiêu đạt 3000 km cao tốc vào năm 2025. Số lượng dự án tăng gấp nhiều lần so với giai đoạn trước, đòi hỏi điều phối nguồn lực và kiểm soát tiến độ chặt chẽ hơn.
Chính quyền địa phương Việt Nam: Trực tiếp phối hợp triển khai các dự án tại từng tỉnh thành. Cần nâng cao năng lực quản lý và sử dụng ngân sách công hiệu quả để đảm bảo tiến độ thi công theo kế hoạch chung quốc gia.
Doanh nghiệp xây dựng Việt Nam: Được hưởng lợi trực tiếp khi nhận khối lượng hợp đồng thi công lớn. Doanh thu và năng lực thi công có thể tăng nhanh hơn so với các giai đoạn trước đây, nhờ nhu cầu đầu tư hạ tầng tăng mạnh.
Ngành giao thông vận tải Việt Nam: Cải thiện hạ tầng cao tốc giúp rút ngắn thời gian di chuyển liên vùng, từ đó nâng cao hiệu suất vận hành và giảm chi phí logistics trên toàn quốc.
Tỉnh Cao Bằng Việt Nam: Là điểm đầu của tuyến cao tốc quốc gia, đóng vai trò đầu mối kết nối vùng Đông Bắc. Hạ tầng mới giúp tăng kết nối, tạo cơ hội thu hút đầu tư và đẩy nhanh tốc độ phát triển kinh tế địa phương.
Tỉnh Cà Mau Việt Nam: Là điểm cuối của tuyến cao tốc, với hệ thống giao thông hiện đại giúp mở rộng thị trường du lịch và phát triển kinh tế vùng Đồng bằng sông Cửu Long. Tạo lợi thế cạnh tranh mới cho địa phương.

[[NEGATIVE]]
Bộ Xây dựng Việt Nam: Rủi ro chậm tiến độ và đội vốn nếu điều phối không hiệu quả do số lượng dự án tăng gấp nhiều lần.
Chính quyền địa phương Việt Nam: Có thể gặp khó khăn trong giải phóng mặt bằng và quản lý vốn đầu tư nếu năng lực tổ chức yếu.
Doanh nghiệp xây dựng Việt Nam: Thi công đồng loạt nhiều dự án có thể làm giãn mỏng năng lực nhân sự và máy móc tăng rủi ro chậm tiến độ hoặc giảm chất lượng.
Doanh nghiệp ngoài ngành xây dựng Việt Nam: Chịu tác động gián tiếp từ chi phí logistics tăng tạm thời hoặc thiếu hụt nguyên vật liệu.

(KẾT THÚC VÍ DỤ)

Ngày đăng: {date}
Loại chủ đề: {group}
Tựa đề: {title}

Mô tả: {description}


Danh sách thực thể sẽ bị ảnh hưởng:
""")

relation_extraction_template = PromptTemplate.from_template("""Bạn đang làm việc dưới bối cảnh phân tích kinh tế.                                                            
Hạn chế tạo mới một thực thể, chỉ được tạo mới tối đa 2 thực thể mới. Chỉ được liên kết tới 4 thực thể khác. Luôn ưu tiên liên kết với các thực thể đã có: {existing_entities}

Dựa trên tác động đến một thực thể, hãy liệt kê các thực thể sẽ bị ảnh hưởng tiêu cực và ảnh hưởng tích cực do hiệu ứng dây chuyền.
Hãy suy luận xem thực thể hiện tại này có thể ảnh hưởng tiếp đến những thực thể khác nào, theo hướng tích cực hoặc tiêu cực.
                                                            
Với mỗi thực thể, ở phần Tên thực thể, hạn chế dùng dấu chấm, gạch ngang, dấu và &, dấu chấm phẩy ;. Cần ghi thêm quốc gia, địa phương cụ thể và ngành nghề của nó (nếu có). 
Tên chỉ nói tới một thực thể duy nhất. Phần Tên không được quá phức tạp, đơn giản nhất có thể.
Nếu thực thể nào thuộc danh mục cổ phiếu sau: {portfolio}, hãy ghi rõ tên cổ phiếu.
Ví dụ: SSI-Chứng khoán; Ngành công nghiệp Việt Nam; Người dùng Mỹ; Ngành thép Châu Á; Ngành du lịch Hạ Long, ...

Ghi nhớ, Hạn chế tạo mới thực thể, chỉ được tạo mới tối đa 2 thực thể mới. Chỉ được liên kết tới 4 thực thể khác. Luôn cố liên kết với các thực thể đã có.

Phần giải thích mỗi thực thể, bắt buộc đánh giá số liệu được ghi, nhiều hoặc ít, tăng hoặc giảm, gấp bao nhiêu lần, ...
Cần cố gắng liên kết với nhiều thực thể khác. Tuy nhiên không suy ngoài phạm vi bài báo. Không tự chèn số liệu ngoài bài báo.
Không dùng dấu hai chấm trong phần giải thích, chỉ dùng hai chấm : để tách giữa Tên thực thể và phần giải thích.

Đưa ra theo định dạng sau:
[[POSITIVE]]
[Entity 1]: [Explanation]
...
[Entity N]: [Explanation]

[[NEGATIVE]]
[Entity A]: [Explanation]
..
[Entity Z]: [Explanation]

(BẮT ĐẦU VÍ DỤ)

Thực thể gốc: Bộ Xây dựng Việt Nam

Ảnh hưởng: Áp lực quản lý 28 dự án với tổng chiều dài 1188 km, nhằm hiện thực hóa mục tiêu đạt 3000 km cao tốc vào năm 2025. Số lượng dự án tăng gấp nhiều lần so với giai đoạn trước, đòi hỏi điều phối nguồn lực và kiểm soát tiến độ chặt chẽ hơn.

Danh sách thực thể sẽ bị ảnh hưởng bởi hiệu ứng dây chuyền:

[[POSITIVE]]
Doanh nghiệp xây dựng Việt Nam: Có cơ hội mở rộng hợp đồng thi công, tăng doanh thu nhờ số lượng dự án cao tốc lớn đang triển khai đồng loạt.
Người lao động Việt Nam: Có thêm nhiều cơ hội việc làm từ các dự án thi công trải dài khắp cả nước.

[[NEGATIVE]]
Bộ Giao thông Vận tải Việt Nam: Chịu áp lực phối hợp và giám sát hiệu quả giữa các bên liên quan, có nguy cơ bị chỉ trích nếu dự án chậm tiến độ.
Doanh nghiệp xây dựng Việt Nam: Có thể chịu áp lực tăng giá nguyên vật liệu và thiếu hụt nguồn cung do nhu cầu tăng đột biến.

(KẾT THÚC VÍ DỤ)

Thực thể gốc: {entities}

Ảnh hưởng: {description}

Danh sách thực thể sẽ bị ảnh hưởng bởi hiệu ứng dây chuyền:
""")

# Update the batch relation extraction template to better handle larger batches
batch_relation_extraction_template = PromptTemplate.from_template("""Bạn đang làm việc dưới bối cảnh phân tích kinh tế.
Hạn chế tạo mới thực thể, chỉ được tạo mới tối đa 2 thực thể mới cho mỗi thực thể gốc. Chỉ được liên kết tối đa 3 thực thể khác cho mỗi thực thể gốc. Luôn ưu tiên liên kết với các thực thể đã có: {existing_entities}

Dựa trên tác động đến các thực thể đầu vào, hãy phân tích hiệu ứng dây chuyền. 
Hãy suy luận xem mỗi thực thể hiện tại có thể ảnh hưởng tiếp đến những thực thể khác nào, theo hướng tích cực hoặc tiêu cực.

Với mỗi thực thể, ở phần Tên thực thể, hạn chế dùng dấu chấm, gạch ngang, dấu và &, dấu chấm phẩy ;. Cần ghi thêm quốc gia, địa phương cụ thể và ngành nghề của nó (nếu có).
Tên chỉ nói tới một thực thể duy nhất. Phần Tên không được quá phức tạp, đơn giản nhất có thể.
Nếu thực thể nào thuộc danh mục cổ phiếu sau: {portfolio}, hãy ghi rõ tên cổ phiếu.
Ví dụ: SSI-Chứng khoán; Ngành công nghiệp Việt Nam; Người dùng Mỹ; Ngành thép Châu Á; Ngành du lịch Hạ Long, ...

Phần giải thích mỗi thực thể, bắt buộc đánh giá số liệu được ghi, nhiều hoặc ít, tăng hoặc giảm, gấp bao nhiêu lần...
Cần cố gắng liên kết với nhiều thực thể khác. Tuy nhiên không suy ngoài phạm vi bài báo. Không tự chèn số liệu ngoài bài báo.
Không dùng dấu hai chấm trong phần giải thích, chỉ dùng hai chấm : để tách giữa Tên thực thể và phần giải thích.

Đưa ra theo định dạng sau cho mỗi thực thể nguồn:

[[SOURCE: Tên thực thể nguồn]]
[[IMPACT: POSITIVE/NEGATIVE]]

[[POSITIVE]]
[Thực thể ảnh hưởng 1]: [Giải thích]
[Thực thể ảnh hưởng 2]: [Giải thích]
[Thực thể ảnh hưởng 3]: [Giải thích]

[[NEGATIVE]]
[Thực thể ảnh hưởng A]: [Giải thích]
[Thực thể ảnh hưởng B]: [Giải thích]
[Thực thể ảnh hưởng C]: [Giải thích]

Bạn sẽ phân tích nhiều thực thể gốc một lúc. Với TỪNG thực thể, chỉ chọn CHÍNH XÁC 2-3 thực thể ảnh hưởng tích cực và 2-3 thực thể ảnh hưởng tiêu cực quan trọng nhất.

LƯU Ý: Có thể có RẤT NHIỀU thực thể đầu vào, hãy phân tích CẨN THẬN từng thực thể để không bỏ sót. Không được tạo thêm thực thể gốc.
                                                                  
(BẮT ĐẦU VÍ DỤ)
Danh sách thực thể nguồn:

Thực thể gốc: Bộ Xây dựng Việt Nam

Ảnh hưởng: NEGATIVE, Áp lực quản lý 28 dự án với tổng chiều dài 1188 km, nhằm hiện thực hóa mục tiêu đạt 3000 km cao tốc vào năm 2025. Số lượng dự án tăng gấp nhiều lần so với giai đoạn trước, đòi hỏi điều phối nguồn lực và kiểm soát tiến độ chặt chẽ hơn.

---

Danh sách thực thể sẽ bị ảnh hưởng bởi hiệu ứng dây chuyền:

[[SOURCE: Bộ Xây dựng Việt Nam]]
[[IMPACT: NEGATIVE]]

[[POSITIVE]]
Doanh nghiệp xây dựng Việt Nam: Có cơ hội mở rộng hợp đồng thi công, tăng doanh thu nhờ số lượng dự án cao tốc lớn đang triển khai đồng loạt.
Người lao động Việt Nam: Có thêm nhiều cơ hội việc làm từ các dự án thi công trải dài khắp cả nước.

[[NEGATIVE]]
Bộ Giao thông Vận tải Việt Nam: Chịu áp lực phối hợp và giám sát hiệu quả giữa các bên liên quan, có nguy cơ bị chỉ trích nếu dự án chậm tiến độ.
Doanh nghiệp xây dựng Việt Nam: Có thể chịu áp lực tăng giá nguyên vật liệu và thiếu hụt nguồn cung do nhu cầu tăng đột biến.

(KẾT THÚC VÍ DỤ)

Danh sách thực thể nguồn:

{input_entities}

Danh sách thực thể sẽ bị ảnh hưởng bởi hiệu ứng dây chuyền:
""")

time.sleep(1)
chain_entity = entity_extraction_template | model
time.sleep(1)
chain_relation = relation_extraction_template | model
time.sleep(1)

# Create chain for batch processing
chain_batch_relation = batch_relation_extraction_template | model
time.sleep(1)

def read_news_data(csv_path="cleaned_posts.csv"):
    """
    Reads the CSV file and converts the ISO date strings using dateutil.
    """
    df = pd.read_csv(csv_path)
    df['parsed_date'] = df['date'].apply(isoparse)
    return df
def parse_entity_response(response):
    """
    Parses the response from the entity extraction prompt.
    """
    if response is None:
        print("Response is None")
        return {"POSITIVE": [], "NEGATIVE": []}
        
    sections = {"POSITIVE": [], "NEGATIVE": []}
    current_section = None
    str_resp = response.content
    
    for line in str(str_resp).splitlines():
        line = line.strip()
        if not line:
            continue
        if "[[POSITIVE]]" in line.upper():
            current_section = "POSITIVE"
            continue
        if "[[NEGATIVE]]" in line.upper():
            current_section = "NEGATIVE"
            continue
        if current_section and ':' in line:
            entity = line.split(":", 1)[0].strip()
            # Skip invalid entities
            if not entity or "không có thực thể nào" in entity.lower():
                continue
            # content = all line except entity
            content = line.split(entity, 1)[-1].strip(':').strip()
            sections[current_section].append((entity, content))

    return sections

def merge_entity(entity, canonical_set):
    """
    Returns the canonical version of the entity if already present (case-insensitive),
    otherwise adds and returns the new entity.
    """
    normalized_entity = str(entity).strip('[').strip(']').strip(' ').lower()
    for exist in canonical_set:
        if exist.lower() == normalized_entity:
            return exist
    canonical_set.add(normalized_entity)
    return normalized_entity

def add_edge(G, source, target, impact, timestamp):
    """
    Adds an edge to the graph if it does not already exist.
    """
    if not G.has_edge(source, target):
        G.add_edge(source, target, impact=impact, timestamp=timestamp)

def graph_to_tuples(G):
    """
    Converts the graph to a string of tuples (date, source, impact, target).
    """
    tuples = []
    for u, v, data in G.edges(data=True):
        # Extract the date as a string, handling different timestamp formats
        timestamp = data.get("timestamp")
        if timestamp is None:
            # Skip edges without timestamps
            continue
            
        try:
            # Handle different timestamp formats
            if isinstance(timestamp, pd.Timestamp):
                date_str = timestamp.date().isoformat()
            elif hasattr(timestamp, "date"):  # datetime object
                date_str = timestamp.date().isoformat()
            elif isinstance(timestamp, (int, float)):  # Unix timestamp
                date_str = pd.Timestamp(timestamp, unit='s').date().isoformat()
            else:  # Try to parse as string
                parsed_date = pd.to_datetime(timestamp)
                date_str = parsed_date.date().isoformat()
                
            # Skip invalid entities in output tuples
            if "không có thực thể nào" in str(u).lower() or "không có thực thể nào" in str(v).lower():
                continue
                
            tuples.append(f"({date_str}, {u}, {data.get('impact')} TO, {v})")
        except Exception as e:
            print(f"Error processing edge ({u}, {v}): {e}, timestamp: {timestamp}, type: {type(timestamp)}")
            continue

    # Sort tuples by ascending order of date

    return "\n".join(sorted(tuples))

def graph_entities_to_str(G):
    """
    Get string of "entity" type nodes from graph for context
    """
    entities = [node for node in G.nodes() if G.nodes[node].get("type") == "entity"]
    graph_str = ", ".join(entities)
    return graph_str[:20000] if len(graph_str) > 20000 else graph_str

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
            
def process_entity_relationships(entity_info, G: nx.DiGraph, canonical_entities, portfolio, portfolio_str_full, article_timestamp):
    """
    Process a single entity's relationships using the relation extraction chain
    Returns a list of new entities to process
    """
    entity, impact, content = entity_info
    next_entities = []
    
    # Create prompt for relation extraction
    prompt_rel = {
        "entities": entity,
        "portfolio": portfolio_str_full,
        "description": impact + ", " + content,
        "existing_entities": graph_entities_to_str(G)
    }
    
    # Get relationships with retry logic
    response_rel = invoke_chain_with_retry(chain_relation, prompt_rel)
    time.sleep(1)  # Rate limiting
    
    if response_rel is None:
        return []
        
    # Process the response
    rel_dict = parse_entity_response(response_rel)
    
    # Add edges and collect new entities
    for new_ent, content_new in rel_dict.get(impact, []):
        canon_new = merge_entity(new_ent, canonical_entities)
        node_type = "stock" if any(str(canon_new).lower().find(stock.lower()) != -1 for stock in portfolio) else "entity"
        
        # Add node if it doesn't exist
        if not G.has_node(canon_new):
            G.add_node(canon_new, type=node_type, timestamp=article_timestamp)
            
        # Add edge
        add_edge(G, entity, canon_new, impact, article_timestamp)
        
        # Add to frontier if it's an entity (not a stock)
        if node_type == "entity":
            next_entities.append((canon_new, impact, content_new))
            
    return next_entities

def parse_batch_entity_response(response):
    """
    Parses the response from the batch entity relation extraction prompt.
    Returns a list of tuples (source_entity, impact, target_entity, content)
    """
    if response is None:
        print("Response is None")
        return []
        
    results = []
    current_source = None
    current_impact = None
    current_section = None
    
    str_resp = str(response.content)
    lines = str_resp.splitlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for source entity marker
        if line.startswith("[[SOURCE:") or "[[SOURCE:" in line:
            source_text = line.replace("[[SOURCE:", "").replace("]]", "").strip()
            if source_text and "không có thực thể nào" not in source_text.lower():
                current_source = source_text
            else:
                current_source = None  # Skip invalid source entities
            continue
            
        # Check for impact marker
        if line.startswith("[[IMPACT:") or "[[IMPACT:" in line:
            impact_str = line.replace("[[IMPACT:", "").replace("]]", "").strip()
            current_impact = impact_str.upper()
            continue
            
        # Check for positive/negative section markers
        if "[[POSITIVE]]" in line.upper():
            current_section = "POSITIVE"
            continue
            
        if "[[NEGATIVE]]" in line.upper():
            current_section = "NEGATIVE"
            continue
            
        # Process entity and explanation if we're in a valid context
        if current_source and current_section and ':' in line:
            # Extract entity name and content
            try:
                entity, *content_parts = line.split(":", 1)
                entity = entity.strip().strip('[]')  # Remove any potential brackets
                
                # Skip invalid target entities
                if not entity or "không có thực thể nào" in entity.lower():
                    continue
                    
                if entity and content_parts:
                    content = content_parts[0].strip()
                    # Use the current_impact if specified, otherwise use the current section
                    actual_impact = current_impact if current_impact else current_section
                    results.append((current_source, actual_impact, entity, content))
            except Exception as e:
                print(f"Error parsing line: {line}. Error: {e}")
                continue
    
    if not results:
        print("Warning: No relationships were parsed from the response")
        print(f"Response content: {str_resp[:500]}...")
        
    return results


def batch_process_entity_relationships(entity_batch, G, canonical_entities, portfolio, portfolio_str_full, article_timestamp):
    """
    Process multiple entities in a single API call
    Returns a list of new entities to process
    """
    if not entity_batch:
        return []
    
    # Maximum retries for batch processing    
    max_batch_retries = 2
    batch_retry_count = 0
    relationships = []
    
    while batch_retry_count < max_batch_retries:
        # Format the input entities for the prompt more explicitly
        input_entities_text = ""
        for entity, impact, content in entity_batch:
            # Format similar to the relation_extraction_template structure
            input_entities_text += f"Thực thể gốc: {entity}\n\nẢnh hưởng: {impact}, {content}\n\n---\n\n"
        
        # Create prompt for batch relation extraction
        prompt_batch = {
            "input_entities": input_entities_text,
            "portfolio": portfolio_str_full,
            "existing_entities": graph_entities_to_str(G)
        }
        
        # Get relationships with retry logic
        response = invoke_chain_with_retry(chain_batch_relation, prompt_batch)
        time.sleep(1)  # Rate limiting
        
        if response is None:
            return []
        
        # Parse the response to get all relationships
        relationships = parse_batch_entity_response(response)
        
        # Check if we got any relationships
        if len(relationships) > 0:
            break
            
        batch_retry_count += 1
        print(f"Batch processing returned 0 relationships. Retry {batch_retry_count}/{max_batch_retries}")
        time.sleep(BASE_DELAY)  # Wait before retrying
    
    # Debug print
    print(f"Processing batch with {len(relationships)} relationships using timestamp: {article_timestamp}")
    
    # Process the relationships to update the graph and collect new entities
    next_entities = []
    
    for source, impact, target, content in relationships:
        # Skip invalid relationships
        if "không có thực thể nào" in source.lower() or "không có thực thể nào" in target.lower():
            continue
            
        canon_source = source  # Source entity should already be canonical
        canon_target = merge_entity(target, canonical_entities)
        
        # Determine if target is a stock
        node_type = "stock" if any(str(canon_target).lower().find(stock.lower()) != -1 for stock in portfolio) else "entity"
        
        # Add node if it doesn't exist
        if not G.has_node(canon_target):
            G.add_node(canon_target, type=node_type, timestamp=article_timestamp)
            
        # Add edge from source to target, ensuring we use the article timestamp
        add_edge(G, canon_source, canon_target, impact, article_timestamp)
        
        # Add to frontier if it's an entity (not a stock)
        if node_type == "entity":
            next_entities.append((canon_target, impact, content))
    
    return next_entities
# ---------------------------
# Knowledge Graph Construction
# ---------------------------
def process_article(idx, row, G, canonical_entities, portfolio, portfolio_sector, max_frontier_size=15):
    """
    Process a single article to extract entities and build relationships
    Thread-safe function to be used with multithreading
    """
    # Build portfolio string
    portfolio_str_full = ", ".join([f"{stock}-{sector}" for stock, sector in zip(portfolio, portfolio_sector)])
    
    article_node = f"Article_{idx}: {row['title']}"
    # article_timestamp = row['parsed_date']  # Store article timestamp for later use
    article_timestamp = row['date'] 
    
    # Thread-safe add node to graph
    if not G.has_node(article_node):
        G.add_node(article_node, type="article", timestamp=article_timestamp)
    
    # Phase 1: Extract initial entities
    max_entity_retries = 3
    entity_retry_count = 0
    entities_dict = {"POSITIVE": [], "NEGATIVE": []}
    
    while entity_retry_count < max_entity_retries:
        prompt_text = {
            "portfolio": portfolio_str_full,
            "date": row['date'],
            "group": row['group'],
            "title": row['title'],
            "description": row['description'],
            "existing_entities": graph_entities_to_str(G)
        }
        
        response_text = invoke_chain_with_retry(chain_entity, prompt_text)
        time.sleep(1)  # Rate limiting
        
        if response_text is None:
            print(f"Skipping article {idx} due to API errors")
            return 0, 0  # Return zeros for new nodes/edges counts
        
        entities_dict = parse_entity_response(response_text)
        
        # Check if we got any entities
        total_entities = len(entities_dict.get("POSITIVE", [])) + len(entities_dict.get("NEGATIVE", []))
        if total_entities > 0:
            break
            
        entity_retry_count += 1
        print(f"Article {idx} returned 0 entities. Retry {entity_retry_count}/{max_entity_retries}")
        time.sleep(BASE_DELAY)  # Wait before retrying
    
    if entity_retry_count == max_entity_retries and total_entities == 0:
        print(f"Failed to extract entities from article {idx} after {max_entity_retries} attempts")
        return 0, 0
    
    # Process initial entities
    initial_entities = []
    new_nodes = 0
    new_edges = 0
    
    for impact in ["POSITIVE", "NEGATIVE"]:
        for ent, content in entities_dict.get(impact, []):
            # Skip invalid entities
            if not ent or "không có thực thể nào" in ent.lower():
                continue
                
            canon_ent = None
            
            # Thread-safe check for existing entity
            normalized_entity = str(ent).strip('[').strip(']').strip(' ').lower()
            existing = False
            for exist in canonical_entities:
                if exist.lower() == normalized_entity:
                    canon_ent = exist
                    existing = True
                    break
            
            if not existing:
                # Thread-safe update of canonical_entities
                canonical_entities.add(normalized_entity)
                canon_ent = normalized_entity
                
            # Determine node type
            node_type = "stock" if any(str(canon_ent).lower().find(stock.lower()) != -1 for stock in portfolio) else "entity"
            
            # Thread-safe add node to graph
            if not G.has_node(canon_ent):
                G.add_node(canon_ent, type=node_type, timestamp=article_timestamp)
                new_nodes += 1
            
            if node_type == "entity":
                initial_entities.append((canon_ent, impact, content))
                
            # Thread-safe add edge
            if not G.has_edge(article_node, canon_ent):
                G.add_edge(article_node, canon_ent, impact=impact, timestamp=article_timestamp)
                new_edges += 1
    
    print(f"Index: {idx}, initial entities: {len(initial_entities)}")

    # Phase 2: Iterative expansion with single batch per iteration
    frontier = initial_entities
    iter_count = 0
    max_iterations = MAX_ITER  # Maximum frontier expansion iterations
    
    while frontier and iter_count < max_iterations:
        # Limit frontier size to prevent model context overload
        if len(frontier) > max_frontier_size:
            frontier = random.sample(frontier, max_frontier_size)
            print(f"Index: {idx}, limited frontier to {max_frontier_size} entities")
            
        # Process the entire frontier in a single API call
        next_frontier = batch_process_entity_relationships(
            frontier, G, canonical_entities, portfolio, portfolio_str_full, article_timestamp
        )
        
        # Remove duplicates by entity name
        frontier = list({ent: (ent, imp, txt) for ent, imp, txt in next_frontier}.values())
        
        # Limit frontier size again if needed
        if len(frontier) > max_frontier_size:
            frontier = random.sample(frontier, max_frontier_size)
        
        print(f"Index: {idx}, next frontier: {len(frontier)}")
        iter_count += 1
    
    return new_nodes, new_edges

def build_knowledge_graph(df, portfolio, portfolio_sector, skip=0, use_threading=True, max_frontier_size=10, max_workers=5, 
                     graph_checkpoint_path=None, canonical_checkpoint_path=None, output_graph_dir="knowledge_graphs"):
    """
    Builds a knowledge graph from articles with optimized batch processing.
    Processes multiple articles in parallel using threads.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing articles to process (expected to be summarized news)
    portfolio : list
        List of portfolio stock symbols
    portfolio_sector : list
        List of portfolio sectors
    skip : int, optional
        Number of articles to skip from the beginning of the DataFrame.
    use_threading : bool, optional
        Whether to use multithreading for processing articles.
    max_frontier_size : int, optional
        Maximum size of frontier to limit context for relation extraction.
    max_workers : int, optional
        Maximum number of worker threads to use if use_threading is True.
    graph_checkpoint_path : str, optional
        Path to knowledge graph checkpoint file to load (if continuing).
    canonical_checkpoint_path : str, optional
        Path to canonical entities checkpoint file to load (if continuing).
    output_graph_dir : str, optional
        Directory to save the final graph and checkpoints.
    
    Returns:
    --------
    nx.DiGraph
        The constructed knowledge graph
    """
    # Initialize graph and canonical entities
    G = nx.DiGraph()
    canonical_entities = set()
    
    # Create output directory for graphs
    os.makedirs(output_graph_dir, exist_ok=True)

    # Load from checkpoints if provided
    if graph_checkpoint_path and os.path.exists(graph_checkpoint_path):
        print(f"Loading knowledge graph from checkpoint: {graph_checkpoint_path}")
        try:
            with open(graph_checkpoint_path, "rb") as f:
                G = pickle.load(f)
            print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        except Exception as e:
            print(f"Error loading graph checkpoint: {e}")
            print("Starting with empty graph.")
    
    if canonical_checkpoint_path and os.path.exists(canonical_checkpoint_path):
        print(f"Loading canonical entities from checkpoint: {canonical_checkpoint_path}")
        try:
            with open(canonical_checkpoint_path, "rb") as f:
                canonical_entities = pickle.load(f)
            print(f"Loaded {len(canonical_entities)} canonical entities")
        except Exception as e:
            print(f"Error loading canonical entities checkpoint: {e}")
            print("Starting with empty canonical entities set.")
    
    # Filter dataframe to skip articles if needed
    if skip > 0:
        print(f"Skipping first {skip} articles from input DataFrame.")
        df = df.iloc[skip:].copy() # Use .copy() to avoid SettingWithCopyWarning
        df.reset_index(drop=True, inplace=True) # Reset index after skipping
    
    # Define chunk size for processing (for saving checkpoints)
    # This chunk_size is for saving progress, not for LLM batching
    chunk_size = 10 
    
    total_articles_to_process = len(df)
    print(f"Total articles to process for graph building: {total_articles_to_process}")

    # Process in chunks to allow for saving checkpoints
    for chunk_start_idx in range(0, total_articles_to_process, chunk_size):
        chunk_end_idx = min(chunk_start_idx + chunk_size, total_articles_to_process)
        chunk_df = df.iloc[chunk_start_idx:chunk_end_idx]
        
        print(f"Processing articles {skip + chunk_start_idx} to {skip + chunk_end_idx - 1} for graph building.")
        
        # Process articles (either in parallel or sequentially)
        if use_threading and len(chunk_df) > 1:
            articles_processed_in_chunk = 0
            with ThreadPoolExecutor(max_workers=min(max_workers, len(chunk_df))) as executor:
                futures = [
                    executor.submit(
                        process_article, 
                        row.name, # Pass original index (row.name) as article ID
                        row, 
                        G, 
                        canonical_entities, 
                        portfolio, 
                        portfolio_sector, 
                        max_frontier_size
                    ) 
                    for idx, row in chunk_df.iterrows()
                ]
                
                for future in futures:
                    future.result() # Wait for results (errors will propagate)
                    articles_processed_in_chunk += 1
                    
            print(f"Processed {articles_processed_in_chunk} articles in parallel for this chunk.")
        else:
            for idx, row in chunk_df.iterrows():
                process_article(
                    row.name, # Pass original index (row.name) as article ID
                    row, 
                    G, 
                    canonical_entities, 
                    portfolio, 
                    portfolio_sector, 
                    max_frontier_size
                )
        
        # Save checkpoint after each chunk
        # Use a more descriptive checkpoint name including the range of articles processed
        checkpoint_name = f"knowledge_graph_checkpoint_{skip+chunk_start_idx}_to_{skip+chunk_end_idx-1}.pkl"
        canonical_name = f"canonical_set_checkpoint_{skip+chunk_start_idx}_to_{skip+chunk_end_idx-1}.pkl"
        
        graph_save_path = os.path.join(output_graph_dir, checkpoint_name)
        canonical_save_path = os.path.join(output_graph_dir, canonical_name)

        with open(graph_save_path, "wb") as f:
            pickle.dump(G, f)
        with open(canonical_save_path, "wb") as f:
            pickle.dump(canonical_entities, f)
            
        print(f"Saved checkpoint for articles {skip+chunk_start_idx} to {skip+chunk_end_idx-1}.")
        print(f"Graph now has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # Determine the final output file names based on the processed range
    # It's better to get the actual range of articles processed by the current df_to_process
    # rather than assuming the original df's min/max date, especially with --skip
    if not df.empty and 'date' in df.columns:
        first_article_date_processed = df['date'].min().strftime('%Y%m%d')
        last_article_date_processed = df['date'].max().strftime('%Y%m%d')
    else:
        first_article_date_processed = 'unknown_start'
        last_article_date_processed = 'unknown_end'
        
    final_graph_name = f"knowledge_graph_final_{first_article_date_processed}_to_{last_article_date_processed}.pkl"
    final_canonical_name = f"canonical_set_final_{first_article_date_processed}_to_{last_article_date_processed}.pkl"

    final_graph_path = os.path.join(output_graph_dir, final_graph_name)
    final_canonical_path = os.path.join(output_graph_dir, final_canonical_name)

    with open(final_graph_path, "wb") as f:
        pickle.dump(G, f)
    with open(final_canonical_path, "wb") as f:
        pickle.dump(canonical_entities, f)
    
    print(f"Completed graph building with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print(f"Final graph saved to: {final_graph_path}")
    print(f"Final canonical set saved to: {final_canonical_path}")
    return G

# ---------------------------
# Main Execution for Building Knowledge Graph
# ---------------------------
def main():
    """
    Main entry point for building the knowledge graph.
    """
    parser = argparse.ArgumentParser(description="Build Knowledge Graph from summarized news articles.")
    parser.add_argument("--news_from", type=int, default=0, help="Starting index for summarized news articles (0-based).")
    parser.add_argument("--news_to", type=int, default=9400, help="Ending index for summarized news articles (inclusive).")
    parser.add_argument("--input_summarized_file", type=str, default="summarized_articles.csv", 
                        help="Path to the input summarized news file.")
    parser.add_argument("--load_graph", action="store_true", help="Load existing knowledge graph from checkpoint to continue building.")
    parser.add_argument("--graph_checkpoint", type=str, default=None, help="Path to knowledge graph checkpoint file to load (if --load_graph is used).")
    parser.add_argument("--canonical_checkpoint", type=str, default=None, help="Path to canonical entities checkpoint file to load (if --load_graph is used).")
    parser.add_argument("--max_frontier_size", type=int, default=10, help="Maximum number of entities to process in a single batch for relation extraction.")
    parser.add_argument("--no_threading", action="store_true", help="Disable multithreading for article processing.")
    parser.add_argument("--max_workers", type=int, default=5, help="Maximum number of worker threads to use if multithreading is enabled.")
    parser.add_argument("--skip", type=int, default=0, help="Number of articles to skip from the beginning of the input summarized DataFrame.")
    parser.add_argument("--output_graph_dir", type=str, default="knowledge_graphs", help="Directory to save the final graph and checkpoints.")

    args = parser.parse_args()

    # Read summarized news data
    print(f"Reading summarized news data from {args.input_summarized_file}...")
    df_summarized = read_news_data(args.input_summarized_file)
    if df_summarized.empty:
        print("No summarized news data to process. Exiting.")
        return
    
    # Filter and preprocess data according to range
    # Ensure 'date' or 'parsed_date' is available and sorted for chronological processing
    if 'date' in df_summarized.columns:
        df_summarized['date'] = pd.to_datetime(df_summarized['date'], errors='coerce')
        df_summarized = df_summarized.sort_values(by='date', ascending=True).reset_index(drop=True)
    elif 'parsed_date' in df_summarized.columns:
        df_summarized['parsed_date'] = pd.to_datetime(df_summarized['parsed_date'], errors='coerce')
        df_summarized = df_summarized.sort_values(by='parsed_date', ascending=True).reset_index(drop=True)
    else:
        print("Warning: Neither 'date' nor 'parsed_date' column found for sorting. Data might not be in chronological order.")

    # Apply news_from and news_to after initial sorting and parsing
    df_to_process = df_summarized.iloc[args.news_from : args.news_to + 1].copy()
    df_to_process.fillna("", inplace=True) # Fill NaNs for consistency

    if df_to_process.empty:
        print("No articles found in the specified range after preprocessing. Exiting.")
        return

    print(f"Building knowledge graph for {len(df_to_process)} articles...")
    
    # Call the build_knowledge_graph function
    G = build_knowledge_graph(
        df=df_to_process,
        portfolio=PORTFOLIO_STOCKS, # Use global constants
        portfolio_sector=PORTFOLIO_SECTOR, # Use global constants
        skip=args.skip,
        use_threading=not args.no_threading,
        max_frontier_size=args.max_frontier_size,
        max_workers=args.max_workers,
        graph_checkpoint_path=args.graph_checkpoint if args.load_graph else None,
        canonical_checkpoint_path=args.canonical_checkpoint if args.load_graph else None,
        output_graph_dir=args.output_graph_dir
    )
    
    print("\nKnowledge graph building process complete!")

if __name__ == "__main__":
    main()
