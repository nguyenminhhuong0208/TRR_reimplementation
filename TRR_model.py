import math
import re
import time
import os
import random
import pickle
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

import tiktoken
tokenize = tiktoken.get_encoding('cl100k_base')

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

# Initialize models globally once
model = ChatGoogleGenerativeAI(model = "gemini-2.0-flash-lite", temperature= 0.02)
time.sleep(1)
model_more_temperature = ChatGoogleGenerativeAI(model = "gemini-2.0-flash-lite", temperature= 0.1)
time.sleep(1)
model2 = ChatGoogleGenerativeAI(model = "gemini-2.5-pro-preview-05-06", temperature= 0.1)
time.sleep(1)

import pandas as pd
from dateutil.parser import isoparse
import networkx as nx
import json
import argparse

# ---------------------------
# Parameters and Setup
# ---------------------------
MAX_ITER = 3
PORTFOLIO_STOCKS = ["FPT", "SSI", "VCB", "VHM", "HPG", "GAS", "MSN", "MWG", "GVR", "VCG"]
PORTFOLIO_SECTOR = ["Công nghệ", "Chứng khoán", "Ngân hàng", "Bất động sản", "Vật liệu cơ bản", "Dịch vụ Hạ tầng", "Tiêu dùng cơ bản", "Bán lẻ", "Chế biến", "Công nghiệp"]
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

# reasoning_template = PromptTemplate.from_template("""
# Cho danh mục cổ phiếu sau:
# {portfolio}

# Cho một đồ thị tri thức được biểu diễn dưới dạng quan hệ thời gian và tác động, được biểu diễn dưới dạng các tuple 
# (thời gian, nguồn, hành động, đích), cần dự đoán liệu danh mục cổ phiếu đã nêu có "sập giá" trong ngày tiếp theo hay không.
                                                  
# Dưới đây là đồ thị tri thức được biểu diễn dưới dạng các tuple (thời gian, nguồn, hành động, đích):
# {tuples}

# Lưu ý rằng "sập giá" ở đây biểu thị cho một đợt giảm giá của cổ phiếu RẤT MẠNH (lên tới 5%). Vì vậy cần có phân tích đa chiều, từ nhiều phía khác nhau.
# Giải thích theo từng bước cho lựa chọn đó.

# Sử dụng lý luận của riêng bạn trên đồ thị được đưa, và không đề cập đến các sự kiện khác ngoài đồ thị có trong quá khứ.

# Dự đoán dưới định dạng sau:
# Explanation: [Lý do]
# Crash: [Yes/No]

# Lý do sập giá cho chuỗi sự kiện trên: """)
# Số bài báo cố định cho mỗi ngày dự đoán
ARTICLES_PER_DATE = 20

# Sửa reasoning_template để thêm prediction_date
reasoning_template = PromptTemplate.from_template("""
Dự đoán liệu danh mục cổ phiếu sau có sập giá vào ngày {prediction_date} hay không:

Danh mục cổ phiếu:
{portfolio}

Cho một đồ thị tri thức được biểu diễn dưới dạng quan hệ thời gian và tác động, được biểu diễn dưới dạng các tuple 
(thời gian, nguồn, hành động, đích), cần dự đoán liệu danh mục cổ phiếu đã nêu có sập giá vào ngày {prediction_date} hay không.
                                                  
Dưới đây là đồ thị tri thức được biểu diễn dưới dạng các tuple (thời gian, nguồn, hành động, đích):
{tuples}

Lưu ý rằng "sập giá" ở đây biểu thị cho một đợt giảm giá của cổ phiếu RẤT MẠNH (lên tới 5%). Vì vậy cần có phân tích đa chiều, từ nhiều phía khác nhau.
Giải thích theo từng bước cho lựa chọn đó, và nêu rõ rằng dự đoán là cho ngày {prediction_date}.

Sử dụng lý luận của riêng bạn trên đồ thị được đưa, và không đề cập đến các sự kiện khác ngoài đồ thị có trong quá khứ.

Dự đoán dưới định dạng sau:
Explanation: [Lý do]
Crash: [Yes/No]

Lý do sập giá cho chuỗi sự kiện vào ngày {prediction_date}: 
""")

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

# Create separate chains for each prompt
chain_summary = news_summarize_template | model
time.sleep(1)  # Delay sau khi khởi tạo chain
chain_summary_more_temperature = news_summarize_template | model_more_temperature
time.sleep(1)
chain_summary_pro = news_summarize_template | model2
time.sleep(1)
chain_entity = entity_extraction_template | model
time.sleep(1)
chain_relation = relation_extraction_template | model
time.sleep(1)
chain_reasoning = reasoning_template | model_more_temperature
time.sleep(1)

# Create chain for batch processing
chain_batch_relation = batch_relation_extraction_template | model
time.sleep(1)


# ---------------------------
# Helper Functions
# ---------------------------
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

def parse_summary_response(response, date, starting_index=1):
    """
    Phân tích đầu ra của LLM theo định dạng.
    """
    pattern = r'^(.+?):\s*(.+?)\s*\|\s*(.+)$'
    response_text = str(response.content)
    lines = response_text.splitlines()
    articles = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = re.match(pattern, line)
        if match:
            group, title, description = match.groups()
            articles.append({
                "postID": starting_index,
                "title": title.strip('[').strip(']').strip(),
                "description": description.strip('[').strip(']').strip(),
                "date": date,
                "group": group.strip('[').strip(']').strip()
            })
            starting_index += 1
    
    return pd.DataFrame(articles)

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

def update_edge_decay_weights(G: nx.DiGraph, current_time=None, lambda_decay=1) -> nx.DiGraph:
    """
    Updates each edge's weight based on exponential decay from its timestamp.
    """
    if current_time is None:
        print("\n\nCurrent time is None, using current timestamp\n")
        current_time = pd.Timestamp.now()
    
    decay_weights = dict()
    for u, v, data in G.edges(data=True):
        u_timestamp = G.nodes[u].get("timestamp")
        v_timestamp = G.nodes[v].get("timestamp")
        # test print
        delta = (int)((v_timestamp - u_timestamp).total_seconds()/86400)
        R_decay = math.exp(-delta / (lambda_decay))
        data["weight"] = 0.0 if delta < 0 else R_decay
        decay_weights[delta] = data["weight"]
    
    # Print decay weights after sort by key
    print(f"Decay weights: {sorted(decay_weights.items())}")
    return G

# def attention_phase(G, current_time, lambda_decay, q=6):
#     """
#     Uses TPPR (Temporal Personalized PageRank) to find important entities and their connections.
#     Filters out future edges and applies time-decayed weights.
#     """

#     import pandas as pd
#     import networkx as nx
#     import math

#     # 🔧 First, create a copy of the graph to avoid modifying the original
#     G_filtered = G.copy()

#     # 🔧 Filter out future edges
#     edges_to_remove = []
#     for u, v, data in G_filtered.edges(data=True):
#         edge_time = data.get("timestamp")

#         if edge_time is None:
#             continue

#         try:
#             if isinstance(edge_time, pd.Timestamp) or hasattr(edge_time, "timestamp"):
#                 edge_timestamp = edge_time.timestamp()
#             elif isinstance(edge_time, (int, float)):
#                 edge_timestamp = edge_time
#             else:
#                 edge_timestamp = pd.Timestamp(edge_time).timestamp()

#             current_timestamp = current_time if isinstance(current_time, (int, float)) else pd.Timestamp(current_time).timestamp()
#             if edge_timestamp > current_timestamp:
#                 edges_to_remove.append((u, v))
#         except Exception as e:
#             print(f"Warning: Could not process timestamp {edge_time} for edge ({u}, {v}): {e}")
#             continue

#     for u, v in edges_to_remove:
#         G_filtered.remove_edge(u, v)

#     print(f"Filtered out {len(edges_to_remove)} future edges from graph for TPPR calculation")

#     # 🔧 Apply time decay to remaining edges
#     def update_edge_decay_weights(G, current_time, lambda_decay):
#         for u, v, data in G.edges(data=True):
#             ts = data.get("timestamp")
#             if ts is None:
#                 continue
#             try:
#                 ts_val = ts if isinstance(ts, (int, float)) else pd.Timestamp(ts).timestamp()
#                 cur_time_val = current_time if isinstance(current_time, (int, float)) else pd.Timestamp(current_time).timestamp()
#                 decay = math.exp(-lambda_decay * (cur_time_val - ts_val))
#                 G[u][v]['weight'] = G[u][v].get('weight', 1.0) * decay
#             except:
#                 continue
#         return G

#     G_filtered = update_edge_decay_weights(G_filtered, current_time, lambda_decay)  # 🔧

#     # 🔧 Temporal Personalized PageRank (TPPR)
#     def temporal_pagerank(G, alpha=0.85, max_iter=100, tol=1.0e-6):
#         nodes = list(G.nodes())
#         N = len(nodes)
#         rank = dict.fromkeys(nodes, 1.0 / N)
#         for _ in range(max_iter):
#             prev_rank = rank.copy()
#             for n in nodes:
#                 rank[n] = (1.0 - alpha) / N
#             for u, v in G.edges():
#                 if G.out_degree(u) > 0:
#                     rank[v] += alpha * (prev_rank[u] * G[u][v].get("weight", 1.0)) / G.out_degree(u)
#             diff = sum(abs(rank[n] - prev_rank[n]) for n in nodes)
#             if diff < tol:
#                 break
#         return rank

#     pr_scores = temporal_pagerank(G_filtered)  # 🔧

#     # Filter to entity or stock nodes
#     filtered_scores = {node: score for node, score in pr_scores.items()
#                        if G_filtered.nodes[node].get("type") in ["entity", "stock"]}

#     top_nodes = sorted(filtered_scores, key=filtered_scores.get, reverse=True)[:q]

#     for node, score in sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)[:q]:
#         print(f"{node}: {score}")

#     selected_nodes = set(top_nodes)
#     for node in top_nodes:
#         if node in G_filtered:
#             selected_nodes.update(G_filtered.predecessors(node))
#             selected_nodes.update(G_filtered.successors(node))

#     sub_G = G_filtered.subgraph(selected_nodes).copy()

#     print(f"Created subgraph with {sub_G.number_of_nodes()} nodes and {sub_G.number_of_edges()} edges")
#     print(f"Top 10 nodes by number of incoming edges: {sorted(sub_G.in_degree(), key=lambda x: x[1], reverse=True)[:10]}")

#     return sub_G
def apply_tppr_decay_weights(G, current_time, lambda_decay):
    """
    Applies time decay to edge weights based on the TPPR formula.
    
    Parameters:
    -----------
    G : nx.DiGraph
        The input graph (G_temporal)
    current_time : float or datetime
        The current timestamp for decay calculation
    lambda_decay : float
        Decay factor for edge weights
    
    Returns:
    --------
    nx.DiGraph
        Graph with updated edge weights
    """
    for u, v, data in G.edges(data=True):
        ts = data.get("timestamp")
        if ts is None:
            continue
        try:
            ts_val = ts if isinstance(ts, (int, float)) else pd.Timestamp(ts).timestamp()
            cur_time_val = current_time if isinstance(current_time, (int, float)) else pd.Timestamp(current_time).timestamp()
            decay = math.exp(-lambda_decay * (cur_time_val - ts_val))
            G[u][v]['weight'] = G[u][v].get('weight', 1.0) * decay
        except Exception as e:
            print(f"Warning: Could not apply decay for edge ({u}, {v}): {e}")
            continue
    return G
def attention_phase(G, current_time, lambda_decay, q=6):
    """
    Uses Temporal Personalized PageRank (TPPR) to find important entities and their connections.
    Creates a filtered copy that only uses edges dated before or on the prediction date.
    Applies time-decayed weights and personalizes ranking based on portfolio stocks and sectors.
    
    Parameters:
    - G: NetworkX DiGraph, the temporal-relational graph (G_temporal)
    - current_time: Timestamp or datetime, the current time for filtering future edges
    - lambda_decay: Float, decay rate for memory retention (default=1.0 as per paper)
    - q: Int, number of top entities to select (default=6 as per paper)
    
    Returns:
    - sub_G: NetworkX DiGraph, the subgraph (G_TRR) containing top-q entities and their neighbors
    """
    #  Create a copy of the graph to avoid modifying the original
    G_filtered = G.copy()
    
    #  Filter out future edges
    edges_to_remove = []
    for u, v, data in G_filtered.edges(data=True):
        edge_time = data.get("timestamp")
        
        # Skip edges without timestamps
        if edge_time is None:
            continue
            
        # Convert to timestamp for comparison
        try:
            if isinstance(edge_time, pd.Timestamp) or hasattr(edge_time, "timestamp"):
                edge_timestamp = edge_time.timestamp()
            elif isinstance(edge_time, (int, float)):
                edge_timestamp = edge_time
            else:
                edge_timestamp = pd.Timestamp(edge_time).timestamp()
            
            current_timestamp = current_time if isinstance(current_time, (int, float)) else pd.Timestamp(current_time).timestamp()
            if edge_timestamp > current_timestamp:
                edges_to_remove.append((u, v))
        except Exception as e:
            print(f"Warning: Could not process timestamp {edge_time} for edge ({u}, {v}): {e}")
            continue
            
    # Remove future edges
    for u, v in edges_to_remove:
        G_filtered.remove_edge(u, v)
        
    print(f"Filtered out {len(edges_to_remove)} future edges from graph for TPPR calculation")
    
    # Step 1: Apply update_edge_decay_weights to create G_temporal
    print("Creating G_temporal by applying temporal decay weights...")
    G_temporal = update_edge_decay_weights(G_filtered, current_time=current_time, lambda_decay=lambda_decay)
    
    # Step 2: Apply TPPR decay weights to G_temporal
    print("Applying TPPR decay weights to G_temporal...")
    G_temporal = apply_tppr_decay_weights(G_temporal, current_time, lambda_decay)

    #  Create personalization vector for TPPR
    personalization = {}
    total_nodes = len(G_filtered.nodes())
    for node in G_filtered.nodes():
        node_type = G_filtered.nodes[node].get("type")
        node_sector = G_filtered.nodes[node].get("sector", "")
        
        # Prioritize portfolio stocks
        if node in PORTFOLIO_STOCKS:
            personalization[node] = 0.1  # High priority for portfolio stocks
        # Prioritize entities in portfolio sectors
        elif node_sector in PORTFOLIO_SECTOR:
            personalization[node] = 0.05  # Medium priority for related sectors
        # Default for other nodes
        else:
            personalization[node] = 1.0 / total_nodes  # Low priority for others
    
    #  Compute Temporal Personalized PageRank (TPPR) scores
    pr_scores = nx.pagerank(G_filtered, alpha=0.85, personalization=personalization, weight="weight")
    
    # Filter to entity or stock nodes
    filtered_scores = {node: score for node, score in pr_scores.items()
                      if G_filtered.nodes[node].get("type") in ["entity", "stock"]}
    
    #  Get top q nodes
    top_nodes = sorted(filtered_scores, key=filtered_scores.get, reverse=True)[:q]
    
    # Print top nodes and their scores
    for node, score in sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)[:q]:
        print(f"{node}: {score}")
    
    #  Include their immediate neighbors
    selected_nodes = set(top_nodes)
    for node in top_nodes:
        if node in G_filtered:
            selected_nodes.update(G_filtered.predecessors(node))
            selected_nodes.update(G_filtered.successors(node))
    
    #  Create the subgraph (G_TRR)
    sub_G = G_filtered.subgraph(selected_nodes).copy()
    
    print(f"Created subgraph with {sub_G.number_of_nodes()} nodes and {sub_G.number_of_edges()} edges")
    print(f"Top 10 nodes by number of incoming edges: {sorted(sub_G.in_degree(), key=lambda x: x[1], reverse=True)[:10]}")
    
    return sub_G

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
                     graph_checkpoint=None, canonical_checkpoint=None):
    """
    Builds a knowledge graph from articles with optimized batch processing.
    Processes multiple articles in parallel using threads.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing articles to process
    portfolio : list
        List of portfolio stock symbols
    portfolio_sector : list
        List of portfolio sectors
    skip : int, optional
        Number of articles to skip
    use_threading : bool, optional
        Whether to use multithreading for processing
    max_frontier_size : int, optional
        Maximum size of frontier to prevent model context overload
    max_workers : int, optional
        Maximum number of worker threads to use
    graph_checkpoint : str, optional
        Path to knowledge graph checkpoint file to load
    canonical_checkpoint : str, optional
        Path to canonical entities checkpoint file to load
    
    Returns:
    --------
    nx.DiGraph
        The constructed knowledge graph
    """
    # Initialize graph and canonical entities
    G = nx.DiGraph()
    canonical_entities = set()
    
    # Load from checkpoints if provided
    if graph_checkpoint and os.path.exists(graph_checkpoint):
        print(f"Loading knowledge graph from checkpoint: {graph_checkpoint}")
        try:
            with open(graph_checkpoint, "rb") as f:
                G = pickle.load(f)
            print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        except Exception as e:
            print(f"Error loading graph checkpoint: {e}")
            print("Starting with empty graph")
    
    if canonical_checkpoint and os.path.exists(canonical_checkpoint):
        print(f"Loading canonical entities from checkpoint: {canonical_checkpoint}")
        try:
            with open(canonical_checkpoint, "rb") as f:
                canonical_entities = pickle.load(f)
            print(f"Loaded {len(canonical_entities)} canonical entities")
        except Exception as e:
            print(f"Error loading canonical entities checkpoint: {e}")
            print("Starting with empty canonical entities set")
    
    # Filter dataframe to skip articles if needed
    if skip > 0:
        print(f"Skipping first {skip} articles")
        df = df.iloc[skip:]
    
    # Define chunk size for processing (for checkpoints)
    chunk_size = 10
    
    # Process in chunks to allow for saving checkpoints
    for chunk_start in range(0, len(df), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(df))
        chunk_df = df.iloc[chunk_start:chunk_end]
        
        print(f"Processing articles {skip+chunk_start} to {skip+chunk_end-1}")
        
        # Process articles (either in parallel or sequentially)
        if use_threading and len(chunk_df) > 1:
            # Use ThreadPoolExecutor for parallel processing
            articles_processed = 0
            with ThreadPoolExecutor(max_workers=min(max_workers, len(chunk_df))) as executor:
                # Submit tasks
                futures = [
                    executor.submit(
                        process_article, 
                        idx, 
                        row, 
                        G, 
                        canonical_entities, 
                        portfolio, 
                        portfolio_sector, 
                        max_frontier_size
                    ) 
                    for idx, row in chunk_df.iterrows()
                ]
                
                # Process results as they complete
                for future in futures:
                    new_nodes, new_edges = future.result()
                    articles_processed += 1
                    
            print(f"Processed {articles_processed} articles in parallel")
        else:
            # Process sequentially
            for idx, row in chunk_df.iterrows():
                new_nodes, new_edges = process_article(
                    idx, 
                    row, 
                    G, 
                    canonical_entities, 
                    portfolio, 
                    portfolio_sector, 
                    max_frontier_size
                )
        
        # Save checkpoint after each chunk
        checkpoint_file = f"knowledge_graph_p3_checkpoint_{skip+chunk_end}.pkl"
        with open(checkpoint_file, "wb") as f:
            pickle.dump(G, f)
        
        canonical_set_file = f"canonical_set_checkpoint_{skip+chunk_end}.pkl"
        with open(canonical_set_file, "wb") as f:
            pickle.dump(canonical_entities, f)
            
        print(f"Saved checkpoint after processing {skip+chunk_end} articles")
        print(f"Graph now has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Save final graph
    with open("knowledge_graph_p3.pkl", "wb") as f:
        pickle.dump(G, f)
    with open("canonical_set.pkl", "wb") as f:
        pickle.dump(canonical_entities, f)
    
    print(f"Completed graph building with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

# ---------------------------
# Final Reasoning
# ---------------------------
def final_reasoning(G, portfolio, portfolio_sector, prediction_date):
    """
    Chuyển đồ thị thành chuỗi tuple và chạy chuỗi suy luận cuối cùng, truyền ngày dự đoán.
    
    Parameters:
    -----------
    G : nx.DiGraph
        Đồ thị con G_TRR
    portfolio : list
        Danh sách cổ phiếu trong danh mục
    portfolio_sector : list
        Danh sách lĩnh vực của danh mục
    prediction_date : str
        Ngày dự đoán ở định dạng ISO (ví dụ: '2025-03-26T01:00:00+07:00')
    
    Returns:
    --------
    Response từ LLM
    """
    tuples_str = graph_to_tuples(G)
    
    with open("tuples.txt", "w", encoding="utf-8") as f:
        f.write(tuples_str)
    
    portfolio_str_full = ", ".join([f"{stock}-{sector}" for stock, sector in zip(portfolio, portfolio_sector)])

    print("\nTuple đầu vào cho suy luận:")
    print(f"Số cạnh: {G.number_of_edges()}")
    
    reasoning_prompt = {
        "tuples": tuples_str,
        "portfolio": portfolio_str_full,
        "prediction_date": prediction_date
    }
    
    response = invoke_chain_with_retry(chain_reasoning, reasoning_prompt, max_retries=MAX_RETRIES*2)
    time.sleep(2)
    return response

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

def trr(df, prediction_date, load_saved_graph=False, lambda_decay=1.0, q=6, max_frontier_size=10, 
        use_threading=True, max_workers=5, skip=0, graph_checkpoint=None, canonical_checkpoint=None):
    """
    Main TRR function to build knowledge graph and make predictions
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing summarized articles
    prediction_date : str
        Date for which to make predictions
    load_saved_graph : bool, optional
        Whether to load an existing graph or build a new one
    lambda_decay : float, optional
        Decay factor for edge weights
    q : int, optional
        Number of top-ranked entities to include in subgraph
    max_frontier_size : int, optional
        Maximum number of entities to process in a single batch
    use_threading : bool, optional
        Whether to use multithreading for processing
    max_workers : int, optional
        Maximum number of worker threads to use
    skip : int, optional
        Number of articles to skip in processing
    graph_checkpoint : str, optional
        Path to knowledge graph checkpoint file to load
    canonical_checkpoint : str, optional
        Path to canonical entities checkpoint file to load
    
    Returns:
    --------
    The prediction result
    """
    try:
        pred_ts = isoparse(str(prediction_date)).timestamp()
        print(f"Timestamp dự đoán: {pred_ts}")
    except Exception as e:
        print(f"Lỗi khi phân tích ngày dự đoán: {e}. Sử dụng prediction_date như hiện tại.")
        pred_ts = prediction_date
    
    if load_saved_graph:
        print("Tải đồ thị tri thức hiện có...")
        try:
            with open(graph_checkpoint or "knowledge_graph_p3_fixed_0227-0407.pkl", "rb") as f:
                G = pickle.load(f)
            print(f"Đã tải đồ thị với {G.number_of_nodes()} đỉnh và {G.number_of_edges()} cạnh")
        except Exception as e:
            print(f"Lỗi khi tải đồ thị: {e}")
            print("Xây dựng đồ thị tri thức mới...")
            G = build_knowledge_graph(
                df, PORTFOLIO_STOCKS, PORTFOLIO_SECTOR, skip=skip, max_frontier_size=max_frontier_size, 
                use_threading=use_threading, max_workers=max_workers, 
                graph_checkpoint=graph_checkpoint, canonical_checkpoint=canonical_checkpoint
            )
    else:
        print("Xây dựng đồ thị tri thức mới...")
        G = build_knowledge_graph(
            df, PORTFOLIO_STOCKS, PORTFOLIO_SECTOR, skip=skip, max_frontier_size=max_frontier_size, 
            use_threading=use_threading, max_workers=max_workers, 
            graph_checkpoint=graph_checkpoint, canonical_checkpoint=canonical_checkpoint
        )
    
    print(f"Áp dụng giai đoạn chú ý cho ngày: {prediction_date}...")
    G_sub = attention_phase(G, current_time=pred_ts, lambda_decay=lambda_decay, q=q)
    
    print("Thực hiện suy luận cuối cùng...")
    prediction = final_reasoning(G_sub, PORTFOLIO_STOCKS, PORTFOLIO_SECTOR, prediction_date)
    
    print("\nDự đoán cuối cùng:")
    print(prediction.content if prediction else "Không có dự đoán")
    
    return prediction

def evaluate_date_range(start_date, end_date, lambda_decay=1, q=6, graph_checkpoint=None, canonical_checkpoint=None):
    """
    Evaluates crash predictions for each day in a date range
    
    Parameters:
    -----------
    start_date : str
        Start date for evaluation in ISO format
    end_date : str
        End date for evaluation in ISO format
    lambda_decay : float
        Decay factor for edge weights
    q : int
        Number of top-ranked entities to include in subgraph
    graph_checkpoint : str
        Path to knowledge graph checkpoint file to load
    canonical_checkpoint : str
        Path to canonical entities checkpoint file to load
    """
    # Convert to datetime objects for iteration
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Load the graph once
    if not graph_checkpoint:
        print("Error: Graph checkpoint required for evaluation mode")
        return
        
    print(f"Loading knowledge graph from {graph_checkpoint}...")
    try:
        with open(graph_checkpoint, "rb") as f:
            G = pickle.load(f)
        print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    except Exception as e:
        print(f"Error loading graph: {e}")
        return
        
    # Set up results storage
    results = []
    results_file = f"crash_predictions_{start_dt.date()}_to_{end_dt.date()}.csv"
    
    # Check if we're continuing an existing evaluation
    if os.path.exists(results_file):
        existing_results = pd.read_csv(results_file)
        # Get the last evaluated date
        if len(existing_results) > 0:
            last_date = pd.to_datetime(existing_results.iloc[-1]['prediction_date'])
            start_dt = last_date + pd.Timedelta(days=1)
            print(f"Continuing evaluation from {start_dt.date()}")
            results = existing_results.to_dict('records')
    
    # Iterate through each day in the range
    current_dt = start_dt
    while current_dt <= end_dt:
        # Set prediction time to 1 AM GMT+7
        prediction_time = current_dt.replace(hour=1, minute=0, second=0)
        prediction_date = prediction_time.isoformat()
        
        print(f"\n{'='*50}")
        print(f"Evaluating prediction for {prediction_date}")
        print(f"{'='*50}")
        
        # Convert to timestamp for evaluation
        pred_ts = prediction_time.timestamp()
        
        # Apply attention phase
        print("Applying attention phase (PageRank-based filtering)...")
        G_sub = attention_phase(G, current_time=pred_ts, lambda_decay=1.0, q=6)
        print(f"Created subgraph with {G_sub.number_of_nodes()} nodes and {G_sub.number_of_edges()} edges")
        
        # Final reasoning
        print("Running final reasoning...")
        prediction = final_reasoning(G_sub, PORTFOLIO_STOCKS, PORTFOLIO_SECTOR)
        
        if prediction:
            # Parse the prediction result
            response_text = prediction.content.strip().lower()
            print("\nPrediction Response:")
            print(response_text)
            
            # Simple parsing: check if 'yes' appears before 'no'
            yes_pos = response_text.find('yes')
            no_pos = response_text.find('no')
            
            # Determine crash prediction
            if yes_pos != -1 and (no_pos == -1 or yes_pos < no_pos):
                crash_prediction = "Yes"
            elif no_pos != -1:
                crash_prediction = "No"
            else:
                crash_prediction = "Unclear"
            
            # Record the result
            result = {
                'prediction_date': prediction_date,
                'crash_prediction': crash_prediction,
                'full_response': prediction.content
            }
            
            results.append(result)
            
            # Save to CSV after each prediction for backup
            pd.DataFrame(results).to_csv(results_file, index=False)
            print(f"Saved prediction to {results_file} - Prediction: {crash_prediction}")
        else:
            print("Error: No prediction available")
            # Still record the failure
            result = {
                'prediction_date': prediction_date,
                'crash_prediction': "Error",
                'full_response': ""
            }
            results.append(result)
            pd.DataFrame(results).to_csv(results_file, index=False)
            
        # Move to next day
        # ignore saturday and sunday
        current_dt += pd.Timedelta(days=1)
        while current_dt.weekday() >= 5:
            current_dt += pd.Timedelta(days=1)
            continue
        
    print(f"\nEvaluation complete. Processed {len(results)} days.")
    return results

def main():
    """
    Điểm vào chính với xử lý tham số cải tiến, hỗ trợ dự đoán cho chuỗi ngày và lưu dự đoán vào cash_prediction.txt
    """
    parser = argparse.ArgumentParser(description="Mô hình Temporal Relational Reasoning")
    parser.add_argument("--news_from", type=int, default=1, help="Chỉ số bắt đầu của bài báo")
    parser.add_argument("--news_to", type=int, default=9400, help="Chỉ số kết thúc của bài báo (bị bỏ qua nếu dùng pred_date_range)")
    parser.add_argument("--pred_date", type=str, default="2025-04-02T01:00:00+07:00", help="Ngày dự đoán (dùng nếu không có pred_date_range)")
    parser.add_argument("--pred_date_range", type=str, help="Khoảng ngày dự đoán (định dạng: start_date,end_date, ví dụ: 2025-03-26,2025-03-31)")
    parser.add_argument("--summarize", action="store_true", help="Tạo tin tức tóm tắt")
    parser.add_argument("--load_graph", action="store_true", help="Tải đồ thị tri thức hiện có")
    parser.add_argument("--lambda_decay", type=float, default=1.0, help="Tham số suy giảm lambda")
    parser.add_argument("--q", type=int, default=6, help="Số thực thể top-q được chọn")
    parser.add_argument("--max_frontier_size", type=int, default=10, help="Số thực thể tối đa xử lý trong một lô")
    parser.add_argument("--batch_size", type=int, default=5, help="Kích thước lô cho tóm tắt tin tức")
    parser.add_argument("--no_threading", action="store_true", help="Tắt đa luồng")
    parser.add_argument("--max_workers", type=int, default=5, help="Số luồng công nhân tối đa")
    parser.add_argument("--skip", type=int, default=0, help="Số bài báo cần bỏ qua trong xử lý")
    parser.add_argument("--graph_checkpoint", type=str, help="Đường dẫn đến file checkpoint của đồ thị tri thức")
    parser.add_argument("--canonical_checkpoint", type=str, help="Đường dẫn đến file checkpoint của tập thực thể chuẩn hóa")
    
    args = parser.parse_args()
    
    # Đọc dữ liệu tin tức
    print(f"Đọc dữ liệu tin tức từ chỉ số {args.news_from}...")
    df = read_news_data("cleaned_posts.csv")
    
    # Lọc và tiền xử lý
    print("Tiền xử lý dữ liệu tin tức...")
    df = df[df['group'] != "Doanh nghiệp"]
    df = df.iloc[::-1]  # Đảo ngược để theo thứ tự thời gian
    df.fillna("", inplace=True)
    
    # Tạo hoặc tải tin tức tóm tắt
    if args.summarize:
        print("Tạo tin tức tóm tắt...")
        df_summary = make_summarized_news(df, batch_size=args.batch_size)
    else:
        if not os.path.exists("summarized_articles.csv"):
            print("Lỗi: File summarized_articles.csv không tồn tại. Vui lòng chạy với --summarize hoặc cung cấp file.")
            return
        print("Tải tin tức tóm tắt hiện có...")
        df_summary = pd.read_csv("summarized_articles.csv")
    
    # File để lưu dự đoán
    prediction_file = "crash_prediction.txt"
    
    # Chế độ dự đoán cho chuỗi ngày
    if args.pred_date_range:
        try:
            start_date, end_date = args.pred_date_range.split(',')
            print(f"Chạy dự đoán cho khoảng ngày từ {start_date} đến {end_date}")
            
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            results = []
            
            current_dt = start_dt
            news_idx = args.news_from
            while current_dt <= end_dt:
                if news_idx + ARTICLES_PER_DATE - 1 >= len(df_summary):
                    print(f"Hết bài báo tại chỉ số {news_idx} cho ngày {current_dt.date()}")
                    break
                
                prediction_time = current_dt.replace(hour=1, minute=0, second=0)
                prediction_date = prediction_time.isoformat()
                
                print(f"\n{'='*50}")
                print(f"Dự đoán cho ngày {prediction_date}")
                print(f"Sử dụng bài báo từ {news_idx} đến {news_idx + ARTICLES_PER_DATE - 1}")
                print(f"{'='*50}")
                
                df_subset = df_summary.iloc[news_idx:news_idx + ARTICLES_PER_DATE]
                if len(df_subset) < ARTICLES_PER_DATE:
                    print(f"Lỗi: Không đủ {ARTICLES_PER_DATE} bài báo từ chỉ số {news_idx}")
                    break
                
                prediction = trr(
                    df_subset, 
                    prediction_date,
                    load_saved_graph=args.load_graph,
                    lambda_decay=args.lambda_decay,
                    q=args.q,
                    max_frontier_size=args.max_frontier_size,
                    use_threading=not args.no_threading,
                    max_workers=args.max_workers,
                    skip=args.skip,
                    graph_checkpoint=args.graph_checkpoint,
                    canonical_checkpoint=args.canonical_checkpoint
                )
                
                # Ghi dự đoán vào cash_prediction.txt
                with open(prediction_file, "a", encoding="utf-8") as pred_f:
                    pred_f.write(f"Prediction for {prediction_date}:\n")
                    pred_f.write(f"{prediction.content if prediction else 'Error: No prediction available'}\n\n")
                    pred_f.flush()  # Đảm bảo ghi ngay lập tức
                
                print("\nPhản hồi dự đoán:")
                response_text = prediction.content.strip().lower() if prediction else ""
                print(response_text)
                
                # Lưu kết quả cho CSV
                crash_prediction = "Error"
                if prediction:
                    yes_pos = response_text.find('yes')
                    no_pos = response_text.find('no')
                    crash_prediction = "Yes" if yes_pos != -1 and (no_pos == -1 or yes_pos < no_pos) else "No" if no_pos != -1 else "Unclear"
                
                result = {
                    'prediction_date': prediction_date,
                    'news_indices': f"{news_idx}-{news_idx + ARTICLES_PER_DATE - 1}",
                    'crash_prediction': crash_prediction,
                    'full_response': prediction.content if prediction else ""
                }
                results.append(result)
                
                # Lưu vào CSV
                results_file = f"crash_predictions_{start_dt.date()}_to_{end_dt.date()}.csv"
                pd.DataFrame(results).to_csv(results_file, index=False)
                print(f"Lưu dự đoán vào {results_file} - Dự đoán: {crash_prediction}")
                print(f"Lưu dự đoán vào {prediction_file}")
                
                # Chuyển sang ngày tiếp theo (bỏ qua thứ Bảy và Chủ Nhật)
                current_dt += pd.Timedelta(days=1)
                while current_dt.weekday() >= 5:
                    current_dt += pd.Timedelta(days=1)
                news_idx += ARTICLES_PER_DATE
            
            print(f"\nHoàn thành đánh giá. Đã xử lý {len(results)} ngày")
            return results
            
        except ValueError:
            print("Lỗi: pred_date_range phải có định dạng 'start_date,end_date' (ví dụ: '2025-03-26,2025-03-31')")
            return
    
    # Chế độ dự đoán đơn ngày
    print(f"Chạy TRR cho ngày dự đoán: {args.pred_date}")
    news_to = args.news_from + ARTICLES_PER_DATE - 1
    df_subset = df_summary.iloc[args.news_from:news_to + 1]
    if len(df_subset) < ARTICLES_PER_DATE:
        print(f"Lỗi: Không đủ {ARTICLES_PER_DATE} bài báo từ chỉ số {args.news_from}")
        return
    
    prediction = trr(
        df_subset, 
        args.pred_date,
        load_saved_graph=args.load_graph,
        lambda_decay=args.lambda_decay,
        q=args.q,
        max_frontier_size=args.max_frontier_size,
        use_threading=not args.no_threading,
        max_workers=args.max_workers,
        skip=args.skip,
        graph_checkpoint=args.graph_checkpoint,
        canonical_checkpoint=args.canonical_checkpoint
    )
    
    # Ghi dự đoán đơn ngày vào cash_prediction.txt
    with open(prediction_file, "a", encoding="utf-8") as pred_f:
        pred_f.write(f"Prediction for {args.pred_date}:\n")
        pred_f.write(f"{prediction.content if prediction else 'Error: No prediction available'}\n\n")
        pred_f.flush()
        print(f"Lưu dự đoán vào {prediction_file}")
    
    return prediction
if __name__ == "__main__":
    main()
    