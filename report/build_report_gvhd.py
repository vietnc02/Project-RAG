"""Báo cáo tiến độ ý 1 gửi thầy hướng dẫn (đã làm được + khó khăn + xin hướng giải quyết)
-> report/Bao_cao_tien_do_Y1.docx"""
import json
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
S = json.loads((ROOT / "report" / "stats.json").read_text(encoding="utf-8"))
OUT = ROOT / "report" / "Bao_cao_tien_do_Y1.docx"

BLUE = RGBColor(0x1C, 0x5C, 0xAB)
GREY = RGBColor(0x52, 0x51, 0x4E)
FONT = "Times New Roman"

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.left_margin, sec.right_margin = Cm(3), Cm(2)
sec.top_margin = sec.bottom_margin = Cm(2)


# ---------------- styles ----------------
def set_font(style, size, bold=None, color=None, italic=None):
    style.font.name = FONT
    style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if italic is not None:
        style.font.italic = italic
    if color is not None:
        style.font.color.rgb = color
    rpr = style.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), FONT)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]


normal = doc.styles["Normal"]
set_font(normal, 13)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.3
normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
for lvl, size in [(1, 15), (2, 13.5), (3, 13)]:
    st = doc.styles[f"Heading {lvl}"]
    set_font(st, size, bold=True, color=BLUE, italic=(lvl == 3))
    st.paragraph_format.space_before = Pt(16 if lvl == 1 else 10)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.keep_with_next = True
    st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
set_font(doc.styles["Caption"], 11.5, bold=False, color=GREY, italic=True)
for name in ("List Bullet", "List Number"):
    set_font(doc.styles[name], 13)


# ---------------- helpers ----------------
def rich(para, text, size=None, italic=False, color=None):
    for tok in re.split(r"(\*\*[^*]+\*\*|_[^_ ][^_]*_)", text):
        if not tok:
            continue
        if tok.startswith("**"):
            r = para.add_run(tok[2:-2]); r.bold = True
        elif tok.startswith("_") and tok.endswith("_") and len(tok) > 2:
            r = para.add_run(tok[1:-1]); r.italic = True
        else:
            r = para.add_run(tok)
        if size:
            r.font.size = Pt(size)
        if italic:
            r.italic = True
        if color:
            r.font.color.rgb = color


def p(text="", align=None, size=None, italic=False, indent=True, color=None):
    para = doc.add_paragraph()
    rich(para, text, size=size, italic=italic, color=color)
    if align is not None:
        para.alignment = align
    if indent and align is None:
        para.paragraph_format.first_line_indent = Cm(1)
    return para


def bullets(items):
    for it in items:
        para = doc.add_paragraph(style="List Bullet")
        para.paragraph_format.space_after = Pt(3)
        if isinstance(it, tuple):
            r = para.add_run(it[0]); r.bold = True
            rich(para, it[1])
        else:
            rich(para, it)


def h(text, lvl):
    return doc.add_heading(text, lvl)


def shade(cell, fill):
    tcPr = cell._element.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


TAB, FIGN = [0], [0]


def table(caption, header, rows, widths, font=11.5, bold_first=True, center_from=None):
    TAB[0] += 1
    cap = doc.add_paragraph(style="Caption"); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.add_run(f"Bảng {TAB[0]}. {caption}")
    cap.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
    for j, txt in enumerate(header):
        c = t.rows[0].cells[j]; c.text = ""
        para = c.paragraphs[0]; para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = para.add_run(txt); r.bold = True; r.font.size = Pt(font); r.font.color.rgb = RGBColor(255, 255, 255)
        shade(c, "1C5CAB")
    for i, row in enumerate(rows, 1):
        for j, v in enumerate(row):
            c = t.rows[i].cells[j]; c.text = ""
            para = c.paragraphs[0]
            para.alignment = (WD_ALIGN_PARAGRAPH.CENTER if (center_from is not None and j >= center_from)
                              else WD_ALIGN_PARAGRAPH.LEFT)
            rich(para, str(v), size=font)
            if bold_first and j == 0:
                for r in para.runs:
                    r.bold = True
            if i % 2 == 0:
                shade(c, "F0F5FC")
    trPr = t.rows[0]._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true"); trPr.append(th)
    for row in t.rows:
        cs = OxmlElement("w:cantSplit"); cs.set(qn("w:val"), "true")
        row._tr.get_or_add_trPr().append(cs)
        for j, w in enumerate(widths):
            row.cells[j].width = Cm(w)
        for c in row.cells:
            for para in c.paragraphs:
                para.paragraph_format.space_after = Pt(1)
                para.paragraph_format.line_spacing = 1.1
                para.paragraph_format.first_line_indent = None
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def figure(name, caption, width=16):
    FIGN[0] += 1
    para = doc.add_paragraph(); para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.keep_with_next = True
    para.add_run().add_picture(str(FIG / name), width=Cm(width))
    cap = doc.add_paragraph(style="Caption"); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.add_run(f"Hình {FIGN[0]}. {caption}")


def field(run, instr, placeholder=""):
    for tag in ("begin", None, "separate", "end"):
        if tag is None:
            it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = instr; run._r.append(it)
            continue
        fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), tag); run._r.append(fc)
        if tag == "separate" and placeholder:
            t = OxmlElement("w:t"); t.text = placeholder; run._r.append(t)


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def label(text):
    """Nhãn in đậm đầu đoạn (Hiện tượng / Nguyên nhân / ...)."""
    para = doc.add_paragraph()
    para.paragraph_format.space_after = Pt(3)
    rich(para, text)
    return para


ft = sec.footer.paragraphs[0]; ft.alignment = WD_ALIGN_PARAGRAPH.CENTER
fr = ft.add_run(); field(fr, "PAGE"); fr.font.size = Pt(11)


def pct(x, d=1):
    return f"{x * 100:.{d}f}%".replace(".", ",")


def num(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def ci(v, as_pct=False):
    m, lo, hi = v
    return f"{pct(m)} [{pct(lo)}; {pct(hi)}]" if as_pct else f"{num(m)} [{num(lo)}; {num(hi)}]"


M, E, PD, MC = S["main"], S["extra"], S["paired"], S["mcnemar"]
lc, sr = M["lc"], M["sr"]
el, es = E["lc"], E["sr"]
NP = sum(v["papers"] for v in S["corpus"].values())
NC = sum(v["chunks"] for v in S["corpus"].values())
RV, RC, EN = S["review"], S["sr_refusals"], S["sr_behaviour"]["ends"]
NREF = sum(len(v) for v in RC.values())

# ================= BÌA =================
for _ in range(6):
    doc.add_paragraph()
p("BÁO CÁO TIẾN ĐỘ ĐỒ ÁN TỐT NGHIỆP", align=WD_ALIGN_PARAGRAPH.CENTER, size=15).runs[0].bold = True
doc.add_paragraph()
p("NỘI DUNG 1", align=WD_ALIGN_PARAGRAPH.CENTER, size=14, color=GREY)
t = p("So sánh LangChain RAG thuần và LangGraph Self-RAG", align=WD_ALIGN_PARAGRAPH.CENTER, size=22, color=BLUE)
t.runs[0].bold = True
p("Kết quả đã đạt được và khó khăn gặp phải", align=WD_ALIGN_PARAGRAPH.CENTER, size=15, italic=True)
for _ in range(10):
    doc.add_paragraph()
p("Tháng 10 năm 2026", align=WD_ALIGN_PARAGRAPH.CENTER, size=13, color=GREY)
page_break()

# ================= 1. TỔNG QUAN =================
h("1. TỔNG QUAN TIẾN ĐỘ", 1)
p("**Yêu cầu nội dung 1:** so sánh LangChain RAG thuần và LangGraph Self-RAG trên cùng bộ bài báo, đo faithfulness, answer relevancy và hallucination rate; hai hệ thống chạy bằng server LLM cục bộ.")
p(f"Em đã hoàn thành phần xây dựng hệ thống và chạy xong một vòng đánh giá đầy đủ trên {NP} bài báo với {S['n']} câu hỏi. "
  "Tuy nhiên kết quả vòng 1 chưa đủ tin cậy để kết luận, do một số vấn đề về cài đặt Self-RAG với mô hình nhỏ và về phương pháp chấm điểm. "
  "Các vấn đề này được trình bày ở mục 3.")
table("Tình trạng các hạng mục", ["Hạng mục", "Trạng thái", "Ghi chú"],
      [["Môi trường LLM cục bộ (Ollama)", "Hoàn thành", "qwen2.5:7b, llama3.1:8b, nomic-embed-text"],
       ["Tiền xử lý dữ liệu, vector DB", "Hoàn thành", f"{NP} bài báo, {NC:,} chunk".replace(",", ".")],
       ["Hệ A - LangChain RAG thuần", "Hoàn thành", "Đã kiểm thử"],
       ["Hệ B - LangGraph Self-RAG", "Hoàn thành", "Bước tự đánh giá còn vấn đề (mục 3.2)"],
       ["Bộ câu hỏi đánh giá", "Hoàn thành", f"{RV['kept']} câu, đã duyệt thủ công"],
       ["Quy trình đánh giá 3 chỉ số", "Hoàn thành", "Có khoảng tin cậy và kiểm định thống kê"],
       ["Đánh giá vòng 1", "Hoàn thành", "Kết quả chưa đủ tin cậy (mục 3)"]],
      [5.6, 3.2, 7.2], font=11.5)

# ================= 2. ĐÃ LÀM ĐƯỢC =================
h("2. NHỮNG VIỆC ĐÃ LÀM ĐƯỢC", 1)

h("2.1. Môi trường chạy cục bộ", 2)
p("Toàn bộ hệ thống chạy trên máy cá nhân (CPU i3-12100F, RAM 16 GB, GPU GTX 1660 Ti 6 GB) qua Ollama, không dùng API bên ngoài. "
  "Do GPU chỉ có 6 GB, em chọn các mô hình 7–8B lượng tử hoá Q4:")
bullets([("qwen2.5:7b: ", "mô hình sinh câu trả lời, dùng chung cho cả hai hệ (temperature = 0)."),
         ("llama3.1:8b: ", "mô hình chấm điểm; em chọn khác họ với mô hình sinh để tránh thiên lệch \"tự chấm bài mình\"."),
         ("nomic-embed-text: ", "mô hình nhúng; cơ sở dữ liệu vector là Chroma.")])

h("2.2. Tiền xử lý dữ liệu", 2)
p(f"Từ 96 tệp PDF, em trích văn bản, loại phần tài liệu tham khảo cuối bài, loại 1 tệp trùng lặp, chia thành các đoạn 1000 ký tự "
  f"(chồng lấn 150) và nhúng vào Chroma, thu được {NP} bài báo và {NC:,} chunk.".replace(",", "."))
figure("fig10_corpus.png", "Phân bố bài báo và chunk theo chủ đề", width=14.5)

h("2.3. Xây dựng hai hệ thống", 2)
figure("fig1_architecture.png", "Kiến trúc (a) LangChain RAG thuần và (b) LangGraph Self-RAG", width=16)
bullets([("Hệ A - LangChain RAG thuần: ", "truy xuất 4 chunk gần nhất rồi sinh câu trả lời, 1 lần gọi LLM mỗi câu hỏi."),
         ("Hệ B - LangGraph Self-RAG: ", "theo ý tưởng của Asai và cộng sự (2023). Do không có mô hình được huấn luyện reflection token, em hiện thực các bước tự đánh giá bằng prompt: chấm độ liên quan của từng tài liệu (ISREL), kiểm tra câu trả lời có căn cứ (ISSUP) và có hữu ích (ISUSE). Nếu không đạt thì viết lại truy vấn, sinh lại hoặc từ chối trả lời; mỗi vòng lặp tối đa 2 lần.")])
p("Để so sánh công bằng, hai hệ dùng chung dữ liệu, bộ truy xuất (top-k = 4), mô hình sinh và prompt sinh câu trả lời; chỉ khác ở vòng tự đánh giá. "
  "Khi kiểm thử với câu hỏi nằm ngoài tài liệu, Self-RAG viết lại truy vấn hai lần rồi từ chối trả lời, đúng với thiết kế.")

h("2.4. Bộ câu hỏi đánh giá", 2)
p(f"Với mỗi bài báo, mô hình sinh tự động một câu hỏi kèm đáp án tham chiếu từ một đoạn văn ngẫu nhiên ({RV['generated']} câu). "
  "Sau đó em duyệt thủ công từng câu bằng cách đối chiếu với đoạn văn gốc: sửa câu hỏi cho rõ nghĩa và tự đứng được, "
  "loại các câu hỏi về công thức bị lỗi ký tự khi trích PDF hoặc chỉ hỏi tên tác giả.")
figure("fig2_testset_review.png", "Kết quả duyệt thủ công bộ câu hỏi", width=14)

h("2.5. Quy trình đánh giá", 2)
table("Định nghĩa ba chỉ số (theo RAGAS)", ["Chỉ số", "Cách tính"],
      [["Faithfulness ↑", "Mô hình chấm tách câu trả lời thành các claim, kiểm tra từng claim có suy ra được từ tài liệu truy xuất. Điểm = số claim được hỗ trợ / tổng số claim."],
       ["Answer relevancy ↑", "Mô hình chấm sinh ngược 3 câu hỏi từ câu trả lời; điểm = độ tương đồng cosine trung bình với câu hỏi gốc. Câu từ chối được 0 điểm."],
       ["Hallucination rate ↓", "Tỉ lệ câu hỏi có câu trả lời chứa ít nhất một claim không được tài liệu hỗ trợ."]],
      [4.0, 12.0], font=11.5)
p("Kết quả được báo cáo kèm khoảng tin cậy 95% (bootstrap) và kiểm định theo cặp trên cùng câu hỏi (bootstrap, McNemar). "
  "Trước khi chạy, em kiểm tra mô hình chấm bằng một câu trả lời cố ý chứa 4 thông tin bịa đặt: mô hình phát hiện đúng cả 4.")

h("2.6. Kết quả đánh giá vòng 1", 2)
table("Kết quả ba chỉ số chính, giá trị trung bình [khoảng tin cậy 95%], n = 86",
      ["Hệ thống", "Faithfulness ↑", "Answer relevancy ↑", "Hallucination rate ↓"],
      [["LangChain RAG", ci(lc["faithfulness"]), ci(lc["answer_relevancy"]), ci(lc["hallucination"], True)],
       ["LangGraph Self-RAG", ci(sr["faithfulness"]), ci(sr["answer_relevancy"]), ci(sr["hallucination"], True)]],
      [3.8, 4.0, 4.0, 4.2], font=11, center_from=1)
figure("fig3_main_metrics.png", "Ba chỉ số chính của hai hệ thống, thanh lỗi là khoảng tin cậy 95%")
fp, ap = PD["faithfulness"], PD["answer_relevancy"]
p(f"Self-RAG có faithfulness cao hơn ({num(sr['faithfulness'][0])} so với {num(lc['faithfulness'][0])}) và hallucination rate thấp hơn "
  f"({pct(sr['hallucination'][0])} so với {pct(lc['hallucination'][0])}), nhưng các chênh lệch này **không có ý nghĩa thống kê** "
  f"(p ≈ {num(fp['p'], 2)} và p ≈ {num(MC['p'], 2)}). Answer relevancy của Self-RAG **thấp hơn có ý nghĩa** (p ≈ {num(ap['p'], 2)}), "
  f"chủ yếu vì Self-RAG từ chối trả lời nhiều hơn ({pct(es['refusal_rate'])} so với {pct(el['refusal_rate'])}). "
  f"Ngoài ra Self-RAG tốn trung bình {num(es['llm_calls'], 1)} lần gọi LLM và {num(es['latency_s'], 1)} giây mỗi câu, so với 1 lần và {num(el['latency_s'], 1)} giây của RAG thuần.")
figure("fig5_secondary.png", "Các chỉ số phụ (trái) và thời gian xử lý mỗi câu hỏi (phải)")

# ================= 3. VẤN ĐỀ GẶP PHẢI =================
h("3. CÁC VẤN ĐỀ GẶP PHẢI", 1)
p("Kết quả vòng 1 trái với kỳ vọng rằng Self-RAG giảm hallucination rõ rệt. Em đã phân tích từng câu hỏi có kết quả khác nhau giữa hai hệ và gặp các vấn đề sau.")

h("3.1. Self-RAG chưa cho thấy cải thiện rõ ràng", 2)
figure("fig6_halluc_transitions.png", "So khớp hallucination theo từng câu hỏi giữa hai hệ", width=15)
p(f"Self-RAG sửa được {MC['only_lc']} câu mà RAG thuần bị hallucination, nhưng lại phát sinh {MC['only_sr']} câu mới mà RAG thuần trả lời đúng, nên hai chiều gần như bù trừ nhau. "
  f"Nếu chỉ tính các câu có trả lời, hallucination rate của hai hệ gần bằng nhau ({pct(el['halluc_among_answered'])} và {pct(es['halluc_among_answered'])}), "
  "tức là phần giảm hallucination của Self-RAG chủ yếu đến từ việc từ chối trả lời.")

h("3.2. Bước tự đánh giá của Self-RAG với mô hình 7B chưa chính xác", 2)
figure("fig8_refusals.png", f"Phân loại {NREF} câu Self-RAG từ chối trả lời", width=15)
p(f"Trong {NREF} câu Self-RAG từ chối trả lời, có {len(RC['false_abstain'])} câu ({', '.join(RC['false_abstain'])}) bị từ chối không cần thiết: "
  "RAG thuần trả lời đúng và bám tài liệu, nhưng bước kiểm tra căn cứ (ISSUP) của Self-RAG liên tục chấm \"không có căn cứ\" rồi từ chối. "
  "Ví dụ ở câu q037, mô hình giải thích: _\"The answer does not mention the end-to-end framework\"_ - tức là mô hình đang chấm câu trả lời **có đủ ý hay không**, thay vì **có bịa thông tin hay không**. "
  "Việc sinh lại không cần thiết còn làm phát sinh lỗi: ở câu q026, câu trả lời đầu đúng bị loại, câu sinh lại thêm thông tin không có trong tài liệu.")
p(f"Ngược lại, cơ chế này cũng có tác dụng ở {len(RC['lc_hallucinated_or_wrong'])} câu: RAG thuần bịa đặt, Self-RAG phát hiện và từ chối đúng. "
  "Bài báo gốc huấn luyện riêng mô hình để sinh reflection token, còn em hiện thực bằng prompt với mô hình 7B có sẵn.")

h("3.3. Mô hình chấm điểm 8B có nhiễu", 2)
p("Có những câu hai hệ trả lời gần như giống hệt nhau nhưng nhận điểm khác nhau. Ví dụ ở câu q014 và q087, câu trả lời của hai hệ gần như trùng từng chữ, được chấm trên cùng tài liệu, "
  "nhưng RAG thuần được faithfulness 1,0 còn Self-RAG chỉ 0,75–0,8. Trong 11 câu Self-RAG \"phát sinh hallucination mới\", có 8 câu thuộc dạng này. "
  "Mức nhiễu của mô hình chấm xấp xỉ mức chênh lệch cần đo giữa hai hệ, nên với cách chấm hiện tại em chưa thể kết luận hệ nào tốt hơn. "
  "Hiện em cũng chưa có đánh giá của con người để đối chứng với mô hình chấm.")

h("3.4. Hai hệ được chấm trên tài liệu khác nhau", 2)
p("Self-RAG được chấm faithfulness trên các tài liệu đã lọc (thường chỉ 1–2 đoạn), còn RAG thuần được chấm trên đủ 4 đoạn. "
  "Khi em chấm lại 11 câu hallucination phát sinh của Self-RAG trên cùng 4 đoạn như RAG thuần, có 3/11 câu không còn bị tính là hallucination. "
  "Như vậy cách chọn tài liệu để chấm ảnh hưởng trực tiếp đến kết quả so sánh.")

h("3.5. Giới hạn phần cứng", 2)
p("GPU 6 GB chỉ chạy được mô hình 7–8B lượng tử hoá. Mỗi vòng đánh giá đầy đủ (2 hệ × 86 câu, cả sinh và chấm) mất khoảng 2,5 giờ. "
  "Mô hình lớn hơn (ví dụ 14B) không vừa GPU, phải chạy một phần trên CPU nên thời gian chấm tăng lên khoảng 4–6 giờ mỗi vòng. "
  "Điều này giới hạn số lần thử nghiệm và cỡ bộ câu hỏi.")

h("3.6. Dữ liệu và bộ câu hỏi", 2)
bullets([f"Bộ câu hỏi chỉ có {S['n']} câu (mỗi bài 1 câu) nên khoảng tin cậy còn rộng, khó phát hiện chênh lệch nhỏ giữa hai hệ.",
         "Các câu hỏi hiện tại chủ yếu là hỏi sự kiện trong một đoạn văn - dạng câu hỏi dễ, nơi RAG thuần đã làm tốt; chưa có câu hỏi tổng hợp nhiều tài liệu hoặc câu hỏi nằm ngoài tài liệu.",
         "Một số PDF có công thức bị lỗi ký tự khi trích văn bản, một bài là bản scan không trích được chữ; thư mục Ref gồm nhiều chủ đề khác nhau và có vài bài gần trùng nội dung."])

doc.save(OUT)
print("saved", OUT)
