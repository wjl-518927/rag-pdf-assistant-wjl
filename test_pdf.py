import fitz  # 这是 pymupdf 的导入名

# 1. 打开 PDF 文件
doc = fitz.open("data/UmuGmPJC.pdf")

print(f"✅ 成功打开 PDF，共 {len(doc)} 页")

# 2. 读取第一页的文本
first_page = doc[0]
text = first_page.get_text()

print("\n📄 第一页的内容预览：")
print(text[:500]) # 打印前 500 个字符看看效果

doc.close()