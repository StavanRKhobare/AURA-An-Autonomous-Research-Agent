import fitz  # PyMuPDF
import requests
import io
import os

class PDFSkill:
    def __init__(self, output_dir: str = "workspace/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def fetch_and_parse(self, pdf_url: str) -> str:
        """
        Downloads a PDF from a URL and extracts its text.
        """
        print(f"Downloading PDF from {pdf_url}...")
        headers = {'User-Agent': 'Aura-Research-Agent/1.0'}
        response = requests.get(pdf_url, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Failed to download PDF: Status code {response.status_code}")

        # Load PDF from memory
        pdf_stream = io.BytesIO(response.content)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        
        full_text = ""
        print(f"Parsing {len(doc)} pages...")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            full_text += f"\n--- Page {page_num + 1} ---\n"
            full_text += text
            
            # Note: Image extraction is available but commented out for initial speed
            # image_list = page.get_images(full=True)
            # for img_index, img in enumerate(image_list):
            #     xref = img[0]
            #     base_image = doc.extract_image(xref)
            #     image_bytes = base_image["image"]
            #     image_ext = base_image["ext"]
            #     image_path = os.path.join(self.output_dir, f"page{page_num+1}_img{img_index}.{image_ext}")
            #     with open(image_path, "wb") as f:
            #         f.write(image_bytes)

        return full_text

if __name__ == "__main__":
    # Quick test using the famous "Attention Is All You Need" paper
    skill = PDFSkill()
    test_url = "https://arxiv.org/pdf/1706.03762.pdf" 
    print(f"Testing PDF parsing on: {test_url}")
    try:
        text = skill.fetch_and_parse(test_url)
        print(f"\nSuccessfully extracted {len(text)} characters.")
        print("Preview (First 500 characters):")
        print("-" * 40)
        print(text[:500])
        print("-" * 40)
    except Exception as e:
        print(f"Error: {e}")
