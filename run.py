from pathlib import Path
from mistralai import Mistral
from mistralai import DocumentURLChunk
import json

api_key = "hoge"  # ここにご自身の API キーを設定
client = Mistral(api_key=api_key)

def ocr_pdf_to_markdown(pdf_path: Path) -> str:
    """
    PDF ファイルに OCR をかけ、Markdown（base64 埋め込み画像付き）を返す。
    """
    # ファイルをアップロード
    uploaded = client.files.upload(
        file={
            "file_name": pdf_path.stem,
            "content": pdf_path.read_bytes(),
        },
        purpose="ocr",
    )
    signed = client.files.get_signed_url(file_id=uploaded.id, expiry=1)
    # OCR 処理
    ocr_resp = client.ocr.process(
        document=DocumentURLChunk(document_url=signed.url),
        model="mistral-ocr-latest",
        include_image_base64=True,
    )
    # JSON → OCRResponse オブジェクトに変換
    from mistralai.models import OCRResponse
    resp_obj = OCRResponse.model_validate_json(ocr_resp.model_dump_json())

    # 画像プレースホルダーを base64 に置換
    def replace_images(md: str, images: dict[str, str]) -> str:
        for img_id, b64 in images.items():
            md = md.replace(f"![{img_id}]({img_id})", f"![{img_id}]({b64})")
        return md

    markdown_pages: list[str] = []
    for page in resp_obj.pages:
        imgs = {img.id: img.image_base64 for img in page.images}
        markdown_pages.append(replace_images(page.markdown, imgs))

    return "\n\n".join(markdown_pages)

def main():
    # 現在のディレクトリ内のすべての .pdf を対象
    for pdf_path in Path(".").glob("*.pdf"):
        try:
            print(f"Processing: {pdf_path.name} …")
            md_content = ocr_pdf_to_markdown(pdf_path)
            out_file = pdf_path.with_suffix(".md")
            out_file.write_text(md_content, encoding="utf-8")
            print(f" -> Saved to {out_file.name}")
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e!r}")

if __name__ == "__main__":
    main()
