import os
import nltk
import docx2txt
from pdfminer.high_level import extract_text
import text_processor

print("\nDownloading NLTK libraries...\n\n")

# Download NLTK libraries.
nltk.download("punkt")
nltk.download("words")
nltk.download("maxent_ne_chunker")
nltk.download("averaged_perceptron_tagger")
nltk.download("names")
nltk.download("stopwords")

print("\n\nDownload complete.\n\n\n")

# For converting PDF files.
def pdf_conversion(file):
  return extract_text(file)

# For converting DOCX files.
def docx_conversion(file):
  doc = docx2txt.process(file)
  if doc:
    return doc.replace("\t", " ")
  return None

# Take the input from the command line.
file = input("Enter the file name (only .PDF and .DOCX extension supported): ")
if len(file) < 1:
  output("No file name entered.")
else:
  if not os.path.exists(file):
    print("File not found")
  else:
    # Fetch the extension of the file and parse accordingly.
    file_ext = file.split(".")[-1]
    if file_ext == "pdf":
      print(text_processor.parse_resume(pdf_conversion(file)))
    elif file_ext == "docx":
      print(text_processor.parse_resume(docx_conversion(file)))
    elif file_ext == "txt":
      with open(file, "r") as f:
        print(text_processor.parse_resume(f.read()))
    else:
      print("File format not supported.")
