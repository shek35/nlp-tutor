import os
import textract
import re
import fitz

TB_DIR = "textbooks"
TEXT_DIR = "textbooks_text"

if not os.path.exists(TB_DIR):
    os.makedirs(TB_DIR)

if not os.path.exists(TEXT_DIR):
    os.makedirs(TEXT_DIR)

def get_pdf_text(file):
    doc = os.path.join(TB_DIR, file)
    try:
        text = textract.process(doc, encoding="utf-8").decode("utf-8")
    except Exception as err:
        print(err)
        text = ""
    return text

def clean_text(text):

    text = text.encode('utf-8', 'ignore').decode('utf-8').strip()
    text = text.replace("", "")
    text = text.replace("", "")
    text = text.replace("", "")
    text = text.replace("", "")
    text = text.replace("", "")
    text = text.replace("", "")
    text = text.replace("", "")
    text = text.replace("", "")
    text = text.replace("", "")
    text = text.replace("•", "")
    text = text.replace("·", "")

    # Remove header and footer patterns
    text = re.sub(r'^\d+\s+|^\w+\s+\|\s+\w+\s+\d+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+\s*$', '', text, flags=re.MULTILINE)

    # Remove page numbers
    text = re.sub(r'\d+', '', text)

    pagenumber = re.compile(
        r'^[(]?\d{1,3}[)]?[\.]?$|^.[(]?\d{1,3}[)]?[\.]?$|^[(]?\d{1,3}[)]?.?[(]?\d{1,3}[)]?[\.]?$')
    cont = re.compile(r'^\(continued\)$|^continued:$|^continued: \(\d+\)$')
    allspecialchars = re.compile(r'^[^\w\s ]*$')

    lines = []

    for line in text.split('\n'):
        copy = line
        line = line.lower().strip()
        # skip lines with one char since they're likely typos
        if len(line) == 1:
            if line.lower() != 'a' or line.lower() != 'i':
                continue
        # skip lines containing page numbers
        if pagenumber.match(line):
            continue
        # Lines which just say continued
        if cont.match(line):
            continue
        # skip lines containing just special characters
        if line != '' and allspecialchars.match(line):
            continue
        # Lines which just say continued
        if cont.match(line):
            continue
        if line == "omitted":
            continue
        lines.append(copy.strip())

    final_data = '\n'.join(lines)
    final_data = re.sub(r'\n\n+', '\n\n', final_data).strip()
    return final_data

def extract_chapters_from_pdf(pdf_path, output_folder):
    # Open the PDF file
    with fitz.open(pdf_path) as pdf_file:
        # Load the table of contents
        toc = pdf_file.get_toc(False)
        # print(toc)

        # Find the starting page of the actual content
        for chapter in toc:
            if "chapter" in chapter[3]["nameddest"]:
                first_chapter_text = chapter[1]
                break
        print(first_chapter_text)
        for page_index in range(pdf_file.page_count):
            page = pdf_file[page_index]
            page_text = page.get_text()
            if re.search(first_chapter_text, page_text, re.IGNORECASE):
                start_page = page_index
                break
        else:
            print("Could not find the starting page of the actual content.")
            return

        # Loop through each entry in the table of contents
        for toc_entry in toc:
            # Get the chapter title and relative page offset
            chapter_title = toc_entry[1].replace("/", "_").replace("*", "_").replace(":", "_") \
            .replace("?", "_").replace("<", "_").replace(">", "_").replace("|", "_").replace('"', "_")\
                .replace("\\", "_").replace(" ", "_")
            relative_page_offset = toc_entry[2]

            # Calculate the starting page for the chapter
            chapter_start_page = start_page + relative_page_offset

            # Extract the text for the chapter
            chapter_text = ""
            for page_index in range(chapter_start_page, pdf_file.page_count):
                page = pdf_file[page_index]
                page_text = page.get_text()

                # Check if the next entry in the table of contents has been reached
                if toc_entry != toc[-1] and page_index + 1 >= start_page + toc[toc.index(toc_entry) + 1][2]:
                    break

                chapter_text += page_text


            # Save the chapter text to a file
                
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            output_file = os.path.join(output_folder, chapter_title + ".txt")
            with open(output_file, "w", encoding="utf-8") as text_file:
                text_file.write(chapter_text)

            # print(f"Chapter '{chapter_title}' extracted and saved to '{output_file}'")

for file in os.listdir(TB_DIR):
    if file.endswith(".pdf"):
        file_name = file.strip(".pdf")
        print(f"Processing '{file}'")
        
        doc = fitz.open(os.path.join(TB_DIR, file))
        text = ""
        for page in doc:
            text += page.get_text("text", sort=True) + "\n"

        with open(os.path.join(TEXT_DIR, file_name + ".txt"), "w", errors="ignore") as out:
            out.write(clean_text(text))

        # extract_chapters_from_pdf(os.path.join(TB_DIR, file), os.path.join(TEXT_DIR, file_name))