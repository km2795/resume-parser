import re
import csv
import nltk
import json

# Extract the skills from the text.
def extract_skills(text):

  # Database of skills.
  SKILLS_DB = []

  # Load the skills in a cache.
  with open("skills_db.csv", "r") as file:
    reader = csv.reader(file, delimiter="\n")
    for row in reader:
      for item in row:
        SKILLS_DB.append(item.lower())

  # Load the stopwords.
  stopwords = set(nltk.corpus.stopwords.words("english"))

  # Process tokens from the text.
  tokens = nltk.word_tokenize(text)

  # Remove the stopwords and non-alphanumeric characters from the token list.
  tokens = [token for token in tokens if token not in stopwords]
  tokens = [token for token in tokens if token.isalpha()]

  bi_tri_grams  = list(map(" ".join, nltk.everygrams(tokens, 2, 3)))

  found_skills = set()

  for token in tokens:
    if token.lower() in SKILLS_DB:
      found_skills.add(token)

  for ngram in bi_tri_grams:
      if ngram.lower() in SKILLS_DB:
          found_skills.add(ngram)

  return list(found_skills)

# Extract name.
def extract_name(text):

  # Database of names.
  NAMES_DB = []

  # Load the names in a cache.
  with open("./dataset/names.csv", "r") as file:
    reader = csv.reader(file, delimiter=",")
    for row in reader:
      for item in row:
        NAMES_DB.append(item.lower())

  # For name selection.
  name_grammar = r"NAME: {<NN|NNP> <NN|NNP>*}"
  chunk_parser = nltk.RegexpParser(name_grammar)

  found_names = []

  # Don't lowercase letter for 'name' field,
  # it helps in rooting out proper nouns (names are generally written as proper nouns).
  name_tokens = chunk_parser.parse(nltk.pos_tag(nltk.word_tokenize(text)))
  for subtree in name_tokens.subtrees():
    if subtree.label() == "NAME":
      for index, leaf in enumerate(subtree.leaves()):
        if (leaf[0].lower() in NAMES_DB and ("NN" in leaf[1] or "NNP" in leaf[1])):
          found_names.append(leaf[0])

  # It's not very accurate, hence returning only the first
  # entry, since most resume formats have the name at the top.
  # This is just a temporary fix.
  return found_names[0]


# Main process to start the extraction.
def parse_resume(text):

  nltk.download("punkt")
  nltk.download("words")
  nltk.download("maxent_ne_chunker")
  nltk.download("averaged_perceptron_tagger")
  nltk.download("names")
  nltk.download("stopwords")

  # Fields that needs to be matched, regardless of resume type.
  resume_fields = {
    "name": "",
    "email": "",
    "phone": "",
    "education": [],
    "projects": [],
    "work experience": [],
    "achievements": [],
    "certifications": [],
    "hobbies": []
  }

  # Cache the individual sentences.
  lines = []

  # Cache the POS tags.
  pos_tags = []

  # Cache the tokens.
  tokens = []

  # Enumerate each sentence to fetch tokens and POS tags.
  sentences = nltk.sent_tokenize(text)

  # Cache the whole raw text as string too.
  doc = ""
  for sent in sentences:
    lines.append(sent)
    _tokens = nltk.word_tokenize(sent)
    tokens.extend(_tokens)
    pos_tags.append(nltk.pos_tag(_tokens))
    doc += sent

  # For mostly regex type text extraction.
  doc = doc.lower()

  # Stores the headlines (if found) of the resume, based on certain pre-defined criterion.
  # Along with headlines, it will store from where the heading starts.
  indices = {}

  # Criterion for "work experience" section.
  found_experience = re.search(r"experience![d]|work experience|work history|work engagement|professional engagement", doc)
  if found_experience != None:
    indices["experience"] = found_experience.end()

  # Criterion for "education" section.
  found_education = re.search(r"education[:]?|graduation[:]?|qualification[:]?", doc)
  if found_education != None:
    indices["education"] = found_education.end()

  # Criterion for "achievements" section.
  found_achievements = re.search(r"achievements|accomplishments", doc)
  if found_achievements != None:
    indices["achievements"] = found_achievements.end()

  # Criterion for "certifications" section.
  found_certifications = re.search(r"certifications", doc)
  if found_certifications != None:
    indices["certifications"] = found_certifications.end()

  # Criterion for "projects" sections.
  found_projects = re.search(r"project[s]?", doc)
  if found_projects != None:
    indices["projects"] = found_projects.end()


  # Each headline will sorted according to their index in the document.
  # The next headline's index will the ending index for the previous headlines.
  # E.g., { education: 45, skills: 89, projects: 129 ... }
  # In this, education section will start from 45 index and will at end at 89,
  # skills will start at 89 and end at 129 and so on.
  indices = sorted(indices.items(), key = lambda kv:(kv[1], kv[0]))

  i = 0
  j = len(indices) - 1

  for item in indices:
    if (i < j):
      # Ending index will contain the previous index's key name, so remove that.
      resume_fields[item[0]] = doc[item[1]:(indices[i + 1][1] - len(indices[i + 1][0]))]
    else:
      # Problem with the last entry. It will include all
      # other fields coming after it, if they have not been explicitly identified.
      resume_fields[item[0]] = doc[item[1]:]
    i += 1

  # Extract the name.
  resume_fields["name"] = extract_name(text)

  # Regex for E-mail.
  email = re.search(r"[\w\d\.\-\#\_\$]+@[\w\d.]+", text)
  resume_fields["email"] = email.group(0) if email != None else ""

  # Regex for phone number.
  phone = re.search(r"([(\+]?\d{1,3}[)]?[- ]?[.]?)?\d{10}", text)
  resume_fields["phone"] = phone.group(0) if phone != None else ""

  # Extract the skills.
  resume_fields["skills"] = extract_skills(text)


  return resume_fields
