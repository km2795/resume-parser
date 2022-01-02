import re
import csv
import nltk
import json


# Extract the skills from the text.
def extract_skills(text, stopwords, tokens):

  # Database of skills.
  SKILLS_DB = []

  # Load the skills in a cache.
  for row in csv.reader(open("skills_db.csv", "r"), delimiter="\n"):
    for item in row:
      SKILLS_DB.append(item.lower())

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
def extract_name(text, pos_tags):

  # Database of names.
  NAMES_DB = []

  # Load the names in a cache.
  for row in csv.reader(open("./dataset/names.csv", "r"), delimiter=","):
    for item in row:
      NAMES_DB.append(item.lower())

  # For name selection.
  name_grammar = r"NAME: {<NN|NNP> <NN|NNP>*}"
  chunk_parser = nltk.RegexpParser(name_grammar)

  found_names = []

  # Don't lowercase letter for 'name' field,
  # it helps in rooting out proper nouns (names are generally written as proper nouns).
  name_tokens = chunk_parser.parse(pos_tags)
  for subtree in name_tokens.subtrees():
    if subtree.label() == "NAME":
      for index, leaf in enumerate(subtree.leaves()):
        if (leaf[0].lower() in NAMES_DB and ("NN" in leaf[1] or "NNP" in leaf[1])):
          found_names.append(leaf[0])

  # It's not very accurate, hence returning only the first
  # entry, since most resume formats have the name at the top.
  # This is just a temporary fix.
  return found_names[0]


# Extract the education.
# It will extract only the name of the institute,
# other information will be skipped.
def extract_education(text):

  # Database of education related terms.
  EDUCATION_DB = [
    "university",
    "school",
    "college",
    "institute",
    "polytechnic",
    "campus"
  ]

  found_education = set()

  # Split the text into sentences, and then
  # search for education related terms in each
  # sentence. If found, add that sentence.
  for sent in text.split("\n"):
    for item in EDUCATION_DB:
      if item in sent.lower():
        found_education.add(sent)

  return list(found_education)


"""
 Extract work experiences.
 Extracts only the names of the places worked.
 It can't extract, college type work places,
 since they would overlap with education database
 (hence are skipped).
"""
def extract_work_experience(text, stopwords):

  # Should contain education type work places.
  # Need a fix for that.
  WORK_EX_DB = [
    "inc", "society", "societies", "forces", "force",
    "conglomerate", "conglomerates", "enterprice",
    "enterprices" "industry", "industries", "service",
    "services", "private limited", "pvt ltd", "pte ltd",
    "technologies"
  ]

  found_work_ex = set()

  # Split the text into sentences in order to find the whole
  # name of the work place. Not doing so, would only add the
  # words present in the database that matched the name in the
  # work place's string.
  for sent in text.split("\n"):
    for item in WORK_EX_DB:
      if item in sent.lower():

        # If certain lines contain stopwords, that means it's
        # not the name of the company, hence don't include them.
        if len([word for word in sent.lower().split(" ") if word in stopwords]) < 1:
          found_work_ex.add(sent.strip())

  return list(found_work_ex)

# Main process to start the extraction.
def parse_resume(text):

  # Load the stopwords.
  stopwords = nltk.corpus.stopwords.words("english")

  # Fetch the word sized tokens from the resume text.
  tokens = nltk.word_tokenize(text)

  # Fetch the POS tags from the word tokens.
  pos_tags = nltk.pos_tag(tokens)

  # Fields that needs to be matched, regardless of resume type.
  resume_fields = {
    "name": "",
    "email": "",
    "phone": "",
    "education": [],
    "projects": [],
    "work_experience": [],
    "achievements": [],
    "certifications": [],
    "hobbies": []
  }

  # Stores the headlines (if found) of the resume, based on certain pre-defined criterion.
  # Along with headlines, it will store from where the heading starts.
  indices = {}

  # Criterion for "achievements" section.
  found_achievements = re.search(r"achievements|accomplishments", text)
  if found_achievements != None:
    indices["achievements"] = found_achievements.end()

  # Criterion for "certifications" section.
  found_certifications = re.search(r"certifications", text)
  if found_certifications != None:
    indices["certifications"] = found_certifications.end()

  # Criterion for "projects" sections.
  found_projects = re.search(r"project[s]?", text)
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
      resume_fields[item[0]] = text[item[1]:(indices[i + 1][1] - len(indices[i + 1][0]))]
    else:
      # Problem with the last entry. It will include all
      # other fields coming after it, if they have not been explicitly identified.
      resume_fields[item[0]] = text[item[1]:]
    i += 1

  # Extract the name.
  resume_fields["name"] = extract_name(text, list(pos_tags))

  # Regex for E-mail.
  email = re.search(r"[\w\d\.\-\#\_\$]+@[\w\d.]+", text)
  resume_fields["email"] = email.group(0) if email != None else ""

  # Regex for phone number.
  phone = re.search(r"([(\+]?\d{1,3}[)]?[- ]?[.]?)?\d{10}", text)
  resume_fields["phone"] = phone.group(0) if phone != None else ""

  # Extract the skills.
  resume_fields["skills"] = extract_skills(text, list(stopwords), list(tokens))

  # Extract the education.
  resume_fields["education"] = extract_education(text)

  # Extract the work experience.
  resume_fields["work_experience"] = extract_work_experience(text, list(stopwords))

  return resume_fields
