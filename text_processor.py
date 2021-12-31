import re
import csv
import nltk
import json

def parse_resume(text):

  nltk.download("punkt")
  nltk.download("words")
  nltk.download("maxent_ne_chunker")
  nltk.download("averaged_perceptron_tagger")
  nltk.download("names")

  # Database of names.
  NAME_DATABASE = []

  # Load the names in a cache.
  with open("./dataset/names.csv", "r") as file:
    reader = csv.reader(file, delimiter=",")
    for row in reader:
      for item in row:
        NAME_DATABASE.append(item.lower())

  resume_fields = {
    "name": "",
    "email": "",
    "phone": ""
  }

  name_grammar = r"NAME: {<NN|NNP>+}"
  chunk_parser = nltk.RegexpParser(name_grammar)

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
    # These lines are not useful, until an NLP type solution is found.
    # lines.append(sent)
    # _tokens = nltk.word_tokenize(sent)
    # tokens.extend(_tokens)
    # pos_tags.append(nltk.pos_tag(_tokens))
    doc += sent

  found_names = []

  # Don't lowercase letter for 'name' field,
  # it helps in rooting out proper nouns (names are generally written as proper nouns).
  name_tokens = chunk_parser.parse(nltk.pos_tag(nltk.word_tokenize(doc)))
  for subtree in name_tokens.subtrees():
    if subtree.label() == "NAME":
      for index, leaf in enumerate(subtree.leaves()):
        if (leaf[0].lower() in NAME_DATABASE and ("NN" in leaf[1] or "NNP" in leaf[1])):
          found_names.append(leaf[0])

  # Since the method is not accurate, it will pick other English type
  # names too, hence only keeping the first entry.
  # Since in most cases, name is present at the top.
  found_names = found_names[0]

  # For mostly regex type text extraction.
  doc = doc.lower()

  # Stores the headlines (if found) of the resume, based on certain pre-defined criterion.
  # Along with headlines, it will store from where the heading starts.
  indices = {}

  # Criterion for "personal information" or "about me" section.
  found_personal_info = re.search(r"personal info[r]?[m]?[a]?[t]?[i]?[o]?[n]?|about me|personal detail[s]?", doc)
  if found_personal_info != None:
    key_name = doc[found_personal_info.start():found_personal_info.end()]
    indices[key_name] = found_personal_info.end()

  # Criterion for "work experience" section.
  found_experience = re.search(r"experience![d]|work experience|work history|work engagement|professional engagement", doc)
  if found_experience != None:
    key_name = doc[found_experience.start():found_experience.end()]
    indices[key_name] = found_experience.end()

  # Criterion for "skills" section.
  found_skills = re.search(r"skill|skill.*[:]?|technologies:", doc)
  if found_skills != None:
    key_name = doc[found_skills.start():found_skills.end()]
    indices[key_name] = found_skills.end()

  # Criterion for "education" section.
  found_education = re.search(r"education[:]?|graduation[:]?|qualification[:]?", doc)
  if found_education != None:
    key_name = doc[found_education.start():found_education.end()]
    indices[key_name] = found_education.end()

  # Criterion for "achievements" section.
  found_achievements = re.search(r"achievements|accomplishments", doc)
  if found_achievements != None:
    key_name = doc[found_achievements.start():found_achievements.end()]
    indices[key_name] = found_achievements.end()

  # Criterion for "certifications" section.
  found_certifications = re.search(r"certifications", doc)
  if found_certifications != None:
    key_name = doc[found_certifications.start():found_certifications.end()]
    indices[key_name] = found_certifications.end()

  # Criterion for "projects" sections.
  found_projects = re.search(r"project", doc)
  if found_projects != None:
    key_name = doc[found_projects.start():found_projects.end()]
    indices[key_name] = found_projects.end()


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

  resume_fields["name"] = found_names

  # Regex for E-mail.
  email = re.search(r"[\w\d\.\-\#\_\$]+@[\w\d.]+", text)
  resume_fields["email"] = email.group(0) if email != None else ""

  # Regex for phone number.
  phone = re.search(r"^([(\+]?\d{1,3}[)]?[- ]?[.]?)?\d{10}$", text)
  resume_fields["phone"] = phone.group(0) if phone != None else ""

  return resume_fields
