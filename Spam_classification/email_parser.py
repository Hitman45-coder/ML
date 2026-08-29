import os

ham_dir = "Spam_classification/data/ham/" 
spam_dir = "Spam_classification/data/spam/"
spam_file_content = []
ham_file_content = []

def extract_file_names(spam_dir: str, ham_dir: str):
    spam_file_names = None 
    non_spam_file_names = None 
    for (root, dirs, files) in os.walk(ham_dir, topdown=True):
        non_spam_file_names = files

    for (root, dirs, files) in os.walk(spam_dir, topdown=True):
        spam_file_names = files
        
    return spam_file_names, non_spam_file_names

def extract_file_content(file_name: str, content: list):
    with open(file_name, 'r', errors="ignore") as file:
        for line in file:
            line = line.strip()
            content.append(line)
    return content


spam_file_names, ham_file_names = extract_file_names(spam_dir, ham_dir)

for i in range(len(spam_file_names)):
    extract_file_content(os.path.join(spam_dir, spam_file_names[i]), spam_file_content )
    extract_file_content(os.path.join(ham_dir, ham_file_names[i]), ham_file_content)