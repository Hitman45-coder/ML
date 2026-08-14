import os

ham_dir = "/home/hitman45/Personal/ML/rep/Spam_classification/data/easy_ham/" 
spam_dir = "/home/hitman45/Personal/ML/rep/Spam_classification/data/spam/"

def extract_file_names(non_spam_dir: str):
    spam_file_names = None  
    non_spam_file_names = None 
    for (root, dirs, files) in os.walk(non_spam_dir, topdown=True):
        non_spam_file_names = files

    for (root, dirs, files) in os.walk(spam_dir, topdown=True):
        spam_file_names = files
        
    return spam_file_names, non_spam_file_names

spam_files, non_spam_files = extract_file_names(ham_dir)

print(spam_files[0])
print()
print(non_spam_files[0])
