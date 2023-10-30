import os
import numpy as np

# Function to count English characters in a text file
def count_english_characters(filename, char_count):
    with open(filename, 'r') as file:
        content = file.read()
        for char in content:
            if char == '\n':
                continue
            if char in char_count:
                char_count[char] += 1
            else:
                char_count[char] = 1
    return char_count

def factorized_prob(count_list, factor_rate = 0):
    #print('factor_rate', factor_rate)
    total_num = sum(count_list) + factor_rate * len(count_list)
    factored_count_list = [ele + factor_rate for ele in count_list]
    result = [ele / total_num for ele in factored_count_list]
    return result

# Directory where the files are located
directory = r'C:\Users\Zhuoming Liu\Desktop\course_resources\UWM courses\(23fall)CS760\homework\hw4\languageID\languageID'
file_list = os.listdir(directory)

training_list = []
testing_list = []
# filter the trainingfile and the testing file
for file_name in file_list:
    the_real_name = file_name.split('.')[0]
    if the_real_name[-1].isdigit() and the_real_name[-2].isdigit():
        testing_list.append(file_name)
    else:
        training_list.append(file_name)

print('training_list', training_list)
print('testing_list', testing_list)

# all file type
file_starting_characters = ['e', 'j', 's']
# all english character
english_char = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']

# Count and number of files that start with each character
file_counts = {}

# Count the English characters in each kind of the text file
english_char_counts = {}
for char in file_starting_characters:
    char_files = [file for file in training_list if file.startswith(char) and file.endswith('.txt')]
    file_counts[char] = len(char_files)
    char_english_char_counts = {}
    for file in char_files:
        file_path = os.path.join(directory, file)
        char_english_char_counts = count_english_characters(file_path, char_english_char_counts)
    english_char_counts[char] = char_english_char_counts



# calculate the prior
all_file_num = []
for ele in file_starting_characters:
    all_file_num.append(file_counts[ele])
# calculate the prob of prior
prior = factorized_prob(all_file_num, factor_rate = 0.5)
print(prior)

from_file_type_to_prob = {}
for now_file_type, count in file_counts.items():
    now_file_type_char_counts = english_char_counts[now_file_type]
    # obtain the value of each character using a vector
    all_value = []
    for char in english_char:
        if char in now_file_type_char_counts:
            all_value.append(now_file_type_char_counts[char])
        else:
            all_value.append(0)
    # calculate the prob
    print('now_file_type:', now_file_type)
    result_per_file_type = factorized_prob(all_value, factor_rate = 0.5)
    #print('result:', result_per_file_type)
    from_file_type_to_prob[now_file_type] = result_per_file_type
    for char, ele in zip(english_char, result_per_file_type):
        print('character ' + str(char) + ' theta_i: ' + str(ele))

# calculate the log prob of the prior
log_prior_prob = {k: np.log(v) for k, v in zip(file_starting_characters, prior)}    
# calculate log p(x|y)
log_per_char_probs = {}
for label in file_starting_characters:
    log_per_char_probs[label] = {}
    for i, char in enumerate(english_char):
        log_per_char_probs[label][char] = np.log(from_file_type_to_prob[label][i])
    
#################################### do the testing for the q4 #############################################
test_file_path = r'C:\Users\Zhuoming Liu\Desktop\course_resources\UWM courses\(23fall)CS760\homework\hw4\languageID\languageID\e10.txt'  # Path to the test document
with open(test_file_path, 'r') as f:
    test_text = f.read()

log_x_given_y = {label: 0.0 for label in file_starting_characters}
# Calculate log likelihoods for the test document
for label in file_starting_characters:
    for char in test_text:
        if char in english_char:
            log_prior_prob[label] += log_per_char_probs[label][char]

# Determine the predicted class
predicted_class = max(log_prior_prob, key=log_prior_prob.get)

# Print the predicted class
print("Predicted Class:", predicted_class)
print('prob for each class:', log_prior_prob)

#################################### do the prediction for all test file #############################
for label in file_starting_characters:
    # select the test file for each category 
    selected_file_list = [file for file in testing_list if file.startswith(label)]
    print('currect test type:', label, "current testing file list:", selected_file_list)
    
    predicted_result = {key:0 for key in file_starting_characters}
    # for each file do the prediction
    for filename in selected_file_list:
        with open(os.path.join(directory, filename), 'r') as f:
            # load the text file content
            test_text = f.read()
        # do the prediction
        log_x_given_y = {label: 0.0 for label in file_starting_characters}
        # Calculate log likelihoods for the test document
        for label in file_starting_characters:
            for char in test_text:
                if char in english_char:
                    log_x_given_y[label] += log_per_char_probs[label][char]

        # Determine the predicted class
        predicted_class = max(log_x_given_y, key=log_x_given_y.get)
        #print("Predicted Class:", predicted_class)
        #print('prob for each class:', log_x_given_y)        
        
        predicted_result[predicted_class] += 1
    print('the prediction: ', predicted_result)
    
