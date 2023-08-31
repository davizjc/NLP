# Number of files to combine
num_files = 4 

with open('allresult.txt', 'w', encoding='utf-8') as combined_file:
    for i in range(1, num_files + 1):
        # Construct the filename
        filename = f"result{i}.txt"
        
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
            for line in lines:
                # Strip the newline at the end and add the prefix
                combined_line = f"{i} {line.strip()}\n"
                combined_file.write(combined_line)
            
            combined_file.write("\n")
