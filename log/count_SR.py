def calculate_count(file_t, file_q):
    try:
        with open(file_t, 'r') as file_t, open(file_q, 'r') as file_q:
            lines_t = file_t.readlines()
            lines_q = file_q.readlines()


            count = 0
            min_lines = min(len(lines_t), len(lines_q))  


            for i in range(min_lines):
                try:
                    value_t = float(lines_t[i].strip())  
                    value_q = float(lines_q[i].strip())  


                    if value_t < 2 and value_q < 5:
                        count += 1
                except ValueError:

                    continue

            return count / min_lines

    except FileNotFoundError:
        print("Can't find File")
        return None

file_t = 'error_t.txt'
file_q = 'error_q.txt'

ratio = calculate_count(file_t, file_q)
if ratio is not None:
    print(f"SR ratio: {ratio}")

