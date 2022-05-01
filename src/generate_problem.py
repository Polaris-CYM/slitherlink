import requests
import numpy as np
from problem_define import slitherlink
from bs4 import BeautifulSoup


def get_string_from_url(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    table_element = soup.find('table', id='LoopTable')
    all_rows = table_element.findAll('tr')
    rows_with_number = all_rows[1::2]  # Converted all rows with numerical restrictions to a list,
                                       # skip rows where the horizontal lines need to be drawn

    overall_limit = ''
    nrow = 0
    for row in rows_with_number:
        nrow += 1
        all_cols = row.findAll('td')
        col_with_number = all_cols[1::2]  # Get numbers in each row (delete the part where the vertical lines need to be drawn)
        row_content = ''
        for col in col_with_number:
            cell_value = col.string
            # cell_value = None if <td align="center"></td>
            # cell_value = x    if <td align="center">x</td>, where x = 0,1,2,3
            if not cell_value:  # cell_value == None
                cell_value = '*'  # Use the * symbol to represent squares without numerical restrictions
            row_content += cell_value
        overall_limit += row_content
    
    return overall_limit, nrow


def generate_problem_from_url(url):
    overall_limit, nrow = get_string_from_url(url)
    while '3' not in overall_limit:  # Guaranteed at least one "3" in the obtained puzzle
        overall_limit, nrow = get_string_from_url(url)

    ncol = int(len(overall_limit) / nrow)
    problem = slitherlink(nrow, ncol, constraint=np.array(list(overall_limit)).reshape(nrow, ncol))
    return problem


if __name__ == "__main__":
    # Get a random puzzle each time size=0--5x5 normal, size=4--5x5 hard, size=1--10x10 normal, size=10--7x7 normal
    problem = generate_problem_from_url(url='http://www.puzzle-loop.com/?v=0&size=0')
    problem.print_problem()
    problem.print_solution()

    
