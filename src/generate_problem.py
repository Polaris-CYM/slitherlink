import requests
import numpy as np
from problem_define import slitherlink
from bs4 import BeautifulSoup


def get_string_from_url(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    table_element = soup.find('table', id='LoopTable')
    content = table_element.findAll('tr')  
    content_list = content[1::2]  # Converted to list. even rows from row 0 are the area where the lines are drawn,
                                  # so skip them and put numbers in the odd rows.
    
    overall_limit = ''
    nrow = 0
    for row in content_list:  # Parse each row (with data)
        nrow += 1
        col_content = row.findAll('td')
        col_content_list = col_content[1::2]
        row_content = ''
        for col in col_content_list:
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
    problem = generate_problem_from_url(url='http://www.puzzle-loop.com/?v=0&size=0') # Get a random puzzle each time (10*10)
    problem.print_problem()
    problem.print_solution()

    
