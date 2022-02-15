import requests
import numpy as np
from problem_define import slitherlink
from bs4 import BeautifulSoup


def generate_problem_from_url(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    table_element = soup.find('table', id='LoopTable')
    content = table_element.findAll('tr')  
    content_list = content[1::2]  # 转化为list。第0行是线，所以跳过；间隔有线，所以每次跳两个。
    
    overall_limit = ''
    nrow = 0
    for row in content_list:  # 对每一行（有数据的）进行解析
        nrow += 1
        col_content = row.findAll('td')
        col_content_list = col_content[1::2]
        row_content = ''
        for col in col_content_list:
            cell_value = col.string  
            # cell_value = None if <td align="center"></td>
            # cell_value = x    if <td align="center">x</td>, where x = 0,1,2,3
            if not cell_value:  # cell_value == None
                cell_value = '*'  # 用*符号去表示无提示的数（不能用0来表示，因为0具有实际意义）
            row_content += cell_value
        overall_limit += row_content
    
    ncol = int(len(overall_limit) / nrow)
    problem = slitherlink(nrow, ncol, constraint=np.array(list(overall_limit)).reshape(nrow, ncol))
    return problem


if __name__ == "__main__":
    problem = generate_problem_from_url(url='http://www.puzzle-loop.com/?v=0&size=5')
    problem.print_problem()
    problem.print_solution()
