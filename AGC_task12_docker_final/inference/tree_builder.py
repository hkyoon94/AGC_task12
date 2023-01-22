from xml.etree.ElementTree import XMLParser, TreeBuilder

class CustomTreeBuilder(TreeBuilder):
    def __init__(self, title=''):
        super().__init__()

        self.body_start = False
        self.depth_l = 0
        self.depth_t = 0
        self.lines = 1
        self.lines_t = []
        self.list_item = []
        self.list_header = []
        self.rows = []
        self.cols = []
        self.table_number = 0
        self.leaf_table = False
        self.leaf_lines = False
        
        self.temp_text = ""
        self.temp_lines = []
        self.temp_list = []
        self.temp_table = []
        self.temp_row = []
        self.temp_rowspan = []
        self.temp_colspan = []
        self.temp_caption = []

        self.parsing = {'p': 0,
                        'span': 0,
                        'note': 0,
                        'custom-shape': 0}

        self.custom_shape_concat = False
        
        self.result_dict = {0: {'type': 'text',
                                'value': title}}

    def recursive_reader(self, value_dict):
        for idx in value_dict:
            if value_dict[idx]['type'] == 'text':
                self.cell_text.append(value_dict[idx]['value'])
            elif value_dict[idx]['type'] != 'img':
                self.recursive_reader(value_dict[idx]['value'])

    def to_html(self, table):
        html_string = ""

        html_string = ''.join([html_string, "<table><tbody>"])
        for tr in table:
            html_string = ''.join([html_string, "<tr>"])
            for td in table[tr]:
                cell_tag = "<td"
                if int(table[tr][td]['rowspan']) > 1:
                    cell_tag = ''.join([cell_tag, ' rowspan=\'', table[tr][td]['rowspan'], '\''])
                if int(table[tr][td]['colspan']) > 1:
                    cell_tag = ''.join([cell_tag, ' colspan=\'', table[tr][td]['colspan'], '\''])
                cell_tag = ''.join([cell_tag, '>'])

                html_string = ''.join([html_string, cell_tag])

                self.cell_text = []
                self.recursive_reader(table[tr][td]['value'])

                html_string = ''.join([html_string, '\n'.join(self.cell_text), "</td>"])
            html_string = ''.join([html_string, "</tr>"])
        html_string = ''.join([html_string, "</tbody></table>"])

        return html_string

    def to_line(self):
        self.temp_text = self.temp_text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        
        if self.temp_text.strip() != "":
            if self.depth_t > 0 or self.depth_l > 0:
                self.temp_lines[-1][self.lines_t[-1]] = {'type': 'text',
                                                        'value': self.temp_text.strip()}
                self.lines_t[-1] += 1
            else:
                self.result_dict[self.lines] = {'type': 'text',
                                                'value': self.temp_text.strip()}
                self.lines += 1

            self.temp_text = ""
    
    def start(self, tag, attrs):
        tag = tag.split('}')[1] if '}' in tag else tag

        if tag in self.parsing:
            self.parsing[tag] += 1

            if tag=='custom-shape':
                self.custom_shape_concat=True

        if tag!='custom-shape' and self.parsing['custom-shape']==0 and self.custom_shape_concat:
            self.custom_shape_concat = False
            self.to_line()
        
        if tag=='image':
            for attr in attrs:
                if attr.endswith('href'):
                    if self.depth_t > 0 or self.depth_l > 0:
                        self.temp_lines[-1][self.lines_t[-1]] = {'type':'img',
                                                                 'value': attrs[attr]}
                        self.lines_t[-1] += 1
                    else:
                        self.result_dict[self.lines] = {'type': 'img',
                                                        'value': attrs[attr]}
                        self.lines += 1
        
        elif tag=='line-break':
            self.to_line()

        elif tag in ['tab', 's']:
            self.temp_text = ''.join([self.temp_text, ' '])
            
        elif tag=='list-item':
            self.temp_lines.append({})
            self.lines_t.append(0)

        elif tag=='list-header':
            self.temp_lines.append({})
            self.lines_t.append(0)
            self.list_header.append(0)

        elif tag=='list':
            self.list_item.append(0)
            self.temp_list.append({})
            self.depth_l += 1

        elif tag=='table-cell':
            self.temp_rowspan.append('1')
            self.temp_colspan.append('1')
            for attr in attrs:
                if attr.endswith('number-rows-spanned'):
                    self.temp_rowspan[-1] = attrs[attr]
                elif attr.endswith('number-columns-spanned'):
                    self.temp_colspan[-1] = attrs[attr]
            
            self.temp_lines.append({})
            self.lines_t.append(0)

        elif tag=='table-row':
            self.cols.append(0)
            self.temp_row.append({})

        elif tag=='table':
            self.temp_caption.append('')
            for attr in attrs:
                if attr.endswith('}name'):
                    self.temp_caption[-1] = (attrs[attr])

            self.rows.append(0)
            self.temp_table.append({})
            self.depth_t += 1
            self.leaf_table = True

            if self.leaf_lines:
                self.leaf_lines = False
        
        return TreeBuilder.start(self, tag, attrs)

    def end(self, tag):
        tag = tag.split('}')[1]  if '}' in tag else tag

        if tag == 'automatic-styles':
            self.body_start = True
        
        elif tag=='g':
            self.custom_shape_concat=False
            self.to_line()

        elif (not self.custom_shape_concat) and tag=='p' and self.parsing['note']==0:
            self.to_line()

        elif tag=='list-item':
            self.temp_list[-1][self.list_item[-1]] = {'type': 'list-item',
                                                      'value': self.temp_lines[-1]}
            
            self.list_item[-1] += 1
            self.temp_lines = self.temp_lines[:-1]
            self.lines_t = self.lines_t[:-1]

        elif tag=='list-header':
            self.temp_list[-1][self.list_header[-1]] = {'type': 'list-header',
                                                        'value': self.temp_lines[-1]}
            self.list_header[-1] += 1
            self.temp_lines = self.temp_lines[:-1]
            self.lines_t = self.lines_t[:-1]

        elif tag=='list':
            if self.depth_t > 0 or self.depth_l > 1:
                self.temp_lines[-1][self.lines_t[-1]] = {'type': 'list',
                                                         'value': self.temp_list[-1]}
                self.lines_t[-1] += 1
            else:
                self.result_dict[self.lines] = {'type': 'list',
                                                'value': self.temp_list[-1]}
                self.lines += 1
            
            self.temp_list = self.temp_list[:-1]
            self.depth_l -= 1
            self.list_item = self.list_item[:-1]

        elif tag=='table-cell':
            if self.custom_shape_concat:
                self.to_line()
                
            if self.leaf_lines:
                self.leaf_lines = False

                if len(self.temp_lines) > 0:
                    table_idx = 0
                    
                    for temp_line in self.temp_lines[-1]:
                        if self.temp_lines[-1][temp_line]['type'] == 'table':
                            table_idx = temp_line

                    for temp_line in range(table_idx, len(self.temp_lines[-1])):
                        self.result_dict[self.lines] = self.temp_lines[-1][temp_line]
                        self.lines += 1
            
            self.temp_row[-1][self.cols[-1]] = {'rowspan': self.temp_rowspan[-1],
                                                'colspan': self.temp_colspan[-1],
                                                'value': self.temp_lines[-1]}
            
            self.cols[-1] += 1

            self.temp_rowspan = self.temp_rowspan[:-1]
            self.temp_colspan = self.temp_colspan[:-1]
            self.temp_lines = self.temp_lines[:-1]
            self.lines_t = self.lines_t[:-1]

        elif tag=='table-row':
            self.temp_table[-1][self.rows[-1]] = self.temp_row[-1]

            self.temp_row = self.temp_row[:-1]
            self.rows[-1] += 1
            self.cols = self.cols[:-1]

        elif tag=='table':
            caption = self.temp_caption[-1]
            self.temp_caption = self.temp_caption[:-1]

            if self.leaf_table:
                self.leaf_table = False
                self.leaf_lines = True

                if len(self.temp_lines) > 0:
                    for temp_line in self.temp_lines[-1]:
                        self.result_dict[self.lines] = self.temp_lines[-1][temp_line]
                        self.lines += 1
                    
                    self.lines_t[-1] = 0

                html_string = self.to_html(self.temp_table[-1])
                 
                if self.depth_t > 1 or self.depth_l > 1:
                    self.temp_lines[-1][self.lines_t[-1]] = {'type': 'table',
                                                             'caption': caption,
                                                             'number': self.table_number,
                                                             'html': html_string,
                                                             'value': self.temp_table[-1]}
                    self.lines_t[-1] += 1
                else:
                    self.result_dict[self.lines] = {'type': 'table',
                                                    'caption': caption,
                                                    'number': self.table_number,
                                                    'html': html_string,
                                                    'value': self.temp_table[-1]}
                    self.lines += 1
                
                self.table_number += 1

            self.temp_table = self.temp_table[:-1]
            self.depth_t -= 1
            self.rows = self.rows[:-1]

        if tag in self.parsing:
            self.parsing[tag] -= 1
            
        return TreeBuilder.end(self, tag)

    def data(self, data):
        if self.parsing['span'] > 0 and self.parsing['note']==0:
            self.temp_text = ''.join([self.temp_text, data])
        elif self.parsing['p'] > 0 and self.parsing['note']==0:
            self.temp_text = ''.join([self.temp_text, data])
        
        return TreeBuilder.data(self, data)

    def close(self):
        return self.result_dict