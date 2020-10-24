import os.path
import random
import numpy as np
import pandas as pd
import xml.dom.minidom
from math import sqrt

from PIL import Image


class Complex(object):

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __str__(self):
        return '(%s, %si)' % (self.a, self.b)

    def __add__(self, rhs):
        a = self.a
        b = self.b
        c = rhs.a
        d = rhs.b
        new_real_part = a + c
        new_imag_part = b + d
        return Complex(new_real_part, new_imag_part)

    def __sub__(self, rhs):
        a = self.a
        b = self.b
        c = rhs.a
        d = rhs.b
        new_real_part = a - c
        new_imag_part = b - d
        return Complex(new_real_part, new_imag_part)

    def __mul__(self, rhs):
        a = self.a
        b = self.b
        c = rhs.a
        d = rhs.b
        new_real_part = (a * c) - (b * d)
        new_imag_part = (a * d) + (c * b)
        return Complex(new_real_part, new_imag_part)


class Calculator(object):

    def __init__(self, z1=None, z2=None, op=None):
        self.z1 = z1
        self.z2 = z2
        self.op = op

    def __call__(self):
        if self.op == '+':
            return self.z1 + self.z2
        elif self.op == '-':
            return self.z1 - self.z2
        elif self.op == '*':
            return self.z1 * self.z2
        else:
            print('Wrong operation sign')

    def parse_oper(self, operation):
        op_elements = operation.split(' ')  # (x1+y1i) * (x2+y2i)
        c1 = self.parse_complex_number(op_elements[0])
        c2 = self.parse_complex_number(op_elements[2])
        op_sign = op_elements[1]
        return Calculator(c1, c2, op_sign)

    @staticmethod
    def parse_complex_number(complex_str):
        bracket_deleter = complex_str[1:-1]
        if '+' in bracket_deleter:
            splitted_op = bracket_deleter.split('+')
            complex_numb = Complex(float(splitted_op[0]), float(splitted_op[1][:-1]))
        else:
            real_minus = False
            if bracket_deleter[0] == '-':
                real_minus = True
                bracket_deleter = bracket_deleter[1:]
            splitted_op = bracket_deleter.split('-')
            real_num = float(splitted_op[0]) if not real_minus else -float(splitted_op[0])
            complex_numb = Complex(real_num, -float(splitted_op[1][:-1]))

        return complex_numb


def data_input():
    x = input("What is your name, surname and year of birth? \n")
    print(x)


def comb_lock():
    code = input("Enter the code: ")
    if code == "2137":
        print("Now you can take your kremówka from the box")
    else:
        print("Wrong code")


def file_count():
    folder_path = 'C:/STUDIA'
    dir_listing = os.listdir(folder_path)

    print(len(dir_listing))


def explore(path):
    files = os.listdir(path)
    for f in files:
        if (os.path.isdir(path + '/' + f)):
            explore(path + '/' + f)
        else:
            print(path + '/' + f)


def jpg_to_png():
    im1 = Image.open(r'C:\STUDIA\VII semestr\Python\obrazki\obrazki do konwersji\jeszcze_nie_rzulta.jpg')
    im1.save(r'C:\STUDIA\VII semestr\Python\obrazki\obrazki skonwertowane\wciaz_nie_rzulta.png')


def words_del():
    file = open('Admiralty, Hong Kong.txt').read()
    words = file.split(' ')
    words_to_delete = ['i', 'oraz', 'nigdy', 'dlaczego']
    punctuation_marks = ['.', ',', '!', '?']
    new_words = []
    for w in words:
        if w == '':
            continue
        if w.lower() not in words_to_delete and not (w[-1] in punctuation_marks and w[:-1].lower() in words_to_delete):
            new_words.append(w)
    new_file = ' '.join(new_words)
    print(new_file)


def replacing_words():
    text = open('Admiralty, Hong Kong.txt').read()

    print("The original string is: " + str(text))

    replaced_text = {"i": "oraz", "oraz": "i", "nigdy": "prawie nigdy", "dalczego": "czemu"}

    temp = text.split()
    res = []
    for wrd in temp:
        res.append(replaced_text.get(wrd, wrd))

    res = ' '.join(res)
    print("Replaced Strings: " + str(res))


def find_roots():
    print("Quadratic function: (a * x^2) + b*x + c")
    a = float(input("a: "))
    b = float(input("b: "))
    c = float(input("c: "))

    r = b ** 2 - 4 * a * c

    if r > 0:
        num_roots = 2
        x1 = (((-b) + sqrt(r)) / (2 * a))
        x2 = (((-b) - sqrt(r)) / (2 * a))
        print("There are 2 roots: %f and %f" % (x1, x2))
    elif r == 0:
        num_roots = 1
        x = (-b) / 2 * a
        print("There is one root: ", x)
    else:
        num_roots = 0
        print("No roots, discriminant < 0.")
        exit()


def sort_numbers():
    randomlist = []
    for i in range(0, 50):
        n = random.randint(1, 1000)
        randomlist.append(n)
    new_list = []
    # randomlist.sort(reverse=True)
    # print(randomlist)

    while randomlist:
        maximum = randomlist[0]
        for x in randomlist:
            if x > maximum:
                maximum = x
        new_list.append(maximum)
        randomlist.remove(maximum)

    print(new_list)


def scalar_prod():
    a = [1, 2, 12, 4]
    b = [2, 4, 2, 8]
    print(np.dot(a, b))


def matrix_sum():
    A = np.random.randint(1, 100, size=(128, 128))
    B = np.random.randint(1, 100, size=(128, 128))
    print(A + B)


def multiply_matrix():
    X = np.random.randint(1, 100, size=(8, 8))
    Y = np.random.randint(1, 100, size=(8, 8))
    result = [[0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]]

    # iterate through rows of X
    for i in range(len(X)):
        for j in range(len(Y[0])):
            for k in range(len(Y)):
                result[i][j] += X[i][k] * Y[k][j]
    for r in result:
        print(r)


def det_matrix():
    X = np.random.randint(1, 10, size=(2, 2))
    print(X)
    det = np.linalg.det(X)
    print(det)


def task_14():
    doc = xml.dom.minidom.parse("example.xml")
    print("Tag name: ", doc.firstChild.tagName)

    doc.firstChild.tagName = "changed_tag"
    with open("example_changed.xml", "w+") as doc_out:
        doc_out.write(doc.toxml())

    doc_changed = xml.dom.minidom.parse("example_changed.xml")
    print("Changed tag name:", doc_changed.firstChild.tagName)


def task_15():
    if os.path.isfile("example.csv"):
        try:
            data = pd.read_csv("example.csv")
            print(data)
            i_del = input("Do you want to delete last record? y/n")
            if i_del == "y":
                data.drop(data.tail(1).index, inplace=True)
                print(data)
                data.to_csv("example.csv", sep=",")
        except:
            print("file is empty, data will be added.")
            data = pd.DataFrame()
    else:
        print("file does not exist, example.csv will be created.")
        data = pd.DataFrame()

    i_in = input("Do you want to add new record? y/n")

    if i_in == "y":
        i1 = input("Enter employee's name: ")
        i2 = input("Enter employee's task: ")
        i3 = input("Enter employee's time limit: ")

        df = pd.DataFrame({"name": [i1],
                           "task": [i2],
                           "time limit": [i3]})

        df = df.append(data)
        df.to_csv('example.csv', index=False)


if __name__ == '__main__':
    # data_input()
    # comb_lock()
    # file_count()
    # explore(os.getcwd())
    # jpg_to_png()
    # words_del()
    # replacing_words()
    # find_roots()
    # sort_numbers()
    # scalar_prod()
    # matrix_sum()
    # multiply_matrix()
    # det_matrix()
    '''x = Complex(10, 5)
    y = Complex(1, 6)
    print(x * y)'''
    '''operation = input("write the whole operation but remember that You need to write it like this: "
                      "'(r1+im1) sign (r2+im2)' but You can write '-' except of '+' in the brackets if You like\n")
    calc = Calculator().parse_oper(operation)
    result = calc()
    print(result)'''
    # task_14()
    task_15()
